#!/usr/bin/env python
"""
Random-init GPT-2: Comparing Learning Scope Across Transformer Components

세 가지 regime 을 동일한 random init / 데이터 / optimizer / schedule 로 학습:
    embedding  ->  attention  ->  full

사용 예 (single GPU):
    python train.py --regime embedding --model_size gpt2 --max_tokens 2e9 \
        --push_to_hub --hub_repo <user>/gpt2-scope-embedding

DDP:
    torchrun --nproc_per_node=4 train.py --regime full --model_size gpt2-large ...
"""

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from data import MemmapBlockDataset, StreamingPackedDataset, cycle
from freeze import REGIMES, apply_regime, describe_trainable
from modeling import GPT2_ARCHS, build_config, build_model


# --------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---- model
    p.add_argument("--model_size", default="gpt2",
                   choices=list(GPT2_ARCHS) + ["custom"],
                   help="gpt2 / gpt2-medium / gpt2-large / gpt2-xl / custom")
    p.add_argument("--n_layer", type=int, default=None)
    p.add_argument("--n_head", type=int, default=None)
    p.add_argument("--n_embd", type=int, default=None)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--pad_vocab_multiple", type=int, default=1,
                   help="64 로 주면 vocab 50257 -> 50304 (matmul 효율)")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])

    # ---- regime
    p.add_argument("--regime", required=True, choices=list(REGIMES))

    # ---- data
    p.add_argument("--data_mode", default="bin", choices=["bin", "stream"])
    p.add_argument("--train_bin", default="data/fineweb-edu/train.bin")
    p.add_argument("--val_bin", default="data/fineweb-edu/val.bin")
    p.add_argument("--hf_dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--hf_subset", default="sample-10BT")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_data_shuffle", action="store_true",
                   help="큰 bin(수십 GB 이상)에서 random access IO/permutation 메모리를 피하고 "
                        "순차 읽기로 전환. FineWeb sample 자체가 랜덤 샘플이라 무방함")

    # ---- optimization
    p.add_argument("--max_tokens", type=float, default=2e9)
    p.add_argument("--batch_tokens", type=int, default=524288,
                   help="optimizer step 당 토큰 수 (0.5M = 512 seq x 1024)")
    p.add_argument("--micro_batch_size", type=int, default=8,
                   help="GPU 당 forward 배치 크기. grad accum 은 자동 계산됨")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--compile", action="store_true")

    # ---- eval / logging
    p.add_argument("--eval_interval_tokens", type=float, default=100e6)
    p.add_argument("--eval_batches", type=int, default=40)
    p.add_argument("--eval_at_start", action="store_true", help="학습 전 초기 loss 도 측정")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--run_name", default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--init_path", default=None,
                   help="공유 random init 경로. 기본: init/{model_size}_bs{block}_seed{seed}.pt")
    p.add_argument("--save_optimizer", action="store_true")
    p.add_argument("--keep_last", type=int, default=2, help="로컬에 남길 체크포인트 개수(-1=전부)")
    p.add_argument("--resume", action="store_true")

    # ---- hub
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_repo", default=None, help="user/repo. 생략 시 run_name 으로 자동 생성")
    p.add_argument("--hub_private", action="store_true")
    p.add_argument("--push_every_evals", type=int, default=1, help="N번째 eval마다 push")
    p.add_argument("--push_optimizer", action="store_true")

    # ---- wandb (optional)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="gpt2-scope-ablation")

    return p.parse_args()


# --------------------------------------------------------------------------- utils
def get_lr(step, max_steps, warmup_steps, lr, min_lr):
    if step < warmup_steps:
        return lr * (step + 1) / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    r = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * r)) * (lr - min_lr)


def build_optimizer(model, args):
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else nodecay).append(p)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    fused = "cuda" in str(next(model.parameters()).device)
    try:
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2),
                                eps=args.eps, fused=fused)
    except TypeError:
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)
    return opt


@torch.no_grad()
def evaluate(model, val_iter, device, ctx, n_batches, ddp):
    model.eval()
    tot_loss = torch.zeros((), device=device, dtype=torch.float64)
    tot_tok = torch.zeros((), device=device, dtype=torch.float64)
    for _ in range(n_batches):
        x, y = next(val_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with ctx:
            logits = model(input_ids=x).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        tot_loss += loss.double() * y.numel()
        tot_tok += y.numel()
    if ddp:
        dist.all_reduce(tot_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(tot_tok, op=dist.ReduceOp.SUM)
    model.train()
    val_loss = (tot_loss / tot_tok).item()
    return val_loss, math.exp(min(val_loss, 20.0))


def model_card(args, cfg, stats, last):
    tok_b = args.max_tokens / 1e9
    return f"""---
license: mit
library_name: transformers
datasets:
- {args.hf_dataset}
language:
- en
tags:
- gpt2
- from-scratch
- ablation
- interpretability
---

# {args.run_name}

Random-init **{args.model_size}** trained from scratch on **{tok_b:.2f}B tokens** of
{args.hf_dataset}, with the trainable component set restricted to the
**`{args.regime}`** regime.

## Regimes

| regime | trainable |
|---|---|
| `embedding` | `wte`, `wpe`, `ln_f` |
| `attention` | + per-block `ln_1`, `attn`, `ln_2` (MLP frozen) |
| `full`      | all parameters |

All three runs share the **same random initialization**, data order, batch size,
optimizer and LR schedule; the only variable is *which components may learn*.

## This run

- trainable params: **{stats['trainable_params']:,}** / {stats['total_params']:,}
  ({stats['trainable_params']/stats['total_params']:.1%})
- n_layer={cfg.n_layer}, n_head={cfg.n_head}, n_embd={cfg.n_embd}, ctx={cfg.n_positions}
- optimizer: AdamW lr={args.lr}, wd={args.weight_decay}, warmup+cosine, {args.dtype}
- batch: {args.batch_tokens:,} tokens/step
- final val loss: **{last.get('val_loss', float('nan')):.4f}** (ppl {last.get('val_ppl', float('nan')):.2f})

## Layout

Intermediate checkpoints live under `checkpoints/tok-XXXXXXM/`, and the full
metric history is in `metrics.jsonl` (`event=="eval"` rows: `tokens`, `val_loss`,
`val_ppl`, `tokens_per_sec`, `max_mem_gb`).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("{args.hub_repo}")
t = AutoTokenizer.from_pretrained("{args.hub_repo}")
```
"""


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()

    # ---- DDP setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.run_name is None:
        args.run_name = f"{args.model_size}-{args.regime}-{args.max_tokens/1e9:.1f}B"
    run_dir = Path(args.out_dir) / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    if master:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    if args.init_path is None:
        args.init_path = f"init/{args.model_size}_ctx{args.block_size}_seed{args.seed}.pt"

    # ---- batch / step 계산
    tokens_per_micro = args.micro_batch_size * args.block_size
    assert args.batch_tokens % (tokens_per_micro * world_size) == 0, \
        "batch_tokens 는 micro_batch_size*block_size*world_size 의 배수여야 합니다."
    accum = args.batch_tokens // (tokens_per_micro * world_size)
    max_steps = int(args.max_tokens // args.batch_tokens)
    warmup_steps = max(1, int(max_steps * args.warmup_ratio))
    eval_every_steps = max(1, int(args.eval_interval_tokens // args.batch_tokens))

    if master:
        print(f"[cfg] world={world_size} micro_bs={args.micro_batch_size} accum={accum} "
              f"tokens/step={args.batch_tokens:,} steps={max_steps:,} "
              f"warmup={warmup_steps} eval_every={eval_every_steps} steps")

    # ---- model
    cfg = build_config(
        model_size=args.model_size, block_size=args.block_size, dropout=args.dropout,
        pad_vocab_multiple=args.pad_vocab_multiple, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, attn_impl=args.attn_impl,
    )
    model = build_model(cfg, seed=args.seed, init_path=args.init_path, master=master)
    if ddp:
        dist.barrier()
    apply_regime(model, args.regime)
    stats = describe_trainable(model)
    if master:
        print(f"[regime] {args.regime}{stats['table']}")
        (run_dir / "param_stats.json").write_text(
            json.dumps({k: v for k, v in stats.items() if k != "table"}, indent=2))

    model.to(device)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # ---- resume
    start_step, tokens_seen = 0, 0
    if args.resume:
        cands = sorted(ckpt_dir.glob("tok-*"))
        if cands:
            last_ckpt = cands[-1]
            sd = torch.load(last_ckpt / "train_state.pt", map_location="cpu")
            from transformers import GPT2LMHeadModel
            loaded = GPT2LMHeadModel.from_pretrained(last_ckpt)
            model.load_state_dict(loaded.state_dict())
            model.to(device)
            apply_regime(model, args.regime)
            start_step, tokens_seen = sd["step"], sd["tokens_seen"]
            if master:
                print(f"[resume] {last_ckpt} step={start_step} tokens={tokens_seen:,}")
        else:
            sd = None
    else:
        sd = None

    raw_model = model
    if args.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    optimizer = build_optimizer(raw_model, args)
    if sd is not None and "optimizer" in sd:
        optimizer.load_state_dict(sd["optimizer"])
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16"))
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    ptdtype = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32)[args.dtype]
    ctx = (torch.autocast(device_type="cuda", dtype=ptdtype)
           if device.startswith("cuda") and args.dtype != "float32"
           else torch.autocast(device_type="cpu", enabled=False))

    # ---- data
    if args.data_mode == "bin":
        train_ds = MemmapBlockDataset(args.train_bin, args.block_size, seed=args.seed,
                                      shuffle=not args.no_data_shuffle)
        val_ds = MemmapBlockDataset(args.val_bin, args.block_size, seed=args.seed, shuffle=False)
        if start_step > 0:
            train_ds.rotate(tokens_seen // args.block_size)
        train_sampler = DistributedSampler(train_ds, shuffle=False, drop_last=True) if ddp else None
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=True) if ddp else None
        train_loader = DataLoader(train_ds, batch_size=args.micro_batch_size, shuffle=False,
                                  sampler=train_sampler, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
        val_loader = DataLoader(val_ds, batch_size=args.micro_batch_size, shuffle=False,
                                sampler=val_sampler, num_workers=1, pin_memory=True, drop_last=True)
        if master:
            print(f"[data] train {train_ds.n_tokens/1e9:.2f}B tok, val {val_ds.n_tokens/1e6:.1f}M tok")
    else:
        train_ds = StreamingPackedDataset(args.hf_dataset, args.hf_subset, "train", args.tokenizer,
                                          args.block_size, rank, world_size, skip_docs=5000)
        val_ds = StreamingPackedDataset(args.hf_dataset, args.hf_subset, "train", args.tokenizer,
                                        args.block_size, rank, world_size, skip_docs=0)
        train_loader = DataLoader(train_ds, batch_size=args.micro_batch_size,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.micro_batch_size,
                                num_workers=1, pin_memory=True, drop_last=True)
        train_sampler = val_sampler = None
        if master:
            print("[data] streaming mode: 데이터 순서 재현성은 world_size/num_workers 에 의존합니다.")

    train_iter = cycle(train_loader, train_sampler)
    val_iter = cycle(val_loader, val_sampler)

    # ---- hub
    api = repo_id = None
    futures = []
    if args.push_to_hub and master:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_id = args.hub_repo or f"{api.whoami()['name']}/{args.run_name}"
        args.hub_repo = repo_id
        api.create_repo(repo_id, private=args.hub_private, exist_ok=True, repo_type="model")
        print(f"[hub] {repo_id}")

    # ---- wandb
    wb = None
    if args.wandb and master:
        import wandb as wb_
        wb = wb_
        wb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    metrics_path = run_dir / "metrics.jsonl"

    def log_metric(rec):
        if not master:
            return
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if wb is not None:
            wb.log(rec, step=rec.get("step", 0))

    def save_and_push(step, tokens, last_metrics, final=False):
        if not master:
            return
        tag = "final" if final else f"tok-{tokens//10**6:06d}M"
        d = ckpt_dir / tag
        d.mkdir(parents=True, exist_ok=True)
        raw_model.save_pretrained(d, safe_serialization=True)
        tokenizer.save_pretrained(d)
        (d / "run_meta.json").write_text(json.dumps({
            "regime": args.regime, "model_size": args.model_size, "step": step,
            "tokens_seen": tokens, **last_metrics,
            "trainable_params": stats["trainable_params"],
            "total_params": stats["total_params"],
        }, indent=2))
        if args.save_optimizer:
            torch.save({"optimizer": optimizer.state_dict(), "step": step,
                        "tokens_seen": tokens}, d / "train_state.pt")
        else:
            torch.save({"step": step, "tokens_seen": tokens}, d / "train_state.pt")

        if api is not None:
            ignore = None if args.push_optimizer else ["train_state.pt"]
            futures.append(api.upload_folder(
                repo_id=repo_id, folder_path=str(d),
                path_in_repo=("." if final else f"checkpoints/{tag}"),
                ignore_patterns=ignore, run_as_future=True,
                commit_message=f"{args.regime} @ {tokens/1e9:.3f}B tokens"))
            if metrics_path.exists():
                futures.append(api.upload_file(
                    path_or_fileobj=str(metrics_path), path_in_repo="metrics.jsonl",
                    repo_id=repo_id, run_as_future=True))
            if final:
                card = run_dir / "README.md"
                card.write_text(model_card(args, cfg, stats, last_metrics))
                futures.append(api.upload_file(
                    path_or_fileobj=str(card), path_in_repo="README.md",
                    repo_id=repo_id, run_as_future=True))

        # 로컬 디스크 정리
        if args.keep_last > 0:
            saved = sorted([p for p in ckpt_dir.glob("tok-*") if p.is_dir()])
            for old in saved[:-args.keep_last]:
                shutil.rmtree(old, ignore_errors=True)

    # ---- initial eval
    if args.eval_at_start and start_step == 0:
        vl, vp = evaluate(model, val_iter, device, ctx, args.eval_batches, ddp)
        if master:
            print(f"[eval] step 0 | tokens 0 | val_loss {vl:.4f} | ppl {vp:.2f}")
        log_metric({"event": "eval", "step": 0, "tokens": 0, "val_loss": vl, "val_ppl": vp})

    # ---- train loop
    model.train()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    window_tokens = 0
    last_metrics = {}

    for step in range(start_step, max_steps):
        lr = get_lr(step, max_steps, warmup_steps, args.lr, args.lr * args.min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = lr

        loss_accum = torch.zeros((), device=device)
        for micro in range(accum):
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if ddp:
                model.require_backward_grad_sync = (micro == accum - 1)
            with ctx:
                logits = model(input_ids=x).logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
            loss_accum += loss.detach() / accum
            scaler.scale(loss / accum).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad], args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen += args.batch_tokens
        window_tokens += args.batch_tokens

        # ---- logging
        if (step + 1) % args.log_interval == 0:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = time.time() - t0
            tps = window_tokens / max(dt, 1e-6)
            if ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0.0
            if master:
                print(f"step {step+1:>6}/{max_steps} | tok {tokens_seen/1e9:6.3f}B | "
                      f"loss {loss_accum.item():.4f} | lr {lr:.2e} | "
                      f"{tps/1e3:7.1f}k tok/s | mem {mem:.1f}GB")
            log_metric({"event": "train", "step": step + 1, "tokens": tokens_seen,
                        "train_loss": loss_accum.item(), "lr": lr,
                        "tokens_per_sec": tps, "max_mem_gb": mem})
            t0, window_tokens = time.time(), 0

        # ---- eval + checkpoint
        is_last = (step + 1) == max_steps
        if (step + 1) % eval_every_steps == 0 or is_last:
            vl, vp = evaluate(model, val_iter, device, ctx, args.eval_batches, ddp)
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0.0
            last_metrics = {"train_loss": loss_accum.item(), "val_loss": vl, "val_ppl": vp,
                            "max_mem_gb": mem}
            if master:
                print(f"[eval] step {step+1} | tokens {tokens_seen/1e9:.3f}B | "
                      f"val_loss {vl:.4f} | ppl {vp:.2f}")
            log_metric({"event": "eval", "step": step + 1, "tokens": tokens_seen,
                        "regime": args.regime, **last_metrics})

            n_eval = (step + 1) // eval_every_steps
            if is_last or n_eval % args.push_every_evals == 0:
                save_and_push(step + 1, tokens_seen, last_metrics, final=is_last)
            t0, window_tokens = time.time(), 0

    if master:
        print(f"[done] {args.run_name} | tokens {tokens_seen/1e9:.3f}B | {last_metrics}")
        for f in futures:  # hub 업로드 완료 대기
            try:
                f.result()
            except Exception as e:
                print(f"[hub] upload failed: {e}")
        if wb is not None:
            wb.finish()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
