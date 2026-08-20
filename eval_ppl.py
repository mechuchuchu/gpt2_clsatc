#!/usr/bin/env python
"""
동일한 val.bin 에 대해 pretrained GPT-2 계열(및 내 체크포인트)의 PPL 측정.

train.py 의 evaluate() 와 완전히 동일한 프로토콜:
  - val.bin 을 block_size 단위로 packing (shuffle=False -> 항상 같은 블록, 같은 순서)
  - 1024 토큰 컨텍스트에서 전 위치 next-token CE 평균 (sliding window 아님)

  python eval_ppl.py --val_bin data/fineweb-edu/val.bin --eval_batches 200
  python eval_ppl.py --models gpt2 runs/gpt2-full-2.0B/checkpoints/final <user>/repo
"""

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from data import MemmapBlockDataset


@torch.no_grad()
def eval_one(name, loader, device, ptdtype, max_batches, subfolder=None):
    kw = {"subfolder": subfolder} if subfolder else {}
    model = AutoModelForCausalLM.from_pretrained(name, **kw).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    ctx = (torch.autocast("cuda", dtype=ptdtype)
           if device.startswith("cuda") and ptdtype != torch.float32
           else torch.autocast("cpu", enabled=False))

    tot_loss, tot_tok, t0 = 0.0, 0, time.time()
    for i, (x, y) in enumerate(loader):
        if 0 < max_batches <= i:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with ctx:
            logits = model(input_ids=x).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        tot_loss += loss.double().item() * y.numel()
        tot_tok += y.numel()

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    nll = tot_loss / tot_tok
    return {"model": name + (f"/{subfolder}" if subfolder else ""),
            "params": n_params, "loss": nll, "ppl": math.exp(min(nll, 20.0)),
            "eval_tokens": tot_tok, "sec": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_bin", default="data/fineweb-edu/val.bin")
    ap.add_argument("--models", nargs="+",
                    default=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                    help="HF id 또는 로컬 체크포인트 경로")
    ap.add_argument("--subfolder", default=None, help="예: checkpoints/tok-000500M")
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batches", type=int, default=200, help="-1 이면 val 전체")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out", default=None, help="결과 json 저장 경로")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ptdtype = dict(bfloat16=torch.bfloat16, float16=torch.float16,
                   float32=torch.float32)[args.dtype]

    ds = MemmapBlockDataset(args.val_bin, args.block_size, shuffle=False)
    print(f"[val] {args.val_bin}: {ds.n_tokens/1e6:.1f}M tokens, {ds.n_blocks:,} blocks")

    rows = []
    for name in args.models:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, drop_last=True)
        r = eval_one(name, loader, device, ptdtype, args.eval_batches, args.subfolder)
        rows.append(r)
        print(f"  {r['model']:>40} | {r['params']/1e6:8.1f}M | "
              f"loss {r['loss']:.4f} | ppl {r['ppl']:8.2f} | {r['sec']}s")

    # random-init 상한선 (uniform over vocab)
    print(f"\n{'random init (ln 50257)':>40} | {'-':>9} | loss 10.8248 | ppl 50257.00")

    print(f"\n| model | params | val loss | ppl |\n|---|---:|---:|---:|")
    for r in rows:
        print(f"| `{r['model']}` | {r['params']/1e6:.0f}M | {r['loss']:.4f} | {r['ppl']:.2f} |")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"val_bin": args.val_bin, "block_size": args.block_size,
                       "eval_tokens": rows[0]["eval_tokens"] if rows else 0,
                       "results": rows}, f, indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
