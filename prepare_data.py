#!/usr/bin/env python
"""
FineWeb-Edu 를 GPT-2 BPE 로 토크나이즈해서 uint16 memmap(.bin)으로 저장.

  python prepare_data.py --train_tokens 2.1e9 --val_tokens 10e6 \
      --out_dir data/fineweb-edu

- 스트림의 앞부분을 val 로 떼어내고(held-out), 나머지를 train 으로 쓴다.
- 세 regime 이 완전히 동일한 데이터/순서를 보게 하려면 이 방식을 권장.
- 2B 토큰 = 약 4GB (uint16).

TIP: export TOKENIZERS_PARALLELISM=true 로 batch encode 가 멀티스레드로 돈다.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


def write_split(ds_iter, tok, out_path, target_tokens, batch_docs=1000, text_key="text"):
    eos = tok.eos_token_id
    written = 0
    n_docs = 0
    buf_texts = []

    with open(out_path, "wb") as f, tqdm(total=int(target_tokens), unit="tok",
                                         unit_scale=True, desc=out_path.name) as pbar:
        def flush():
            nonlocal written, buf_texts
            if not buf_texts:
                return
            enc = tok(buf_texts)["input_ids"]
            flat = []
            for ids in enc:
                flat.extend(ids)
                flat.append(eos)
            arr = np.asarray(flat, dtype=np.uint16)
            take = min(len(arr), int(target_tokens) - written)
            arr[:take].tofile(f)
            written += take
            pbar.update(take)
            buf_texts = []

        for ex in ds_iter:
            buf_texts.append(ex[text_key])
            n_docs += 1
            if len(buf_texts) >= batch_docs:
                flush()
                if written >= target_tokens:
                    break
        flush()
    return written, n_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--train_tokens", type=float, default=2.1e9,
                    help="학습 예산보다 살짝 크게 잡는 것을 권장 (기본 2.1B)")
    ap.add_argument("--val_tokens", type=float, default=10e6)
    ap.add_argument("--out_dir", default="data/fineweb-edu")
    ap.add_argument("--batch_docs", type=int, default=1000)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_bin, val_bin = out / "train.bin", out / "val.bin"

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok.model_max_length = int(1e9)

    ds = load_dataset(args.dataset, name=args.subset, split=args.split, streaming=True)
    it = iter(ds)

    if val_bin.exists() and train_bin.exists():
        print(f"[skip] {val_bin}, {train_bin} 가 이미 존재합니다.")
        return

    print("== validation split ==")
    v_tokens, v_docs = write_split(it, tok, val_bin, args.val_tokens,
                                   args.batch_docs, args.text_key)
    print("== train split ==")
    t_tokens, t_docs = write_split(it, tok, train_bin, args.train_tokens,
                                   args.batch_docs, args.text_key)

    meta = dict(
        dataset=args.dataset, subset=args.subset, tokenizer=args.tokenizer,
        dtype="uint16", val_tokens=int(v_tokens), train_tokens=int(t_tokens),
        val_docs=v_docs, train_docs=t_docs,
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
