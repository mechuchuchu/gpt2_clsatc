#!/usr/bin/env python
"""
세 regime 의 metrics.jsonl 을 읽어 "validation loss vs tokens seen" 를 한 장에 그린다.

  python plot_results.py --runs runs/gpt2-embedding-2.0B runs/gpt2-attention-2.0B runs/gpt2-full-2.0B \
      --out figs/val_loss.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(run_dir, event="eval"):
    p = Path(run_dir) / "metrics.jsonl"
    rows = []
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r.get("event") == event:
                rows.append(r)
    rows.sort(key=lambda r: r["tokens"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="figs/val_loss.png")
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--train_curve", action="store_true", help="train loss 도 함께 표시")
    args = ap.parse_args()

    labels = args.labels or [Path(r).name for r in args.runs]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run, lab in zip(args.runs, labels):
        ev = load(run, "eval")
        x = [r["tokens"] / 1e9 for r in ev]
        axes[0].plot(x, [r["val_loss"] for r in ev], marker="o", ms=3, label=lab)
        axes[1].plot(x, [r["val_ppl"] for r in ev], marker="o", ms=3, label=lab)
        if args.train_curve:
            tr = load(run, "train")
            axes[0].plot([r["tokens"] / 1e9 for r in tr],
                         [r["train_loss"] for r in tr], alpha=0.25, lw=0.8)
        if ev:
            print(f"{lab:>30} | final val_loss {ev[-1]['val_loss']:.4f} "
                  f"| ppl {ev[-1]['val_ppl']:.2f}")

    axes[0].set_xlabel("tokens seen (B)"); axes[0].set_ylabel("validation loss")
    axes[0].set_title("Validation loss vs tokens")
    axes[1].set_xlabel("tokens seen (B)"); axes[1].set_ylabel("perplexity")
    axes[1].set_yscale("log"); axes[1].set_title("Perplexity vs tokens")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend()
        if args.ymax:
            ax.set_ylim(top=args.ymax)
    fig.suptitle("Random-init GPT-2: learning scope ablation (FineWeb-Edu)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
