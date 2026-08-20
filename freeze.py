"""
학습 범위(regime) 별 파라미터 freeze 로직.

  embedding : wte, wpe, ln_f 만 학습
  attention : + 각 block 의 ln_1, attn, ln_2 학습 (MLP freeze)
  full      : 전체 학습

주의: freeze 되어 있어도 gradient 는 그 층을 "통과"해서 흐른다.
      (embedding regime 이라도 backward 는 12개 block 전체를 거친다)
"""

from collections import OrderedDict

REGIMES = ("embedding", "attention", "full")

# regime 과 무관하게 항상 학습되는 것들
ALWAYS_TRAINABLE_PREFIX = (
    "transformer.wte",
    "transformer.wpe",
    "transformer.ln_f",
    "lm_head",  # wte 와 tied
)


def _is_attention_group(name: str) -> bool:
    """block 내부에서 attention regime 에 포함되는 파라미터."""
    return (".attn." in name) or (".ln_1." in name) or (".ln_2." in name)


def apply_regime(model, regime: str):
    if regime not in REGIMES:
        raise ValueError(f"unknown regime: {regime} (choices: {REGIMES})")

    for name, p in model.named_parameters():
        if regime == "full":
            p.requires_grad_(True)
            continue
        base = name.startswith(ALWAYS_TRAINABLE_PREFIX)
        if regime == "embedding":
            p.requires_grad_(bool(base))
        else:  # attention
            p.requires_grad_(bool(base or _is_attention_group(name)))
    return model


def _group_of(name: str) -> str:
    if "wte" in name:
        return "token_embedding(wte)"
    if "wpe" in name:
        return "pos_embedding(wpe)"
    if ".attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    if "ln_f" in name:
        return "ln_f"
    if "ln_1" in name or "ln_2" in name:
        return "block_layernorm"
    return "other"


def describe_trainable(model):
    """그룹별 trainable / total 파라미터 수 요약."""
    stats = OrderedDict()
    for name, p in model.named_parameters():
        g = _group_of(name)
        d = stats.setdefault(g, {"total": 0, "trainable": 0})
        d["total"] += p.numel()
        if p.requires_grad:
            d["trainable"] += p.numel()

    total = sum(v["total"] for v in stats.values())
    trainable = sum(v["trainable"] for v in stats.values())

    lines = ["", f"{'group':>22} | {'trainable':>14} | {'total':>14} | ratio"]
    lines.append("-" * 70)
    for g, v in stats.items():
        r = v["trainable"] / max(v["total"], 1)
        lines.append(f"{g:>22} | {v['trainable']:>14,} | {v['total']:>14,} | {r:6.1%}")
    lines.append("-" * 70)
    lines.append(f"{'TOTAL':>22} | {trainable:>14,} | {total:>14,} | {trainable/total:6.1%}")
    lines.append("")

    return {
        "trainable_params": trainable,
        "total_params": total,
        "by_group": {k: v for k, v in stats.items()},
        "table": "\n".join(lines),
    }
