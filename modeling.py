"""
GPT-2 계열 모델 생성 유틸.

- gpt2 / gpt2-medium / gpt2-large / gpt2-xl 를 이름만 바꿔서 사용 가능
- custom 사이즈도 --n_layer / --n_head / --n_embd 로 지정 가능
- 세 regime(embedding / attention / full)이 "완전히 동일한 random init"에서
  출발하도록, 초기화 state_dict 를 파일로 저장/재사용한다.
"""

from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

# HuggingFace 이름 -> 아키텍처 하이퍼파라미터
GPT2_ARCHS = {
    "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),    # ~124M
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),   # ~355M
    "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),   # ~774M
    "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),   # ~1.56B
}


def build_config(
    model_size="gpt2",
    block_size=1024,
    vocab_size=50257,
    dropout=0.0,
    pad_vocab_multiple=1,
    n_layer=None,
    n_head=None,
    n_embd=None,
    attn_impl="sdpa",
):
    """GPT2Config 를 만든다. model_size='custom' 이면 n_layer/n_head/n_embd 필수."""
    if model_size == "custom":
        assert None not in (n_layer, n_head, n_embd), \
            "model_size=custom 이면 --n_layer --n_head --n_embd 를 모두 지정해야 합니다."
        arch = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd)
    else:
        arch = dict(GPT2_ARCHS[model_size])
        # 개별 override 허용
        for k, v in [("n_layer", n_layer), ("n_head", n_head), ("n_embd", n_embd)]:
            if v is not None:
                arch[k] = v

    if pad_vocab_multiple > 1:
        rem = vocab_size % pad_vocab_multiple
        if rem:
            vocab_size += pad_vocab_multiple - rem  # matmul 효율용 padding (미사용 row)

    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=block_size,
        n_embd=arch["n_embd"],
        n_layer=arch["n_layer"],
        n_head=arch["n_head"],
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        bos_token_id=50256,
        eos_token_id=50256,
        use_cache=False,  # 학습 중엔 불필요
    )
    try:
        cfg._attn_implementation = attn_impl  # "sdpa" | "eager" | "flash_attention_2"
    except Exception:
        pass
    return cfg


def build_model(cfg, seed=1234, init_path=None, master=True):
    """
    동일 seed -> 동일 random init.
    init_path 가 주어지고 파일이 있으면 그걸 로드(= 세 regime 간 bit-level 동일 보장),
    없으면 master rank 가 저장한다.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = GPT2LMHeadModel(cfg)

    if init_path is not None:
        p = Path(init_path)
        if p.exists():
            sd = torch.load(p, map_location="cpu")
            model.load_state_dict(sd, strict=True)
            print(f"[init] loaded shared random init from {p}")
        elif master:
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), p)
            print(f"[init] saved shared random init to {p}")

    # GPT-2 는 wte 와 lm_head 가 tied 되어 있다 (확인용)
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr(), \
        "lm_head 와 wte 가 tied 되어 있지 않습니다."
    return model


def n_params(model, non_embedding=False):
    n = sum(p.numel() for p in model.parameters())
    if non_embedding:
        n -= model.transformer.wpe.weight.numel()
        n -= model.transformer.wte.weight.numel()
    return n
