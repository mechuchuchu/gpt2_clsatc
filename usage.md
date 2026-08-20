# Random-init GPT-2: Learning Scope Ablation

A direct implementation of the design document (§1–§9). Starting from the **same random initialization**, the experiment progressively expands the set of trainable components — **embedding → +attention → full** — and compares pretraining on 2B tokens from FineWeb-Edu.

```text
modeling.py       GPT2Config/model creation (gpt2 / medium / large / xl / custom)
freeze.py         regime-specific requires_grad settings + parameter statistics by group
data.py           uint16 memmap dataset + streaming packing dataset
prepare_data.py   FineWeb-Edu → train.bin / val.bin tokenization
train.py          training loop + token-based evaluation + HF Hub checkpoint push
plot_results.py   val loss / ppl vs. tokens comparison plots
run_all.sh        run all three regimes
```

## 1. Installation & Data

```bash
pip install -r requirements.txt
huggingface-cli login          # required for pushing to the Hub

# 2.1B train + 10M validation tokens (uint16, ~4.2GB)
python prepare_data.py --train_tokens 2.1e9 --val_tokens 10e6 --out_dir data/fineweb-edu
```

You can also run without writing the dataset to disk using `--data_mode stream`, but **`bin` mode is recommended for comparing the three regimes**, since data ordering is a controlled variable (§4-5).

## 2. Training

```bash
python train.py --regime embedding --model_size gpt2 --max_tokens 2e9 \
  --micro_batch_size 8 --batch_tokens 524288 --lr 6e-4 \
  --eval_interval_tokens 100e6 --eval_at_start \
  --push_to_hub --hub_repo <user>/gpt2-scope-embedding
```

Change `--regime` to `attention` and `full` to run the other two experiments. Alternatively, run all three at once with `bash run_all.sh`.

### Identical Initialization

On the first run, the random initialization state dict is saved to:

```text
init/{model_size}_ctx1024_seed1234.pt
```

Subsequent runs load this file. Therefore, all three regimes start from **bit-level identical initial weights**. Using the same seed would also reproduce the initialization, but sharing the saved file is safer across changes in PyTorch versions.

### Regime Definitions (`freeze.py`)

| Regime      | Trainable parameters                              |
| ----------- | ------------------------------------------------- |
| `embedding` | `wte`, `wpe`, `ln_f` (`lm_head` is tied to `wte`) |
| `attention` | + `ln_1`, `attn`, `ln_2` in every block           |
| `full`      | everything                                        |

Even when a layer is frozen, gradients still propagate through it. Therefore, in the embedding regime, the backward pass still goes through all 12 Transformer blocks. In other words, the speedup comes primarily from reduced optimizer and gradient-storage overhead, not from skipping the forward/backward computation through frozen layers.

For GPT-2 Small, the approximate number of trainable parameters is:

* Embedding: ~39.4M
* Attention: ~67.8M
* Full: ~124M

The exact numbers are printed in the console and saved to `runs/<run>/param_stats.json`.

## 3. Changing Model Size

```bash
--model_size gpt2         # 12L/12H/768
--model_size gpt2-medium  # 24L/16H/1024
--model_size gpt2-large   # 36L/20H/1280
--model_size gpt2-xl      # 48L/25H/1600
--model_size custom --n_layer 16 --n_head 12 --n_embd 768
```

Keep `--batch_tokens` (= tokens per optimizer step) fixed and change only `--micro_batch_size`. Gradient accumulation is automatically recalculated, allowing the optimization hyperparameters to remain consistent when scaling up the model.

Recommended starting points for a single A100 80GB, adjusting as necessary:

| Model       | Micro batch size |     LR | Notes                                         |
| ----------- | ---------------: | -----: | --------------------------------------------- |
| gpt2        |             8–16 |   6e-4 | Default                                       |
| gpt2-medium |              4–8 |   3e-4 |                                               |
| gpt2-large  |              2–4 | 2.5e-4 | `--grad_checkpointing` recommended            |
| gpt2-xl     |              1–2 |   2e-4 | `--grad_checkpointing`, multi-GPU recommended |

## 4. Evaluation & Logging

* Evaluation is performed based on **tokens, not steps** (`--eval_interval_tokens 100e6`): 100M → … → 2B.
* All three runs evaluate on the same batches from the beginning of the same held-out `val.bin`.
* `runs/<run>/metrics.jsonl` is appended to throughout training:

  * `event="train"`: `step`, `tokens`, `train_loss`, `lr`, `tokens_per_sec`, `max_mem_gb`
  * `event="eval"`: `tokens`, `val_loss`, `val_ppl`, `max_mem_gb`
* All items listed in §7 — trainable parameters, throughput, GPU memory, and checkpoints — are recorded here or in `param_stats.json` and `checkpoints/`.

```bash
python plot_results.py \
  --runs runs/gpt2-embedding-2.0B runs/gpt2-attention-2.0B runs/gpt2-full-2.0B \
  --labels embedding "embedding+attention" full --out figs/val_loss.png
```

## 5. HF Hub Push

With `--push_to_hub`, the repository will contain:

```text
<repo>/
  ├── config.json, model.safetensors, tokenizer files   # final checkpoint (root)
  ├── README.md                                        # automatically generated model card
  ├── metrics.jsonl                                    # full training curves
  └── checkpoints/
        ├── tok-000100M/  (+ run_meta.json)
        ├── tok-000200M/
        └── ...
```

* Uploads are performed in the background using `run_as_future=True`, so training is not blocked.
* Use `--push_every_evals 2` to push every 200M tokens, for example, to reduce repository size.
* Optimizer states are not uploaded by default (`--push_optimizer` includes them).
* Locally, `--keep_last N` keeps only the most recent N checkpoints.

Load an intermediate checkpoint with:

```python
from transformers import AutoModelForCausalLM

m = AutoModelForCausalLM.from_pretrained(
    "<user>/<repo>",
    subfolder="checkpoints/tok-000500M",
)
```

## 6. Resume / Distributed Training

```bash
python train.py --regime full --resume --save_optimizer ...

torchrun --standalone --nproc_per_node=4 \
  train.py --regime full --model_size gpt2-large ...
```

When using `bin` mode, resume rotates the data index by the number of blocks already consumed, preserving the data order. Enable `--save_optimizer` if you also want to restore the optimizer state.

## 7. Important Notes

* `--pad_vocab_multiple 64` (50257 → 50304) can improve matmul efficiency, but **the same value must be used across all three regimes** for a valid comparison. The initialization file also depends on the vocabulary size.
* Dropout defaults to 0.0. For 2B-token pretraining, leaving it disabled is the preferred setting.
* `--compile` can improve training speed, but the initial compilation takes time, and the computation graphs differ across regimes, so the compilation cache cannot be shared between them.
* The embedding regime may plateau at a substantially higher loss. When plotting it together with the full model, using `plot_results.py --ymax` to limit the y-axis can make the differences between the curves easier to see.
