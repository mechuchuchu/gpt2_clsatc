#!/usr/bin/env bash
# 세 regime 을 순서대로 학습 (동일 init / 동일 데이터 순서 / 동일 schedule)
set -euo pipefail

MODEL=${MODEL:-gpt2}              # gpt2 | gpt2-medium | gpt2-large | gpt2-xl
TOKENS=${TOKENS:-2e9}
MICRO_BS=${MICRO_BS:-8}           # OOM 나면 줄이고, batch_tokens 는 그대로 두면 됨
BATCH_TOKENS=${BATCH_TOKENS:-524288}
LR=${LR:-6e-4}
HF_USER=${HF_USER:-your-hf-username}
SEED=${SEED:-1234}
NPROC=${NPROC:-1}

if [ "$NPROC" -gt 1 ]; then
  LAUNCH="torchrun --standalone --nproc_per_node=$NPROC"
else
  LAUNCH="python"
fi

# 0) 데이터 준비 (한 번만)
python prepare_data.py --train_tokens 2.1e9 --val_tokens 10e6 --out_dir data/fineweb-edu

# 1) 세 regime 학습
for REGIME in embedding attention full; do
  RUN="${MODEL}-${REGIME}-$(python -c "print('%.1f' % (float('$TOKENS')/1e9))")B"
  echo "=== ${RUN} ==="
  $LAUNCH train.py \
    --regime "$REGIME" \
    --model_size "$MODEL" \
    --max_tokens "$TOKENS" \
    --batch_tokens "$BATCH_TOKENS" \
    --micro_batch_size "$MICRO_BS" \
    --lr "$LR" \
    --seed "$SEED" \
    --eval_interval_tokens 100e6 \
    --eval_at_start \
    --dtype bfloat16 \
    --run_name "$RUN" \
    --push_to_hub --hub_repo "${HF_USER}/${RUN}"
done

# 2) 비교 플롯
python plot_results.py \
  --runs runs/${MODEL}-embedding-*B runs/${MODEL}-attention-*B runs/${MODEL}-full-*B \
  --labels embedding "embedding+attention" full \
  --out figs/${MODEL}_val_loss.png
