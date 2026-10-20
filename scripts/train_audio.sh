#!/usr/bin/env bash
# Run from the repository root. Extra Trainer/Moss arguments may follow OUT_DIR.
set -euo pipefail
if (( $# < 4 )); then
  echo "Usage: bash scripts/train_audio.sh {pretrain|sft} MODEL MANIFEST OUT_DIR [extra arguments]" >&2
  exit 2
fi
stage=$1
model_path=$2
manifest_path=$3
output_path=$4
shift 4
case "$stage" in
  pretrain) stage_args=(--learning_rate 4e-4 --max_steps 40000) ;;
  sft) stage_args=(--learning_rate 5e-5 --num_train_epochs 1) ;;
  *) echo "Stage must be pretrain or sft" >&2; exit 2 ;;
esac
python -m "ntpp.train.$stage" \
  --model_name_or_path "$model_path" \
  --train_file "$manifest_path" \
  --output_dir "$output_path" \
  --use_moss_audio True \
  --moss_device cuda --dataloader_num_workers 0 \
  --moss_model_name_or_path youspeech/moss_nano_trainer_unofficial \
  --moss_num_codebooks 4 --block_size 1024 \
  --do_train --do_eval --bf16 True \
  --gradient_checkpointing True \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 64 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr": 4e-6}' \
  --eval_strategy steps --eval_steps 1000 \
  --save_steps 1000 --save_total_limit 3 \
  --validation_split_percentage 5 \
  --max_eval_samples 10 \
  --logging_steps 10 --report_to tensorboard \
  --logging_dir "$output_path/tensorboard" \
  "${stage_args[@]}" "$@"
