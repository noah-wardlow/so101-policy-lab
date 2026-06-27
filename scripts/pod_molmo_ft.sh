#!/usr/bin/env bash
# Runs ON the pod: LoRA fine-tune MolmoAct2-SO100_101 on our SO-101 dataset.
# Args: STEPS BATCH OUTPUT_DIR  (env overridable)
set -e
cd /workspace
VENV=/workspace/venv
STEPS="${STEPS:-4000}"
BATCH="${BATCH:-4}"
OUT="${OUT:-/workspace/molmo_ft}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/data/lerobot}"

echo "[molmo-ft] steps=$STEPS batch=$BATCH out=$OUT dataset=$DATASET_ROOT"
"$VENV/bin/lerobot-train" \
  --dataset.repo_id=local/so101-pickplace \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=molmoact2 \
  --policy.checkpoint_path=allenai/MolmoAct2-SO100_101 \
  --policy.norm_tag=so100_so101_molmoact2 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.normalize_gripper=true \
  --policy.enable_lora_vlm=true \
  --policy.enable_lora_action_expert=false \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name=so101_molmo_ft \
  --steps="$STEPS" \
  --batch_size="$BATCH" \
  --save_freq=2000 \
  --wandb.enable=false
echo "[molmo-ft] DONE -> $OUT/checkpoints/last/pretrained_model"
