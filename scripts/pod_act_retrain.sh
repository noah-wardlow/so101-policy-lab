#!/usr/bin/env bash
# Lean ACT retrain on the big dataset (venv + ffmpeg already set up). Exports ONNX.
set -e
cd /workspace
VENV=/workspace/venv
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-8}"
OUT="${OUT:-/workspace/act_out}"
echo "[act] retrain steps=$STEPS batch=$BATCH"
DATASET_ROOT=/workspace/data/lerobot OUTPUT_DIR="$OUT" STEPS=$STEPS BATCH_SIZE=$BATCH \
  CHUNK=50 N_ACTION_STEPS=50 DEVICE=cuda PYTHON="$VENV/bin/python" TRAIN_CMD="$VENV/bin/lerobot-train" \
  bash scripts/train_act.sh
echo "[act] exporting ONNX..."
"$VENV/bin/python" scripts/export_act_to_onnx.py --policy "$OUT/checkpoints/last/pretrained_model" --out /workspace/browser-act
echo "[act] ACT_DONE"
