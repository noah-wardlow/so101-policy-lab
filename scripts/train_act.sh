#!/usr/bin/env bash
# Train an ACT policy on the recorded SO-101 pick-place dataset.
# Runs on CUDA (RunPod), MPS (Mac), or CPU. Defaults tuned for a small,
# controlled sim distribution (use_vae=false → clean decoder-only ONNX export).
#
#   PYTHON=.venv/bin/python scripts/train_act.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"
TRAIN_CMD="${TRAIN_CMD:-lerobot-train}"   # console script from `pip install lerobot`
DATASET_REPO_ID="${DATASET_REPO_ID:-local/so101-pickplace}"
DATASET_ROOT="${DATASET_ROOT:-data/lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/so101-act}"
JOB_NAME="${JOB_NAME:-so101_act}"
POLICY_REPO_ID="${POLICY_REPO_ID:-local/so101-act}"
STEPS="${STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CHUNK="${CHUNK:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
DEVICE="${DEVICE:-auto}"

if [[ "${DEVICE}" == "auto" ]]; then
  DEVICE="$("${PYTHON}" - <<'PY'
import torch
print("cuda" if torch.cuda.is_available()
      else "mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available()
      else "cpu")
PY
)"
fi
echo "→ training ACT on device=${DEVICE} steps=${STEPS} batch=${BATCH_SIZE} chunk=${CHUNK}"

"${TRAIN_CMD}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=act \
  --policy.chunk_size="${CHUNK}" \
  --policy.n_action_steps="${N_ACTION_STEPS}" \
  --policy.use_vae=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --policy.push_to_hub=false \
  --policy.device="${DEVICE}" \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --wandb.enable=false

echo "→ trained policy in ${OUTPUT_DIR}/checkpoints/last/pretrained_model"
echo "  export with: ${PYTHON} scripts/export_act_to_onnx.py --policy ${OUTPUT_DIR}/checkpoints/last/pretrained_model --out public/models/browser-act"
