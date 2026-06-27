#!/usr/bin/env bash
# Rebuild the training venv on a fresh RunPod pod. Run ON the pod.
# The pod has no persistent volume, so /workspace is wiped on every stop — this
# reconstructs the env from scratch. Keep datasets LOCAL (off-pod) to avoid loss.
set -e
cd /workspace
# ffmpeg shared libs are required by torchcodec (LeRobot's video backend) to
# decode the dataset's video-encoded camera frames during training.
apt-get update -qq && apt-get install -y -qq ffmpeg
rm -rf /workspace/venv
# lerobot requires Python >=3.12, but the RunPod image ships 3.11. Use uv to fetch
# a 3.12 build and create the venv with it (fast + reliable, no apt PPA needed).
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv python install 3.12
uv venv --python 3.12 /workspace/venv
PIP=(uv pip install --python /workspace/venv/bin/python)
"${PIP[@]}" -U pip wheel
# MolmoAct2 lives in mainline LeRobot behind the `molmoact2` extra. Pin the
# known-good commit so the train flags in pod_molmo_ft.sh stay compatible.
"${PIP[@]}" "lerobot[molmoact2,dataset] @ git+https://github.com/huggingface/lerobot.git@536b962"
# ONNX export deps (used by export_act_to_onnx.py; harmless for Molmo-only runs).
"${PIP[@]}" onnx onnxruntime
echo "[setup] DONE -> /workspace/venv/bin/lerobot-train"
venv/bin/lerobot-train --help >/dev/null 2>&1 && echo "[setup] lerobot-train OK" || echo "[setup] WARN: lerobot-train smoke failed"
