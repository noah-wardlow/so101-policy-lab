#!/usr/bin/env bash
# Runs ON the pod (nohup): wait for the Molmo FT to finish saving, then serve it.
# Lets the FT->serve handoff happen without babysitting the ~hours-long training.
cd /workspace
LOG="${LOG:-/workspace/molmo_ft.log}"
while ! grep -qiE "molmo-ft. DONE" "$LOG" 2>/dev/null; do
  if grep -qiE "No space left|Error while|Traceback|out of memory" "$LOG" 2>/dev/null; then
    echo "FT_FAILED" > /workspace/autoserve.status; exit 1
  fi
  sleep 60
done
# FT exits after DONE, freeing the GPU; give it a moment, then serve.
sleep 20
echo "SERVING" > /workspace/autoserve.status
CKPT=/workspace/molmo_ft/checkpoints/last/pretrained_model \
  CAMERAS=wrist,front,side PORT=8000 DEVICE=cuda \
  /workspace/venv/bin/python /workspace/server/molmo_ft_server.py >> /workspace/molmo_server.log 2>&1
