#!/usr/bin/env python3
"""Provision a RunPod GPU, train ACT on the local LeRobot dataset, export ONNX,
and pull the model back — then terminate the pod.

  RUNPOD_API_KEY=... python3 scripts/runpod_train.py \
      --dataset data/lerobot --out public/models/browser-act --steps 20000

Requires ~/.ssh/id_ed25519.pub + rsync/ssh. Uses the RunPod REST API.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import runpod_api as rp


def sh(cmd: list[str]) -> None:
    print("$", " ".join(cmd[:6]), "…" if len(cmd) > 6 else "", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/lerobot")
    ap.add_argument("--out", default="public/models/browser-act")
    ap.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fp16", action="store_true", help="also export an fp16 ONNX")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--terminate", metavar="POD_ID")
    args = ap.parse_args()
    if not rp.KEY:
        sys.exit("set RUNPOD_API_KEY")
    if args.terminate:
        rp.delete_pod(args.terminate)
        print(f"terminated {args.terminate}")
        return

    pubkey = Path("~/.ssh/id_ed25519.pub").expanduser().read_text().strip()
    pod = rp.create_pod("so101-act-train", args.gpu, pubkey, ["22/tcp"], disk_gb=40, cloud="COMMUNITY")
    pod_id = pod.get("id")
    print(f"→ deployed pod {pod_id} ({args.gpu})", flush=True)

    try:
        ep = None
        for _ in range(60):
            try:
                ep = rp.ssh_endpoint(pod_id)
            except Exception:  # noqa: BLE001
                pass
            if ep:
                break
            time.sleep(10)
        if not ep:
            raise RuntimeError("pod SSH never came up")
        ip, port = ep
        ssh = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-p", str(port), f"root@{ip}"]
        rsync_e = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {port}"
        print(f"→ SSH {ip}:{port}; waiting for sshd…", flush=True)
        for _ in range(40):
            if subprocess.run([*ssh, "echo ok"], capture_output=True).returncode == 0:
                break
            time.sleep(10)

        sh(["rsync", "-az", "-e", rsync_e, f"{args.dataset}/", f"root@{ip}:/workspace/data/lerobot/"])
        sh(["rsync", "-az", "-e", rsync_e, "scripts/", f"root@{ip}:/workspace/scripts/"])

        fp16flag = "--fp16" if args.fp16 else ""
        remote = f"""set -e
cd /workspace
pip install -q 'lerobot==0.5.2' onnx onnxscript huggingface_hub pillow
export DATASET_ROOT=/workspace/data/lerobot OUTPUT_DIR=/workspace/outputs STEPS={args.steps} BATCH_SIZE={args.batch} DEVICE=cuda
bash scripts/train_act.sh
python scripts/export_act_to_onnx.py --policy /workspace/outputs/checkpoints/last/pretrained_model --out /workspace/browser-act {fp16flag}
"""
        sh([*ssh, remote])

        Path(args.out).mkdir(parents=True, exist_ok=True)
        sh(["rsync", "-az", "-e", rsync_e, f"root@{ip}:/workspace/browser-act/", f"{args.out}/"])
        print(f"→ model pulled to {args.out}", flush=True)
        print(json.dumps({"pod_id": pod_id, "out": args.out}, indent=2))
    finally:
        if not args.keep:
            try:
                rp.delete_pod(pod_id)
                print(f"→ terminated pod {pod_id}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"!! terminate failed for {pod_id}: {e} — terminate manually", flush=True)
        else:
            print(f"→ pod {pod_id} left running (--keep)", flush=True)


if __name__ == "__main__":
    main()
