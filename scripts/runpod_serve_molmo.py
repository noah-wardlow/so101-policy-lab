#!/usr/bin/env python3
"""Provision a RunPod GPU and serve MolmoAct2-SO100_101 (server/molmo_server.py).

  RUNPOD_API_KEY=... python3 scripts/runpod_serve_molmo.py            # deploy + start
  RUNPOD_API_KEY=... python3 scripts/runpod_serve_molmo.py --terminate <podId>

Leaves the pod RUNNING and prints the public proxy URL
(https://<podId>-8000.proxy.runpod.net/infer). Terminate when done.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import runpod_api as rp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA A40")
    ap.add_argument("--terminate", metavar="POD_ID")
    args = ap.parse_args()
    if not rp.KEY:
        sys.exit("set RUNPOD_API_KEY")
    if args.terminate:
        rp.delete_pod(args.terminate)
        print(f"terminated {args.terminate}")
        return

    pubkey = Path("~/.ssh/id_ed25519.pub").expanduser().read_text().strip()
    pod = rp.create_pod("molmoact2-so101", args.gpu, pubkey, ["22/tcp", "8000/http"], disk_gb=60)
    pod_id = pod.get("id")
    print(f"→ deployed pod {pod_id} ({args.gpu})", flush=True)
    print("  create response:", json.dumps({k: pod.get(k) for k in ("id", "desiredStatus", "machine")})[:200], flush=True)
    proxy = f"https://{pod_id}-8000.proxy.runpod.net"

    ep = None
    for _ in range(60):
        try:
            ep = rp.ssh_endpoint(pod_id)
        except Exception as e:  # noqa: BLE001
            print("  (waiting)", e)
        if ep:
            break
        time.sleep(10)
    if not ep:
        print(f"!! SSH never came up. Pod {pod_id} running — terminate manually.")
        return
    ip, port = ep
    print(f"→ SSH {ip}:{port}", flush=True)
    ssh = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-p", str(port), f"root@{ip}"]
    rsync_e = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {port}"
    for _ in range(30):
        if subprocess.run([*ssh, "echo ok"], capture_output=True).returncode == 0:
            break
        time.sleep(10)

    subprocess.run(["rsync", "-az", "-e", rsync_e, "server/", f"root@{ip}:/workspace/server/"], check=True)
    remote = """set -e
cd /workspace
pip install -q 'transformers>=4.46' accelerate fastapi 'uvicorn[standard]' pillow einops timm
nohup env MODEL_ID=allenai/MolmoAct2-SO100_101 DEVICE=cuda DTYPE=bf16 PORT=8000 \
  python server/molmo_server.py > /workspace/molmo.log 2>&1 &
sleep 3; echo started
"""
    subprocess.run([*ssh, remote], check=True)
    print(json.dumps({"pod_id": pod_id, "ssh": f"{ip}:{port}",
                      "infer_url": f"{proxy}/infer", "health_url": f"{proxy}/health"}, indent=2))
    print("\nModel download+load ~10-15 min. Poll health_url until {ok:true}.")
    print(f"Terminate: RUNPOD_API_KEY=... python3 scripts/runpod_serve_molmo.py --terminate {pod_id}")


if __name__ == "__main__":
    main()
