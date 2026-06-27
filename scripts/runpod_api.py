"""Minimal RunPod REST API helpers (the GraphQL deploy mutations are blocked)."""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

BASE = "https://rest.runpod.io/v1"
KEY = os.environ.get("RUNPOD_API_KEY", "")
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def _req(method: str, path: str, body: dict | None = None) -> dict | list:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} -> {e.code}: {e.read().decode()[:400]}") from e


def create_pod(name: str, gpu_type: str, pubkey: str, ports: list[str],
               disk_gb: int = 60, cloud: str = "SECURE") -> dict:
    body = {
        "name": name,
        "imageName": IMAGE,
        "gpuTypeIds": [gpu_type],
        "gpuCount": 1,
        "cloudType": cloud,
        "computeType": "GPU",
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ports,
        "supportPublicIp": True,
        "env": {"PUBLIC_KEY": pubkey},
    }
    return _req("POST", "/pods", body)


def get_pod(pod_id: str) -> dict:
    return _req("GET", f"/pods/{pod_id}")


def delete_pod(pod_id: str) -> None:
    _req("DELETE", f"/pods/{pod_id}")


def ssh_endpoint(pod_id: str):
    """Return (ip, port) for the public TCP 22 mapping, or None if not ready.
    RunPod exposes this as pod.publicIp + pod.portMappings["22"]."""
    pod = get_pod(pod_id)
    ip = pod.get("publicIp")
    mappings = pod.get("portMappings") or {}
    port = mappings.get("22")
    if ip and port:
        return ip, int(port)
    return None
