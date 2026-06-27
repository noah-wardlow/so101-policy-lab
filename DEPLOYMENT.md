# Live deployment (RunPod)

## Pod
- `<POD_ID>` — NVIDIA A40 (48GB), SECURE, CA-MTL-1
- SSH: `ssh -p <SSH_PORT> root@<POD_IP>`  (key: ~/.ssh/id_ed25519)
- Only port **8000** is proxied: `https://<POD_ID>-8000.proxy.runpod.net`
- Training venv: `/workspace/venv` (py3.12, lerobot@536b962 + molmoact2 extra + onnx)

## MolmoAct2 — FINE-TUNED, LIVE
- LoRA fine-tuned `allenai/MolmoAct2-SO100_101` on our 52-ep SO-101 dataset (loss 1.16).
- Checkpoint: `/workspace/molmo_ft/checkpoints/last/pretrained_model` (model.safetensors, 11G)
- Served by `server/molmo_ft_server.py` on port 8000 (CORS enabled).
  `CKPT=... DEVICE=cuda CAMERAS=side,front PORT=8000 python molmo_ft_server.py`
- Health: https://<POD_ID>-8000.proxy.runpod.net/health
- Infer:  POST {images:{side,front}, state:[6 sim-deg], task, reset} -> {actions:[30][6 sim-deg]}
- KEY: the FT model speaks OUR sim-degree convention, so the browser uses an
  IDENTITY transform (degrees<->radians only) — no hand-tuned sign/offset hacks.
- Verified: browser reaches the cube precisely + attempts pick-place. ~3s/infer (GPU+net).
- Cloudflare proxy 403s non-browser User-Agents; browser fetch is fine.

## Browser ACT
- Trained on our scripted-expert dataset, exported to ONNX (`public/models/browser-act/`),
  run in-browser via onnxruntime-web (single-thread SIMD WASM).
- CRITICAL FIX: the manifest image-norm stats are nested [3,1,1]; the browser JS
  must flatten them (np.ravel equivalent) or images normalize to garbage -> degenerate
  output. See BrowserActPolicy `flat()`.

## Cleanup (stop billing)
```
ssh -p <SSH_PORT> root@<POD_IP> 'pkill -f molmo_ft_server'   # stop server
RUNPOD_API_KEY=... python3 scripts/runpod_serve_molmo.py --terminate <POD_ID>
# or: curl -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" https://rest.runpod.io/v1/pods/<POD_ID>
```
