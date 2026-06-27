# Live deployment

## Web app — Cloudflare Workers + R2
Live: **https://so101-policy-lab.nmwardlow.workers.dev**

A single Worker (`worker.js` + `wrangler.jsonc`) serves the static build from
Workers Assets and the ~137 MB ACT ONNX from R2 — **same origin**, so the
cross-origin-isolation headers (`public/_headers` → COOP/COEP, needed for
threaded WASM) hold without any CORS/CORP juggling. The ONNX is too big for a
static asset (25 MiB/file limit), hence R2 + the fetch handler in `worker.js`.

Deploy from scratch:
```bash
npm run build                                              # → dist/ (incl. 137MB onnx)
wrangler r2 bucket create so101-policy-lab                 # once
wrangler r2 object put so101-policy-lab/act/act.onnx \
  --file dist/models/act/act.onnx \
  --content-type application/octet-stream --remote         # upload model to R2
rm dist/models/act/act.onnx                                # keep it out of Assets (>25MiB)
wrangler deploy                                            # ship Worker + assets
```
Re-deploy after a code change is just `npm run build && rm
dist/models/act/act.onnx && wrangler deploy` (the model only needs re-uploading
when it changes). `VITE_MODEL_BASE` is unset — everything is same-origin under
`/models/act/`.

# RunPod (Molmo / training)

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
