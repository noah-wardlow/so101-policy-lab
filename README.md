# SO-101 Policy Lab

Train and run robot manipulation policies for the [SO-101](https://github.com/TheRobotStudio/SO-ARM100)
arm **entirely in the browser**, on a MuJoCo-WASM simulation rendered with
[`mujoco-react`](https://www.npmjs.com/package/mujoco-react). Task: pick up the
red cube and place it on the green target.

[Live](https://so101-policy-lab.nmwardlow.workers.dev/?mode=act&run=false&cams=true&cams3=false)

Two policy backends, selectable in the HUD:

- **Browser ACT** — a LeRobot ACT policy trained on scripted demonstrations,
  exported to ONNX, run **client-side** with `onnxruntime-web`. No backend → the
  app is a static site. Uses three cameras (wrist + front + side).
- **MolmoAct2** — AllenAI's `MolmoAct2-SO100_101` vision-language-action model,
  LoRA-fine-tuned on the same data, served on a GPU and called over HTTP (set
  `VITE_MOLMO_ENDPOINT` or enter the URL in the HUD).

## How it works

```
MujocoProvider → MujocoCanvas (sim) → the selected policy drives ctrl each step
```

- **`src/robot/so101.ts`** — single source of truth: joint order, ctrl ranges,
  gripper, home pose, scene geometry, the camera registry, and the
  sim↔policy-degree conversion. Every robot constant lives here.
- **`src/controllers/ScriptedExpert.tsx`** — the IK "teacher", used **only** to
  generate training data: pocket-aligned grasp → lift → carry → place → retreat
  home, gated by a physics verifier (`src/robot/verifier.ts`). The trained
  policies drive the arm with **no IK** — the real test of the controller.
- **`src/policies/`** — `BrowserActPolicy` (manifest-driven ONNX, one component
  per selected model) and `MolmoPolicy` (remote VLA over HTTP).
- **`src/controllers/PolicyAutoFinish.tsx`** — pauses the sim once the cube is
  placed, so an out-of-distribution policy can't nudge it afterward.

Search params (zod-validated, `src/router.tsx`): `?mode=act|molmo|expert|teleop`,
`?run=1`, `?cams=0`, `?molmo=<url>`.

## Develop

```bash
npm install
npm run dev          # http://localhost:3000
```

Requires `mujoco-react` ≥ 10.7 — its Vite plugin generates the typed model
register (control names etc.) from `public/models/so101/SO101.xml`. The
cross-origin-isolation headers in `vite.config.ts` enable threaded WASM.

## Train your own

```bash
# 1. Record verifier-filtered demos (headless, scripted expert):
node scripts/record.mjs --episodes 90 --out data/raw

# 2. Build a LeRobotDataset (auto-detects cameras; --cameras to subset):
python scripts/build_lerobot_dataset.py --raw data/raw --root data/lerobot --overwrite

# 3. Train ACT (CUDA / Mac MPS / CPU auto-detected) → export ONNX:
bash scripts/train_act.sh
python scripts/export_act_to_onnx.py --policy <checkpoint> --out public/models/act
```

Reload the app — `BrowserActPolicy` loads `public/models/<id>/policy.json`.
`scripts/runpod_*.py` provision a RunPod GPU for ACT / Molmo training (needs
`RUNPOD_API_KEY`); the Molmo server lives in `server/`.

## Deploy

The client is static — host it on Cloudflare Pages (or any static host):

- **App + WASM** → Pages. Add a `public/_headers` file (included) so the
  COOP/COEP headers are set in production (threaded WASM needs them).
- **ONNX models** → object storage (Cloudflare R2 / S3 / a CDN). Each ACT model
  is ~137MB, over Pages' 25 MiB-per-file limit, so they can't be Pages assets.
  Set `VITE_MODEL_BASE=<bucket-url>` and the app loads `<base>/act/policy.json`.
- **MolmoAct2 server** (`server/`) → a GPU host (RunPod). It is Python + a 5B
  model, so it never runs on Cloudflare; the browser just calls its URL. Molmo is
  optional — ACT works with no backend at all.
