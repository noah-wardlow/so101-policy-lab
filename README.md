# SO-101 Policy Lab

Train and run robot manipulation policies for the [SO-101](https://github.com/TheRobotStudio/SO-ARM100)
arm **entirely in the browser**, on a MuJoCo-WASM simulation rendered with
[`mujoco-react`](https://www.npmjs.com/package/mujoco-react). Task: pick up the
red cube and place it on the green target.

**▶ [Live demo](https://so101-policy-lab.nmwardlow.workers.dev)** — the 3-cam ACT
policy running entirely in your browser (first load fetches a ~137 MB ONNX, then
hit **Run**).

Two policy backends, selectable in the HUD:

- **Browser ACT** — a LeRobot ACT policy trained on scripted demonstrations,
  exported to ONNX, run **client-side** with `onnxruntime-web`. No backend → the
  app is a static site. Uses three cameras (wrist + front + side).
- **MolmoAct2** — AllenAI's `MolmoAct2-SO100_101` vision-language-action model,
  LoRA-fine-tuned on the same data, served on a GPU and called over HTTP (set
  `VITE_MOLMO_ENDPOINT` or enter the URL in the HUD).

### Molmo: camera count matters — use 2, not 3

Molmo works **well** in `MolmoAct2-SO100_101`'s pretrained camera configuration —
**2 cameras (wrist + front)** — where a light LoRA fine-tune (even on a small
dataset) grasps reliably. Adding a **3rd camera (side)** to match the ACT setup
**regressed it badly** — twitchy, near-zero grasps — even though training loss
still converged (loss is computed on the training data and doesn't reveal a
mishandled extra input the base model wasn't pretrained for).

So: **the 3-camera setup is for ACT** (there it's the *more* reliable model);
**Molmo should stay on 2 cameras (wrist + front)**. To restore the good Molmo,
re-fine-tune on the `wrist,front` subset of the dataset
(`build_lerobot_dataset.py --cameras wrist,front`) and serve with
`CAMERAS=wrist,front`; `MolmoPolicy` then sends those two views.

Independent of camera count, Molmo is **slow + bursty by construction**:
~3s/inference over the network, run receding-horizon (infer a chunk → replay →
replace), so the arm moves in stop-start bursts. That's inherent to a big remote
VLA and no client change removes it — it's a tradeoff against ACT's free,
fast, in-browser execution.

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
