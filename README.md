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

For this task, **ACT clearly wins** — it grasps reliably (~7/8) and runs free in
the browser. The fine-tuned Molmo is included as a comparison and is, honestly,
the weaker option here (see below).

### Molmo: status, limitations, next steps

A light LoRA fine-tune (4,000 steps, 101 sim episodes, loss → 0.087) of a 5B
general VLA does **not** match the task-specific ACT on precise grasping:

- **Twitchy, low grasp rate.** The action *scale* is correct (the server
  denormalizes with the policy's own `norm_tag` post-processor, so degrees round
  trip), so this is under-adaptation, not a wiring bug — a generalist nudged
  toward the task vs a small network trained end-to-end on it.
- **Slow + bursty by construction.** ~3s/inference over the network, run
  receding-horizon (infer a chunk → replay → replace), so the arm moves in
  stop-start bursts. No client change removes this; it's inherent to a big
  remote VLA.

To make Molmo competitive (in rough priority order):
1. **Train much longer / more data** — 4k steps is light; try 15–30k steps and a
   larger, more varied dataset (more cube positions, distractors).
2. **Tune the receding-horizon** — match the replay frequency to the dataset rate
   and the inference latency to cut the burstiness (`MolmoPolicy` config).
3. **Quantize / host closer** to cut the ~3s latency, or run a smaller VLA.
4. Consider whether a VLA is even the right tool here — its payoff is *language
   generalization* across tasks, which this single fixed task doesn't exercise.

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
