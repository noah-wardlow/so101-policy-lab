# CLAUDE.md — so101-policy-lab

Guidance for coding agents working in this repo. For the human-facing pitch see
`README.md`; for live-server/pod details see `DEPLOYMENT.md`.

## What this is

An in-browser SO-101 pick-and-place lab. A MuJoCo WASM sim runs entirely
client-side; you can drive the arm three ways and compare them on the same scene:

- **teleop** — IK gizmo / keyboard.
- **expert** — a scripted pick-place controller (also the data generator).
- **act** — a LeRobot **ACT** policy running in-browser via onnxruntime-web (no
  backend). 3-cam (wrist+front+side).
- **molmo** — a fine-tuned **MolmoAct2** VLA served over HTTP on a GPU pod
  (the only mode that needs a URL).

The app is built on **`mujoco-react`** (the sibling library at
`../mujoco-react`, published to npm — this repo depends on the *published*
package, not the local path). When debugging library behavior, read
`node_modules/mujoco-react/dist/*.d.ts` for the installed API surface.

## Run it

```bash
npm install
npm run dev        # Vite dev server (port 3000, or next free port)
npm run build      # tsc + vite production build
npm run typecheck  # tsc --noEmit
npm run lint
```

State lives in the URL (TanStack Router + zod, see `src/router.tsx`):
`?mode=act&run=1&cams=1&cams3=0&molmo=<url>`. `mode` invalid → catches to
`act`. So `localhost:3000/?mode=act&run=1` boots straight into a running ACT.

## Architecture / data flow

```
src/main.tsx → Root (RouterProvider) → App
App
 └─ MujocoProvider (loads WASM)                         [mujoco-react]
     └─ MujocoCanvas (R3F canvas; paused={autoFinished})
         ├─ SceneChildren (needs IK context)
         │   ├─ useIkController(ikConfig)               5-DOF, pos-weighted
         │   ├─ teleop: IkGizmo + SO101Controller
         │   ├─ ScriptedExpert (always mounted, idle unless episode runs)
         │   ├─ BrowserActPolicy  (mounted only while mode=act AND run)
         │   ├─ PolicyAutoFinish  (detects place → pauses sim, flips button)
         │   └─ MolmoPolicy       (mounted only while mode=molmo AND run)
         ├─ Recorder (samples frames on the physics clock)
         └─ CameraStreamer (live panes, when cams=1)
 └─ WindowBridge (exposes window.__lab for scripts/eval)
 └─ Hud (mode buttons, Run/Reset, Molmo URL field)
```

Physics each frame is owned by `mujoco-react`'s `MujocoSimProvider`. The arm's 5
actuators are driven by the **IK controller**; the **gripper** actuator is driven
separately (see "Controls ownership" gotcha). Policies write `ctrl` via
`applyPolicyActionToControls`.

## Source layout (what each file is for)

- `src/App.tsx` — top-level wiring, URL state → props, the `__lab` bridge, the
  auto-finish pause/Reset logic.
- `src/router.tsx` — zod `searchSchema`, `ControlMode` type, the single `/` route.
- `src/Hud.tsx` — overlay UI; polls `window.__actStatus` for ACT load status.
- `src/robot/so101.ts` — **the scene + robot constants hub.** Scene geometry
  (`DEFAULT_CUBE_POS`, `DEFAULT_TARGET_POS`, `sampleCubeXY`), camera defs
  (`CAMERA_DEFS`, `SO101_CAMERAS`, `SO101_CAMERAS_3`, `camerasForKeys`), joint
  constants (`SO101_ARM_JOINTS`, `SO101_HOME`, `SO101_ARM_DOF`), the
  `simToPolicyDegrees`/`policyDegreesToSim` conversions, `createSo101SceneConfig`.
- `src/robot/controls.ts` — `GRIPPER_CONTROLS = defineControls({ gripper: 'gripper' })`.
- `src/robot/verifier.ts` — `isCubeGrasped`, `qposIsSane`, `EpisodeMetrics`.
- `src/controllers/ScriptedExpert.tsx` — scripted pick-place + the `ExpertHandle`
  (`runEpisode`, `placeCube`, `getCube`, `reachTest`). Pocket-centering grasp,
  two-sided grip gating, clean home-and-settle ending.
- `src/controllers/PolicyAutoFinish.tsx` — watches the cube; on a successful
  place calls `onFinish()` (App pauses the sim, no joint manipulation).
- `src/controllers/SO101Controller.tsx`, `useArmController.ts`,
  `ik2link.ts`/`fk2link.ts` — teleop arm control.
- `src/policies/actModels.ts` — `ACT_MODELS` registry (id, label, manifestUrl,
  cameraKeys). `MODEL_BASE = VITE_MODEL_BASE ?? '/models'`.
- `src/policies/BrowserActPolicy.tsx` — loads the ONNX policy, captures camera
  tensors, runs `usePolicy`, applies manifest normalization.
- `src/policies/MolmoPolicy.tsx` — `useRemotePolicy` client for the Molmo server.
- `src/recording/Recorder.tsx` — captures `RecordedFrame`s (qpos/ctrl + camera
  JPEGs) for dataset export.
- `src/mujoco-register.gen.ts` — **generated** by the `mujoco-react/vite` plugin
  (do not edit). Populates the typed model register (`ModelCameras.so101.*`,
  `Cameras`, etc). The plugin also injects its import, so nothing imports it by
  hand.
- `public/models/<id>/` — `act.onnx` + `policy.json` per ACT model, served at
  `/models/...` in dev. `so101/` holds the MJCF + meshes.

## Scripts & server

- `scripts/record.mjs` — drives a headless browser through `window.__lab` to
  record expert episodes (→ raw frames). Uses Playwright.
- `scripts/build_lerobot_dataset.py` — frames → LeRobot dataset.
  `--cameras wrist,front` (A) or `wrist,front,side` (C) selects the camera subset
  from one raw recording.
- `scripts/train_act.sh` — LeRobot ACT training (env-driven:
  `DATASET_ROOT`, `OUTPUT_DIR`, `STEPS`, ...).
- `scripts/export_act_to_onnx.py` — checkpoint → `act.onnx` + `policy.json`.
- `scripts/eval_policy.mjs` — headless eval; runs N episodes, scores whether the
  cube was actually moved onto the target. `node scripts/eval_policy.mjs --mode
  act --url http://localhost:3000/ --episodes 8`.
- `scripts/pod_molmo_ft.sh`, `server/molmo_ft_server.py` — Molmo LoRA fine-tune +
  inference server (GPU pod).
- `scripts/runpod_*.py` — pod lifecycle helpers.

Pod/host specifics (IPs, ports, proxy URLs) live in `DEPLOYMENT.md`, **not** in
source. Source reads endpoints from env (`VITE_MODEL_BASE`,
`VITE_MOLMO_ENDPOINT`) or the `?molmo=` query param.

## End-to-end workflow (data → policy → verify)

1. **Record** (3-cam recording): `node scripts/record.mjs` → raw frames.
2. **Build dataset**: `build_lerobot_dataset.py --cameras wrist,front,side`.
3. **Train** ACT (`train_act.sh`), **export** to ONNX (`export_act_to_onnx.py`).
   On the pod this is chained in `/workspace/run_all.sh` (ACT → Molmo FT,
   sequential).
4. **Deploy** to the app: copy `act.onnx` + `policy.json` into
   `public/models/act/`. In prod, host the ONNX on an object store and set
   `VITE_MODEL_BASE` (Cloudflare Pages' 25 MiB/file limit blocks the ~137 MB ONNX).
5. **Verify**: `npm run dev` then `eval_policy.mjs`. A clean ACT model places the
   cube within ~3–5 cm of the target.

Molmo: fine-tune on the same data, serve `molmo_ft_server.py` on the pod, point
the app at it via `?molmo=<proxy-url>` or `VITE_MOLMO_ENDPOINT`.

## Conventions & gotchas (read before editing)

- **Sim-degree convention.** The dataset/state/actions are sim-joint **degrees**.
  Everything funnels through `simToPolicyDegrees`/`policyDegreesToSim` (degrees↔
  radians). The Molmo FT model was trained on this convention, so its transform is
  **identity** — do not add sign flips or hand-tuned offsets.
- **Controls ownership.** The IK controller claims the 5 arm actuators. The
  gripper is a separate `defineControls`/`useControls` group so it never conflicts
  with IK ownership. Never put arm joints in a control group while IK is active.
- **No `as any` / `as never`.** Two type-safe libs (mujoco-react, zod) — keep it
  that way. Camera names are typed: `cameraName?: Cameras`, values come from
  `ModelCameras.so101.wrist_cam` (from the generated register), not string
  literals. Residual casts are only for parsing untyped manifest JSON / HTTP
  bodies.
- **Stop-on-place (not pause).** ACT is memoryless with no notion of "done", so
  after placing it can loop the grasp. This is an *inference* artifact, not bad
  data — the expert demos are clean (verified: cube drifts ~1cm, gripper cycles
  exactly 1 close + 1 open), so re-recording does NOT fix it. `BrowserActPolicy`
  watches for a genuine completed place — cube **lifted → gripper released
  (ctrl open) → settled on the table**, held ~0.4s — then fires `onDone` →
  `setRunning(false)`, which only stops inference (no pause, no joint manipulation;
  the arm holds, cube stays). Gate on the **release**, not just "cube near target":
  the earlier removed auto-finish fired mid-place because it skipped that check.
  Grasp *failures* at some cube positions (policy never lifts the cube) are a
  separate model-competence gap, not the terminal loop — the stop correctly won't
  fire there.
- **Warm policy session.** `BrowserActPolicy` mounts whenever its model is the
  selected mode (keyed by id) and is gated by `enabled={running}`. The ~137 MB
  ONNX session loads **once on select** and stays warm across Run/Stop/Reset.
  `usePolicy` keys inference off `data.time - lastActionTime`; after a sim reset
  that ref is stale, so the policy calls `policy.reset()` on each run-start — this
  (not a remount) is what fixes "can't re-run after reset". Do not re-gate the
  mount on `running` (that reintroduces a full reload per run).
- **Don't edit `src/` during an active recording** — HMR reloads disrupt the
  deterministic-step rollout. Record first, edit after.
- **Pocket-centering grasp.** The expert targets the geometric center between the
  jaw collision geoms (not the gripper site, which is offset toward the fixed
  jaw), and requires a genuine two-sided grip for `success`. Don't "fix" the
  grasp by retargeting the site.
- **`window.__lab`** is the only automation surface (typed in `src/global.d.ts`):
  `reset`, `setRunning`, `placeCube`, `getCube`, `reachTest`, `recordEpisode`.
  `reachTest(x,y,z)` is a one-shot IK reachability probe (via `ik.solveIK`, no arm
  motion).
