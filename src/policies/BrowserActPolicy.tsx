import { useEffect, useRef, useState } from 'react';
import * as ort from 'onnxruntime-web/wasm';
import ortWasmMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import {
  usePolicy,
  usePolicyCameraTensors,
  applyPolicyActionToControls,
  useBeforePhysicsStep,
  findBodyByName,
} from 'mujoco-react';
import {
  createOnnxPolicySession,
  onnxTensorToPolicyActionChunk,
} from 'mujoco-react/onnx';
import type { OnnxPolicySession } from 'mujoco-react/onnx';
import {
  camerasForKeys,
  CAM_RENDER_ISOLATION,
  SO101_NUM_ACTUATORS,
  GRIPPER_INDEX,
  CUBE_REST_Z,
  simToPolicyDegrees,
  policyDegreesToSim,
} from '../robot/so101';

interface NormSpec {
  mode?: string;
  stats?: { mean?: number[]; std?: number[]; min?: number[]; max?: number[] };
  eps?: number;
}

/** Deep-flatten a possibly-nested numeric array (image stats are [3,1,1]). */
function flat(a: unknown): number[] {
  const out: number[] = [];
  const rec = (x: unknown) => {
    if (Array.isArray(x)) x.forEach(rec);
    else if (typeof x === 'number') out.push(x);
  };
  rec(a);
  return out;
}

/** Apply (or invert) the manifest's normalization for one tensor. */
function normalize(values: number[], spec: NormSpec | undefined, channels = 1, inverse = false): number[] {
  if (!spec || spec.mode === 'identity' || !spec.stats) return values;
  const n = values.length;
  // Stats may be nested (image MEAN_STD stats are [C,1,1]); flatten like np.ravel.
  const mean = flat(spec.stats.mean);
  const std = flat(spec.stats.std);
  const eps = spec.eps ?? 1e-8;
  const pick = (arr: number[] | undefined, i: number, fallback: number) => {
    if (!arr || arr.length === 0) return fallback;
    if (arr.length === n) return arr[i];
    if (channels > 1 && arr.length === channels) return arr[Math.floor(i / (n / channels))];
    return arr[0];
  };
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const m = pick(mean, i, 0);
    const s = pick(std, i, 1);
    out[i] = inverse ? values[i] * s + m : (values[i] - m) / (s + eps);
  }
  return out;
}

export function BrowserActPolicy({
  enabled,
  manifestUrl,
  cameraKeys,
  onDone,
}: {
  enabled: boolean;
  manifestUrl: string;
  /** Camera keys this model consumes (must match its policy.json image inputs). */
  cameraKeys: readonly string[];
  /** Fired once the cube has been lifted then released and left settled — used to
   *  stop the rollout so ACT's terminal grip-looping never gets a chance to run. */
  onDone?: () => void;
}) {
  const sessionRef = useRef<OnnxPolicySession | null>(null);
  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState('loading…');
  // This component is mounted with a per-model `key`, so cameraKeys is stable for
  // its lifetime — resolve once.
  const modelCameras = camerasForKeys(cameraKeys);

  // Capture this model's observation cameras as CHW [0,1] tensors at model res.
  const cameras = usePolicyCameraTensors({
    streams: modelCameras.map((c) => ({
      key: c.key,
      width: c.width,
      height: c.height,
      channels: 3 as const,
      layout: 'CHW' as const,
      range: [0, 1] as [number, number],
      cameraName: c.cameraName,
      position: c.position,
      lookAt: c.lookAt,
      up: c.up,
      fov: c.fov,
      near: c.near,
      far: c.far,
      renderIsolation: CAM_RENDER_ISOLATION,
    })),
  });

  // Load once when this component mounts. It only mounts while its model is the
  // selected mode, so mounting IS the lazy gate — the ~137MB session then stays
  // warm across Run/Stop/Reset (no reload per run).
  useEffect(() => {
    let cancelled = false;
    setReady(false);
    setStatus('loading…');
    // Single-thread SIMD WASM with the exact bundled artifacts. In-browser speed
    // is bound by the MuJoCo sim + offscreen camera renders, not the ONNX run.
    ort.env.wasm.simd = true;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = { mjs: ortWasmMjsUrl, wasm: ortWasmUrl };
    createOnnxPolicySession({
      manifestUrl,
      runtime: ort,
      sessionOptions: { executionProviders: ['wasm'] },
    })
      .then((s) => {
        if (cancelled) return;
        sessionRef.current = s;
        setReady(true);
        const ep = (s.session as unknown as { executionProviderName?: string }).executionProviderName ?? '?';
        setStatus(`${s.manifest.model ?? 'act.onnx'} ${s.manifest.output.shape.join('x')} [${ep}]`);
      })
      .catch((e) => !cancelled && setStatus(`ACT load failed: ${e.message ?? e}`));
    return () => {
      cancelled = true;
    };
  }, [manifestUrl]);

  const policy = usePolicy({
    enabled: enabled && ready,
    // Match the dataset's effective capture rate (~15Hz sim-time) so the action
    // chunk is replayed at training speed. usePolicy decimates on sim time.
    frequency: 15,
    queueStrategy: 'append',
    prefetchThreshold: 8,
    onObservation: ({ data }) => simToPolicyDegrees(data.qpos),
    infer: async ({ observation }) => {
      const s = sessionRef.current!;
      const norm = s.manifest.normalization as { inputs?: Record<string, NormSpec>; output?: NormSpec } | undefined;
      const tensors = cameras.capture().tensors;

      const feeds: Record<string, ort.Tensor> = {};
      const stateNorm = normalize(Array.from(observation as ArrayLike<number>), norm?.inputs?.state, 1);
      feeds.state = new ort.Tensor('float32', Float32Array.from(stateNorm), [1, SO101_NUM_ACTUATORS]);
      for (const cam of modelCameras) {
        const t = tensors[cam.key];
        const data = normalize(Array.from(t.data), norm?.inputs?.[cam.key], 3);
        feeds[cam.key] = new ort.Tensor('float32', Float32Array.from(data), [1, 3, cam.height, cam.width]);
      }

      const result = await s.session.run(feeds);
      const outName = s.manifest.output.name;
      const chunk = onnxTensorToPolicyActionChunk(result[outName], SO101_NUM_ACTUATORS);
      const n = s.manifest.n_action_steps ?? chunk.length;
      // Denormalize each action vector, then convert policy degrees -> sim radians.
      return chunk.slice(0, n).map((a) => policyDegreesToSim(normalize(Array.from(a as ArrayLike<number>), norm?.output, 1, true)));
    },
    onAction: ({ action, model, data }) => {
      applyPolicyActionToControls(model, data, action, {
        actuatorOffset: 0,
        actionSize: SO101_NUM_ACTUATORS,
        clamp: true,
        skipInvalid: true,
      });
    },
    onError: (e) => setStatus(`infer error: ${e}`),
  });

  // Per-frame: (1) reset the policy when the sim clock jumps backwards (a scene
  // reset) so re-runs work without reloading the session, and (2) detect a genuine
  // completed place and fire onDone — which stops the rollout before ACT's terminal
  // grip-looping (it keeps re-grasping because it can't perceive "done") can swipe
  // the cube. "Done" = the cube was lifted, is now back down on the table, AND the
  // gripper is open (released) — held for a short dwell so a transient doesn't fire.
  const lastTimeRef = useRef(0);
  const cubeRef = useRef(-1);
  const liftedRef = useRef(false);
  const doneSinceRef = useRef<number | null>(null);
  const firedRef = useRef(false);
  useBeforePhysicsStep(({ model, data }) => {
    if (data.time < lastTimeRef.current - 1e-6) {
      policy.reset();
      liftedRef.current = false;
      doneSinceRef.current = null;
      firedRef.current = false;
    }
    lastTimeRef.current = data.time;
    if (!enabled || firedRef.current || !onDone) return;

    if (cubeRef.current < 0) cubeRef.current = findBodyByName(model, 'red_cube');
    const id = cubeRef.current;
    if (id < 0) return;
    const cz = data.xpos[id * 3 + 2];
    if (cz > CUBE_REST_Z + 0.03) liftedRef.current = true;
    const released = data.ctrl[GRIPPER_INDEX] > 1.0; // gripper commanded open
    const settled = cz < CUBE_REST_Z + 0.02; // cube back down on the table
    if (liftedRef.current && released && settled) {
      if (doneSinceRef.current == null) doneSinceRef.current = data.time;
      else if (data.time - doneSinceRef.current > 0.4) {
        firedRef.current = true;
        onDone();
      }
    } else {
      doneSinceRef.current = null;
    }
  });

  // Surface status to the HUD (polled there).
  useEffect(() => {
    window.__actStatus = status;
  }, [status]);

  return null;
}
