import { createRef, useEffect, useMemo, useRef, useState } from 'react';
import { OrbitControls, Html, Stats, Environment } from '@react-three/drei';
import {
  MujocoProvider,
  MujocoCanvas,
  useIkController,
  IkGizmo,
  DragInteraction,
  useMujoco,
} from 'mujoco-react';
import type { MujocoSimAPI, IkContextValue, IkConfig } from 'mujoco-react';
import {
  createSo101SceneConfig,
  IK_SITE,
  SO101_ARM_JOINTS,
  SO101_CAMERAS,
  SO101_CAMERAS_3,
  GRIPPER_INDEX,
  camerasForKeys,
} from './robot/so101';
import type { So101CameraDef } from './robot/so101';
import { CameraStreamer, CameraPanel } from './CameraPanes';
import { SO101Controller } from './controllers/SO101Controller';
import { ScriptedExpert } from './controllers/ScriptedExpert';
import type { ExpertHandle } from './controllers/ScriptedExpert';
import { Recorder } from './recording/Recorder';
import type { RecorderHandle } from './recording/Recorder';
import { BrowserActPolicy } from './policies/BrowserActPolicy';
import { ACT_MODELS, isActModel } from './policies/actModels';
import { MolmoPolicy } from './policies/MolmoPolicy';
import { sampleCubeXY } from './robot/so101';
import { getRouteApi } from '@tanstack/react-router';
import { Hud } from './Hud';
import type { ControlMode } from './router';

const route = getRouteApi('/');

const Z_UP: [number, number, number] = [0, 0, 1];
const CAMERA_POS: [number, number, number] = [0.95, -1.05, 1.35];
const ORBIT_TARGET: [number, number, number] = [0.42, -0.28, 0.82];

function LoadingPanel({ error }: { error?: string }) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 12,
        color: error ? '#f87171' : '#94a3b8',
        fontFamily: 'system-ui, sans-serif',
      }}
    >
      {error ? (
        <span style={{ fontSize: 14 }}>{error}</span>
      ) : (
        <>
          <div
            style={{
              width: 32,
              height: 32,
              border: '3px solid #334155',
              borderTop: '3px solid #38bdf8',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite',
            }}
          />
          <span style={{ fontSize: 14 }}>Loading SO-101…</span>
          <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
        </>
      )}
    </div>
  );
}

function LoadingOverlay() {
  const sim = useMujoco();
  if (sim.isReady) return null;
  return (
    <Html center>
      <LoadingPanel error={sim.isError ? sim.error : undefined} />
    </Html>
  );
}

/** Scene children that need the IK context. */
function SceneChildren({
  showGizmo,
  controlMode,
  running,
  expertRef,
  molmoEndpoint,
  onDone,
}: {
  showGizmo: boolean;
  controlMode: ControlMode;
  running: boolean;
  expertRef: React.RefObject<ExpertHandle | null>;
  molmoEndpoint: string;
  onDone: () => void;
}) {
  const ikConfig = useMemo(
    () => ({
      siteName: IK_SITE,
      joints: [...SO101_ARM_JOINTS],
      // SO-101 is a 5-DOF arm: a full 6-DOF pose is over-constrained, so weight
      // position far above orientation. This lets the EE actually reach grasp
      // points (approach stays ~downward but yaw/tilt give as needed).
      posWeight: 1.0,
      rotWeight: 0.5,
      damping: 0.05,
      maxIterations: 150,
    } satisfies IkConfig),
    [],
  );
  const ik: IkContextValue | null = useIkController(ikConfig);

  return (
    <>
      {ik && showGizmo && controlMode === 'teleop' && (
        <IkGizmo controller={ik} scale={0.08} />
      )}
      {controlMode === 'teleop' && <SO101Controller ik={ik} />}
      {/* Expert is always mounted but idle unless an episode is running. */}
      <ScriptedExpert ik={ik} ref={expertRef} />
      {/* Learned ACT policy — mounted whenever its model is selected (keyed by id
          so switching models swaps the ONNX session) and gated by `enabled`. The
          ~137MB session loads once on select and stays warm across Run/Stop/Reset;
          on each run-start the policy resets its decimation clock + queue, which is
          what actually fixes "can't re-run after reset" (no remount needed). */}
      {isActModel(controlMode) && (
        <BrowserActPolicy
          key={controlMode}
          enabled={running}
          manifestUrl={ACT_MODELS[controlMode].manifestUrl}
          cameraKeys={ACT_MODELS[controlMode].cameraKeys}
          onDone={onDone}
        />
      )}
      <MolmoPolicy enabled={controlMode === 'molmo' && running} endpoint={molmoEndpoint} />
    </>
  );
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Exposes an imperative bridge on window for Playwright + the recording pipeline. */
function WindowBridge({
  apiRef,
  expertRef,
  recorderRef,
  setRunning,
}: {
  apiRef: React.RefObject<MujocoSimAPI | null>;
  expertRef: React.RefObject<ExpertHandle | null>;
  recorderRef: React.RefObject<RecorderHandle | null>;
  setRunning: (r: boolean) => void;
}) {
  useEffect(() => {
    /** Reset, randomize the cube for episode `i`, record the expert rollout at 30Hz. */
    const recordEpisode = async (i: number, fps = 30) => {
      const expert = expertRef.current;
      const recorder = recorderRef.current;
      if (!expert || !recorder) return { ok: false, reason: 'not-ready' };
      apiRef.current?.reset();
      const [cx, cy] = sampleCubeXY(i);
      expert.placeCube(cx, cy);
      await sleep(400); // let the cube settle and FK update

      // Recorder self-samples on the physics clock at `fps` (sim-time aligned).
      recorder.begin(fps);
      let metrics = null;
      try {
        metrics = await expert.runEpisode();
      } finally {
        /* recorder.end() disarms below */
      }
      const frames = recorder.end();
      return { ok: true, success: !!metrics?.success, metrics, cube: [cx, cy], frames };
    };

    window.__lab = {
      reset: () => apiRef.current?.reset(),
      setRunning: (r: boolean) => setRunning(r),
      // Terminal diagnostic: gripper command + sim time (cube via getCube). A
      // re-closing gripper after a place == the policy is looping the grasp.
      probe: () => {
        const api = apiRef.current;
        return api ? { t: api.getTime(), grip: api.getCtrl()[GRIPPER_INDEX] } : null;
      },
      placeCube: (x: number, y: number) => expertRef.current?.placeCube(x, y),
      getCube: () => expertRef.current?.getCube() ?? null,
      reachTest: (x: number, y: number, z: number) => expertRef.current?.reachTest(x, y, z) ?? false,
      recordEpisode,
    };
  }, [apiRef, expertRef, recorderRef, setRunning]);
  return null;
}

export function App() {
  const apiRef = useRef<MujocoSimAPI>(null);
  const expertRef = useRef<ExpertHandle | null>(null);
  const recorderRef = useRef<RecorderHandle | null>(null);
  const sceneConfig = useMemo(() => createSo101SceneConfig(), []);
  const [status, setStatus] = useState('idle');

  // Type-safe, validated search params via TanStack Router (mode + run live in
  // the URL → shareable/benchmarkable; transient UI prefs stay local state).
  const search = route.useSearch();
  const navigate = route.useNavigate();
  const mode = search.mode;
  const running = search.run;
  // Switching policy STOPS the current run and resets the scene (clean A/B/C).
  const setMode = (m: ControlMode) => {
    navigate({ search: (p) => ({ ...p, mode: m, run: false }) });
    apiRef.current?.reset();
    setStatus('idle');
  };
  const setRunning = (r: boolean) => navigate({ search: (p) => ({ ...p, run: r }) });

  // The active camera set: the selected ACT model's cameras (so the live panes
  // show exactly what that model sees), else wrist+front — or wrist+front+side
  // when recording the 3-cam dataset (?cams3=1).
  const activeCameras: So101CameraDef[] = useMemo(() => {
    if (isActModel(mode)) return camerasForKeys(ACT_MODELS[mode].cameraKeys);
    return search.cams3 ? SO101_CAMERAS_3 : SO101_CAMERAS;
  }, [mode, search.cams3]);
  const camRefs = useMemo(
    () => Object.fromEntries(activeCameras.map((c) => [c.key, createRef<HTMLCanvasElement>()])),
    [activeCameras],
  );

  const [gizmo, setGizmo] = useState(true);
  // Stop the rollout once the policy completes a place (cube lifted → released →
  // settled). ACT can't perceive "done" and otherwise loops the grasp; this just
  // stops inference (no pause, no joint manipulation — the arm holds, cube stays).
  const onDone = () => setRunning(false);
  const resetScene = () => {
    setRunning(false);
    apiRef.current?.reset();
    setStatus('idle');
  };
  // Molmo endpoint: ?molmo=<url> overrides, else VITE_MOLMO_ENDPOINT, else blank
  // (set it in the HUD). No machine-specific URL is baked into the source.
  const [molmoUrl, setMolmoUrl] = useState(
    search.molmo ?? import.meta.env.VITE_MOLMO_ENDPOINT ?? '',
  );

  const randomizeCube = () => {
    apiRef.current?.reset();
    const i = Math.floor(performance.now()) % 997;
    const [x, y] = sampleCubeXY(i);
    setTimeout(() => expertRef.current?.placeCube(x, y), 200);
  };
  const runEpisode = async () => {
    // Mirror the recording path exactly: reset → place a fresh cube → settle → run.
    // Without the reset+placeCube, a second click would run from wherever the last
    // episode ended (cube on the pad, arm parked) and look like errant "swiping".
    // The dataset path (recordEpisode) already does this; the data is unaffected.
    apiRef.current?.reset();
    const [x, y] = sampleCubeXY(Math.floor(performance.now()) % 997);
    expertRef.current?.placeCube(x, y);
    setStatus('running episode…');
    await sleep(400); // let the cube settle + FK update before the expert reads it
    expertRef.current?.runEpisode().then(
      (r) => setStatus(r.success ? 'episode: success ✓' : 'episode: failed'),
      (e) => setStatus(`error: ${e}`),
    );
  };

  return (
    <MujocoProvider>
      <MujocoCanvas
        ref={apiRef}
        config={sceneConfig}
        loadingFallback={
          <Html center>
            <LoadingPanel />
          </Html>
        }
        camera={{ position: CAMERA_POS, up: Z_UP, fov: 45, near: 0.01, far: 100 }}
        speed={1}
        dpr={[1, 2]}
        gl={{ preserveDrawingBuffer: true }}
        shadows
        style={{ width: '100%', height: '100%' }}
      >
        <OrbitControls enableDamping dampingFactor={0.1} target={ORBIT_TARGET} makeDefault />

        <LoadingOverlay />

        <SceneChildren
          showGizmo={gizmo}
          controlMode={mode}
          running={running}
          expertRef={expertRef}
          molmoEndpoint={molmoUrl}
          onDone={onDone}
        />

        <Recorder apiRef={apiRef} ref={recorderRef} cameras={activeCameras} />
        {search.cams && <CameraStreamer refs={camRefs} cameras={activeCameras} />}

        <DragInteraction />

        <Environment
          preset="lobby"
          background
          backgroundBlurriness={1}
          backgroundIntensity={0.6}
          environmentIntensity={0.5}
        />
        <ambientLight intensity={0.4} />
        <directionalLight position={[2, -2, 5]} intensity={1.5} castShadow />
        <directionalLight position={[-1, 1, 3]} intensity={0.3} />
        <gridHelper
          args={[4, 40, '#64748b', '#94a3b8']}
          rotation={[Math.PI / 2, 0, 0]}
          position={[0, 0, 0.001]}
        />
        <Stats />
      </MujocoCanvas>

      <WindowBridge apiRef={apiRef} expertRef={expertRef} recorderRef={recorderRef} setRunning={setRunning} />
      {search.cams && <CameraPanel refs={camRefs} cameras={activeCameras} />}

      <Hud
        mode={mode}
        setMode={setMode}
        running={running}
        setRunning={setRunning}
        gizmo={gizmo}
        setGizmo={setGizmo}
        molmoUrl={molmoUrl}
        setMolmoUrl={setMolmoUrl}
        onReset={resetScene}
        onRandomize={randomizeCube}
        onRunEpisode={runEpisode}
        episodeStatus={status}
      />
    </MujocoProvider>
  );
}
