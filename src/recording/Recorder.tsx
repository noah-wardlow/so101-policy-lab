import { forwardRef, useImperativeHandle, useRef } from 'react';
import { useAfterPhysicsStep } from 'mujoco-react';
import type {
  MujocoSimAPI,
  MujocoModel,
  MujocoData,
  CameraFrameCaptureSession,
} from 'mujoco-react';
import { CAM_RENDER_ISOLATION, simToPolicyDegrees } from '../robot/so101';
import type { So101CameraDef } from '../robot/so101';

export interface RecordedFrame {
  /** seconds since episode start */
  t: number;
  /** observation.state — 6 joint angles, degrees */
  state: number[];
  /** action — 6 commanded joint targets, degrees */
  action: number[];
  /** PNG data URLs keyed by camera (wrist, front) */
  images: Record<string, string>;
}

export interface RecorderHandle {
  /** Arm sim-time sampling at `fps` (captures in the physics-step callback). */
  begin: (fps?: number) => void;
  end: () => RecordedFrame[];
  frameCount: () => number;
}

/**
 * Buffers (state, action, wrist+front PNG) frames in memory during an episode.
 * The dataset recorder script (scripts/record.mjs) drives begin()/sample()/end()
 * over the window bridge and writes raw episodes to disk; build_lerobot_dataset.py
 * converts them into a LeRobotDataset for ACT training.
 */
export const Recorder = forwardRef<
  RecorderHandle,
  { apiRef: React.RefObject<MujocoSimAPI | null>; cameras: So101CameraDef[] }
>(function Recorder({ apiRef, cameras }, ref) {
  const modelRef = useRef<MujocoModel | null>(null);
  const dataRef = useRef<MujocoData | null>(null);
  const sessionsRef = useRef<Record<string, CameraFrameCaptureSession> | null>(null);
  const framesRef = useRef<RecordedFrame[]>([]);
  const armedRef = useRef(false);
  const fpsRef = useRef(30);
  const startTimeRef = useRef(0); // sim time at episode start
  const lastSampleRef = useRef(-Infinity); // sim time of last captured frame

  // Sample on the physics clock, not wall-clock: capture image + state together
  // at fixed SIMULATION-time intervals so frames are evenly spaced and each
  // image pairs exactly with its state/action (what LeRobot fps=30 assumes).
  useAfterPhysicsStep(({ model, data }) => {
    modelRef.current = model;
    dataRef.current = data;
    if (!armedRef.current) return;
    const dt = 1 / fpsRef.current;
    if (data.time - lastSampleRef.current + 1e-9 < dt) return;
    lastSampleRef.current = data.time;
    captureFrame(data.time - startTimeRef.current);
  });

  const ensureSessions = (): Record<string, CameraFrameCaptureSession> | null => {
    if (sessionsRef.current) return sessionsRef.current;
    const api = apiRef.current;
    if (!api) return null;
    const sessions: Record<string, CameraFrameCaptureSession> = {};
    for (const cam of cameras) {
      sessions[cam.key] = api.createCameraFrameCaptureSession({
        width: cam.width,
        height: cam.height,
        cameraName: cam.cameraName,
        position: cam.position,
        lookAt: cam.lookAt,
        up: cam.up,
        fov: cam.fov,
        near: cam.near,
        far: cam.far,
        type: 'image/png',
        renderIsolation: CAM_RENDER_ISOLATION,
      });
    }
    sessionsRef.current = sessions;
    return sessions;
  };

  const captureFrame = (t: number) => {
    const data = dataRef.current;
    const sessions = ensureSessions();
    if (!data || !sessions) return;
    const state = simToPolicyDegrees(data.qpos);
    const action = simToPolicyDegrees(data.ctrl);
    const images: Record<string, string> = {};
    for (const cam of cameras) {
      images[cam.key] = sessions[cam.key].captureDataUrl().dataUrl;
    }
    framesRef.current.push({ t, state, action, images });
  };

  const begin = (fps = 30) => {
    framesRef.current = [];
    fpsRef.current = fps;
    startTimeRef.current = dataRef.current?.time ?? 0;
    lastSampleRef.current = -Infinity;
    armedRef.current = true;
  };
  const end = () => {
    armedRef.current = false;
    return framesRef.current;
  };
  const frameCount = () => framesRef.current.length;

  useImperativeHandle(ref, () => ({ begin, end, frameCount }), []);
  return null;
});
