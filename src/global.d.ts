import type { ThreeElements } from '@react-three/fiber';
import type { EpisodeMetrics } from './robot/verifier';
import type { RecordedFrame } from './recording/Recorder';

declare module 'react/jsx-runtime' {
  namespace JSX {
    interface IntrinsicElements extends ThreeElements {}
  }
}

/** Headless automation bridge the app exposes on `window` for the .mjs scripts —
 * typed so usage is checked, not cast. */
export interface LabBridge {
  reset(): void;
  setRunning(running: boolean): void;
  placeCube(x: number, y: number): void;
  getCube(): [number, number, number] | null;
  /** True if a top-down grasp at (x,y,z) is IK-reachable (no arm motion). */
  reachTest(x: number, y: number, z: number): boolean;
  /** Terminal diagnostic: current sim time + gripper command value. */
  probe(): { t: number; grip: number } | null;
  recordEpisode(
    i: number,
    fps?: number,
  ): Promise<{
    ok: boolean;
    success?: boolean;
    metrics?: EpisodeMetrics | null;
    cube?: number[];
    frames?: RecordedFrame[];
    reason?: string;
  }>;
}

declare global {
  interface Window {
    __lab?: LabBridge;
    /** Latest in-browser ACT status (loading / model shape / errors). */
    __actStatus?: string;
  }
}
