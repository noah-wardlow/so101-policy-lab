/**
 * SO-101 single source of truth.
 *
 * Every fact about the arm that more than one module needs — joint order,
 * control ranges, the home pose, gripper open/closed targets, the work-surface
 * geometry, camera framings, and the canonical "policy units" conversion —
 * lives here. A classic failure mode is scattering these constants (sign flips,
 * gripper ranges, home poses) across the browser, the ACT server, and the Molmo
 * server with no agreement between them. Keep them here; each
 * policy client layers its *own* embodiment transform on top (see policies/).
 */

import { ModelCameras } from 'mujoco-react';
import type { SceneConfig, SceneObject, Cameras } from 'mujoco-react';

/** Actuator / qpos order for the 6 SO-101 DOF. Index === ctrl index. */
export const SO101_JOINTS = [
  'shoulder_pan',
  'shoulder_lift',
  'elbow_flex',
  'wrist_flex',
  'wrist_roll',
  'gripper',
] as const;
export type So101Joint = (typeof SO101_JOINTS)[number];

export const SO101_NUM_ACTUATORS = 6;
export const SO101_ARM_DOF = 5; // first five drive the IK chain
export const GRIPPER_INDEX = 5;

/** Actuator ctrlrange (radians), straight from SO101.xml `<actuator>`. */
export const SO101_CTRL_RANGE: Record<So101Joint, [number, number]> = {
  shoulder_pan: [-1.91986, 1.91986],
  shoulder_lift: [-1.74533, 1.74533],
  elbow_flex: [-1.69, 1.69],
  wrist_flex: [-1.65806, 1.65806],
  wrist_roll: [-2.74385, 2.84121],
  gripper: [-0.17453, 1.74533],
};

/** Gripper control targets (radians). min == closed, max == open. */
export const GRIPPER_CLOSED = -0.1; // clamp onto the cube (full shut is -0.17453)
export const GRIPPER_OPEN = 1.5;

/** IK chain. We solve to the stock `gripperframe` site (empirically the grasp
 * point for a top-down approach with this rotating-plate gripper). */
export const IK_SITE = 'gripperframe';
export const SO101_ARM_JOINTS = SO101_JOINTS.slice(0, SO101_ARM_DOF);

/**
 * Home pose (radians), chosen so the gripper starts above the work surface,
 * pointing down — a clean starting point for top-down IK grasps.
 */
export const SO101_HOME: number[] = [
  0.0, // shoulder_pan
  -1.2, // shoulder_lift
  1.2, // elbow_flex
  0.9, // wrist_flex
  0.0, // wrist_roll
  GRIPPER_OPEN, // gripper starts open
];

// --- Scene geometry ----------------------------------------------------------
// The base is mounted on the table at (0.35, -0.3, 0.80) per SO101.xml. The
// work surface (table top) is at z = 0.80. Keep manipulands within ~0.20 m of
// the base in XY so they stay inside the arm's reach.

export const BASE_XY: [number, number] = [0.35, -0.3];
export const TABLE_TOP_Z = 0.8;
// A 3cm cube (half-extent 0.015) — the manipuland the gripper grasps reliably
// top-down at the gripperframe site.
export const CUBE_SIZE: [number, number, number] = [0.015, 0.015, 0.015];
export const CUBE_REST_Z = TABLE_TOP_Z + CUBE_SIZE[2];

// Layout: place pad back-right, cube front-right, with the robot base between
// them along the reach axis. The cube sits at the right-side top-down grasp
// limit (~x=0.50 at y=-0.40); the place pad only needs to be reachable for a drop.
//   place (0.62,-0.25, BACK-right) — base (0.35,-0.30) — cube (0.50,-0.40, FRONT-right)
export const DEFAULT_CUBE_POS: [number, number, number] = [0.50, -0.40, CUBE_REST_Z];
export const DEFAULT_TARGET_POS: [number, number, number] = [0.62, -0.25, TABLE_TOP_Z + 0.004];
/** Bigger drop pad (5cm) — easier place target + clearer visual. */
export const TARGET_HALF = 0.05;

/**
 * Sampling box for episode randomization of the cube (forward-RIGHT, ±~2cm),
 * kept inside the reliable right grasp zone (grasp maxes ~x=0.50 at this y). The
 * spread forces the policy to localize the cube from vision; the verifier
 * discards missed attempts.
 */
export const CUBE_SAMPLE_BOX = {
  x: [0.48, 0.51] as [number, number],
  y: [-0.42, -0.38] as [number, number],
  z: CUBE_REST_Z,
};

export const TASK_PROMPT = 'pick up the red cube and place it on the green target';

// --- Cameras -----------------------------------------------------------------
// Canonical observation cameras, matching the LeRobot SO-101 convention
// (binhpham/sim_so101_cubes): two 320x240 RGB views, keyed `wrist` and `side`.
// One definition drives BOTH the dataset recorder and live policy inference.
export const CAM_W = 320;
export const CAM_H = 240;

// Shared render-isolation settings for every camera consumer (live panes, ACT
// policy capture, dataset recorder). Isolation = a dedicated offscreen renderer
// so the camera renders don't disturb the main view. Keep ONE stable object so
// the capture layer can cache the isolated renderer (it warns on unstable settings).
export const CAM_RENDER_ISOLATION = { enabled: true, antialias: true } as const;

export interface So101CameraDef {
  key: string;
  width: number;
  height: number;
  /** Named MJCF camera (typed against the model), or omit for a free framing. */
  cameraName?: Cameras;
  position?: [number, number, number];
  lookAt?: [number, number, number];
  up?: [number, number, number];
  fov?: number;
  /** Near/far clip. Crucial for the eye-in-hand wrist cam (objects ~5cm away). */
  near?: number;
  far?: number;
}

// Camera registry, keyed by name. Each ACT model declares which of these it
// consumes (A/B = wrist+front, C = wrist+front+side) — the recorder, live panes,
// and the in-browser policy all resolve their camera list from these defs.
//   - wrist: eye-in-hand `wrist_cam`, verified to look down on the cube at grasp
//   - front: third-person framing the right-side work area
//   - side:  third-person from the right, for the 3-cam ablation
export const CAMERA_DEFS: Record<'wrist' | 'front' | 'side', So101CameraDef> = {
  wrist: {
    key: 'wrist',
    width: CAM_W,
    height: CAM_H,
    cameraName: ModelCameras.so101.wrist_cam,
    // Override the MJCF's narrow 48.5° fovy with a wider fov:55 — shows more of
    // the cube + table context.
    fov: 55,
    near: 0.01,
    far: 100,
  },
  front: {
    key: 'front',
    width: CAM_W,
    height: CAM_H,
    // Framed for the right-side work area (cube ~0.50,-0.40 + place ~0.62,-0.25).
    position: [0.30, -0.90, 1.05],
    lookAt: [0.56, -0.33, 0.81],
    up: [0, 0, 1],
    fov: 52,
    near: 0.01,
    far: 100,
  },
  side: {
    key: 'side',
    width: CAM_W,
    height: CAM_H,
    // Looks across the work area from the right side (a third view for the 3-cam
    // model — complements the front view with depth the front cam can't see).
    position: [1.08, -0.58, 1.06],
    lookAt: [0.56, -0.33, 0.81],
    up: [0, 0, 1],
    fov: 50,
    near: 0.01,
    far: 100,
  },
};

/** Resolve a camera-key list into defs (skips unknown keys). */
export const camerasForKeys = (keys: readonly string[]): So101CameraDef[] =>
  keys.map((k) => CAMERA_DEFS[k as keyof typeof CAMERA_DEFS]).filter(Boolean);

// Default policy/recorder cameras (Models A & B): wrist + front. Model C adds a
// side view (SO101_CAMERAS_3). One definition drives recorder + panes + policy.
export const SO101_CAMERAS: So101CameraDef[] = [CAMERA_DEFS.wrist, CAMERA_DEFS.front];
export const SO101_CAMERAS_3: So101CameraDef[] = [CAMERA_DEFS.wrist, CAMERA_DEFS.front, CAMERA_DEFS.side];

/**
 * Build the scene objects. The SO-101 base is mounted at (0.35,-0.3,0.80) in
 * SO101.xml but the supporting table + floor are NOT in the MJCF — we add them
 * here so the table top lands exactly at z=0.80 under the base.
 */
export function createSceneObjects(
  cubePos: [number, number, number] = DEFAULT_CUBE_POS,
  targetPos: [number, number, number] = DEFAULT_TARGET_POS,
): SceneObject[] {
  return [
    {
      name: 'floor',
      type: 'box',
      size: [2, 2, 0.005],
      position: [0, 0, -0.005],
      rgba: [0.13, 0.14, 0.18, 1],
    },
    {
      name: 'table',
      type: 'box',
      size: [0.4, 0.4, 0.4],
      position: [BASE_XY[0], BASE_XY[1], 0.4],
      rgba: [0.82, 0.7, 0.55, 1],
      friction: '1.5 0.3 0.1',
      solref: '0.01 1',
      solimp: '0.95 0.99 0.001 0.5 2',
      condim: 4,
    },
    {
      name: 'red_cube',
      type: 'box',
      size: CUBE_SIZE,
      position: cubePos,
      rgba: [0.92, 0.12, 0.08, 1],
      mass: 0.02,
      freejoint: true,
      contype: 1,
      conaffinity: 1,
      // Grasp-friendly contacts.
      friction: '2 0.3 0.1',
      solref: '0.01 1',
      solimp: '0.95 0.99 0.001 0.5 2',
      condim: 4,
    },
    {
      name: 'green_target',
      type: 'box',
      size: [TARGET_HALF, TARGET_HALF, 0.004],
      position: targetPos,
      rgba: [0.12, 0.72, 0.38, 0.6],
      // Visual pad only — no contact so it never nudges the cube.
      contype: 0,
      conaffinity: 0,
    },
  ];
}

export function createSo101SceneConfig(
  cubePos?: [number, number, number],
  targetPos?: [number, number, number],
): SceneConfig {
  return {
    src: '/models/so101/',
    sceneFile: 'SO101.xml',
    homeJoints: SO101_HOME,
    sceneObjects: createSceneObjects(cubePos, targetPos),
  };
}

// --- Policy units ------------------------------------------------------------
// Canonical "policy units" for a self-trained policy = sim joint angles in
// DEGREES, no sign flips. (LeRobot SO-101 datasets are in degrees, so training
// data we record matches that scale; ACT mean/std-normalizes internally either
// way.) External pretrained checkpoints (MolmoAct2) need the *real* SO-101 sign
// convention — that transform is isolated in policies/molmo.ts, not here.

const RAD2DEG = 180 / Math.PI;
const DEG2RAD = Math.PI / 180;

export const radToDeg = (rad: number) => rad * RAD2DEG;
export const degToRad = (deg: number) => deg * DEG2RAD;

/** Sim ctrl/qpos (radians) -> policy state vector (degrees), all 6 joints. */
export function simToPolicyDegrees(simJoints: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let i = 0; i < SO101_NUM_ACTUATORS; i++) out.push(radToDeg(simJoints[i]));
  return out;
}

/** Policy action vector (degrees) -> sim ctrl targets (radians), clamped. */
export function policyDegreesToSim(actionDeg: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let i = 0; i < SO101_NUM_ACTUATORS; i++) {
    const joint = SO101_JOINTS[i];
    const [lo, hi] = SO101_CTRL_RANGE[joint];
    out.push(Math.min(hi, Math.max(lo, degToRad(actionDeg[i]))));
  }
  return out;
}

export function clampCtrl(index: number, valueRad: number): number {
  const [lo, hi] = SO101_CTRL_RANGE[SO101_JOINTS[index]];
  return Math.min(hi, Math.max(lo, valueRad));
}

/** Teleport the red_cube freejoint to (x,y) on the table, upright + at rest. */
export function placeFreeBody(
  model: {
    body_jntadr: Int32Array;
    jnt_qposadr: Int32Array;
    jnt_dofadr: Int32Array;
  },
  data: { qpos: Float64Array | Float32Array; qvel: Float64Array | Float32Array },
  bodyId: number,
  x: number,
  y: number,
  z: number,
): void {
  if (bodyId < 0) return;
  const jntAdr = model.body_jntadr[bodyId];
  if (jntAdr < 0) return;
  const q = model.jnt_qposadr[jntAdr];
  const d = model.jnt_dofadr[jntAdr];
  data.qpos[q] = x;
  data.qpos[q + 1] = y;
  data.qpos[q + 2] = z;
  data.qpos[q + 3] = 1;
  data.qpos[q + 4] = 0;
  data.qpos[q + 5] = 0;
  data.qpos[q + 6] = 0;
  for (let i = 0; i < 6; i++) data.qvel[d + i] = 0;
}

/** Deterministic cube sample for episode `i` (reproducible datasets). */
export function sampleCubeXY(i: number): [number, number] {
  // Cheap LCG hash so we don't need Math.random (and stays reproducible).
  const h1 = ((i * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const h2 = ((i * 1664525 + 1013904223) & 0x7fffffff) / 0x7fffffff;
  const x = CUBE_SAMPLE_BOX.x[0] + h1 * (CUBE_SAMPLE_BOX.x[1] - CUBE_SAMPLE_BOX.x[0]);
  const y = CUBE_SAMPLE_BOX.y[0] + h2 * (CUBE_SAMPLE_BOX.y[1] - CUBE_SAMPLE_BOX.y[0]);
  return [x, y];
}
