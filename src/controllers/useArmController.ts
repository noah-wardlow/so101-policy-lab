import { useRef, useEffect } from 'react';
import { useBeforePhysicsStep } from 'mujoco-react';
import type { IkContextValue } from 'mujoco-react';
import { inverseKinematics2Link } from './ik2link';
import { forwardKinematics2Link } from './fk2link';
import type { LinkageParams } from './ik2link';

export type { LinkageParams } from './ik2link';

export interface ArmConfig {
  /** Actuator indices: [rotation, pitch, elbow, wristPitch, wristRoll, gripper] */
  indices: number[];
  /** Key bindings: [rotCW, rotCCW, eeForward, eeBack, eeUp, eeDown, pitchUp, pitchDown, rollCW, rollCCW, gripperToggle] */
  keys: string[];
  /** Initial joint values — if provided, arm starts here instead of IK-computed default. */
  initialJoints?: number[];
  /** Initial shoulder rotation. Default: 0 */
  initialRotation?: number;
  /** Initial wrist roll. Default: 0 */
  initialRoll?: number;
  /** Linkage parameters for 2-link IK/FK. Defaults to SO101 linkage. */
  linkage?: LinkageParams;
  /** Tip length from wrist to end-effector (m). Default: 0.108 */
  tipLength?: number;
  /** Gripper open position. Default: 1.5 */
  gripperOpen?: number;
  /** Gripper closed position. Default: -0.25 */
  gripperClosed?: number;
}

export interface BaseConfig {
  /** Actuator indices: [forward, turn] */
  indices: [number, number];
  /** Key bindings: [forward, back, turnLeft, turnRight] */
  keys: [string, string, string, string];
  speed?: number;
}

export interface HeadConfig {
  /** Actuator indices: [pan, tilt] */
  indices: [number, number];
  /** Key bindings: [panLeft, panRight, tiltUp, tiltDown] */
  keys: [string, string, string, string];
}

export interface ArmControllerConfig {
  /** Total number of actuators */
  numActuators: number;
  /** Base (mobile) drive config — omit for fixed-base robots */
  base?: BaseConfig;
  /** Arm configs — one per arm */
  arms: ArmConfig[];
  /** Head config — omit if no head */
  head?: HeadConfig;
}

const JOINT_STEP = 0.01;
const EE_STEP = 0.001;
const PITCH_STEP = 0.012;
const DEFAULT_TIP_LENGTH = 0.108;
const DEFAULT_GRIPPER_OPEN = 1.5;
const DEFAULT_GRIPPER_CLOSED = -0.25;
const INITIAL_EE: [number, number] = [0.162, 0.118];

/**
 * Generic arm controller hook — configure arms, base, head via a single config.
 * Handles IK, gripper toggles, base velocity, and head pan/tilt.
 *
 * Automatically disables the library's IK solver when arm keys are pressed,
 * syncing from the current arm position so there's no jump.
 */
export function useArmController(config: ArmControllerConfig, ik?: IkContextValue | null) {
  const keys = useRef<Record<string, boolean>>({});

  const armStates = useRef(
    config.arms.map((arm) => {
      const targetJoints = new Float64Array(arm.indices.length);
      const gripperClosed = arm.gripperClosed ?? DEFAULT_GRIPPER_CLOSED;
      let eePos = [INITIAL_EE[0], INITIAL_EE[1]];
      let pitch = 0;

      if (arm.initialJoints) {
        for (let j = 0; j < Math.min(arm.initialJoints.length, targetJoints.length); j++) {
          targetJoints[j] = arm.initialJoints[j];
        }
        if (arm.initialJoints.length >= 3) {
          const [x, y] = forwardKinematics2Link(arm.initialJoints[1], arm.initialJoints[2], arm.linkage);
          eePos = [x, y];
        }
        if (arm.initialJoints.length >= 4) {
          pitch = arm.initialJoints[3] - arm.initialJoints[1] + arm.initialJoints[2];
        }
      } else {
        const [j2, j3] = inverseKinematics2Link(INITIAL_EE[0], INITIAL_EE[1], arm.linkage);
        targetJoints[0] = arm.initialRotation ?? 0;
        targetJoints[1] = j2;
        targetJoints[2] = j3;
        targetJoints[3] = j2 - j3;
        targetJoints[4] = arm.initialRoll ?? 0;
        targetJoints[5] = gripperClosed;
      }

      return { targetJoints, eePos, pitch, gripperOpen: false, gripperKeyWasDown: false, controlActive: false, ikWasEnabled: false };
    }),
  );

  const baseState = useRef({ prevActive: [false, false] });
  const headState = useRef(new Float64Array(2));

  // Keyboard listeners
  useEffect(() => {
    const onDown = (e: KeyboardEvent) => { keys.current[e.code] = true; };
    const onUp = (e: KeyboardEvent) => { keys.current[e.code] = false; };
    const onBlur = () => { keys.current = {}; };
    window.addEventListener('keydown', onDown);
    window.addEventListener('keyup', onUp);
    window.addEventListener('blur', onBlur);
    return () => {
      window.removeEventListener('keydown', onDown);
      window.removeEventListener('keyup', onUp);
      window.removeEventListener('blur', onBlur);
    };
  }, []);

  useBeforePhysicsStep(({ data }) => {
    const k = keys.current;

    // === Base ===
    if (config.base) {
      const b = config.base;
      const bs = baseState.current;
      const speed = b.speed ?? 1;

      if (k[b.keys[0]]) {
        data.ctrl[b.indices[0]] = -speed;
        bs.prevActive[0] = true;
      } else if (k[b.keys[1]]) {
        data.ctrl[b.indices[0]] = speed;
        bs.prevActive[0] = true;
      } else if (bs.prevActive[0]) {
        data.ctrl[b.indices[0]] = 0;
        bs.prevActive[0] = false;
      }

      if (k[b.keys[2]]) {
        data.ctrl[b.indices[1]] = speed;
        bs.prevActive[1] = true;
      } else if (k[b.keys[3]]) {
        data.ctrl[b.indices[1]] = -speed;
        bs.prevActive[1] = true;
      } else if (bs.prevActive[1]) {
        data.ctrl[b.indices[1]] = 0;
        bs.prevActive[1] = false;
      }
    }

    // === Arms ===
    for (let i = 0; i < config.arms.length; i++) {
      const arm = config.arms[i];
      const s = armStates.current[i];
      const ak = arm.keys;
      const tipLength = arm.tipLength ?? DEFAULT_TIP_LENGTH;
      const gripperOpen = arm.gripperOpen ?? DEFAULT_GRIPPER_OPEN;
      const gripperClosed = arm.gripperClosed ?? DEFAULT_GRIPPER_CLOSED;

      // Check if any arm movement key is pressed (not gripper)
      let anyArmKey = false;
      for (let j = 0; j < 10 && j < ak.length; j++) {
        if (k[ak[j]]) { anyArmKey = true; break; }
      }

      // On transition from idle to keyboard: sync state from current ctrl
      // (picks up IK gizmo position, post-reset homeJoints, or any external ctrl change)
      if (anyArmKey && !s.controlActive) {
        for (let j = 0; j < arm.indices.length; j++) {
          s.targetJoints[j] = data.ctrl[arm.indices[j]];
        }
        const [x, y] = forwardKinematics2Link(s.targetJoints[1], s.targetJoints[2], arm.linkage);
        s.eePos[0] = x;
        s.eePos[1] = y;
        s.pitch = s.targetJoints[3] - s.targetJoints[1] + s.targetJoints[2];
        s.ikWasEnabled = ik?.ikEnabledRef.current ?? false;
        if (s.ikWasEnabled) ik!.setIkEnabled(false);
        s.controlActive = true;
      }

      // On transition from keyboard to idle: re-sync IK target so arm holds position
      if (!anyArmKey) {
        if (s.controlActive && s.ikWasEnabled && ik) {
          ik.syncTargetToSite();
          ik.setIkEnabled(true);
        }
        s.controlActive = false;
      }

      // Shoulder rotation
      if (k[ak[0]]) s.targetJoints[0] += JOINT_STEP;
      if (k[ak[1]]) s.targetJoints[0] -= JOINT_STEP;

      // EE position
      if (k[ak[2]]) s.eePos[0] += EE_STEP;
      if (k[ak[3]]) s.eePos[0] -= EE_STEP;
      if (k[ak[4]]) s.eePos[1] += EE_STEP;
      if (k[ak[5]]) s.eePos[1] -= EE_STEP;

      // Wrist pitch
      if (k[ak[6]]) s.pitch += PITCH_STEP;
      if (k[ak[7]]) s.pitch -= PITCH_STEP;

      // Wrist roll
      if (k[ak[8]]) s.targetJoints[4] += JOINT_STEP * 3;
      if (k[ak[9]]) s.targetJoints[4] -= JOINT_STEP * 3;

      // Gripper toggle — edge detection (only on key-down transition)
      const gripperDown = !!k[ak[10]];
      if (gripperDown && !s.gripperKeyWasDown) {
        s.gripperOpen = !s.gripperOpen;
        s.targetJoints[5] = s.gripperOpen ? gripperOpen : gripperClosed;
      }
      s.gripperKeyWasDown = gripperDown;

      // Solve IK
      const compY = s.eePos[1] + tipLength * Math.sin(s.pitch);
      const [j2, j3] = inverseKinematics2Link(s.eePos[0], compY, arm.linkage);
      s.targetJoints[1] = j2;
      s.targetJoints[2] = j3;
      s.targetJoints[3] = j2 - j3 + s.pitch;

      // Write arm joints to ctrl only when keyboard is active
      if (s.controlActive) {
        for (let j = 0; j < arm.indices.length; j++) {
          data.ctrl[arm.indices[j]] = s.targetJoints[j];
        }
      }
      // Always write gripper (last index) — independent of IK
      data.ctrl[arm.indices[arm.indices.length - 1]] = s.targetJoints[arm.indices.length - 1];
    }

    // === Head ===
    if (config.head) {
      const h = config.head;
      const hs = headState.current;

      if (k[h.keys[0]]) hs[0] += JOINT_STEP * 2;
      if (k[h.keys[1]]) hs[0] -= JOINT_STEP * 2;
      if (k[h.keys[2]]) hs[1] += JOINT_STEP * 2;
      if (k[h.keys[3]]) hs[1] -= JOINT_STEP * 2;

      data.ctrl[h.indices[0]] = hs[0];
      data.ctrl[h.indices[1]] = hs[1];
    }
  });
}
