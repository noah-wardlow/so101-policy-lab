/**
 * Type-safe SO-101 control definitions. `defineControls` maps friendly aliases to
 * model actuator names; the mujoco-react Vite plugin constrains those names to the
 * loaded model (a wrong/stale name is a compile error) — no manual register import.
 * Controllers read/write by alias via `useControls()` (`controls.set('gripper', v)`).
 *
 * NOTE: `useControls` claims cooperative ownership of its actuators on mount, and
 * the library's `useIkController` already owns the ARM actuators — so controllers
 * that coexist with IK only declare the actuators they actually drive (the gripper).
 */
import { defineControls } from 'mujoco-react';

/** Gripper-only — safe to mount alongside the IK controller (which owns the arm). */
export const GRIPPER_CONTROLS = defineControls({ gripper: 'gripper' });
