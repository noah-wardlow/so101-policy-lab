import { forwardRef, useImperativeHandle, useRef } from 'react';
import * as THREE from 'three';
import { useBeforePhysicsStep, useControls, findBodyByName, findGeomByName } from 'mujoco-react';
import { GRIPPER_CONTROLS } from '../robot/controls';
import type { IkContextValue, MujocoModel, MujocoData } from 'mujoco-react';
import {
  GRIPPER_CLOSED,
  GRIPPER_OPEN,
  GRIPPER_INDEX,
  SO101_ARM_DOF,
  SO101_HOME,
  BASE_XY,
  CUBE_REST_Z,
  clampCtrl,
  placeFreeBody,
} from '../robot/so101';
import { isCubeGrasped, qposIsSane } from '../robot/verifier';
import type { EpisodeMetrics } from '../robot/verifier';

const LIFT_THRESHOLD = 0.03; // cube must rise this far to count as lifted

export interface ExpertHandle {
  runEpisode: () => Promise<EpisodeMetrics>;
  placeCube: (x: number, y: number) => void;
  getCube: () => [number, number, number] | null;
  /** One-shot reachability probe for a top-down grasp at (x,y,z) — no arm motion. */
  reachTest: (x: number, y: number, z: number) => boolean;
}

/** Gripper points straight down (tool z aligned with -world z) for top-down grasp. */
const GRASP_DOWN = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI, 0, 0));

const HOVER_DZ = 0.06; // hover height above grasp point (z reach maxes ~0.90)
const GRASP_DZ = -0.006; // gripperframe at block mid-height for a stable grip
const LIFT_DZ = 0.055;

function sleep(ms: number) {
  return new Promise((res) => setTimeout(res, ms));
}

export const ScriptedExpert = forwardRef<ExpertHandle, { ik: IkContextValue | null }>(
  function ScriptedExpert({ ik }, ref) {
    const modelRef = useRef<MujocoModel | null>(null);
    const dataRef = useRef<MujocoData | null>(null);
    const cubeBodyRef = useRef<number>(-1);
    const jawBodiesRef = useRef<number[]>([]);
    // Jaw collision geoms whose world positions define the grasp POCKET (the gap
    // between the two jaws). Targeting the pocket (not the gripperframe site,
    // which is offset toward the fixed jaw) is what makes the cube land between
    // both jaws instead of getting poked by the fixed one.
    const pocketGeomsRef = useRef<{ fixed: number[]; moving: number[] }>({ fixed: [], moving: [] });
    const gripperTargetRef = useRef<number>(GRIPPER_OPEN);
    const activeRef = useRef(false); // only drive the gripper while an episode runs
    // Type-safe gripper writes by name. Gripper-only group so it never conflicts
    // with the IK controller's ownership of the arm actuators.
    const gripperControls = useControls(GRIPPER_CONTROLS);

    // Each physics step: cache model/data and (while active) hold the gripper at
    // its target. IK owns the 5 arm actuators; we own the gripper. When idle we
    // touch nothing so teleop/gizmo control stays free.
    useBeforePhysicsStep(({ model, data }) => {
      modelRef.current = model;
      dataRef.current = data;
      if (cubeBodyRef.current < 0) {
        cubeBodyRef.current = findBodyByName(model, 'red_cube');
        jawBodiesRef.current = [
          findBodyByName(model, 'gripper'), // fixed jaw + structure
          findBodyByName(model, 'moving_jaw_so101_v1'), // moving jaw
        ].filter((id) => id >= 0);
        pocketGeomsRef.current = {
          fixed: ['fixed_jaw_box3', 'fixed_jaw_box4', 'fixed_jaw_box5', 'fixed_jaw_box6']
            .map((n) => findGeomByName(model, n)).filter((id) => id >= 0),
          moving: ['moving_jaw_box2', 'moving_jaw_box3']
            .map((n) => findGeomByName(model, n)).filter((id) => id >= 0),
        };
      }
      if (activeRef.current) {
        gripperControls.set('gripper', clampCtrl(GRIPPER_INDEX, gripperTargetRef.current));
      }
    });

    const grasped = (): boolean => {
      const model = modelRef.current;
      const data = dataRef.current;
      if (!model || !data) return false;
      return isCubeGrasped(model, data, cubeBodyRef.current, jawBodiesRef.current);
    };

    /** Sample the two-sided (both-jaw) grasp over a few frames — majority wins. */
    const sampleTwoSided = async (samples: number): Promise<boolean> => {
      let hits = 0;
      for (let i = 0; i < samples; i++) {
        if (grasped()) hits++;
        await sleep(40);
      }
      return hits * 2 > samples;
    };

    const readEePos = (): THREE.Vector3 | null => {
      const data = dataRef.current;
      const siteId = ik?.siteIdRef.current ?? -1;
      if (!data || siteId < 0) return null;
      const p = data.site_xpos.subarray(siteId * 3, siteId * 3 + 3);
      return new THREE.Vector3(p[0], p[1], p[2]);
    };

    /** World-space center of the gripper pocket (mean of fixed + moving jaw geoms). */
    const readPocketCenter = (): THREE.Vector3 | null => {
      const data = dataRef.current;
      const { fixed, moving } = pocketGeomsRef.current;
      if (!data || fixed.length === 0 || moving.length === 0) return null;
      const meanOf = (ids: number[]) => {
        const v = new THREE.Vector3();
        for (const id of ids) v.add(new THREE.Vector3(data.geom_xpos[id * 3], data.geom_xpos[id * 3 + 1], data.geom_xpos[id * 3 + 2]));
        return v.multiplyScalar(1 / ids.length);
      };
      // Pocket = midpoint between the fixed-jaw cluster and the moving-jaw cluster.
      return meanOf(fixed).add(meanOf(moving)).multiplyScalar(0.5);
    };

    const readCubePos = (): THREE.Vector3 | null => {
      const data = dataRef.current;
      const id = cubeBodyRef.current;
      if (!data || id < 0) return null;
      const p = data.xpos.subarray(id * 3, id * 3 + 3);
      return new THREE.Vector3(p[0], p[1], p[2]);
    };

    /** Command an IK target pose and wait until the EE settles near it. */
    const moveTo = async (
      label: string,
      pos: THREE.Vector3,
      { eps = 0.012, timeoutMs = 2500 }: { eps?: number; timeoutMs?: number } = {},
    ) => {
      if (!ik) return false;
      ik.setIkEnabled(true);
      const target = ik.ikTargetRef.current;
      target.position.copy(pos);
      target.quaternion.copy(GRASP_DOWN);
      const start = performance.now();
      // Poll until the site reaches the commanded position (or timeout).
      // eslint-disable-next-line no-constant-condition
      while (true) {
        await sleep(33);
        const ee = readEePos();
        const d = ee ? ee.distanceTo(pos) : 999;
        if (d < eps) {
          console.log(
            `EXPERT ${label}: reached target=(${pos.x.toFixed(3)},${pos.y.toFixed(3)},${pos.z.toFixed(3)}) d=${d.toFixed(4)}`,
          );
          return true;
        }
        if (performance.now() - start > timeoutMs) {
          console.log(
            `EXPERT ${label}: TIMEOUT target=(${pos.x.toFixed(3)},${pos.y.toFixed(3)},${pos.z.toFixed(3)}) ee=(${ee?.x.toFixed(3)},${ee?.y.toFixed(3)},${ee?.z.toFixed(3)}) d=${d.toFixed(4)}`,
          );
          return false;
        }
      }
    };

    const setGripper = async (value: number, dwellMs = 350) => {
      gripperTargetRef.current = value;
      await sleep(dwellMs);
    };

    const placeCube = (x: number, y: number) => {
      const model = modelRef.current;
      const data = dataRef.current;
      if (model && data && cubeBodyRef.current >= 0) {
        placeFreeBody(model, data, cubeBodyRef.current, x, y, CUBE_REST_Z);
      }
    };

    const getCube = (): [number, number, number] | null => {
      const p = readCubePos();
      return p ? [p.x, p.y, p.z] : null;
    };

    const FAIL: EpisodeMetrics = {
      bothJaws: false,
      lifted: false,
      held: false,
      placed: false,
      qposSane: true,
      success: false,
    };

    const runEpisode = async (): Promise<EpisodeMetrics> => {
      const cube = readCubePos();
      if (!cube) return FAIL;
      activeRef.current = true;
      try {
        return await runEpisodeInner(cube);
      } finally {
        activeRef.current = false;
      }
    };

    const runEpisodeInner = async (cube: THREE.Vector3): Promise<EpisodeMetrics> => {
      console.log(`EXPERT start: cube=(${cube.x.toFixed(3)},${cube.y.toFixed(3)},${cube.z.toFixed(3)})`);

      // 1-4. Grasp by servoing the POCKET (the gap between the jaws) onto the
      // cube — not the gripperframe site, which sits toward the fixed jaw and
      // makes the cube get poked instead of captured. We close-loop the hover so
      // the pocket centers over the cube in XY, then descend driving the pocket
      // down onto the cube center, then close.
      let bothJaws = false;
      let lifted = false;
      let afterLift: THREE.Vector3 | null = null;
      for (let attempt = 0; attempt < 3 && !(bothJaws && lifted); attempt++) {
        const c = readCubePos() ?? cube; // cube may have shifted on a failed try
        gripperTargetRef.current = GRIPPER_OPEN;

        // Hover: center the pocket over the cube in XY at a safe height. The IK
        // target is the site; we push it by the pocket-vs-cube error each pass.
        const tgt = new THREE.Vector3(c.x, c.y, c.z + HOVER_DZ);
        for (let k = 0; k < 6; k++) {
          await moveTo('align', tgt, { eps: 0.005, timeoutMs: 1500 });
          const pk = readPocketCenter();
          if (!pk) break;
          const ex = c.x - pk.x;
          const ey = c.y - pk.y;
          if (Math.hypot(ex, ey) < 0.004) break;
          tgt.x += ex;
          tgt.y += ey;
        }
        // Descend straight down to the grasp height, keeping the pocket-centered
        // XY target (the site Z floor is enforced by the IK; don't servo Z or it
        // diverges chasing a pocket that can't physically go lower).
        await moveTo('descend', new THREE.Vector3(tgt.x, tgt.y, c.z + GRASP_DZ), { eps: 0.008, timeoutMs: 1800 });
        await sleep(150);
        await setGripper(GRIPPER_CLOSED, 600);
        bothJaws = await sampleTwoSided(5);
        await moveTo('lift', new THREE.Vector3(tgt.x, tgt.y, c.z + LIFT_DZ));
        afterLift = readCubePos();
        lifted = !!afterLift && afterLift.z > cube.z + LIFT_THRESHOLD;
        if (!(bothJaws && lifted) && attempt < 2)
          console.log(`EXPERT regrasp (attempt ${attempt + 1}: bothJaws=${bothJaws} lifted=${lifted})`);
      }

      // 5. move over the drop target, tracking that the grasp is held throughout
      const model = modelRef.current;
      const data = dataRef.current;
      let dropX = 0.42;
      let dropY = -0.18;
      if (model && data) {
        const tId = findBodyByName(model, 'green_target');
        if (tId >= 0) {
          const tp = data.xpos.subarray(tId * 3, tId * 3 + 3);
          dropX = tp[0];
          dropY = tp[1];
        }
      }
      await moveTo('carry', new THREE.Vector3(dropX, dropY, cube.z + LIFT_DZ));
      // held := cube still up at the drop point AND still gripped by both jaws.
      // (We now require a genuine two-sided grip — see `success` — so one-sided
      // pokes that happen to carry the cube no longer pollute the dataset.)
      const carried = readCubePos();
      const held = !!carried && carried.z > cube.z + LIFT_THRESHOLD && grasped();

      // 6. lower + release
      await moveTo('place', new THREE.Vector3(dropX, dropY, cube.z + 0.02), { eps: 0.015 });
      await setGripper(GRIPPER_OPEN, 400);
      // 7. retreat straight UP off the cube (the arm's reach ceiling is ~0.90, so
      //    this clears it by only ~10cm — not enough to fold home directly above).
      await moveTo('retreat', new THREE.Vector3(dropX, dropY, cube.z + 0.12));
      // 7b. pull the gripper BACK toward the base (−X) at the same height, so the
      //     home-fold below doesn't start directly above the cube. Makes the terminal
      //     pose (gripper retracted, far from the cube) visually distinct from the
      //     "about to place" pose — which should help the policy tell "done" apart
      //     from "place again" instead of looping a second place at the terminal.
      await moveTo('clear', new THREE.Vector3(BASE_XY[0] + 0.13, dropY, cube.z + 0.12), {
        eps: 0.02,
      });
      // 8. settle to the home/neutral pose. Gentle ramp (small gain, more steps) so
      //    the demo terminal is smooth, not a jerk. (Arm via direct ctrl with IK
      //    disabled — IK owns these actuators cooperatively.)
      ik?.setIkEnabled(false);
      for (let s = 0; s < 34; s++) {
        const d = dataRef.current;
        if (d) for (let i = 0; i < SO101_ARM_DOF; i++) d.ctrl[i] += (SO101_HOME[i] - d.ctrl[i]) * 0.13;
        await sleep(33);
      }
      await sleep(700); // hold at neutral so the recorder captures a stable end

      const finalCube = readCubePos();
      const placed =
        !!finalCube &&
        Math.hypot(finalCube.x - dropX, finalCube.y - dropY) < 0.06 &&
        finalCube.z < cube.z + 0.02; // settled back on the table
      const qposSane = !!data && qposIsSane(data);
      // Require a genuine two-sided grasp so the recorded demonstrations show a
      // clean pick (not a fixed-jaw poke-and-carry) for the policy to imitate.
      const success = bothJaws && lifted && held && placed && qposSane;
      const metrics: EpisodeMetrics = { bothJaws, lifted, held, placed, qposSane, success };
      console.log(`EXPERT done: ${JSON.stringify(metrics)} finalCube=(${finalCube?.x.toFixed(3)},${finalCube?.y.toFixed(3)},${finalCube?.z.toFixed(3)})`);
      return metrics;
    };

    // Reachability via the lib's one-shot IK solve (returns null if no solution)
    // — the clean replacement for the old "drive the arm and measure residual"
    // probe: same answer, no arm motion, no scene disturbance.
    const reachTest = (x: number, y: number, z: number) =>
      ik?.solveIK({
        position: [x, y, z],
        quaternion: GRASP_DOWN,
        currentQ: SO101_HOME.slice(0, SO101_ARM_DOF),
      }) != null;

    useImperativeHandle(ref, () => ({ runEpisode, placeCube, getCube, reachTest }), [ik]);

    return null;
  },
);
