/**
 * Physics verifier — scores a scripted rollout so only genuine successes get
 * recorded. Gates (per the training-data spec):
 *   - both gripper jaws contact the cube (not a lucky nudge)
 *   - the cube lifts above a threshold
 *   - the cube stays held through the carry
 *   - the cube ends inside the place region
 *   - no invalid qpos (NaN / Inf / blow-up)
 */
import { getContact } from 'mujoco-react';
import type { MujocoModel, MujocoData } from 'mujoco-react';

export interface EpisodeMetrics {
  bothJaws: boolean;
  lifted: boolean;
  held: boolean;
  placed: boolean;
  qposSane: boolean;
  success: boolean;
}

/** Which of the given jaw bodies are currently touching the cube. */
export function jawCubeContacts(
  model: MujocoModel,
  data: MujocoData,
  cubeBodyId: number,
  jawBodyIds: number[],
): boolean[] {
  const touched = jawBodyIds.map(() => false);
  const bodyOf = (geom: number) => model.geom_bodyid[geom];
  const contacts = data.contact; // copied WASM handle — release when done
  try {
    for (let i = 0; i < data.ncon; i++) {
      const c = getContact(contacts, i);
      if (!c) continue;
      const b1 = bodyOf(c.geom1);
      const b2 = bodyOf(c.geom2);
      let other = -1;
      if (b1 === cubeBodyId) other = b2;
      else if (b2 === cubeBodyId) other = b1;
      else continue;
      const j = jawBodyIds.indexOf(other);
      if (j >= 0) touched[j] = true;
    }
  } finally {
    contacts.delete?.();
  }
  return touched;
}

/** True when the cube touches at least two distinct jaw bodies (opposing grasp). */
export function isCubeGrasped(
  model: MujocoModel,
  data: MujocoData,
  cubeBodyId: number,
  jawBodyIds: number[],
): boolean {
  const touched = jawCubeContacts(model, data, cubeBodyId, jawBodyIds);
  return touched.filter(Boolean).length >= 2;
}

export function qposIsSane(data: MujocoData, n = 6): boolean {
  for (let i = 0; i < n; i++) {
    const v = data.qpos[i];
    if (!Number.isFinite(v) || Math.abs(v) > 1e3) return false;
  }
  return true;
}
