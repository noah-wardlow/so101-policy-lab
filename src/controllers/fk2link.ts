/**
 * 2-link forward kinematics — inverse of inverseKinematics2Link.
 * Given (shoulder, elbow) joint angles, returns end-effector (x, y) in the sagittal plane.
 */

import type { LinkageParams } from './ik2link';
import { SO101_LINKAGE } from './ik2link';

export function forwardKinematics2Link(
  joint2: number,
  joint3: number,
  p: LinkageParams = SO101_LINKAGE,
): [number, number] {
  const { l1, l2, theta1Offset, theta2Offset } = p;

  const theta1 = joint2 - theta1Offset;
  const theta2 = joint3 - theta2Offset;

  const r = Math.sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * Math.cos(theta2));
  const gamma = Math.atan2(l2 * Math.sin(theta2), l1 + l2 * Math.cos(theta2));
  const beta = theta1 - gamma;

  return [r * Math.cos(beta), r * Math.sin(beta)];
}
