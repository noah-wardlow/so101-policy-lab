/**
 * 2-link analytical inverse kinematics for planar arms.
 * Converts end-effector (x, y) position to (shoulder, elbow) joint angles.
 */

export interface LinkageParams {
  /** Upper arm link length */
  l1: number;
  /** Forearm link length */
  l2: number;
  /** Shoulder angle offset (rad) */
  theta1Offset: number;
  /** Elbow angle offset (rad) */
  theta2Offset: number;
  /** Joint 2 clamp range [min, max] */
  joint2Range: [number, number];
  /** Joint 3 clamp range [min, max] */
  joint3Range: [number, number];
}

/** Default linkage for SO101 / XLeRobot arms */
export const SO101_LINKAGE: LinkageParams = {
  l1: 0.1159,
  l2: 0.1350,
  theta1Offset: Math.atan2(0.028, 0.11257),
  theta2Offset: Math.atan2(0.0052, 0.1349) + Math.atan2(0.028, 0.11257),
  joint2Range: [-0.1, 3.45],
  joint3Range: [-0.2, Math.PI],
};

export function inverseKinematics2Link(
  x: number,
  y: number,
  p: LinkageParams = SO101_LINKAGE,
): [number, number] {
  const { l1, l2, theta1Offset, theta2Offset, joint2Range, joint3Range } = p;

  let r = Math.sqrt(x * x + y * y);
  const rMax = l1 + l2;
  const rMin = Math.abs(l1 - l2);

  if (r > rMax) {
    const s = rMax / r;
    x *= s;
    y *= s;
    r = rMax;
  }
  if (r < rMin && r > 0) {
    const s = rMin / r;
    x *= s;
    y *= s;
    r = rMin;
  }

  const cosTheta2 = -(r * r - l1 * l1 - l2 * l2) / (2 * l1 * l2);
  const theta2 = Math.PI - Math.acos(Math.max(-1, Math.min(1, cosTheta2)));

  const beta = Math.atan2(y, x);
  const gamma = Math.atan2(l2 * Math.sin(theta2), l1 + l2 * Math.cos(theta2));
  const theta1 = beta + gamma;

  const joint2 = Math.max(joint2Range[0], Math.min(joint2Range[1], theta1 + theta1Offset));
  const joint3 = Math.max(joint3Range[0], Math.min(joint3Range[1], theta2 + theta2Offset));

  return [joint2, joint3];
}
