import { useArmController } from './useArmController';
import type { ArmControllerConfig } from './useArmController';
import type { IkContextValue } from 'mujoco-react';

const config: ArmControllerConfig = {
  numActuators: 6,
  arms: [{
    indices: [0, 1, 2, 3, 4, 5],
    keys: ['KeyD', 'KeyA', 'KeyW', 'KeyS', 'KeyQ', 'KeyE', 'KeyR', 'KeyF', 'KeyZ', 'KeyC', 'KeyV'],
    initialJoints: [0, -1.5707963268, 1.5707963268, 0.659999464, 0, -0.17453],
    gripperClosed: -0.17453,
  }],
};

export function SO101Controller({ ik }: { ik?: IkContextValue | null }) {
  useArmController(config, ik);
  return null;
}
