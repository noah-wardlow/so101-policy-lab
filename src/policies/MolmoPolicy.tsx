import { useMemo, useRef } from 'react';
import { useRemotePolicy, useMujoco, applyPolicyActionToControls } from 'mujoco-react';
import type { CameraFrameCaptureSession, RemotePolicyConfig } from 'mujoco-react';
import {
  SO101_NUM_ACTUATORS,
  SO101_CAMERAS,
  CAM_RENDER_ISOLATION,
  simToPolicyDegrees,
  policyDegreesToSim,
  TASK_PROMPT,
} from '../robot/so101';

/**
 * Browser client for the FINE-TUNED MolmoAct2 server (server/molmo_ft_server.py).
 *
 * Because the model was LoRA fine-tuned on our own SO-101 sim data, it speaks our
 * convention directly: state/action are sim-joint DEGREES, cameras are the same
 * `wrist`+`front` views we recorded. So the transform here is IDENTITY — just
 * degrees<->radians (simToPolicyDegrees / policyDegreesToSim), no sign flips or
 * hand-tuned offsets.
 */
export function MolmoPolicy({
  enabled,
  endpoint,
}: {
  enabled: boolean;
  endpoint: string;
}) {
  const sim = useMujoco();
  const apiRef = useRef(sim.api);
  apiRef.current = sim.api;
  // One persistent capture session per camera — render-on-demand, reused, and
  // captured synchronously each inference (no async per-call offscreen render).
  const sessionsRef = useRef<Record<string, CameraFrameCaptureSession> | null>(null);
  const captureFrames = (): Record<string, string> => {
    const api = apiRef.current;
    if (!api) return {};
    if (!sessionsRef.current) {
      const s: Record<string, CameraFrameCaptureSession> = {};
      for (const c of SO101_CAMERAS) {
        s[c.key] = api.createCameraFrameCaptureSession({
          width: c.width,
          height: c.height,
          cameraName: c.cameraName,
          position: c.position,
          lookAt: c.lookAt,
          up: c.up,
          fov: c.fov,
          near: c.near,
          far: c.far,
          type: 'image/jpeg',
          quality: 0.9,
          renderIsolation: CAM_RENDER_ISOLATION,
        });
      }
      sessionsRef.current = s;
    }
    const images: Record<string, string> = {};
    for (const c of SO101_CAMERAS) images[c.key] = sessionsRef.current[c.key].captureDataUrl().dataUrl;
    return images;
  };

  const config = useMemo<RemotePolicyConfig>(
    () => ({
      enabled,
      endpoint,
      method: 'POST',
      frequency: 5, // VLA inference is slow; receding-horizon (replace stale chunk)
      queueStrategy: 'replace',
      prefetchThreshold: 1,
      onObservation: ({ data }) => simToPolicyDegrees(data.qpos),
      buildRequest: ({ observation, requestIndex }) => ({
        images: captureFrames(),
        state: Array.from(observation),
        task: TASK_PROMPT,
        reset: requestIndex === 0,
      }),
      parseResponse: (body) => {
        const actions = (body as { actions?: number[][] }).actions ?? [];
        return actions.map((a) => policyDegreesToSim(a)); // degrees -> sim radians (identity)
      },
      onAction: ({ action, model, data }) => {
        applyPolicyActionToControls(model, data, action, {
          actuatorOffset: 0,
          actionSize: SO101_NUM_ACTUATORS,
          clamp: true,
          skipInvalid: true,
        });
      },
    }),
    [enabled, endpoint],
  );

  useRemotePolicy(config);
  return null;
}
