import type { RefObject } from 'react';
import { useCameraStream } from 'mujoco-react';
import { CAM_RENDER_ISOLATION } from './robot/so101';
import type { So101CameraDef } from './robot/so101';

/** One live camera stream rendered into a DOM <canvas> (mounted inside the R3F Canvas). */
function CameraStream({
  canvasRef,
  cam,
}: {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  cam: So101CameraDef;
}) {
  useCameraStream(canvasRef, {
    width: 256,
    height: 192,
    cameraName: cam.cameraName,
    position: cam.position,
    lookAt: cam.lookAt,
    up: cam.up,
    fov: cam.fov,
    near: cam.near,
    far: cam.far,
    fps: 12,
    renderIsolation: CAM_RENDER_ISOLATION,
  });
  return null;
}

/** Streams every observation camera. Mount INSIDE <MujocoCanvas>. */
export function CameraStreamer({
  refs,
  cameras,
}: {
  refs: Record<string, RefObject<HTMLCanvasElement | null>>;
  cameras: So101CameraDef[];
}) {
  return (
    <>
      {cameras.map((cam) =>
        refs[cam.key] ? <CameraStream key={cam.key} canvasRef={refs[cam.key]} cam={cam} /> : null,
      )}
    </>
  );
}

/** The HTML overlay panel of camera panes. Mount OUTSIDE the R3F Canvas. */
export function CameraPanel({
  refs,
  cameras,
}: {
  refs: Record<string, RefObject<HTMLCanvasElement | null>>;
  cameras: So101CameraDef[];
}) {
  return (
    <section
      style={{
        position: 'fixed',
        left: 14,
        bottom: 14,
        display: 'flex',
        gap: 10,
        padding: 10,
        background: 'rgba(15,23,42,0.9)',
        border: '1px solid rgba(148,163,184,0.18)',
        borderRadius: 12,
        zIndex: 25,
        backdropFilter: 'blur(10px)',
      }}
    >
      {cameras.map((cam) => (
        <div key={cam.key} style={{ width: 168 }}>
          <div
            style={{
              position: 'relative',
              aspectRatio: '4 / 3',
              overflow: 'hidden',
              borderRadius: 8,
              background: '#020617',
              border: '1px solid rgba(148,163,184,0.18)',
            }}
          >
            <canvas
              ref={refs[cam.key]}
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
            />
          </div>
          <div
            style={{
              marginTop: 5,
              color: '#94a3b8',
              font: '11px system-ui, sans-serif',
              textTransform: 'uppercase',
              letterSpacing: 0.5,
            }}
          >
            {cam.key} cam · policy view
          </div>
        </div>
      ))}
    </section>
  );
}
