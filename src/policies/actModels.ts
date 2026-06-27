/**
 * The ACT models exposed in the HUD for comparison. Each has its own ONNX +
 * manifest under public/models/<id>/ and declares which cameras it consumes
 * (must match its policy.json image inputs). One `BrowserActPolicy` component is
 * mounted per selected model, keyed by id, so switching fully remounts it
 * (fresh ONNX session + camera streams).
 *
 *   A — scripted-expert demos, wrist+front
 *   C — scripted-expert demos, wrist+front+side (3-cam ablation)
 */
export type ActModelId = 'act-a' | 'act-c';

export interface ActModelDef {
  id: ActModelId;
  label: string;
  manifestUrl: string;
  cameraKeys: string[];
}

// Where the model files live. Local dev serves them from public/models; in prod
// set VITE_MODEL_BASE to an object store (Cloudflare R2 / S3) since the ~137MB
// ONNX files exceed static-host per-file limits. The .onnx path inside each
// policy.json resolves relative to its manifest, so both come from the same base.
const MODEL_BASE = import.meta.env.VITE_MODEL_BASE ?? '/models';
const manifest = (id: ActModelId) => `${MODEL_BASE}/${id}/policy.json`;

export const ACT_MODELS: Record<ActModelId, ActModelDef> = {
  'act-a': {
    id: 'act-a',
    label: 'ACT-A · 2-cam (wrist+front)',
    manifestUrl: manifest('act-a'),
    cameraKeys: ['wrist', 'front'],
  },
  'act-c': {
    id: 'act-c',
    label: 'ACT-C · 3-cam (wrist+front+side)',
    manifestUrl: manifest('act-c'),
    cameraKeys: ['wrist', 'front', 'side'],
  },
};

export const ACT_MODEL_IDS = Object.keys(ACT_MODELS) as ActModelId[];
export const isActModel = (m: string): m is ActModelId => m in ACT_MODELS;
