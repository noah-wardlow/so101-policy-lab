import { useEffect, useState } from 'react';

import { ACT_MODELS, ACT_MODEL_IDS, isActModel } from './policies/actModels';
import type { ControlMode } from './router';

const MODES: { id: ControlMode; label: string }[] = [
  { id: 'teleop', label: 'Teleop (keyboard + gizmo)' },
  { id: 'expert', label: 'Scripted expert (IK teacher)' },
  ...ACT_MODEL_IDS.map((id) => ({ id: id as ControlMode, label: ACT_MODELS[id].label })),
  { id: 'molmo', label: 'MolmoAct2 (remote, fine-tuned)' },
];
const POLICY_MODES: ControlMode[] = [...ACT_MODEL_IDS, 'molmo'];

const panel: React.CSSProperties = {
  position: 'fixed',
  top: 14,
  right: 14,
  width: 290,
  maxHeight: 'calc(100vh - 28px)',
  overflowY: 'auto',
  background: 'rgba(15,23,42,0.92)',
  border: '1px solid rgba(148,163,184,0.18)',
  borderRadius: 12,
  color: '#e2e8f0',
  font: '13px system-ui, sans-serif',
  zIndex: 30,
  backdropFilter: 'blur(12px)',
  boxShadow: '0 10px 40px rgba(0,0,0,0.4)',
};
const section: React.CSSProperties = { padding: '12px 14px', borderTop: '1px solid rgba(148,163,184,0.12)' };
const heading: React.CSSProperties = { fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.6, color: '#94a3b8', marginBottom: 8, fontWeight: 700 };
const btn = (bg: string): React.CSSProperties => ({
  width: '100%', padding: '9px 12px', borderRadius: 8, border: 'none', cursor: 'pointer',
  background: bg, color: '#fff', fontSize: 13, fontWeight: 600, marginBottom: 6,
});
const smallBtn: React.CSSProperties = {
  flex: 1, padding: '7px 8px', borderRadius: 7, border: '1px solid rgba(148,163,184,0.25)',
  cursor: 'pointer', background: 'rgba(51,65,85,0.6)', color: '#e2e8f0', fontSize: 12, fontWeight: 600,
};

export interface HudProps {
  mode: ControlMode;
  setMode: (m: ControlMode) => void;
  running: boolean;
  setRunning: (r: boolean) => void;
  gizmo: boolean;
  setGizmo: (g: boolean) => void;
  molmoUrl: string;
  setMolmoUrl: (u: string) => void;
  onReset: () => void;
  onRandomize: () => void;
  onRunEpisode: () => void;
  episodeStatus: string;
}

export function Hud(props: HudProps) {
  const { mode, setMode, running, setRunning, gizmo, setGizmo } = props;
  const isPolicy = POLICY_MODES.includes(mode);
  const [policyStatus, setPolicyStatus] = useState('');

  // Poll the policy status the runners stash on window.
  useEffect(() => {
    const id = setInterval(() => {
      setPolicyStatus(isActModel(mode) ? window.__actStatus ?? '' : mode === 'molmo' ? 'remote VLA' : '');
    }, 400);
    return () => clearInterval(id);
  }, [mode]);

  return (
    <div style={panel}>
      <div style={{ padding: '13px 14px 4px' }}>
        <div style={{ fontWeight: 800, fontSize: 15, letterSpacing: 0.2 }}>SO-101 Policy Lab</div>
        <div style={{ color: '#64748b', fontSize: 11, marginTop: 1 }}>train + run policies in the browser</div>
      </div>

      {/* Policy */}
      <div style={section}>
        <div style={heading}>Policy</div>
        <select
          value={mode}
          onChange={(e) => { setRunning(false); setMode(e.target.value as ControlMode); }}
          style={{ width: '100%', padding: '8px', borderRadius: 8, background: '#1e293b', color: '#e2e8f0', border: '1px solid rgba(148,163,184,0.25)', fontSize: 13, marginBottom: 8 }}
        >
          {MODES.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
        </select>

        {isPolicy && (
          <>
            <button
              style={btn(running ? '#dc2626' : '#16a34a')}
              onClick={() => setRunning(!running)}
            >
              {running ? '■ Stop Policy' : '▶ Run Policy'}
            </button>
            <div style={{ color: '#94a3b8', fontSize: 11, minHeight: 14 }}>
              {policyStatus || 'idle'}
            </div>
          </>
        )}
        {mode === 'expert' && (
          <>
            <button style={btn('#2563eb')} onClick={props.onRunEpisode}>Run one episode</button>
            <div style={{ color: '#94a3b8', fontSize: 11, minHeight: 14 }}>{props.episodeStatus}</div>
          </>
        )}
        {mode === 'teleop' && (
          <div style={{ color: '#64748b', fontSize: 11 }}>WASD/QE move · RF wrist · ZC roll · V grip · drag the gizmo</div>
        )}
      </div>

      {/* Scene */}
      <div style={section}>
        <div style={heading}>Scene</div>
        <div style={{ display: 'flex', gap: 6 }}>
          <button style={smallBtn} onClick={props.onReset}>Reset</button>
          <button style={smallBtn} onClick={props.onRandomize}>Randomize cube</button>
        </div>
      </div>

      {/* View — the IK gizmo is a teleop tool (it drives the IK target); it has
          no effect while a policy or the scripted expert is driving the arm, so
          only offer it in teleop mode. */}
      {mode === 'teleop' && (
        <div style={section}>
          <div style={heading}>View</div>
          <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}>
            <span>IK gizmo</span>
            <input type="checkbox" checked={gizmo} onChange={(e) => setGizmo(e.target.checked)} />
          </label>
        </div>
      )}

      {mode === 'molmo' && (
        <div style={section}>
          <div style={heading}>Molmo endpoint</div>
          <input
            value={props.molmoUrl}
            onChange={(e) => props.setMolmoUrl(e.target.value)}
            style={{ width: '100%', padding: '6px 8px', borderRadius: 6, background: '#1e293b', color: '#cbd5e1', border: '1px solid rgba(148,163,184,0.2)', fontSize: 11 }}
          />
        </div>
      )}
    </div>
  );
}
