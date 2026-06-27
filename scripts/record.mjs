// Drive the scripted expert headlessly and record a raw dataset to disk.
//   node scripts/record.mjs --episodes 60 --out data/raw --url http://localhost:3000/
// Each episode dir: frames.jsonl (t,state,action,image paths) + wrist/*.png + side/*.png.
// build_lerobot_dataset.py converts data/raw -> a LeRobotDataset for ACT training.
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const arg = (k, d) => {
  const i = process.argv.indexOf(k);
  return i >= 0 ? process.argv[i + 1] : d;
};

const url = arg('--url', 'http://localhost:3000/');
const outDir = path.resolve(arg('--out', 'data/raw'));
const episodes = Number(arg('--episodes', '60'));
const startIdx = Number(arg('--start', '0'));
const fps = Number(arg('--fps', '30'));

fs.mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 900, height: 600 } });
page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
if (process.argv.includes('--debug')) {
  page.on('console', (m) => {
    if (m.text().includes('EXPERT')) console.log('   ', m.text());
  });
}

console.log(`→ loading ${url}`);
// cams=0 disables the live camera panes (extra render load slows the sim and
// breaks the expert's wall-clock grasp timing).
const recUrl = url.includes('?') ? `${url}&cams=0` : `${url}?cams=0`;
await page.goto(recUrl, { waitUntil: 'networkidle', timeout: 60000 });
await page.waitForFunction(() => !!window.__lab?.recordEpisode, null, { timeout: 60000 });
await page.waitForTimeout(4000); // WASM warmup

const writePng = (file, dataUrl) => {
  const b64 = dataUrl.split(',', 2)[1] ?? '';
  fs.writeFileSync(file, Buffer.from(b64, 'base64'));
};

const recorded = [];
let ok = 0;
for (let i = startIdx; i < startIdx + episodes; i++) {
  // Resilient to HMR/navigation: if the page reloads (e.g. a file is saved
  // during a long record), re-wait for the bridge and retry this episode.
  let res;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await page.waitForFunction(() => !!window.__lab?.recordEpisode, null, { timeout: 30000 });
      res = await page.evaluate((idx) => window.__lab.recordEpisode(idx), i);
      break;
    } catch (e) {
      console.log(`  ep ${i}: reload/err (${String(e).slice(0, 50)}), retrying…`);
      await page.waitForTimeout(2000);
    }
  }
  const n = res?.frames?.length ?? 0;
  if (!res?.ok || !res.success || n < 10) {
    console.log(`  ep ${i}: SKIP (success=${res?.success} frames=${n})`);
    continue;
  }
  const epDir = path.join(outDir, `episode_${String(i).padStart(5, '0')}`);
  const camKeys = Object.keys(res.frames[0].images);
  for (const cam of camKeys) fs.mkdirSync(path.join(epDir, cam), { recursive: true });
  const lines = [];
  res.frames.forEach((f, fi) => {
    const name = `${String(fi).padStart(5, '0')}.png`;
    const row = { t: f.t, state: f.state, action: f.action };
    for (const cam of camKeys) {
      writePng(path.join(epDir, cam, name), f.images[cam]);
      row[cam] = `${cam}/${name}`;
    }
    lines.push(JSON.stringify(row));
  });
  fs.writeFileSync(path.join(epDir, 'frames.jsonl'), lines.join('\n') + '\n');
  recorded.push({ episode: i, frames: n, cube: res.cube, metrics: res.metrics });
  ok++;
  console.log(`  ep ${i}: OK ${n} frames`);
}

fs.writeFileSync(
  path.join(outDir, 'manifest.json'),
  JSON.stringify(
    {
      fps,
      robot_type: 'so101',
      task: 'pick up the red cube and place it on the green target',
      joints: ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'],
      state_units: 'degrees',
      action_units: 'degrees',
      cameras: { wrist: [240, 320, 3], front: [240, 320, 3] },
      episodes: recorded,
    },
    null,
    2,
  ),
);
console.log(`→ recorded ${ok}/${episodes} episodes into ${outDir}`);
await browser.close();
