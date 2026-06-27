// Unified policy eval: run a policy across cube positions, score whether the cube
// was actually MOVED onto the green target (not just started there).
//   node scripts/eval_policy.mjs --mode browser-act --runMs 38000 [--episodes 8]
//   node scripts/eval_policy.mjs --mode molmo --runMs 50000
import { chromium } from 'playwright';

const arg = (k, d) => { const i = process.argv.indexOf(k); return i >= 0 ? process.argv[i + 1] : d; };
const mode = arg('--mode', 'browser-act');
const runMs = Number(arg('--runMs', mode === 'molmo' ? 50000 : 38000));
const url = arg('--url', 'http://localhost:3000/');

// Right-side geometry: place pad BACK-right (0.62,-0.25), cube FRONT-right.
const TARGET = [0.62, -0.25];
const PLACED = 0.07;           // cube center within 7cm of target => on the pad
const MOVED = 0.03;            // cube displaced at least 3cm => policy touched it
const cubeBox = { x: [0.48, 0.51], y: [-0.42, -0.38] };
const N = Number(arg('--episodes', '8'));
const positions = Array.from({ length: N }, (_, i) => {
  const h1 = ((i * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const h2 = ((i * 1664525 + 1013904223) & 0x7fffffff) / 0x7fffffff;
  return [
    +(cubeBox.x[0] + h1 * (cubeBox.x[1] - cubeBox.x[0])).toFixed(3),
    +(cubeBox.y[0] + h2 * (cubeBox.y[1] - cubeBox.y[0])).toFixed(3),
  ];
});

const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 900, height: 640 } });
const errs = [];
page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text().slice(0, 100)); });
page.on('pageerror', (e) => errs.push('PE:' + e.message.slice(0, 100)));

await page.goto(`${url}?mode=${mode}&run=1`, { waitUntil: 'networkidle', timeout: 60000 });
await page.waitForFunction(() => !!window.__lab?.getCube, null, { timeout: 30000 });
// Let the policy/model load.
for (let i = 0; i < 30; i++) {
  await page.waitForTimeout(2000);
  const s = await page.evaluate(() => window.__actStatus || 'ready');
  if (s && !s.startsWith('loading')) break;
}
await page.evaluate(() => window.__lab.setRunning(true));
await page.waitForTimeout(1000);

let placed = 0, moved = 0;
const rows = [];
for (let i = 0; i < positions.length; i++) {
  const [x, y] = positions[i];
  await page.evaluate(() => window.__lab.reset());
  await page.evaluate(() => window.__lab.setRunning(true));
  await page.waitForTimeout(600);
  await page.evaluate(([x, y]) => window.__lab.placeCube(x, y), [x, y]);
  await page.waitForTimeout(1200);
  const start = await page.evaluate(() => window.__lab.getCube());
  await page.waitForTimeout(runMs);
  const end = await page.evaluate(() => window.__lab.getCube());
  await page.screenshot({ path: `/tmp/eval-${mode}-${i}.png` });
  const dT = end ? Math.hypot(end[0] - TARGET[0], end[1] - TARGET[1]) : 9;
  const dM = start && end ? Math.hypot(end[0] - start[0], end[1] - start[1]) : 0;
  const ok = dT < PLACED && dM > MOVED;
  if (ok) placed++;
  if (dM > MOVED) moved++;
  rows.push(`pos${i} (${x},${y}): moved ${dM.toFixed(3)}m, to-target ${dT.toFixed(3)}m => ${ok ? 'SUCCESS ✓' : (dM > MOVED ? 'moved, missed' : 'untouched')}`);
}

console.log(`\n=== ${mode} eval (${positions.length} eps, ${runMs / 1000}s each) ===`);
for (const r of rows) console.log(r);
console.log(`\nSUCCESS (placed on target): ${placed}/${positions.length}   |   cube touched: ${moved}/${positions.length}`);
if (errs.length) console.log('errors:', errs.slice(0, 3));
await browser.close();
