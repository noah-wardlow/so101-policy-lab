# Training runbook (record → dataset → train on a GPU pod)

How to (re)train the policies from scratch. The in-browser **scripted expert**
generates the data; a **GPU pod** (e.g. RunPod A40) trains ACT and/or fine-tunes
MolmoAct2. No secrets live here — pod IPs, API keys, and account IDs stay in your
shell env / a local (gitignored) notes file.

## ⚠️ The one rule: keep datasets LOCAL

Cheap pods are provisioned **without a persistent volume** (`volumeInGb: 0`), so
**stopping a pod wipes `/workspace` completely** — venv, datasets, checkpoints,
all gone. So:

- Record + build datasets **locally** and keep them (`data/` is gitignored).
- Treat the pod as disposable compute: rebuild the venv with one script, upload
  the dataset, train, then **pull artifacts off the pod before stopping it.**

## 1. Record the dataset (local)

Run the dev server (`npm run dev`), then drive the scripted expert headlessly:

```bash
node scripts/record.mjs --episodes 160 --out data/raw   # ~95% land; stop ~100+ good eps
```

Notes:
- Default `act` mode captures all 3 cameras (wrist+front+side); each successful
  episode is one `data/raw/episode_*/` dir. Failures are skipped.
- **Don't edit `src/` while recording** — HMR reloads the page and disrupts the
  in-flight episode (the script retries, but it wastes time).

Build the LeRobot dataset (needs `lerobot` — easiest to run this **on the pod**,
see below, or in any env with lerobot installed):

```bash
python scripts/build_lerobot_dataset.py \
  --raw data/raw --root data/lerobot --cameras wrist,front,side --overwrite
```

## 2. Provision + set up a pod

Requires `RUNPOD_API_KEY` in your environment (never commit it).

```bash
# provision (NVIDIA A40, ssh + http:8000), then wait for the SSH endpoint
python - <<'PY'
import os, sys; sys.path.insert(0,'scripts'); import runpod_api as r
pk = open(os.path.expanduser('~/.ssh/id_ed25519.pub')).read().strip()
print(r.create_pod('train', 'NVIDIA A40', pk, ['22/tcp','8000/http'])['id'])
PY
# get (ip, port):  python -c "import sys;sys.path.insert(0,'scripts');import runpod_api as r;print(r.ssh_endpoint('<POD_ID>'))"
```

Build the venv (one script — handles every env snag below):

```bash
ssh -p <PORT> root@<IP> 'mkdir -p /workspace/scripts'
scp -P <PORT> scripts/*.sh scripts/*.py root@<IP>:/workspace/scripts/
ssh -p <PORT> root@<IP> 'cd /workspace && bash scripts/pod_setup_venv.sh'   # ~5 min
```

## 3. Upload the dataset

`data/raw` is ~1 GB / tens of thousands of small files — stream it tarred (avoids
per-file SSH overhead), then build the LeRobot dataset on the pod:

```bash
tar -cf - -C data raw | ssh -p <PORT> root@<IP> 'mkdir -p /workspace/data && tar -xf - -C /workspace/data'
ssh -p <PORT> root@<IP> '/workspace/venv/bin/python /workspace/scripts/build_lerobot_dataset.py \
  --raw /workspace/data/raw --root /workspace/data/lerobot --cameras wrist,front,side --overwrite'
```

To put a built dataset on a **second** pod, stream pod→local→pod (the small built
`data/lerobot` is ~140 MB; pods can't SSH each other):

```bash
ssh -p <A_PORT> root@<A_IP> 'tar -cf - -C /workspace/data lerobot' \
  | ssh -p <B_PORT> root@<B_IP> 'mkdir -p /workspace/data && tar -xf - -C /workspace/data'
```

## 4. Train

**ACT** (3-cam, ~45 min on an A40):

```bash
ssh -p <PORT> root@<IP> 'cd /workspace && env \
  DATASET_REPO_ID=local/so101-pickplace DATASET_ROOT=/workspace/data/lerobot \
  OUTPUT_DIR=/workspace/act_out STEPS=30000 BATCH_SIZE=8 CHUNK=50 N_ACTION_STEPS=50 \
  DEVICE=cuda PYTHON=/workspace/venv/bin/python TRAIN_CMD=/workspace/venv/bin/lerobot-train \
  bash scripts/train_act.sh'
# export to ONNX, then pull act.onnx + policy.json into public/models/act/
ssh -p <PORT> root@<IP> '/workspace/venv/bin/python /workspace/scripts/export_act_to_onnx.py \
  --policy /workspace/act_out/checkpoints/last/pretrained_model --out /workspace/browser-act'
```

**MolmoAct2 LoRA fine-tune** (~4 h on an A40). Validate 3-cam support with a short
dry run first (prior fine-tunes were 2-cam):

```bash
ssh -p <PORT> root@<IP> 'cd /workspace && STEPS=10 BATCH=2 OUT=/workspace/molmo_dry \
  DATASET_ROOT=/workspace/data/lerobot bash scripts/pod_molmo_ft.sh'        # dry run
ssh -p <PORT> root@<IP> 'cd /workspace && DATASET_ROOT=/workspace/data/lerobot \
  bash scripts/pod_molmo_ft.sh'                                            # full run
```

Always **pull checkpoints to local before stopping the pod.**

## File map

- `scripts/pod_setup_venv.sh` — rebuild the pod venv (py3.12 + `lerobot[molmoact2,dataset]@536b962` + ffmpeg + onnx).
- `scripts/record.mjs` — drive the scripted expert headlessly → `data/raw`.
- `scripts/build_lerobot_dataset.py` — `data/raw` → a LeRobotDataset (`--cameras`).
- `scripts/train_act.sh` + `scripts/export_act_to_onnx.py` — ACT train + ONNX export.
- `scripts/pod_molmo_ft.sh` — MolmoAct2 LoRA fine-tune.
- `scripts/runpod_api.py` — pod provision / status / ssh-endpoint helpers (reads `RUNPOD_API_KEY`).
