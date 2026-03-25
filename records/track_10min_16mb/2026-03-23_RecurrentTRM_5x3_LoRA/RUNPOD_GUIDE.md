# RunPod 1×H100 Operator's Guide — Recurrent TRM 9×3 + LoRA

*For a machine learning engineer running a single training job on a 1×H100 80GB RunPod pod, from pod launch to safely stopped.*

---

## Phase 1 — Launch and Connect to the Pod

**What:** Create a 1×H100 pod on RunPod and open an SSH connection.

**Why:** Everything else runs inside the pod. You need SSH set up before you can do anything.

1. Log in to [console.runpod.io](https://console.runpod.io).
2. Deploy using the official Parameter Golf template:
   [Launch Template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th)
   Choose **1×H100 80GB SXM**. Enable **SSH terminal access**. Leave other settings at defaults.
3. Wait for the pod status to show **Running**. Click **Connect** → copy the SSH command. It will look like:
   ```
   ssh root@[IP] -p [PORT]
   ```
4. From your **local terminal**, connect:
   ```bash
   ssh root@[IP] -p [PORT]
   ```
   If you set up an SSH key in RunPod settings, it will authenticate automatically. If not, RunPod shows a one-time password on the Connect screen.

> **Cost note:** The pod starts billing the moment it enters Running state. Check the current 1×H100 rate on RunPod's pricing page before launching. Stop the pod (Phase 8) when done — do not leave it idle.

---

## Phase 2 — Sanity Check the GPU

**What:** Verify the GPU is visible and healthy before doing anything else.

**Why:** Occasionally RunPod allocates a pod whose GPU driver is not ready. Catch this early.

```bash
nvidia-smi
```

**Good output looks like:**

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx    Driver Version: 535.xx    CUDA Version: 12.x          |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| 0  NVIDIA H100 80GB HBM3  Off |  00000000:xx:xx.x Off |                    0 |
| N/A   33C    P0    72W / 700W |      0MiB / 81559MiB |      0%      Default |
+-----------------------------------------------------------------------------+
```

Key things to confirm:
- Name contains **H100**
- Memory shows near **81559 MiB** free (nothing else is using it yet)
- No `ERR!` or `N/A` in the Uncorr. ECC column

If the GPU does not appear at all, terminate the pod and launch a new one.

---

## Phase 3 — Clone or Update the Repository

**What:** Get the latest code from `main` onto the pod. RunPod may already have a stale clone from a previous session — the steps below handle both cases.

**Why:** Your recent fixes (GQA patch, `run_h100.sh` corrections, updated docs) live on `main`. Training against stale code will reproduce the bugs you already fixed.

### If the repo does not exist yet on the pod

```bash
cd /workspace
git clone https://github.com/SonnyZhan/parameter-golf.git
cd parameter-golf
```

### If the repo already exists (stale clone from a previous session)

```bash
cd /workspace/parameter-golf
git fetch origin
git checkout main
git pull origin main
```

If `git pull` complains about local changes that would be overwritten (e.g. generated files), discard them first:

```bash
git checkout main
git fetch origin
git reset --hard origin/main
```

> `git reset --hard origin/main` discards any local modifications and sets the working tree to exactly what is on `main`. Only run this if you are sure you have not made changes on the pod that you need to keep. Log files and model artifacts in `logs/` and `final_model.*` are not tracked by git, so they are unaffected.

**Verify you are on `main` and the experiment folder is current:**

```bash
git branch --show-current       # should print: main
git log --oneline -5            # should show your recent commits
ls records/track_10min_16mb/2026-03-23_RecurrentTRM_5x3_LoRA/
```

Expected files: `train_gpt.py`, `run_h100.sh`, `README.md`, `submission.json`, `architecture_notes.md`, `smoke_test.ps1`, `logs/`.

---

## Phase 4 — Install Dependencies

**What:** The RunPod Parameter Golf template pre-installs most packages. Verify and fill any gaps.

**Why:** `train_gpt.py` imports `sentencepiece` for tokenizer handling; it is sometimes missing from base images.

```bash
python -c "import torch; print(torch.__version__)"
python -c "import sentencepiece; print('sentencepiece ok')"
```

If `sentencepiece` is missing:

```bash
pip install sentencepiece --quiet
```

If `torch` is below 2.5.0, `run_h100.sh` will upgrade it automatically in Phase 6. You can also do it now:

```bash
pip install --upgrade "torch>=2.5.0" --index-url https://download.pytorch.org/whl/cu121 --quiet
```

> `run_h100.sh` checks the PyTorch version at startup and runs this `pip install` automatically if needed. You may see a pip upgrade step before training starts — this is expected, not an error.

---

## Phase 5 — Download the Competition Dataset

**What:** Download the FineWeb dataset with the 1024-token vocabulary to `data/datasets/fineweb10B_sp1024/`.

**Why:** Training data must be local on the pod. This download takes several minutes depending on bandwidth.

From the repo root:

```bash
cd /workspace/parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This downloads:
- Full validation split → `data/datasets/fineweb10B_sp1024/fineweb_val_*.bin`
- 80 training shards (8B tokens) → `data/datasets/fineweb10B_sp1024/fineweb_train_*.bin`
- Tokenizer → `data/tokenizers/fineweb_1024_bpe.model`

**Verify the download completed:**

```bash
ls data/datasets/fineweb10B_sp1024/ | head -5
ls -lh data/tokenizers/fineweb_1024_bpe.model
```

You should see multiple `fineweb_train_*.bin` and `fineweb_val_*.bin` files, and a `.model` file of a few MB.

> If the data was already downloaded in a previous session on a **stopped** (not terminated) pod, it will still be there. The `ls` check above is sufficient to confirm — you do not need to re-download.

---

## Phase 6 — Smoke Test (5 Iterations, No tmux Needed)

**What:** Run 5 training steps to confirm the model initialises, data loads, and a forward+backward pass completes without crashing.

**Why:** Smoke tests catch environment issues (wrong CUDA version, missing data, misconfigured paths) in under a minute, before you commit to a long run.

Navigate to the experiment folder:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-03-23_RecurrentTRM_5x3_LoRA
```

Run the smoke test (single process, no `torchrun`, skips validation):

```bash
RUN_ID=smoke_h100_5iter \
ITERATIONS=5 \
MAX_WALLCLOCK_SECONDS=0 \
WARMUP_STEPS=2 \
WARMDOWN_ITERS=0 \
VAL_LOSS_EVERY=0 \
TRAIN_BATCH_TOKENS=32768 \
TRAIN_SEQ_LEN=512 \
NUM_LAYERS=9 \
NUM_PASSES=3 \
LORA_RANK=4 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
VOCAB_SIZE=1024 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
python train_gpt.py
```

> `MAX_WALLCLOCK_SECONDS=0` disables the wall-clock cap entirely for this check — without it the default 600-second cap would apply but would not be a problem for 5 steps. Setting it to 0 makes the intent explicit.

**Good smoke output looks like:**

```
model_params:21978184
--- param breakdown ---
  param: tok_emb.weight  shape=(1024, 512)  n=524288
  ...
--- end param breakdown ---
step:1/5  train_loss:...
step:2/5  train_loss:...
...
step:5/5  train_loss:...
```

**If it crashes here, see Failure Triage before proceeding.**

---

## Phase 7 — Real Training Run (tmux + 1×H100)

### 7a. Start a tmux session

**What:** Wrap the training in tmux so an SSH disconnect does not kill the job.

**Why:** Training takes up to 10 minutes. Any SSH drop without tmux kills the process and you lose everything.

```bash
tmux new-session -s train
```

You are now inside a tmux session named `train`. All subsequent commands in this phase run here.

**Essential tmux commands (memorise before starting):**

| Action | Keys |
| --- | --- |
| Detach (leave job running) | `Ctrl+B`, then `D` |
| Reattach from outside | `tmux attach -t train` |
| Scroll up in output | `Ctrl+B`, then `[`, then arrow keys or PgUp; `Q` to exit scroll |
| Kill session (only after run is done) | `tmux kill-session -t train` |

### 7b. Launch training

Make sure you are in the experiment folder inside the tmux session:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-03-23_RecurrentTRM_5x3_LoRA
```

Run the production script with **1 GPU**:

```bash
NPROC_PER_NODE=1 bash run_h100.sh
```

> **Critical:** The script defaults to `torchrun --nproc_per_node=8`, which would launch 8 worker processes on a single GPU — this will either fail or produce wrong results. Always pass `NPROC_PER_NODE=1` on a 1×H100 pod.

> The script enforces `MAX_WALLCLOCK_SECONDS=600` (10 minutes) and `ITERATIONS=20000`. Training stops at whichever limit comes first — on 1×H100 it will hit the wall clock, not the iteration count. A "bigger run" means a full 10-minute budgeted session, not more than 20000 steps.

**What you will see:**

1. A PyTorch version check (and a pip install if torch < 2.5 — normal, wait for it).
2. A `========================================` banner with run name and settings.
3. `model_params:21978184` and the parameter breakdown.
4. Periodic `step:N/M  train_loss:X.XXXX` lines every 50 steps.
5. Periodic `val_loss:X.XXXX  val_bpb:X.XXXX` lines every 200 steps.
6. At the end: `final_int8_zlib_roundtrip_exact val_bpb:X.XXXX` and `Total submission size int8+zlib: XXXXXXXX bytes`.

The log is simultaneously written to `logs/<run_name>.log`.

### 7c. Monitor GPU usage (optional, separate SSH session)

Open a **second SSH connection** to the pod (same IP and port), then:

```bash
watch -n 2 nvidia-smi
```

During training, GPU utilisation should be **consistently above 80%** and memory should be most of the 80GB. If utilisation sits near 0% for more than 30 seconds after the first few steps, something is wrong.

---

## Failure Triage

### CUDA out of memory (OOM)

```
torch.OutOfMemoryError: CUDA out of memory.
```

Reduce batch size:

```bash
NPROC_PER_NODE=1 TRAIN_BATCH_TOKENS=262144 bash run_h100.sh
```

Try halving `TRAIN_BATCH_TOKENS` until it fits. Note: this changes the effective batch size and may affect convergence.

### NaN loss

```
step:12  train_loss:nan
```

Most common causes: very high learning rate at init, or a recently changed code path. Try:

```bash
NPROC_PER_NODE=1 MATRIX_LR=0.01 SCALAR_LR=0.01 bash run_h100.sh
```

If NaN appears on step 1, it is likely a data or model initialisation issue — re-run the smoke test and check that the parameter breakdown prints correctly.

### SSH drops without tmux

The job is dead. Re-run from Phase 7. Next time use tmux.

### SSH drops with tmux running

Reconnect:

```bash
ssh root@[IP] -p [PORT]
tmux attach -t train
```

Your job is still running. Scroll up to check for errors.

### `enable_gqa` keyword error

```
TypeError: scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'
```

This should not occur — the fix was applied on `main`. If you see it, the pod has stale code. Go back to Phase 3 and run `git reset --hard origin/main`, then retry.

---

## Phase 8 — Save Results

### 8a. Identify what to save

After a successful run, the experiment folder contains:

| File | Size (approx) | Save? |
| --- | --- | --- |
| `logs/<run_name>.log` | ~1–5 MB | **Yes — required for submission** |
| `final_model.int8.ptz` | ~12–14 MB | **Yes — required for submission** |
| `final_model.pt` | ~84 MB | Optional (raw weights, large) |

The `final_int8_zlib_roundtrip_exact val_bpb:X.XXXX` line near the end of the log is the score you submit.

### 8b. Copy files to your local machine

Run these commands from your **local** terminal. Get `[IP]` and `[PORT]` from the RunPod Connect screen.

```bash
# Copy the log (rename to reflect the seed used — default in run_h100.sh is 1337)
scp -P [PORT] root@[IP]:/workspace/parameter-golf/records/track_10min_16mb/2026-03-23_RecurrentTRM_5x3_LoRA/logs/<run_name>.log ./logs/train_seed1337.log

# Copy the compressed model artifact
scp -P [PORT] root@[IP]:/workspace/parameter-golf/records/track_10min_16mb/2026-03-23_RecurrentTRM_5x3_LoRA/final_model.int8.ptz ./final_model.int8.ptz
```

> The `-P` flag (capital P) sets the SSH port for `scp`. This differs from `ssh` where the port flag is lowercase `-p`.

For the 3-run statistical requirement, repeat with different seeds and rename accordingly:

```bash
# On the pod:
SEED=42 NPROC_PER_NODE=1 bash run_h100.sh
SEED=2024 NPROC_PER_NODE=1 bash run_h100.sh
```

Then `scp` each resulting log as `train_seed42.log` and `train_seed2024.log`.

### 8c. Optional — commit and push from the pod

If you want to push logs or artifacts to GitHub directly from the pod:

```bash
cd /workspace/parameter-golf
git config user.email "you@example.com"
git config user.name "Your Name"
git add records/track_10min_16mb/2026-03-23_RecurrentTRM_5x3_LoRA/logs/
git add records/track_10min_16mb/2026-03-23_RecurrentTRM_5x3_LoRA/final_model.int8.ptz
git commit -m "Add H100 training logs and artifact (seed 1337)"
git push https://<your-github-username>:<your-PAT>@github.com/SonnyZhan/parameter-golf.git HEAD:main
```

Replace `<your-PAT>` with a GitHub Personal Access Token that has `repo` scope. Do not type your password inline — use a PAT.

---

## Phase 9 — Fill in submission.json

Open `submission.json` locally and fill in the fields from the final lines of the log:

```json
{
  "author": "Fuming",
  "github_id": "[your GitHub username]",
  "name": "Recurrent TRM 9x3 + per-pass LoRA",
  "blurb": "...",
  "date": "2026-03-23T00:00:00Z",
  "val_loss": [from log: final_int8_zlib_roundtrip_exact val_loss],
  "val_bpb":  [from log: final_int8_zlib_roundtrip_exact val_bpb],
  "bytes_total": [from log: Total submission size int8+zlib],
  "bytes_code":  [from log: Code size]
}
```

The relevant log lines look like:

```
final_int8_zlib_roundtrip_exact val_bpb:1.XXXX val_loss:X.XXXX
Serialized model int8+zlib: XXXXXXXX bytes
Code size: XXXXX bytes
Total submission size int8+zlib: XXXXXXXX bytes
```

---

## Phase 10 — Stop the Pod

**Do this as soon as you have saved all files you need.**

1. Go to [console.runpod.io](https://console.runpod.io).
2. Find your pod.
3. Click **Stop** (keeps the disk if you want to resume) or **Terminate** (deletes everything permanently).

> Use **Stop** if you might return to this pod within a day or two. Use **Terminate** if you are done entirely — it avoids ongoing storage fees.

---

## End-of-Run Checklist

- [ ] SSH into pod; `nvidia-smi` confirms H100 visible and near-full memory free
- [ ] `cd /workspace/parameter-golf` and either cloned fresh or ran `git reset --hard origin/main`
- [ ] `git branch --show-current` prints `main`; `git log --oneline -3` shows your latest commits
- [ ] Experiment folder contains `train_gpt.py`, `run_h100.sh`, updated `README.md`
- [ ] Dataset present: `fineweb_train_*.bin`, `fineweb_val_*.bin`, `fineweb_1024_bpe.model`
- [ ] Smoke test (5 iterations, single process) completed without error
- [ ] `tmux new-session -s train` started before launching the real run
- [ ] Used `NPROC_PER_NODE=1 bash run_h100.sh` — not bare `bash run_h100.sh`
- [ ] PyTorch upgrade step (if triggered) completed without error
- [ ] Training ran to wall clock (~600s); `final_int8_zlib_roundtrip_exact val_bpb:X.XXXX` printed
- [ ] `logs/<run_name>.log` copied locally and renamed `train_seed1337.log`
- [ ] `final_model.int8.ptz` copied locally
- [ ] `submission.json` filled: `github_id`, `val_bpb`, `val_loss`, `bytes_total`, `bytes_code`
- [ ] Two more runs completed with `SEED=42` and `SEED=2024` and logs saved
- [ ] Pod **stopped or terminated** on RunPod console
