#!/bin/bash
# =============================================================================
# run_h100.sh — Recurrent TRM 7×3 + per-pass LoRA (rank 4)
#
# Two modes:
#
#   1×H100 testing (1200 real iterations, no time cap):
#     NPROC_PER_NODE=1 bash run_h100.sh
#
#   8×H100 competition (20000 iter cap, 600s wall-clock stops it first):
#     bash run_h100.sh
#
# Architecture: 7 physical blocks × 3 passes = 21 effective layers
# Artifact target: under 16,000,000 bytes (decimal) after int8+zlib
# LoRA matrices stored in fp16 (not fp32) to keep artifact size stable
#
# After the run completes, copy the log off RunPod:
#   scp <user>@<pod-ip>:.../logs/<run_name>.log ./logs/train_seed<SEED>.log
# Repeat with SEED=42 and SEED=2024 for the 3-run statistical requirement.
# =============================================================================
set -e

DATA_ROOT="${DATA_PATH:-/workspace/parameter-golf/data}"
SEED="${SEED:-1337}"
NPROC="${NPROC_PER_NODE:-8}"
RUN_NAME="h100_recurrent_lora_7x3_seed${SEED}_$(date +%Y%m%d_%H%M%S)"

mkdir -p logs

# ---------------------------------------------------------------------------
# Per-mode hyperparameters
#
# 1×H100 mode — 3000 real iterations, no wall-clock cap
#   Actual observed step time from previous run: ~1014ms/step
#   → 3000 steps × 1.014s ≈ 51 minutes total wall clock
#   → WARMDOWN_ITERS=500: cooldown starts at step 2500 of 3000 (16.7%)
#   → LR_WARMUP_STEPS=200: ramp up over first 200 steps (6.7%)
#   → MUON_MOMENTUM_WARMUP_STEPS=500: Muon reaches 0.95 at step 500
#   → MAX_WALLCLOCK_SECONDS=0: disabled, runs to ITERATIONS completion
#   Previous 1200-step result: val_bpb=1.3280, artifact=14.79MB (PASS)
#   Expected 3000-step result: val_bpb≈1.18-1.22 (near competition baseline)
#
# 8×H100 mode — competition default, 600s wall-clock cap
#   At ~130ms/step (8 GPUs in parallel, no grad_accum):
#   → 600s ÷ 0.13s ≈ 4600 steps before wall clock stops it
#   → WARMDOWN_ITERS=1200: warmdown starts ~156s before end
#   → LR_WARMUP_STEPS=150: 150 × 0.13s = 19.5s warmup
#   → MUON_MOMENTUM_WARMUP_STEPS=500: 500 × 0.13s = 65s
#   → MAX_WALLCLOCK_SECONDS=600: hard 10-minute cap for competition
# ---------------------------------------------------------------------------

if [ "$NPROC" -eq 1 ]; then
    # 1×H100 — 3000 real iterations, no time cap
    MAX_WALLCLOCK_SECONDS=0
    ITERATIONS=3000
    WARMDOWN_ITERS=500
    LR_WARMUP_STEPS=200
    MUON_MOMENTUM_WARMUP_STEPS=500
    echo "Mode: 1×H100  (3000 iterations, no wall-clock cap, ~51 min)"
else
    # 8×H100 — competition 10-minute mode
    MAX_WALLCLOCK_SECONDS=600
    ITERATIONS=20000
    WARMDOWN_ITERS=1200
    LR_WARMUP_STEPS=150
    MUON_MOMENTUM_WARMUP_STEPS=500
    echo "Mode: 8×H100  (competition schedule, 600s wall-clock cap)"
fi

echo "========================================"
echo "Run: $RUN_NAME"
echo "GPUs: $NPROC  |  Seed: $SEED"
echo "Architecture: 7 layers × 3 passes = 21 effective layers"
echo "LoRA rank: 4  |  Stored as fp16 in artifact"
echo "Iterations: $ITERATIONS  |  Wall-clock cap: ${MAX_WALLCLOCK_SECONDS}s (0=none)"
echo "Warmdown: $WARMDOWN_ITERS steps  |  LR warmup: $LR_WARMUP_STEPS steps"
echo "Muon momentum warmup: $MUON_MOMENTUM_WARMUP_STEPS steps"
echo "Data: $DATA_ROOT"
echo "========================================"

# ---------------------------------------------------------------------------
# Verify critical env vars propagate correctly into Python before spending
# GPU time. Catches the silent failure where NUM_LAYERS=8 was ignored.
# ---------------------------------------------------------------------------
echo "--- Verifying env vars ---"
NUM_LAYERS=7 NUM_PASSES=3 LORA_RANK=4 python3 -c "
import os, sys
checks = {
    'NUM_LAYERS': ('7', os.environ.get('NUM_LAYERS', 'NOT_SET')),
    'NUM_PASSES': ('3', os.environ.get('NUM_PASSES', 'NOT_SET')),
    'LORA_RANK':  ('4', os.environ.get('LORA_RANK',  'NOT_SET')),
}
failed = False
for name, (expected, got) in checks.items():
    status = 'OK' if got == expected else 'FAIL'
    print(f'  {name}: expected={expected}  got={got}  [{status}]')
    if got != expected:
        failed = True
if failed:
    print('ERROR: env vars not propagating into Python. Fix before running.')
    sys.exit(1)
print('All env vars OK — starting training')
"
echo "--- Starting torchrun ---"

SEED=$SEED \
RUN_ID=$RUN_NAME \
MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS \
ITERATIONS=$ITERATIONS \
WARMDOWN_ITERS=$WARMDOWN_ITERS \
WARMUP_STEPS=20 \
LR_WARMUP_STEPS=$LR_WARMUP_STEPS \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
NUM_LAYERS=7 \
NUM_PASSES=3 \
LORA_RANK=4 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
TIED_EMBED_LR=0.05 \
MUON_MOMENTUM=0.95 \
MUON_MOMENTUM_WARMUP_START=0.85 \
MUON_MOMENTUM_WARMUP_STEPS=$MUON_MOMENTUM_WARMUP_STEPS \
MUON_BACKEND_STEPS=5 \
GRAD_CLIP_NORM=1.0 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
MAX_VAL_SEQS=512 \
DATA_PATH="$DATA_ROOT/datasets/fineweb10B_sp1024" \
TOKENIZER_PATH="$DATA_ROOT/tokenizers/fineweb_1024_bpe.model" \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=$NPROC train_gpt.py \
  2>&1 | tee "logs/${RUN_NAME}.log"

# ---------------------------------------------------------------------------
# Post-run: parse artifact size and give a clear pass/fail verdict
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Run complete."
echo "Log: logs/${RUN_NAME}.log"
echo ""

ARTIFACT_LINE=$(grep "Total submission size int8+zlib:" "logs/${RUN_NAME}.log" | tail -1)
ARTIFACT_BYTES=$(echo "$ARTIFACT_LINE" | grep -o '[0-9]* bytes' | head -1 | grep -o '[0-9]*')

if [ -n "$ARTIFACT_BYTES" ]; then
    if [ "$ARTIFACT_BYTES" -lt 16000000 ]; then
        echo "ARTIFACT: PASS  — ${ARTIFACT_BYTES} bytes (limit: 16,000,000)"
    else
        echo "ARTIFACT: FAIL  — ${ARTIFACT_BYTES} bytes (OVER 16,000,000 limit)"
        echo "Action: reduce NUM_LAYERS to 6 and re-run"
    fi
else
    echo "ARTIFACT: could not parse size — check log manually"
fi

BPB_LINE=$(grep "final_int8_zlib_roundtrip_exact" "logs/${RUN_NAME}.log" | tail -1)
echo "VAL BPB: $BPB_LINE"
echo ""
echo "Next steps:"
echo "  1. Copy log: scp ... logs/${RUN_NAME}.log ./logs/train_seed${SEED}.log"
echo "  2. Copy artifact: scp ... final_model.int8.ptz ./"
echo "  3. Repeat with SEED=42 and SEED=2024 for 3-run leaderboard requirement"
echo "  4. Update submission.json with final val_bpb from BPB line above"
echo "========================================"
