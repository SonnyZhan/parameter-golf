#!/bin/bash
# =============================================================================
# run_h100.sh — Competition run for Recurrent TRM 9×3 + per-pass LoRA
#
# Usage (8×H100, competition — default):
#   bash run_h100.sh
#
# Usage (1×H100, single-GPU testing):
#   NPROC_PER_NODE=1 bash run_h100.sh
#
# The script automatically adjusts time-sensitive hyperparameters based on
# GPU count, because warmdown and LR warmup are step-count-based but the
# wall-clock budget is fixed at 600s.  On 1×H100 each step is ~8× slower
# (grad_accum=8 sequential micro-batches) + ~3× recurrent overhead, so
# ~24× fewer steps fit in 600s than on 8×H100.
#
# After the run completes, copy the log off RunPod:
#   scp <user>@<pod-ip>:.../logs/<run_name>.log ./logs/train_seed<SEED>.log
# Repeat with SEED=42 and SEED=2024 for the 3-run statistical requirement.
# =============================================================================
set -e

DATA_ROOT="${DATA_PATH:-/workspace/parameter-golf/data}"
SEED="${SEED:-1337}"
NPROC="${NPROC_PER_NODE:-8}"
RUN_NAME="h100_recurrent_lora_9x3_seed${SEED}_$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------------------
# Option A: ensure PyTorch >=2.5.0.
# The code already uses repeat_interleave instead of enable_gqa (works on any
# version), but >=2.5 also enables FlashAttention-3 improvements on H100.
# ---------------------------------------------------------------------------
python -c "
import torch, sys
major, minor = (int(x) for x in torch.__version__.split('.')[:2])
if (major, minor) < (2, 5):
    print(f'PyTorch {torch.__version__} < 2.5 detected, upgrading...', flush=True)
    sys.exit(1)
else:
    print(f'PyTorch {torch.__version__} OK', flush=True)
" || pip install --upgrade "torch>=2.5.0" --index-url https://download.pytorch.org/whl/cu121

mkdir -p logs

# ---------------------------------------------------------------------------
# Per-GPU-count hyperparameters.
#
# WHY THESE DIFFER:
#   The wall clock is fixed at 600s for both configs.  Warmdown and LR warmup
#   are specified in *steps*, but the script's lr_mul() uses wall-clock time
#   to decide when warmdown begins (warmdown_ms = WARMDOWN_ITERS × ms/step).
#   If WARMDOWN_ITERS × ms/step > 600s, warmdown starts at step 0 — the
#   model never trains at full LR.
#
#   8×H100: ~130ms/step (baseline 43ms × 3 recurrent passes)
#     → 1200 steps × 130ms = 156s  ✓  warmdown occupies last 156s of 600s
#     → 150 LR warmup steps × 130ms = 19.5s  ✓  ~3% of budget
#     → 500 Muon momentum warmup × 130ms = 65s  ✓  reaches target by step 500
#
#   1×H100: ~1000ms/step (8 grad_accum micro-batches × 3 recurrent passes)
#     → 1200 steps × 1000ms = 1200s >> 600s  ✗  warmdown starts immediately
#     → 150 LR warmup steps × 1000ms = 150s = 25% of budget  (too long)
#     → 500 Muon momentum steps × 1000ms = 500s  (momentum never stabilises)
#     Fix: scale everything down ~8× to match expected ~600 total steps.
# ---------------------------------------------------------------------------

if [ "$NPROC" -eq 1 ]; then
    # 1×H100 — ~600 total steps in 600s
    WARMDOWN_ITERS=120       # last ~20% of wall clock (~120s)
    LR_WARMUP_STEPS=60       # ~10% of budget (60s)
    MUON_MOMENTUM_WARMUP_STEPS=100  # reaches 0.95 by step 100 (~100s)
    echo "Mode: 1×H100  (single-GPU schedule)"
else
    # 8×H100 — ~4600 total steps in 600s (competition default)
    WARMDOWN_ITERS=1200      # last ~156s of wall clock
    LR_WARMUP_STEPS=150      # ~19.5s warmup
    MUON_MOMENTUM_WARMUP_STEPS=500  # reaches 0.95 by step 500 (~65s)
    echo "Mode: 8×H100  (competition schedule)"
fi

echo "========================================"
echo "Starting run: $RUN_NAME"
echo "GPUs: $NPROC  Seed: $SEED"
echo "Data root: $DATA_ROOT"
echo "Max wallclock: 600s (10 min)"
echo "Max iterations: 20000 (wallclock will stop it first)"
echo "Layers: 8x3=24 effective  (9 layers = 17.6MB artifact, over 16MB limit)"
echo "Warmdown: $WARMDOWN_ITERS steps  LR warmup: $LR_WARMUP_STEPS steps"
echo "Muon momentum warmup: $MUON_MOMENTUM_WARMUP_STEPS steps"
echo "========================================"

SEED=$SEED \
RUN_ID=$RUN_NAME \
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=20000 \
WARMDOWN_ITERS=$WARMDOWN_ITERS \
WARMUP_STEPS=20 \
LR_WARMUP_STEPS=$LR_WARMUP_STEPS \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
NUM_LAYERS=8 \
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

echo ""
echo "========================================"
echo "Run complete."
echo "Log saved to:  logs/${RUN_NAME}.log"
echo "Model saved to: final_model.int8.ptz"
echo ""
echo "Next steps:"
echo "  1. scp this log to your local logs/ folder"
echo "  2. Rename it: train_seed${SEED}.log"
echo "  3. Repeat with SEED=42 and SEED=2024 for the 3-run requirement"
echo "  4. Fill val_bpb in submission.json from the final_int8_zlib_roundtrip line in the log"
echo "========================================"
