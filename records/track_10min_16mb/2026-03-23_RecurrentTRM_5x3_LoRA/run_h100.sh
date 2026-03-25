#!/bin/bash
# =============================================================================
# run_h100.sh — Competition run for Recurrent TRM 9×3 + per-pass LoRA
#
# Usage (8×H100, competition):
#   bash run_h100.sh
#
# Usage (single GPU, testing / cheaper pods):
#   NPROC_PER_NODE=1 bash run_h100.sh
#
# After the run completes, copy the log off RunPod:
#   scp <user>@<pod-ip>:/workspace/parameter-golf/records/track_10min_16mb/2026-03-23_RecurrentTRM_5x3_LoRA/logs/<run_name>.log ./logs/
# Then rename it to train_seed<SEED>.log (e.g. train_seed1337.log) before submission.
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

echo "========================================"
echo "Starting run: $RUN_NAME"
echo "GPUs: $NPROC  Seed: $SEED"
echo "Data root: $DATA_ROOT"
echo "Max wallclock: 600s (10 min)"
echo "Max iterations: 20000 (wallclock will stop it first)"
echo "Warmdown: 1200 steps"
echo "========================================"

SEED=$SEED \
RUN_ID=$RUN_NAME \
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=20000 \
WARMDOWN_ITERS=1200 \
WARMUP_STEPS=20 \
LR_WARMUP_STEPS=150 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
NUM_LAYERS=9 \
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
MUON_MOMENTUM_WARMUP_STEPS=500 \
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
