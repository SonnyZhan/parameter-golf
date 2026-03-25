#!/bin/bash
set -e
 
DATA_ROOT="${DATA_PATH:-/workspace/parameter-golf/data}"
RUN_NAME="h100_recurrent_lora_1200_$(date +%Y%m%d_%H%M%S)"
 
mkdir -p logs
 
echo "========================================"
echo "Starting H100 run: $RUN_NAME"
echo "Data root: $DATA_ROOT"
echo "Iterations: 1200"
echo "Batch tokens: 524288"
echo "Seq len: 1024"
echo "Matrix LR: 0.04"
echo "========================================"
 
RUN_ID=$RUN_NAME \
ITERATIONS=1200 \
WARMDOWN_ITERS=200 \
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
MUON_MOMENTUM_WARMUP_STEPS=500 \
GRAD_CLIP_NORM=1.0 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
MAX_VAL_SEQS=512 \
MAX_WALLCLOCK_SECONDS=0 \
DATA_PATH="$DATA_ROOT/datasets/fineweb10B_sp1024" \
TOKENIZER_PATH="$DATA_ROOT/tokenizers/fineweb_1024_bpe.model" \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py \
  2>&1 | tee "logs/${RUN_NAME}.log"
 
echo "Run complete. Log saved to logs/${RUN_NAME}.log"
echo "Check artifact: ls -lh final_model.int8.ptz"
