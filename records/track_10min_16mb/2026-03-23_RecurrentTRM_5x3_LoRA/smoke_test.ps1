# RecurrentTRM 9x3 LoRA — local smoke profiles.
# Training uses fineweb_train_*.bin; val_bpb uses fineweb_val_*.bin (see train_gpt.py).
#
# Profiles:
#   fast         — 120 iters, seq 256, 32k batch tokens (~30–90 min on GTX 1660 Ti class).
#   medium300    — 300 iters, seq 512, 32k batch tokens (same 4096 tok/micro as fast; ~fits 6GB VRAM).
#   medium300_large — same as medium300 but 64k batch tokens (needs ~8GB+ VRAM; ~2x activations vs fast).
#   overnight500 — Phases 0–3: 500 iters, seq 512, 32k batch; lower MATRIX_LR/SCALAR_LR, longer LR warmup;
#                  val every 50 steps; post-train FINAL_POST_TRAIN_COMPARE (train subset vs val) if MAX_TRAIN_EVAL_SEQS set.
#
# Usage: .\smoke_test.ps1
#        .\smoke_test.ps1 -Profile fast
#        .\smoke_test.ps1 -Profile medium300_large
#        .\smoke_test.ps1 -Profile overnight500

param(
    [ValidateSet("fast", "medium300", "medium300_large", "overnight500")]
    [string]$Profile = "medium300"
)

$repoRoot = (Resolve-Path "$PSScriptRoot\..\..\..")
$env:DATA_PATH = "$repoRoot\data\datasets\fineweb10B_sp1024"
$env:TOKENIZER_PATH = "$repoRoot\data\tokenizers\fineweb_1024_bpe.model"
$env:VOCAB_SIZE = "1024"

# Shared architecture / optimizer defaults (match prior successful smokes).
$env:SEED = "1337"
$env:MAX_WALLCLOCK_SECONDS = "0"
$env:WARMDOWN_ITERS = "0"
$env:NUM_LAYERS = "9"
$env:NUM_PASSES = "3"
$env:MODEL_DIM = "512"
$env:NUM_HEADS = "8"
$env:NUM_KV_HEADS = "4"
$env:MATRIX_LR = "0.02"
$env:SCALAR_LR = "0.02"
$env:GRAD_CLIP_NORM = "1.0"

switch ($Profile) {
    "fast" {
        $env:RUN_ID = "smoke_recurrent_lora_fast"
        $env:ITERATIONS = "120"
        $env:WARMUP_STEPS = "10"
        $env:VAL_LOSS_EVERY = "40"
        $env:TRAIN_BATCH_TOKENS = "32768"
        $env:TRAIN_SEQ_LEN = "256"
        $env:MAX_VAL_SEQS = "64"
        $env:MAX_TRAIN_EVAL_SEQS = "64"
        $env:TRAIN_LOG_EVERY = "10"
        $env:LR_WARMUP_STEPS = "80"
    }
    "medium300" {
        $env:RUN_ID = "smoke_recurrent_lora_medium300"
        $env:ITERATIONS = "300"
        $env:WARMUP_STEPS = "20"
        $env:VAL_LOSS_EVERY = "100"
        $env:TRAIN_BATCH_TOKENS = "32768"
        $env:TRAIN_SEQ_LEN = "512"
        $env:MAX_VAL_SEQS = "128"
        $env:MAX_TRAIN_EVAL_SEQS = "128"
        $env:TRAIN_LOG_EVERY = "10"
        $env:LR_WARMUP_STEPS = "120"
    }
    "medium300_large" {
        $env:RUN_ID = "smoke_recurrent_lora_medium300_large"
        $env:ITERATIONS = "300"
        $env:WARMUP_STEPS = "20"
        $env:VAL_LOSS_EVERY = "100"
        $env:TRAIN_BATCH_TOKENS = "65536"
        $env:TRAIN_SEQ_LEN = "512"
        $env:MAX_VAL_SEQS = "128"
        $env:MAX_TRAIN_EVAL_SEQS = "128"
        $env:TRAIN_LOG_EVERY = "10"
        $env:LR_WARMUP_STEPS = "120"
    }
    "overnight500" {
        # Phase 0: same architecture as shared env (9x3, d=512, LoRA rank from train_gpt defaults).
        # Phase 1: 500 iterations, same local-safe batch/seq as medium300.
        # Phase 2: gentler matrix/scalar LR; LR warmup ~15% of iterations; GRAD_CLIP from shared (1.0).
        # Phase 3: val every 50 steps — compare train_loss vs val_loss in log for generalization.
        $env:RUN_ID = "smoke_recurrent_lora_overnight500"
        $env:ITERATIONS = "500"
        $env:WARMUP_STEPS = "20"
        $env:VAL_LOSS_EVERY = "50"
        $env:TRAIN_BATCH_TOKENS = "32768"
        $env:TRAIN_SEQ_LEN = "512"
        $env:MAX_VAL_SEQS = "128"
        $env:MAX_TRAIN_EVAL_SEQS = "128"
        $env:TRAIN_LOG_EVERY = "10"
        $env:LR_WARMUP_STEPS = "75"
        $env:MATRIX_LR = "0.014"
        $env:SCALAR_LR = "0.014"
    }
}

Write-Host "Profile: $Profile  RUN_ID=$($env:RUN_ID)  ITERATIONS=$($env:ITERATIONS)  TRAIN_SEQ_LEN=$($env:TRAIN_SEQ_LEN)  TRAIN_BATCH_TOKENS=$($env:TRAIN_BATCH_TOKENS)"

$py = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Push-Location $PSScriptRoot
try {
    & $py train_gpt.py
} finally {
    Pop-Location
}
