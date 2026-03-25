$env:RUN_ID = "smoke_vrl"
$env:ITERATIONS = "500"
$env:WARMUP_STEPS = "20"
$env:VAL_LOSS_EVERY = "100"
$env:TRAIN_BATCH_TOKENS = "65536"
$env:TRAIN_SEQ_LEN = "512"
$env:SEED = "1337"
$env:MAX_WALLCLOCK_SECONDS = "0"
$env:WARMDOWN_ITERS = "0"
$env:MAX_VAL_SEQS = "128"
$env:MATRIX_LR = "0.02"
$env:SCALAR_LR = "0.02"
$repoRoot = (Resolve-Path "$PSScriptRoot\..\..\..")
$env:DATA_PATH = "$repoRoot\data\datasets\fineweb10B_sp1024"
$env:TOKENIZER_PATH = "$repoRoot\data\tokenizers\fineweb_1024_bpe.model"
$env:VOCAB_SIZE = "1024"

$py = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py train_gpt.py
