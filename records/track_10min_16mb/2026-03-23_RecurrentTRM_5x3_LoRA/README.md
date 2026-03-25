# Recurrent Universal Transformer 9×3 with Per-Pass LoRA

## Architecture

This submission uses **9 physical transformer blocks** looped **3 times**, yielding **27 effective layers** while sharing weights across passes. Each recurrent pass receives a learned pass embedding, and per-pass LoRA deltas on Q and V let each pass specialize attention without duplicating full matrices.

**Key dimensions (defaults, overridable via env):**
- `d_model = 512`, `num_heads = 8`, `num_kv_heads = 4`
- `mlp_mult = 3`, `vocab_size = 1024`
- **9 × 3 = 27** effective depth; LoRA rank 4 on Q and V per block per pass
- Tied embeddings

The encoder/decoder split follows the baseline pattern over the **effective** depth (here 13 encoder / 14 decoder steps), with skip connections wired the same way as `train_gpt.py`.

## Parameter and artifact budget (~16MB)

At **512 width** and **9 physical blocks**, total params are **~22M** (vs ~12.4M for the original 5×512 design). That uses more of the **code + int8+zlib ≤ 16,000,000 bytes** budget while staying safely under it (expect on the order of **12–14MB** compressed weights; verify with the log line `Serialized model int8+zlib:`).

**Do not** jump to `MODEL_DIM=768` and many blocks without measuring: a **10×768** configuration is **~52M params** and will **not** fit the competition limit.

To tune toward the cap without overshooting, adjust **`NUM_LAYERS`** (physical blocks) at **`MODEL_DIM=512`** first (e.g. 8 → 9 → 10), re-run export, and read off `Total submission size int8+zlib:`.

## How to run (Windows smoke test)

```powershell
cd records\track_10min_16mb\2026-03-23_RecurrentTRM_5x3_LoRA
powershell -ExecutionPolicy Bypass -File .\smoke_test.ps1
```

Requires the 1-shard data download from repo root:  
`python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`.

## Expected val_bpb

After full 8×H100 training: TBD. Local 300-step smoke is only a sanity check; compare runs using `val_bpb` after int8+zlib round-trip in the log.
