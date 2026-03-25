# Architecture Notes — Recurrent Transformer N×3 + LoRA

Working notes, not polished docs.

## Current default: 9 physical blocks × 3 passes (512 wide)

We moved from **5×512 (~12.4M params, ~3.7MB int8+zlib)** to **9×512** so the submission uses more of the **~16MB** total artifact budget without blowing past it. Rule of thumb from the repo: **~20M params at 512 width → ~13MB int8+zlib** (see VRL logs); **9 blocks** lands near **~22M params** — still under 16MB with `train_gpt.py` bytes, but check every time.

**Bad idea we tried in logs:** `NUM_LAYERS=10`, `MODEL_DIM=768` → **~52M params** — that is not a viable Parameter Golf submission.

## Parameter count (9 blocks, d=512)

Same per-block breakdown as the old 5-block notes (~2.36M per block before LoRA overhead — see previous revision in git if needed).

- **9 physical blocks** × ~2.36M ≈ **21.2M** (block stacks)
- **LoRA:** 9 × 3 passes × ~7,168 ≈ **193.5k**
- **Embeddings + pass_emb + skips:** same order as 5-block case scaled (skip count = min(⌊27/2⌋, ⌈27/2⌉) = **13** skips × 512 = **6,656**)

**Exact count (defaults, CPU init): 21,978,184 params** — `python -c` over `GPT(...)` matches `model_params:` in the training log.

## Compressed size

Expect **int8+zlib** in the **~12–14MB** range for ~22M params — **verify** with:

`Serialized model int8+zlib:` and `Total submission size int8+zlib:` in the log.

## Where the savings still are

Versus **27 unique** layers at this width you would pay ~27 × 2.36M ≈ **63M+** params. Recurrence still buys a large factor; LoRA pays a small overhead for per-pass specialization.

## Known risks

Same as before: LoRA rank-4 may limit pass differentiation; shared weights across passes can yield conflicting gradients. If you add an **10th** physical block at 512, re-check artifact bytes before locking a cloud run.
