# Architecture Notes — Recurrent TRM 9×3 + Per-Pass LoRA

Working notes, not polished docs. See README.md for the submission-facing version.

## Current config: 8 physical blocks × 3 passes, d=512

Moved to 8 blocks after a measured 1×H100 run confirmed that **9 blocks overshoots the 16MB cap**:

- 9 blocks: payload 22.7MB → int8+zlib 17.6MB + 50KB code = **17.6MB  ✗ over limit**
- 8 blocks (estimated): ~15.7MB  ✓
- 7 blocks (estimated): ~13.8MB  ✓ (more headroom, less capacity)

Rule of thumb from measured compression ratio (1.29x on int8+zlib):
each block removed saves ~2.4MB payload → ~1.9MB compressed.

**Do not** jump to MODEL_DIM=768 without checking: 10×768 is ~52M params and will not fit.

## Parameter breakdown (9 blocks, d=512)

- 9 physical blocks × ~2.36M ≈ 21.2M (attn + MLP + norms + scales per block)
- LoRA: 9 blocks × 3 passes × (512×4 + 4×512 + 512×4 + 4×256) ≈ 193.5K
- Embeddings (tok_emb 1024×512) + pass_emb (3×512) + final_norm ≈ ~584K
- **Total: ~21,978,184 params** (matches `model_params:` in training log)

## GQA fix (applied 2026-03-25)

Removed `enable_gqa=True` kwarg from `F.scaled_dot_product_attention` — that argument
requires PyTorch ≥2.5.0 and crashed Dynamo on older RunPod images. Replaced with explicit
KV head expansion using `repeat_interleave` before SDPA. Numerically identical; works on
any PyTorch version. `run_h100.sh` also auto-upgrades torch to ≥2.5 on the pod.

## Artifact size estimate

~22M params × 1 byte (int8) + per-row scales (fp16, ~1/64 of weights) ≈ ~22.3MB raw.
zlib compression on int8 model weights typically achieves ~50–60% reduction → ~11–13MB.
Add ~50KB for train_gpt.py code. Expected total: **12–14MB**, safely under 16MB.
Re-verify after every NUM_LAYERS change.

## Optimizer split rationale

- **Muon** gets all 2D backbone matrices (c_q/k/v/proj, MLP fc/proj). Newton-Schulz
  orthogonalization works best on tall/wide matrices; the 5-step iteration is compiled
  separately via `torch.compile` on bf16 hardware.
- **Adam** gets LoRA at half backbone LR (0.02 vs 0.04). LoRA matrices are small and
  low-rank; large early updates would saturate the low-rank space before the backbone
  has learned useful features to adapt. lora_qB/vB are zero-initialized so pass 0 starts
  as pure backbone and LoRA grows in gradually.
- **Adam** gets scalars (attn_scale, mlp_scale, resid_mix, q_gain, pass_embedding).
  These are 1D or control tensors — Newton-Schulz doesn't apply to 1D params.

## Known risks

- LoRA rank 4 may limit pass differentiation if the tasks each pass needs to specialize
  for are high-rank. Watch per-pass gradient norms if you instrument.
- Shared backbone weights receive conflicting gradients from all 3 passes simultaneously.
  This is expected and manageable, but means the backbone must learn representations that
  are universally useful across all passes — a harder optimization target than a standard
  single-pass model.
- resid_mix gate (lerp between x and x0) is critical for gradient flow across 27 effective
  layers. If training diverges, check whether mix[1] (x0 weight) has grown large.
- If adding a 10th physical block at 512, re-check artifact bytes before launching a cloud
  run — budget is tight (~1MB headroom estimated).
