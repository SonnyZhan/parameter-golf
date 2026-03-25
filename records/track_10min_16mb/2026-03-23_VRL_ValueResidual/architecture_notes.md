# Architecture Notes — Value Residual Learning (VRL)

Working notes. Not polished.

## What's Actually Changed

Two things, both tiny:

1. Each attention block gets `self.vrl_alpha = nn.Parameter(torch.zeros(1))`. That's 11 extra scalars for 11 layers — 44 bytes. Literally nothing in the parameter budget.

2. During forward, layer 0's attention captures the normalized input `x` (the thing right before Q/K/V projections) as `v0`. Every subsequent layer adds `vrl_alpha * v0` to its attention output before the residual connection. `v0` is d_model=512 dimensional, same as the residual stream, so no shape gymnastics needed.

The num_layers default is bumped from 9 to 11 — this is the actual parameter cost increase. 11 blocks at ~1.57M params each (with mlp_mult=2) gives ~17.3M total params, vs ~17M for the baseline 9-layer. The 2 extra layers add about 3.1M params. After int8+zlib that's roughly +3.1MB, which should still be well under 16MB.

Actually let me be more careful: the baseline 9-layer is ~17M and compresses to ~5MB. Adding 2 layers adds 2 × (512×512 + 512×256 + 512×256 + 512×512 + 512×1024 + 1024×512 + scalars) ≈ 2 × 1,573,896 ≈ 3.15M extra params. That's ~3.15MB more int8 before zlib. Total compressed should be around 8MB. Fine.

## Why Not Detaching v0 Matters

If you `v0 = v0.detach()` when passing it to later layers, layer 0 gets no gradient signal about what kind of v0 is useful. It would just produce whatever its normal attention dynamics produce, and later layers would try to use it. By keeping the gradient flow alive, layer 0 learns to produce a v0 that's actually useful as an anchor for later layers. This is important — it turns VRL from a static shortcut into a learned representation.

The gradient cost is not zero: v0's computation graph stays alive through all 11 layers of backward. But v0 is just a single tensor snapshot (bsz, seq, 512), not a full layer's activations. Memory overhead should be small.

## Why Alpha=0 Init Is Important

If alpha starts at 1.0 or some other nonzero value, the model immediately mixes layer-0 features into every layer's attention output on step 1. This could destabilize training because the model architecture effectively changes compared to the baseline. With alpha=0, training starts identical to a normal 11-layer GPT. The model discovers gradually whether and how much to use the v0 anchor. Some layers might push alpha positive, others might keep it near zero or go slightly negative. The learned distribution of alphas would be interesting to look at after training.

## Known Risks

The main risk: if `vrl_alpha` grows too large in some layers, those layers effectively ignore their own computed attention and just copy layer-0 features. This would be a capacity waste — you're paying for the full attention computation but not using it. Might need to add alpha clipping or L2 regularization on vrl_alpha if this happens.

A subtler risk: because we're using the attention INPUT x (which is already through RMSNorm and resid_mix) rather than the literal value projection output, the "value" in "Value Residual Learning" is a slight misnomer. It's really "normalized-input residual learning." The reason is pragmatic: c_v maps to kv_dim (256), not d_model (512), so we'd need a projection to match shapes. Using the d_model-sized input avoids that. The original VRL paper uses actual value projections, but they assume kv_dim == d_model. This might be wrong and worth revisiting if results are underwhelming.

The other consideration is that with 11 layers the encoder/decoder split becomes 5/6 (encoder stores 5 skips, decoder consumes them). The v0 signal is shared across both halves. Not sure if that helps or hurts the skip connection pattern.
