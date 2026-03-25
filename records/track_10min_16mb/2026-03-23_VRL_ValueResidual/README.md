# Value Residual Learning (VRL) — 11-Layer GPT

## Architecture

In deep transformers, hidden states progressively lose the original token signal as they pass through many layers — a phenomenon sometimes called representation drift. Value Residual Learning (VRL) addresses this by capturing the attention input from the very first layer and injecting a learned-weighted copy of it into every subsequent layer's attention output.

The change is surgically small: one scalar `vrl_alpha` per attention block, initialized to 0. At layer 0, the normalized attention input is captured as `v0`. At every subsequent layer, `vrl_alpha * v0` is added to the attention output before the residual connection. Because `vrl_alpha` starts at zero, the model begins training identically to the baseline and gradually learns how much to rely on the original token anchor.

**Key dimensions:**
- `d_model = 512`, `num_heads = 8`, `num_kv_heads = 4`
- `mlp_mult = 2`, `vocab_size = 1024`
- 11 unique transformer blocks (up from 9 in baseline)
- Per-layer VRL alpha initialized to 0.0
- Tied embeddings

## Why This Should Work

The theoretical argument: later layers in deep networks increasingly lose information about the original input tokens. By giving each layer a direct shortcut back to the layer-0 representation — gated by a learned scalar — the model can recover lost token identity when it helps, without being forced to use it. The zero init means the model pays no cost if VRL is unhelpful.

## Parameter Overhead

11 extra float32 scalars. Essentially free — 44 bytes total. The real cost is the 2 extra transformer blocks (11 vs 9), which still fits comfortably under 16MB after int8+zlib.

## How to Run (Windows, local smoke test)

```powershell
cd records\track_10min_16mb\2026-03-23_VRL_ValueResidual
powershell -ExecutionPolicy Bypass -File .\smoke_test.ps1
```

Requires the 1-shard data download from repo root.

## Expected val_bpb

After full cloud training: TBD. The hypothesis is that VRL + 11 layers should beat the 9-layer baseline (~1.22 bpb) with minimal parameter and compute overhead.
