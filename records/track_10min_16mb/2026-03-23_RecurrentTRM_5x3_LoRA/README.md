# Recurrent TRM 9×3 + Per-Pass LoRA

**val_bpb: TBD** (to be filled after H100 run)

## Summary

A ~22M parameter language model that uses **9 physical transformer blocks looped 3 times** (27 effective layers) with full weight sharing across recurrences. Each recurrent pass receives a learned pass embedding to signal its iteration index, and per-pass low-rank adapters (LoRA rank 4) on Q and V let each pass develop distinct attention behavior without duplicating the full weight matrices. GQA (8Q / 4KV heads) is used for memory efficiency.

The key bet: weight sharing across passes gives 27 effective layers of compute for the parameter cost of ~9, and LoRA adapters give the passes enough independence to specialize. The backbone is optimized with Muon (Newton-Schulz orthogonalization); LoRA weights and scalars use Adam at lower LR.

## Architecture

| Hyperparameter | Value |
| --- | --- |
| Physical blocks (`NUM_LAYERS`) | 9 |
| Recurrent passes (`NUM_PASSES`) | 3 |
| Effective depth | 27 |
| Model dim (`MODEL_DIM`) | 512 |
| Q heads / KV heads | 8 / 4 (GQA) |
| Head dim | 64 |
| MLP mult | 3× (hidden = 1536) |
| LoRA rank | 4 (Q and V, per block per pass) |
| Vocabulary | 1024 (SentencePiece BPE) |
| Tied embeddings | Yes |
| Positional encoding | RoPE (base 10000) |
| Logit softcap | 30.0 |

### Recurrent loop

Token embeddings are RMS-normalized, then passed through `num_passes` outer iterations. Before each pass, a learned pass embedding (one vector per pass index, `[num_passes, dim]`) is added to the hidden state to signal which recurrence the model is in. Every physical block runs in sequence within each pass, and a residual mix gate `[2, dim]` in each block lerps between the current hidden state and the original embedding anchor `x0`, stabilizing gradients across the deep effective depth.

### Per-pass LoRA on Q and V

Each `CausalSelfAttention` has shared backbone projections (`c_q`, `c_k`, `c_v`, `proj`) plus per-pass LoRA parameter lists of length `num_passes`:

```python
q = c_q(x) + (x @ lora_qA[pass_idx] @ lora_qB[pass_idx]) / rank
v = c_v(x) + (x @ lora_vA[pass_idx] @ lora_vB[pass_idx]) / rank
```

`lora_qA/vA` are Kaiming-initialized; `lora_qB/vB` are zero-initialized so pass 0 starts as pure backbone. K is not LoRA-adapted (backbone only).

### GQA implementation

`num_kv_heads=4` serves `num_heads=8` query heads. K and V are expanded to full head count via `repeat_interleave` before SDPA (compatible with any PyTorch version; numerically identical to `enable_gqa=True` in PyTorch ≥2.5).

## Parameters (~22M total)

| Component | Count |
| --- | --- |
| 9 blocks × ~2.36M (attn + MLP + norms + scales) | ~21.2M |
| LoRA: 9 blocks × 3 passes × ~7,168 | ~193.5K |
| Embeddings + pass_emb + misc | ~584K |
| **Total** | **~21,978,184** |

Expected artifact: ~12–14MB int8+zlib + ~50KB code = **under 16MB**.

## Training

### Optimizer groups

| Group | Optimizer | Parameters | LR |
| --- | --- | --- | --- |
| Token embedding | Adam (fused) | `tok_emb.weight` | 0.05 |
| Backbone 2D weights | Muon (Newton-Schulz, 5 steps) | `c_q/k/v/proj.weight`, MLP weights | 0.04 |
| Scalars / control | Adam (fused) | `attn_scale`, `mlp_scale`, `resid_mix`, `q_gain`, `pass_embedding` | 0.04 |
| LoRA adapters | Adam (fused) | `lora_qA/B`, `lora_vA/B` across all blocks × passes | 0.02 |

LoRA gets half the backbone LR because it's low-rank and sensitive to large updates early in training.

### Schedule

- LR warmup: 150 steps (linear from 0)
- Constant phase: bulk of training
- Warmdown: final 1200 steps (linear to 0)
- Muon momentum: warmed 0.85 → 0.95 over first 500 steps

### Other

- Gradient clip norm: 1.0
- Batch: 524,288 tokens/step, seq len 1024
- Autocast: bfloat16 (H100)
- `torch.compile(fullgraph=True, dynamic=False)` on the model and Newton-Schulz kernel

## Command (8×H100)

```bash
bash run_h100.sh
# or equivalently:
SEED=1337 \
RUN_ID=h100_recurrent_lora_9x3_seed1337 \
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=20000 \
WARMDOWN_ITERS=1200 \
NUM_LAYERS=9 NUM_PASSES=3 LORA_RANK=4 \
MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
MATRIX_LR=0.04 SCALAR_LR=0.04 TIED_EMBED_LR=0.05 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Hardware

8× NVIDIA H100 80GB SXM (RunPod).

## Metrics

| Metric | Value |
| --- | --- |
| val_bpb (post-quant int8+zlib roundtrip) | TBD |
| val_loss (post-quant) | TBD |
| Model parameters | ~21,978,184 |
| Artifact size (int8+zlib + code) | TBD (expected ~12–14MB) |
| Train steps | TBD |
| Train time | ≤600s |

## Files

- `train_gpt.py` — single-file training script (model, data, optimizers, quantization, eval)
- `submission.json` — leaderboard metadata
- `run_h100.sh` — launch script for RunPod 8×H100
- `smoke_test.ps1` — local Windows smoke test (fast/medium300/overnight500 profiles)
- `architecture_notes.md` — working notes on parameter budget and design decisions
- `logs/` — training logs (smoke test logs included; H100 logs to be added after run)
