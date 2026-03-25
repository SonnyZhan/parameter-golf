from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    iterations: int = 200
    train_batch_tokens: int = 8192
    val_batch_size: int = 8192
    val_loss_every: int = 0
    warmup_steps: int = 5
    train_seq_len: int = 1024
    num_layers: int = 9
    model_dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_mult: int = 2
    matrix_lr: float = 0.04
    scalar_lr: float = 0.04
    tied_embed_lr: float = 0.05

    def env_overrides(self) -> dict[str, str]:
        return {
            "ITERATIONS": str(self.iterations),
            "TRAIN_BATCH_TOKENS": str(self.train_batch_tokens),
            "VAL_BATCH_SIZE": str(self.val_batch_size),
            "VAL_LOSS_EVERY": str(self.val_loss_every),
            "WARMUP_STEPS": str(self.warmup_steps),
            "TRAIN_SEQ_LEN": str(self.train_seq_len),
            "NUM_LAYERS": str(self.num_layers),
            "MODEL_DIM": str(self.model_dim),
            "NUM_HEADS": str(self.num_heads),
            "NUM_KV_HEADS": str(self.num_kv_heads),
            "MLP_MULT": str(self.mlp_mult),
            "MATRIX_LR": str(self.matrix_lr),
            "SCALAR_LR": str(self.scalar_lr),
            "TIED_EMBED_LR": str(self.tied_embed_lr),
            "MAX_WALLCLOCK_SECONDS": "0",
        }


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "baseline_smoke": ExperimentConfig(name="baseline_smoke"),
    "wider_640": ExperimentConfig(name="wider_640", model_dim=640, num_heads=10, num_kv_heads=5),
    "deeper_11l": ExperimentConfig(name="deeper_11l", num_layers=11, matrix_lr=0.035, scalar_lr=0.035),
}
