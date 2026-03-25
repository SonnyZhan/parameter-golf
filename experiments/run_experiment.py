from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import EXPERIMENTS

FINAL_RE = re.compile(r"final_int8_zlib_roundtrip val_loss:([0-9.]+) val_bpb:([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run named train_gpt_local experiments and log results.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["baseline_smoke"],
        help="Experiment names from experiments/config.py",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current interpreter).",
    )
    parser.add_argument(
        "--results-file",
        default="experiments/results.jsonl",
        help="JSONL file to append experiment results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    results_path = (repo_root / args.results_file).resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)

    for config_name in args.configs:
        if config_name not in EXPERIMENTS:
            known = ", ".join(sorted(EXPERIMENTS.keys()))
            raise ValueError(f"Unknown config '{config_name}'. Known configs: {known}")
        cfg = EXPERIMENTS[config_name]
        run_id = f"{cfg.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        env = os.environ.copy()
        env.update(cfg.env_overrides())
        env["RUN_ID"] = run_id
        env.setdefault("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
        env.setdefault("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
        env.setdefault("VOCAB_SIZE", "1024")

        cmd = [args.python, "train_gpt_local.py"]
        print(f"=== Running {cfg.name} (RUN_ID={run_id}) ===")
        print(f"Command: {' '.join(cmd)}")

        started = time.time()
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        ended = time.time()

        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")

        combined = f"{proc.stdout}\n{proc.stderr}"
        match = FINAL_RE.search(combined)
        val_loss = float(match.group(1)) if match else None
        val_bpb = float(match.group(2)) if match else None

        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config_name": cfg.name,
            "run_id": run_id,
            "return_code": proc.returncode,
            "duration_sec": round(ended - started, 3),
            "val_loss": val_loss,
            "val_bpb": val_bpb,
            "env_overrides": cfg.env_overrides(),
        }
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"Result: return_code={proc.returncode} val_loss={val_loss} val_bpb={val_bpb}")
        print(f"Appended: {results_path}")


if __name__ == "__main__":
    main()
