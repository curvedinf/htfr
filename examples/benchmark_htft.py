"""Summarize Hypertensor Field Transformer training metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=str, help="Path to metrics JSONL emitted by train_htft.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.metrics)
    if not path.exists():
        raise SystemExit(f"Metrics file {path} not found")
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        raise SystemExit("Metrics file is empty")
    best = min(entries, key=lambda row: row["eval_perplexity"])
    last = entries[-1]
    print(f"Loaded {len(entries)} metric entries from {path}")
    print("Best student perplexity: {:.3f} at step {}".format(best["eval_perplexity"], best["step"]))
    print("Final student perplexity: {:.3f} at step {}".format(last["eval_perplexity"], last["step"]))
    if "teacher_perplexity" in last:
        print("Teacher perplexity baseline: {:.3f}".format(last["teacher_perplexity"]))
        ratio = last["eval_perplexity"] / max(last["teacher_perplexity"], 1e-12)
        print("Student/teacher perplexity ratio: {:.3f}".format(ratio))


if __name__ == "__main__":
    main()
