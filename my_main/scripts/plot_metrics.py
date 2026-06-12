#!/usr/bin/env python3
"""
duplex_finetune.py の metrics.jsonl から学習曲線を描画する。

Usage:
  python3 my_main/scripts/plot_metrics.py my_main/duplex_lora/metrics.jsonl
  # -> my_main/duplex_lora/curves.png

学習中でも実行可能（その時点までの曲線を描く）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> tuple[list[dict], list[dict]]:
    train, val = [], []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            (train if r.get("split") == "train" else val).append(r)
    return train, val


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <metrics.jsonl> [output.png]")
    path = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else path.parent / "curves.png"

    train, val = load(path)
    if not train:
        sys.exit("no train records found")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ts = [r["step"] for r in train]
    vs = [r["step"] for r in val]

    ax = axes[0][0]
    ax.plot(ts, [r["loss"] for r in train], label="train", alpha=0.8)
    if val:
        ax.plot(vs, [r["loss"] for r in val], "o-", label="val")
    ax.set_title("loss (weighted)")
    ax.set_xlabel("step"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(ts, [r["loss_text"] for r in train], label="train text", alpha=0.8)
    ax.plot(ts, [r["loss_pad"] for r in train], label="train pad", alpha=0.8)
    if val:
        ax.plot(vs, [r["loss_text"] for r in val], "o-", label="val text")
        ax.plot(vs, [r["loss_pad"] for r in val], "o-", label="val pad")
    ax.set_title("loss by slot type")
    ax.set_xlabel("step"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot(ts, [r["acc_text"] for r in train], label="train text", alpha=0.8)
    ax.plot(ts, [r["acc_pad"] for r in train], label="train pad", alpha=0.8)
    if val:
        ax.plot(vs, [r["acc_text"] for r in val], "o-", label="val text")
        ax.plot(vs, [r["acc_pad"] for r in val], "o-", label="val pad")
    ax.set_title("token accuracy")
    ax.set_xlabel("step"); ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.plot(ts, [r.get("gnorm", 0) for r in train], alpha=0.8, label="gnorm")
    ax.set_yscale("log")
    ax.set_title("grad norm (LoRA)")
    ax.set_xlabel("step"); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(ts, [r.get("lr", 0) for r in train], "C1", alpha=0.6, label="lr")
    ax2.set_ylabel("lr", color="C1")

    fig.suptitle(f"{path.parent.name}  ({len(train)} train logs, {len(val)} evals)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
