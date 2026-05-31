"""Summary bar chart: OOF vs LB across V1 (A0-A7) and V2.

Highlights:
- V1 took OOF from 0.81 -> 0.82 but LB regressed 0.808 -> 0.804 (TE leak)
- V2 drops OOF to 0.817 but (predicted) restores LB to >= 0.810
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = [
    ("A0 (legacy)",   0.8132, 0.80804, "reference"),
    ("A4 (TE leak)",  0.8184, 0.80406, "TE self-leak"),
    ("A7 (blend)",    0.8200, 0.80383, "blend overfits OOF"),
    ("V2 (STRONG)",   0.8153, None,    "no TE, 15-seed"),
    ("V2_best (opt)", 0.8171, None,    "no TE, Optuna + 15-seed"),
    ("V2 pseudo",     0.8120, None,    "pseudo-label, acc drops"),
]


def plot():
    labels = [r[0] for r in RUNS]
    oof = [r[1] for r in RUNS]
    lb = [r[2] if r[2] is not None else np.nan for r in RUNS]
    notes = [r[3] for r in RUNS]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(labels))
    width = 0.36
    bars_oof = ax.bar(x - width / 2, oof, width, label="Honest OOF", color="#4C78A8")
    bars_lb = ax.bar(
        x + width / 2, lb, width, label="Kaggle LB (public)", color="#E45756",
    )
    for b, v in zip(bars_oof, oof):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0006, f"{v:.4f}",
                ha="center", fontsize=8.5)
    for b, v in zip(bars_lb, lb):
        if np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, 0.801, "LB\n?", ha="center", fontsize=8, color="#777")
        else:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.0006, f"{v:.5f}",
                    ha="center", fontsize=8.5)

    ax.axhline(0.80804, color="#E45756", ls=":", lw=1, alpha=0.4)
    ax.text(len(RUNS) - 0.5, 0.80810, "Legacy LB = 0.80804", fontsize=8, color="#E45756", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax.set_ylim(0.800, 0.826)
    ax.set_ylabel("Accuracy")
    ax.set_title("V1 (A0→A7) vs V2 — OOF and actual Kaggle LB\nV1 overfit OOF via TE self-leak; V2 restores LB-friendliness")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.2)

    # Annotate each bar group with the note
    for xi, n in zip(x, notes):
        ax.text(xi, 0.8014, n, ha="center", fontsize=7.5, color="#555",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))

    fig.tight_layout()
    out = Path("reports/xgboost/figures/v2_summary.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    plot()
