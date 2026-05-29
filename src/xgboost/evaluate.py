"""Evaluation plots and tables for the XGBoost pipeline.

All plots are saved to ``reports/xgboost/figures`` so they can be embedded in
the final report and slides without re-running the training loop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from . import config
from . import cv as cv_module
from . import data as data_module
from . import model as model_module
from .features import apply_fold_features

sns.set_theme(style="whitegrid", context="talk")


def _savefig(fig: plt.Figure, name: str) -> Path:
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_pr(y_true: np.ndarray, y_proba: np.ndarray, tag: str) -> tuple[Path, Path]:
    """ROC + precision-recall curves for OOF predictions."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C78A8", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", alpha=0.7)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC (OOF) — {tag}")
    ax.legend(loc="lower right")
    roc_path = _savefig(fig, f"{tag}_roc.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#F58518", lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall (OOF) — {tag}")
    ax.legend(loc="lower left")
    pr_path = _savefig(fig, f"{tag}_pr.png")

    return roc_path, pr_path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    tag: str,
) -> Path:
    preds = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not transported", "Transported"],
        yticklabels=["Not transported", "Transported"],
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix @ {threshold:.2f} — {tag}")
    return _savefig(fig, f"{tag}_confusion_matrix.png")


def plot_feature_importance(
    feature_names: Sequence[str],
    importance: np.ndarray,
    tag: str,
    top_k: int = 25,
    importance_type: str = "gain",
) -> Path:
    df = pd.DataFrame({"feature": list(feature_names), "importance": importance})
    df = df.sort_values("importance", ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(df))))
    sns.barplot(data=df, x="importance", y="feature", ax=ax, color="#4C78A8")
    ax.set_title(f"Top {len(df)} features by {importance_type} — {tag}")
    ax.set_xlabel(importance_type)
    return _savefig(fig, f"{tag}_importance_{importance_type}.png")


def plot_threshold_scan(scan: pd.DataFrame, best_threshold: float, tag: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(scan["threshold"], scan["accuracy"], color="#4C78A8", lw=2)
    ax.axvline(best_threshold, color="#E45756", linestyle="--", label=f"best = {best_threshold:.2f}")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("OOF accuracy")
    ax.set_title(f"Threshold scan — {tag}")
    ax.legend()
    return _savefig(fig, f"{tag}_threshold_scan.png")


def plot_learning_curve(
    fold_scores: list[dict],
    tag: str,
) -> Path:
    df = pd.DataFrame(fold_scores)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.bar(df["fold"].astype(str), df["acc"], color="#4C78A8", alpha=0.85, label="acc")
    ax1.set_ylim(0.7, 0.86)
    ax1.set_ylabel("Fold accuracy", color="#4C78A8")
    ax1.set_xlabel("Fold")
    ax1.set_title(f"Per-fold accuracy & best iteration — {tag}")

    ax2 = ax1.twinx()
    ax2.plot(df["fold"].astype(str), df["best_iter"], color="#F58518", marker="o", linewidth=2, label="best_iter")
    ax2.set_ylabel("Best iteration", color="#F58518")
    return _savefig(fig, f"{tag}_fold_curve.png")


def plot_shap_summary(
    params: dict,
    tag: str,
    top_k: int = 25,
    target_encode_cols: tuple[str, ...] | None = None,
    fold_aware: bool = True,
) -> Path | None:
    """Train a *single* model on the full train split, then draw a SHAP beeswarm.

    SHAP values are computed on the fold-0 validation slice so that feature
    values remain unseen to the model; this keeps the explanations honest.
    """
    try:
        import shap
    except ImportError:
        return None

    common = data_module.load_common()
    X, X_test, y, groups = data_module.build_xgb_features(common)
    folds = cv_module.make_folds(y, groups, seed=config.RANDOM_SEED)
    f0 = folds[0]
    Xt = X.iloc[f0.train_idx].reset_index(drop=True)
    Xv = X.iloc[f0.valid_idx].reset_index(drop=True)
    yt = y.iloc[f0.train_idx].reset_index(drop=True)

    if fold_aware:
        raw_tr = common.train.iloc[f0.train_idx].reset_index(drop=True)
        raw_va = common.train.iloc[f0.valid_idx].reset_index(drop=True)
        Xt, Xv, _ = apply_fold_features(
            Xt, Xv, X_test, yt, raw_tr, raw_va, common.test.reset_index(drop=True),
            target_encode_cols=target_encode_cols,
            use_surname_refit=True,
            use_cabin_bin_refit=True,
        )
    Xv = Xv[Xt.columns]

    booster = model_module.build_model(params=params)
    from xgboost.callback import EarlyStopping
    booster.set_params(callbacks=[EarlyStopping(rounds=150, save_best=True)])
    booster.fit(Xt, yt, eval_set=[(Xv, y.iloc[f0.valid_idx].reset_index(drop=True))], verbose=False)

    # SHAP TreeExplainer on categorical features: XGBoost 2.x exposes pred_contribs
    # directly; we pass through shap.TreeExplainer which is the recommended wrapper.
    explainer = shap.TreeExplainer(booster)
    # Subsample the valid slice to keep plot tractable
    sample = Xv.sample(n=min(1500, len(Xv)), random_state=config.RANDOM_SEED).reset_index(drop=True)
    shap_values = explainer.shap_values(sample)
    fig = plt.figure(figsize=(10, max(5, 0.3 * top_k)))
    shap.summary_plot(shap_values, sample, max_display=top_k, show=False, plot_type="dot")
    path = config.FIGURES_DIR / f"{tag}_shap_summary.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Bar-style global importance
    fig = plt.figure(figsize=(9, max(4, 0.3 * top_k)))
    shap.summary_plot(shap_values, sample, max_display=top_k, show=False, plot_type="bar")
    bar_path = config.FIGURES_DIR / f"{tag}_shap_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ablation(ablation_df: pd.DataFrame) -> Path:
    # Canonical stage order so the bars read as a progression
    order = ["A0", "A1", "A2", "A3", "A3b", "A4", "A5", "A6", "A7"]
    df = ablation_df.copy()
    df["_order"] = df["stage"].map({s: i for i, s in enumerate(order)}).fillna(len(order))
    df = df.sort_values("_order")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#4C78A8"] * len(df)
    best_idx = int(df["oof_acc"].values.argmax())
    colors[best_idx] = "#E45756"
    sns.barplot(data=df, x="stage", y="oof_acc", hue="stage", ax=ax, palette=colors, legend=False)
    for i, row in enumerate(df.itertuples()):
        ax.text(i, row.oof_acc + 0.0008, f"{row.oof_acc:.4f}", ha="center", fontsize=10)
    ymin = max(0.78, float(df["oof_acc"].min()) - 0.006)
    ymax = float(df["oof_acc"].max()) + 0.006
    ax.set_ylim(ymin, ymax)
    ax.set_title("Ablation: OOF accuracy per stage (group-aware CV)")
    ax.set_xlabel("Stage")
    ax.set_ylabel("OOF accuracy")
    return _savefig(fig, "ablation.png")
