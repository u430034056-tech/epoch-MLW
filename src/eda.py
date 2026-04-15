from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_TRAIN = ROOT / "data" / "raw" / "train.csv"
DATA_TEST = ROOT / "data" / "raw" / "test.csv"
OUT_DIR = ROOT / "outputs" / "eda"
TOTAL_DATA_OUT = OUT_DIR / "total_data.csv"
NOTES_OUT = OUT_DIR / "eda_notes.md"


def _ensure_outdir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _coerce_bools(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in ["CryoSleep", "VIP", "Transported"]:
        if col not in d.columns:
            continue
        # Kaggle CSV may store booleans as True/False, "True"/"False", or mixed strings.
        if d[col].dtype == bool:
            continue
        if d[col].dtype == object or str(d[col].dtype).startswith("string"):
            s = d[col].astype("string")
            s_norm = s.str.strip().str.lower()
            d[col] = s_norm.map({"true": True, "false": False})
    return d


def _read_total_data() -> pd.DataFrame:
    train = _coerce_bools(pd.read_csv(DATA_TRAIN))
    test = _coerce_bools(pd.read_csv(DATA_TEST))

    # Total data = train + test. Test has no label, we keep Transported as NA for it.
    if "Transported" not in test.columns:
        test["Transported"] = pd.NA
    total_data = pd.concat([train, test], ignore_index=True, sort=False)
    return total_data


def _savefig(name: str) -> Path:
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def _write_notes(lines: list[str]) -> None:
    NOTES_OUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def plot_missingness(df: pd.DataFrame) -> Path:
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=miss.index, y=miss.values, color="#4C78A8")
    ax.set_title("Missing value rate by feature")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Missing rate")
    ax.set_ylim(0, max(0.05, float(miss.max()) * 1.1))
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    for i, v in enumerate(miss.values):
        ax.text(i, v + 0.005, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title("Missing value rate (total_data = train+test)")
    return _savefig("missingness_rate_combined.png")


def plot_age_distribution(df: pd.DataFrame) -> Path:
    plt.figure(figsize=(8, 4.5))
    ax = sns.histplot(data=df, x="Age", bins=30, stat="density", kde=True, color="#54A24B")
    ax.set_title("Age distribution (total_data)")
    ax.set_xlabel("Age")
    ax.set_ylabel("Density")
    return _savefig("age_distribution_combined.png")


def plot_spend_distributions(df: pd.DataFrame) -> Path:
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    spend_cols = [c for c in spend_cols if c in df.columns]
    d = df[spend_cols].copy()

    # Long format for faceting
    long = d.melt(var_name="Feature", value_name="Spend")
    long["Spend"] = long["Spend"].fillna(0.0)

    g = sns.FacetGrid(long, col="Feature", col_wrap=3, sharex=False, sharey=False, height=3)
    g.map_dataframe(sns.histplot, x="Spend", bins=40, color="#F58518")
    g.set_titles("{col_name}")
    g.fig.suptitle("Spend distributions (total_data; missing=0)", y=1.02)
    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.set_ylabel("Count (log)")
    plt.tight_layout()
    path = OUT_DIR / "spend_distributions_logcount_combined.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def _add_deck_side_from_cabin(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Cabin" in d.columns:
        parts = d["Cabin"].astype("string").str.split("/", expand=True)
        if "Deck" not in d.columns:
            d["Deck"] = parts[0]
        if "Side" not in d.columns:
            d["Side"] = parts[2]
    return d


def plot_categoricals(df: pd.DataFrame, top_n: int = 20) -> list[Path]:
    d = _add_deck_side_from_cabin(df)
    cat_cols = ["HomePlanet", "CryoSleep", "VIP", "Deck", "Side", "Destination"]
    out: list[Path] = []

    for col in cat_cols:
        if col not in d.columns:
            continue
        vc = d[col].fillna("Missing").value_counts(dropna=False).head(top_n)
        plt.figure(figsize=(9, max(4, 0.25 * len(vc))))
        ax = sns.barplot(x=vc.values, y=vc.index, color="#4C78A8")
        ax.set_title(f"{col} distribution (total_data, top {len(vc)})")
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        for i, val in enumerate(vc.values):
            ax.text(val, i, f" {int(val)}", va="center", fontsize=8)
        out.append(_savefig(f"{col.lower()}_distribution_combined.png"))

    return out


def plot_target_distribution(df_total: pd.DataFrame) -> Path:
    # total_data includes test rows where Transported is missing (NA).
    d = df_total.copy()
    d["Transported_total"] = d["Transported"].astype("object")
    d.loc[d["Transported_total"].isna(), "Transported_total"] = "Missing"

    order = [False, True, "Missing"]
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(
        data=d,
        x="Transported_total",
        order=[x for x in order if x in set(d["Transported_total"].unique())],
        color="#4C78A8",
    )
    ax.set_title("Transported distribution (total_data: train labels + test Missing)")
    ax.set_xlabel("Transported")
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )
    total_path = _savefig("target_distribution_total_data.png")

    # Also keep the conventional train-only target distribution for reporting.
    train = df_total[df_total["Transported"].notna()].copy()
    plt.figure(figsize=(5, 4))
    ax2 = sns.countplot(
        data=train,
        x="Transported",
        hue="Transported",
        palette="Set2",
        legend=False,
    )
    ax2.set_title("Target distribution (Transported) [train subset of total_data]")
    ax2.set_xlabel("Transported")
    ax2.set_ylabel("Count")
    for p in ax2.patches:
        ax2.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )
    _savefig("target_distribution_train_subset.png")
    return total_path


def plot_numeric_vs_target(df_total: pd.DataFrame) -> list[Path]:
    """
    Numeric features vs target on the train subset of total_data.
    Produces:
    - Distribution of Age by target (box)
    - Total spend by target (box, log1p)
    - Point-biserial correlations (saved as table)
    """
    train = df_total[df_total["Transported"].notna()].copy()
    train["Transported"] = train["Transported"].astype(bool)

    spend_cols = [c for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"] if c in train.columns]
    if spend_cols:
        for c in spend_cols:
            train[c] = pd.to_numeric(train[c], errors="coerce")
        train["TotalSpend"] = train[spend_cols].fillna(0).sum(axis=1)
        train["Log1pTotalSpend"] = np.log1p(train["TotalSpend"])

    out: list[Path] = []

    if "Age" in train.columns:
        plt.figure(figsize=(7, 4.5))
        ax = sns.boxplot(data=train, x="Transported", y="Age", palette="Set2", showfliers=False)
        ax.set_title("Age vs Transported (train subset of total_data)")
        ax.set_xlabel("Transported")
        ax.set_ylabel("Age")
        out.append(_savefig("age_vs_target_box_train_subset.png"))

    if "Log1pTotalSpend" in train.columns:
        plt.figure(figsize=(7, 4.5))
        ax = sns.boxplot(data=train, x="Transported", y="Log1pTotalSpend", palette="Set2", showfliers=False)
        ax.set_title("log1p(TotalSpend) vs Transported (train subset of total_data)")
        ax.set_xlabel("Transported")
        ax.set_ylabel("log1p(TotalSpend)")
        out.append(_savefig("logspend_vs_target_box_train_subset.png"))

    return out


def plot_categoricals_vs_target(df_total: pd.DataFrame) -> list[Path]:
    d = _add_deck_side_from_cabin(df_total)
    train = d[d["Transported"].notna()].copy()
    cat_cols = ["HomePlanet", "CryoSleep", "VIP", "Deck", "Side", "Destination"]
    out: list[Path] = []

    for col in cat_cols:
        if col not in train.columns:
            continue
        ct = pd.crosstab(train[col].fillna("Missing"), train["Transported"], normalize="index")
        # `col` may contain mixed types (e.g., bool + strings like "Missing"), so sort by string key.
        ct = ct.sort_index(key=lambda idx: idx.astype("string"))

        plt.figure(figsize=(8, 4.5))
        ax = ct.plot(kind="bar", stacked=True, color=["#4C78A8", "#54A24B"])
        ax.set_title(f"{col} vs Transported (train subset of total_data, row-normalized)")
        ax.set_xlabel(col)
        ax.set_ylabel("Proportion")
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.legend(title="Transported", loc="best")
        out.append(_savefig(f"{col.lower()}_vs_target_train_subset.png"))

    return out


def main() -> None:
    # Make plots deterministic-ish across machines
    os.environ.setdefault("PYTHONHASHSEED", "0")
    np.random.seed(0)
    sns.set_theme(style="whitegrid")

    _ensure_outdir()
    total_data = _read_total_data()
    total_data.to_csv(TOTAL_DATA_OUT, index=False)

    paths: list[Path] = []
    paths.append(plot_missingness(total_data))
    paths.append(plot_target_distribution(total_data))
    paths.append(plot_age_distribution(total_data))
    paths.append(plot_spend_distributions(total_data))
    paths.extend(plot_categoricals(total_data))
    paths.extend(plot_categoricals_vs_target(total_data))
    paths.extend(plot_numeric_vs_target(total_data))

    # Notes: one sentence per generated figure.
    notes: list[str] = []
    notes.append("## EDA notes (one sentence per figure; based on `total_data = train + test`)\n")

    def one_liner(p: Path) -> str:
        name = p.name
        if name == "missingness_rate_combined.png":
            return "Missing-rate bar chart by feature: quickly identifies the most-missing fields to guide imputation/encoding."
        if name == "target_distribution_total_data.png":
            return "Transported counts on total_data (including test as Missing): verifies the train+test merge and overall label mix."
        if name == "target_distribution_train_subset.png":
            return "Transported counts on the labeled train subset: checks class imbalance and a majority-class baseline."
        if name == "age_distribution_combined.png":
            return "Age histogram (total_data): shows concentration, tails/outliers, and informs binning or scaling."
        if name == "age_vs_target_box_train_subset.png":
            return "Age vs Transported boxplot (train subset): compares age distributions across classes to assess separability."
        if name == "spend_distributions_logcount_combined.png":
            return "Faceted histograms for the 5 spend features (total_data; log-scaled counts): reveals zero-inflation and long tails for feature engineering (e.g., log1p, zero-spend flags)."
        if name == "logspend_vs_target_box_train_subset.png":
            return "log1p(TotalSpend) vs Transported boxplot (train subset): checks whether overall spend correlates with the target."
        if name.endswith("_distribution_combined.png"):
            col = name.replace("_distribution_combined.png", "")
            return f"{col} frequency distribution (total_data): checks imbalance and missing/rare categories."
        if name.endswith("_vs_target_train_subset.png"):
            col = name.replace("_vs_target_train_subset.png", "")
            return f"{col} vs Transported row-normalized stacked bars (train subset): compares the Transported=True proportion across categories."
        return "EDA figure: helps understand distributions and feature–target relationships."

    for p in sorted(paths, key=lambda x: x.name):
        notes.append(f"- `{p.name}`：{one_liner(p)}")
    notes.append("")

    _write_notes(notes)

    print(f"total_data rows: {len(total_data)}")
    print(f"Saved total_data: {TOTAL_DATA_OUT.relative_to(ROOT)}")
    print(f"Saved notes: {NOTES_OUT.relative_to(ROOT)}")
    print("Saved EDA figures:")
    for p in paths:
        print(f"- {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

