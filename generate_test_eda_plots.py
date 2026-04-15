from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

from preprocess import (
    LOG1P_FEATURE_COLUMNS,
    SPEND_COLUMNS,
    _normalize_age_series,
    basic_cleaning,
    build_common_features,
    build_group_features_single_split,
    create_spend_features,
    create_spend_structure_features,
    enforce_dtypes,
    extract_name_features,
    fill_group_consistent_categories,
    get_project_paths,
    load_raw_data,
    split_cabin_features,
)


PROJECT_ROOT = Path(__file__).resolve().parent
EDA_ROOT = PROJECT_ROOT / "eda_outputs"
OUTPUT_DIRS = {
    "A": EDA_ROOT / "00_raw_audit",
    "B": EDA_ROOT / "01_engineered_test",
    "C": EDA_ROOT / "02_rule_diagnostics",
    "D": EDA_ROOT / "03_post_common_validation",
    "E": EDA_ROOT / "04_model_branch_validation",
}
AGE_GROUP_ORDER = ["Child", "Teen", "YoungAdult", "Adult", "MiddleAge", "Senior", "Unknown"]


def _setup_theme() -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9


def _ensure_dirs() -> None:
    EDA_ROOT.mkdir(exist_ok=True)
    for folder in OUTPUT_DIRS.values():
        folder.mkdir(parents=True, exist_ok=True)
        for png in folder.glob("*.png"):
            png.unlink()


def _coerce_string(series: pd.Series, missing_label: str = "Missing") -> pd.Series:
    text = series.astype("string")
    return text.fillna(missing_label).replace({"<NA>": missing_label})


def _annotate_bars(ax: plt.Axes, total: float | None = None, pct: bool = False) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if np.isnan(height):
            continue
        label = f"{int(height)}"
        if pct and total:
            label = f"{int(height)}\n({height / total:.1%})"
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _finalize(fig: plt.Figure, title: str, subtitle: str, note: str, output_path: Path) -> None:
    fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.945, subtitle, ha="center", va="top", fontsize=10)
    if note:
        fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_bar(
    counts: pd.Series,
    output_path: Path,
    title: str,
    subtitle: str = "",
    note: str = "",
    xlabel: str = "",
    ylabel: str = "Count",
    annotate_pct: bool = False,
    rotate: int = 0,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#4C78A8")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate)
    _annotate_bars(ax, total=float(counts.sum()), pct=annotate_pct)
    _finalize(fig, title, subtitle, note, output_path)


def _plot_count_series(
    series: pd.Series,
    output_path: Path,
    title: str,
    subtitle: str = "",
    note: str = "",
    order: list[str] | None = None,
    sort_desc: bool = True,
    annotate_pct: bool = True,
    rotate: int = 0,
) -> None:
    values = _coerce_string(series)
    counts = values.value_counts(dropna=False)
    if order is None:
        counts = counts.sort_values(ascending=not sort_desc)
    else:
        ordered_index = list(order) + [item for item in counts.index.tolist() if item not in order]
        counts = counts.reindex(ordered_index, fill_value=0)
    _plot_bar(
        counts=counts,
        output_path=output_path,
        title=title,
        subtitle=subtitle,
        note=note,
        xlabel=series.name or "",
        annotate_pct=annotate_pct,
        rotate=rotate,
    )


def _plot_hist(
    series: pd.Series,
    output_path: Path,
    title: str,
    subtitle: str = "",
    note: str = "",
    xlabel: str | None = None,
    bins: int = 30,
    kde: bool = True,
    color: str = "#4C78A8",
) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(values, bins=bins, kde=kde, ax=ax, color=color)
    ax.set_xlabel(xlabel or (series.name or "Value"))
    ax.set_ylabel("Count")
    _finalize(fig, title, subtitle, note, output_path)


def _plot_box(
    series: pd.Series,
    output_path: Path,
    title: str,
    subtitle: str = "",
    note: str = "",
    xlabel: str | None = None,
) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.boxplot(x=values, ax=ax, color="#72B7B2")
    ax.set_xlabel(xlabel or (series.name or "Value"))
    ax.set_yticks([])
    _finalize(fig, title, subtitle, note, output_path)


def _plot_box_by_category(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    output_path: Path,
    title: str,
    subtitle: str = "",
    note: str = "",
    order: list[str] | None = None,
) -> None:
    frame = df[[category_col, value_col]].copy()
    frame[category_col] = _coerce_string(frame[category_col])
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.dropna(subset=[value_col])
    if order is None:
        order = frame[category_col].value_counts(dropna=False).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.boxplot(data=frame, x=category_col, y=value_col, order=order, ax=ax)
    ax.set_xlabel(category_col)
    ax.set_ylabel(value_col)
    _finalize(fig, title, subtitle, note, output_path)


def _plot_missing_bar(df: pd.DataFrame, output_path: Path) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    rates = (missing / len(df)).fillna(0)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=missing.index, y=missing.values, ax=ax, color="#E15759")
    ax.set_xlabel("Field")
    ax.set_ylabel("Missing Count")
    ax.tick_params(axis="x", rotation=45)
    for patch, rate in zip(ax.patches, rates.values):
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height,
            f"{int(height)}\n({rate:.1%})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _finalize(
        fig,
        "Stage A (Raw): Missing values by field",
        "Raw / cleaned-only audit before any imputation",
        "Missing values are shown before creating missing indicators.",
        output_path,
    )


def _plot_missing_heatmap(df: pd.DataFrame, output_path: Path, sample_rows: int = 300) -> None:
    sampled = df.copy()
    sampled_note = "Full test set visualized."
    if len(sampled) > sample_rows:
        sampled = sampled.sample(sample_rows, random_state=42).sort_index()
        sampled_note = f"Rows sampled for readability: {sample_rows}/{len(df)}."
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(
        sampled.isna(),
        cmap=ListedColormap(["#4C78A8", "#F58518"]),
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Field")
    ax.set_ylabel("Sampled Row")
    _finalize(
        fig,
        "Stage A (Raw): Missingness heatmap",
        "Missing matrix for structural missingness check",
        sampled_note,
        output_path,
    )


def _raw_boolean_bucket(series: pd.Series) -> pd.Series:
    def classify(value: object) -> str:
        if pd.isna(value):
            return "Missing"
        if isinstance(value, (bool, np.bool_)):
            return "True" if bool(value) else "False"
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return "True"
        if text in {"false", "0", "no", "n"}:
            return "False"
        if text == "":
            return "Missing"
        return "Other"

    return series.map(classify)


def _zero_share_by_column(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = {}
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        observed = values.dropna()
        result[col] = float((observed == 0).mean()) if not observed.empty else 0.0
    return pd.Series(result).sort_values(ascending=False)


def _plot_percent_bar(
    values: pd.Series,
    output_path: Path,
    title: str,
    subtitle: str = "",
    note: str = "",
    xlabel: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(x=values.index.astype(str), y=values.values, ax=ax, color="#54A24B")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share")
    ax.set_ylim(0, max(1.0, float(values.max()) * 1.15 if len(values) else 1.0))
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _finalize(fig, title, subtitle, note, output_path)


def _plot_stacked_bar(
    frame: pd.DataFrame,
    row_col: str,
    col_col: str,
    output_path: Path,
    title: str,
    subtitle: str = "",
    note: str = "",
) -> None:
    table = pd.crosstab(_coerce_string(frame[row_col]), _coerce_string(frame[col_col]))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    table.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel(row_col)
    ax.set_ylabel("Count")
    ax.legend(title=col_col)
    _finalize(fig, title, subtitle, note, output_path)


def _top_other_counts(series: pd.Series, top_n: int = 10) -> pd.Series:
    counts = _coerce_string(series).value_counts(dropna=False)
    if len(counts) <= top_n:
        return counts
    top = counts.iloc[:top_n].copy()
    top.loc["Other"] = counts.iloc[top_n:].sum()
    return top


def _prepare_data(project_root: Path) -> dict[str, object]:
    paths = get_project_paths(project_root)
    train_df, test_df = load_raw_data(paths["data_dir"])
    raw_test = basic_cleaning(test_df.copy())
    typed_test = enforce_dtypes(raw_test.copy())

    stage_b = build_group_features_single_split(typed_test.copy())
    stage_b["GroupSize_test_local"] = stage_b["GroupSize"]
    stage_b = split_cabin_features(stage_b)
    stage_b = extract_name_features(stage_b)
    stage_b = create_spend_features(stage_b)
    stage_b = create_spend_structure_features(stage_b)
    stage_b = _apply_stage_b_complete_case_spend_logic(stage_b)

    common_bundle = build_common_features(train_df, test_df)
    common_test = common_bundle["common_test"].copy()

    return {
        "train_df": train_df,
        "raw_test": raw_test,
        "typed_test": typed_test,
        "stage_b": stage_b,
        "common_bundle": common_bundle,
        "common_test": common_test,
    }


def _apply_stage_b_complete_case_spend_logic(df: pd.DataFrame) -> pd.DataFrame:
    updated = df.copy()
    spend_frame = updated[SPEND_COLUMNS].apply(pd.to_numeric, errors="coerce")
    spend_missing_mask = spend_frame.isna().any(axis=1)

    updated["StageBSpendAnyMissing"] = spend_missing_mask.astype(int)
    updated["TotalSpend"] = spend_frame.sum(axis=1, min_count=len(SPEND_COLUMNS))

    spend_count = pd.Series(spend_frame.gt(0).sum(axis=1), index=updated.index, dtype="Int64")
    spend_count.loc[spend_missing_mask] = pd.NA
    updated["SpendCount"] = spend_count

    is_zero_spend = pd.Series([pd.NA] * len(updated), index=updated.index, dtype="Int64")
    observed_total_mask = updated["TotalSpend"].notna()
    is_zero_spend.loc[observed_total_mask] = (updated.loc[observed_total_mask, "TotalSpend"] == 0).astype("Int64")
    updated["IsZeroSpend"] = is_zero_spend

    updated["LuxurySpend"] = spend_frame[["Spa", "VRDeck"]].sum(axis=1, min_count=2)
    updated["BasicSpend"] = spend_frame[["RoomService", "FoodCourt", "ShoppingMall"]].sum(axis=1, min_count=3)

    updated["LuxuryShare"] = np.nan
    updated.loc[observed_total_mask, "LuxuryShare"] = (
        updated.loc[observed_total_mask, "LuxurySpend"] / (updated.loc[observed_total_mask, "TotalSpend"] + 1.0)
    )

    spend_per_active = pd.Series(np.nan, index=updated.index, dtype="float64")
    zero_count_mask = observed_total_mask & updated["SpendCount"].eq(0).fillna(False)
    nonzero_count_mask = observed_total_mask & updated["SpendCount"].gt(0).fillna(False)
    spend_per_active.loc[zero_count_mask] = 0.0
    spend_per_active.loc[nonzero_count_mask] = (
        updated.loc[nonzero_count_mask, "TotalSpend"] / updated.loc[nonzero_count_mask, "SpendCount"].astype(float)
    )
    updated["SpendPerActiveCategory"] = spend_per_active

    return updated


def generate_stage_a(raw_test: pd.DataFrame, typed_test: pd.DataFrame) -> None:
    out = OUTPUT_DIRS["A"]
    _plot_missing_bar(raw_test, out / "00_M1_missing_bar.png")
    _plot_missing_heatmap(raw_test, out / "00_M2_missing_heatmap.png")

    for column, filename in [
        ("CryoSleep", "00_M3_cryosleep_raw_value_counts.png"),
        ("VIP", "00_M3_vip_raw_value_counts.png"),
    ]:
        counts = _raw_boolean_bucket(raw_test[column]).value_counts().reindex(
            ["True", "False", "Missing", "Other"], fill_value=0
        )
        _plot_bar(
            counts=counts,
            output_path=out / filename,
            title=f"Stage A (Raw): {column} raw value buckets",
            subtitle="Before boolean normalization",
            note="Raw values are grouped into True / False / Missing / Other.",
            xlabel=column,
            annotate_pct=True,
        )

    category_specs = [
        ("HomePlanet", "00_C1_homeplanet_distribution.png"),
        ("Destination", "00_C2_destination_distribution.png"),
        ("CryoSleep", "00_C3_cryosleep_distribution.png"),
        ("VIP", "00_C4_vip_distribution.png"),
    ]
    for column, filename in category_specs:
        _plot_count_series(
            typed_test[column],
            out / filename,
            title=f"Stage A (Raw): {column} distribution",
            subtitle="Missing shown as a standalone category",
            note="Stage A uses raw / typed-only fields, before train-fit preprocessing.",
            rotate=20,
        )

    age_numeric = pd.to_numeric(typed_test["Age"], errors="coerce")
    age_note = f"Missing ages excluded from histogram: {int(age_numeric.isna().sum())}."
    _plot_hist(
        age_numeric,
        out / "00_N1_age_hist.png",
        title="Stage A (Raw): Age distribution",
        subtitle="Histogram with KDE on observed raw ages",
        note=age_note,
        xlabel="Age",
        bins=30,
        kde=True,
    )
    normalized_age, out_mask = _normalize_age_series(typed_test["Age"])
    outlier_count = int(pd.Series(out_mask).sum())
    _plot_box(
        age_numeric,
        out / "00_N2_age_boxplot.png",
        title="Stage A (Raw): Age boxplot",
        subtitle="Observed raw ages before any train-fit filling",
        note=f"Out-of-range candidates (<0 or >100): {outlier_count}.",
        xlabel="Age",
    )

    spend_specs = [
        ("RoomService", "00_N3_roomservice_hist.png", "00_N8_roomservice_log1p_hist.png", "00_N13_roomservice_boxplot.png"),
        ("FoodCourt", "00_N4_foodcourt_hist.png", "00_N9_foodcourt_log1p_hist.png", "00_N14_foodcourt_boxplot.png"),
        ("ShoppingMall", "00_N5_shoppingmall_hist.png", "00_N10_shoppingmall_log1p_hist.png", "00_N15_shoppingmall_boxplot.png"),
        ("Spa", "00_N6_spa_hist.png", "00_N11_spa_log1p_hist.png", "00_N16_spa_boxplot.png"),
        ("VRDeck", "00_N7_vrdeck_hist.png", "00_N12_vrdeck_log1p_hist.png", "00_N17_vrdeck_boxplot.png"),
    ]
    for column, raw_file, log_file, box_file in spend_specs:
        values = pd.to_numeric(typed_test[column], errors="coerce")
        observed = values.dropna()
        note = f"Observed non-missing samples: {len(observed)}. Raw scale retained for long-tail inspection."
        _plot_hist(
            observed,
            out / raw_file,
            title=f"Stage A (Raw): {column} distribution",
            subtitle="Raw spend scale",
            note=note,
            xlabel=column,
            bins=35,
            kde=False,
        )
        _plot_hist(
            np.log1p(observed[observed >= 0]),
            out / log_file,
            title=f"Stage A (Raw): {column} log1p distribution",
            subtitle="LR / XGBoost branch preview, not raw scale",
            note="Only non-negative values are log1p transformed.",
            xlabel=f"log1p({column})",
            bins=35,
            kde=False,
            color="#F58518",
        )
        _plot_box(
            observed,
            out / box_file,
            title=f"Stage A (Raw): {column} boxplot",
            subtitle="Raw spend scale with outlier visibility",
            note=f"Observed non-missing samples: {len(observed)}.",
            xlabel=column,
        )


def generate_stage_b(typed_test: pd.DataFrame, stage_b: pd.DataFrame) -> None:
    out = OUTPUT_DIRS["B"]
    spend_missing_rows = int(stage_b["StageBSpendAnyMissing"].sum())
    zero_share = _zero_share_by_column(typed_test, SPEND_COLUMNS)
    _plot_percent_bar(
        zero_share,
        out / "01_S1_zero_share_by_spend_column.png",
        title="Stage B (Engineered): Zero-share by spend column",
        subtitle="Computed within each observed non-missing spend column",
        note="Supports IsZeroSpend and SpendCount feature construction.",
        xlabel="Spend Column",
    )

    _plot_hist(
        stage_b["TotalSpend"],
        out / "01_S2_totalspend_hist.png",
        title="Stage B (Engineered): TotalSpend distribution",
        subtitle="Complete-case TotalSpend for test-only exploratory stage",
        note=(
            f"Rows with any spend missing are excluded from TotalSpend (n={spend_missing_rows}) "
            "to avoid pandas skipna undercount and false zero-spend interpretation."
        ),
        xlabel="TotalSpend",
        bins=35,
        kde=False,
    )
    _plot_hist(
        np.log1p(pd.to_numeric(stage_b["TotalSpend"], errors="coerce").clip(lower=0)),
        out / "01_S3_totalspend_log1p_hist.png",
        title="Stage B (Engineered): log1p(TotalSpend) distribution",
        subtitle="LR / XGBoost-aligned view, not raw scale",
        note=(
            f"log1p view is applied only to complete-case TotalSpend rows; partial spend-missing rows "
            f"remain excluded (n={spend_missing_rows})."
        ),
        xlabel="log1p(TotalSpend)",
        bins=35,
        kde=False,
        color="#F58518",
    )
    _plot_count_series(
        stage_b["SpendCount"],
        out / "01_S4_spendcount_distribution.png",
        title="Stage B (Engineered): SpendCount distribution",
        subtitle="Number of active spend categories per passenger",
        note=(
            "SpendCount is shown as Missing when any spend field is missing, "
            "instead of silently undercounting positive categories."
        ),
        order=[str(i) for i in range(6)],
        sort_desc=False,
    )
    _plot_count_series(
        stage_b["IsZeroSpend"],
        out / "01_S5_iszerospend_distribution.png",
        title="Stage B (Engineered): IsZeroSpend distribution",
        subtitle="Binary indicator for zero total spend",
        note=(
            "Rows with any spend missing stay in a Missing bucket, "
            "so partial spend rows are not misread as confirmed zero-spend passengers."
        ),
        order=["0", "1"],
        sort_desc=False,
    )
    for column, filename in [
        ("LuxurySpend", "01_S6_luxuryspend_hist.png"),
        ("BasicSpend", "01_S6_basicspend_hist.png"),
    ]:
        _plot_hist(
            stage_b[column],
            out / filename,
            title=f"Stage B (Engineered): {column} distribution",
            subtitle="Spend structure feature from test-only deterministic engineering",
            note=(
                "Component sums require complete observed component fields; "
                "missing component rows are excluded from the histogram."
            ),
            xlabel=column,
            bins=35,
            kde=False,
        )

    _plot_count_series(
        stage_b["GroupSize_test_local"],
        out / "01_G1_group_size_test_local_distribution.png",
        title="Stage B (Engineered): GroupSize_test_local distribution",
        subtitle="test-local exploratory only",
        note="This is not the formal combined-split GroupSize used in the shared pipeline.",
        sort_desc=False,
    )
    _plot_count_series(
        stage_b["GroupMemberNo"],
        out / "01_G2_group_member_no_distribution.png",
        title="Stage B (Engineered): GroupMemberNo distribution",
        subtitle="PassengerId split within test-only data",
        note="GroupMemberNo comes from PassengerId -> GroupID / member index.",
        sort_desc=False,
    )
    _plot_count_series(
        stage_b["Deck"],
        out / "01_G3_deck_distribution.png",
        title="Stage B (Engineered): Deck distribution",
        subtitle="Cabin split into deck / number / side",
        note="Missing deck values are shown as Missing.",
    )
    _plot_count_series(
        stage_b["Side"],
        out / "01_G4_side_distribution.png",
        title="Stage B (Engineered): Side distribution",
        subtitle="Cabin split into deck / number / side",
        note="Missing side values are shown as Missing.",
    )
    cabin_num = pd.to_numeric(stage_b["CabinNum"], errors="coerce")
    _plot_hist(
        cabin_num,
        out / "01_G5_cabin_num_hist.png",
        title="Stage B (Engineered): CabinNum distribution",
        subtitle="Observed cabin number values only",
        note=f"Missing CabinNum excluded from histogram: {int(cabin_num.isna().sum())}.",
        xlabel="CabinNum",
        bins=35,
        kde=False,
    )
    surname_counts = _coerce_string(stage_b["Surname"]).replace({"Missing": pd.NA}).dropna().value_counts()
    _plot_hist(
        surname_counts,
        out / "01_G6_surname_frequency_hist.png",
        title="Stage B (Engineered): Surname frequency distribution",
        subtitle="Distribution of occurrence counts per surname",
        note=f"Missing surnames excluded: {int(_coerce_string(stage_b['Surname']).eq('Missing').sum())}.",
        xlabel="Occurrences per surname",
        bins=20,
        kde=False,
    )


def generate_stage_c(typed_test: pd.DataFrame, stage_b: pd.DataFrame) -> None:
    out = OUTPUT_DIRS["C"]
    cryo_order = ["False", "True", "Missing"]
    _plot_box_by_category(
        stage_b,
        "CryoSleep",
        "TotalSpend",
        out / "02_R1_cryosleep_vs_totalspend_boxplot.png",
        title="Stage C (Rule): CryoSleep vs TotalSpend",
        subtitle="Rule diagnostic for sleep status and spend consistency",
        note=(
            "CryoSleep missing is shown as Missing; TotalSpend uses Stage B complete-case aggregation, "
            "so rows with any spend missing are excluded here."
        ),
        order=cryo_order,
    )
    _plot_box_by_category(
        stage_b,
        "CryoSleep",
        "SpendCount",
        out / "02_R2_cryosleep_vs_spendcount.png",
        title="Stage C (Rule): CryoSleep vs SpendCount",
        subtitle="Rule diagnostic for sleep status and active spend categories",
        note=(
            "SpendCount is complete-case in Stage B; rows with any spend missing are kept as Missing "
            "rather than being silently undercounted."
        ),
        order=cryo_order,
    )
    stacked = stage_b.copy()
    stacked["IsZeroSpendLabel"] = stage_b["IsZeroSpend"].map({0: "NonZeroSpend", 1: "ZeroSpend"}).astype("string")
    _plot_stacked_bar(
        stacked,
        "CryoSleep",
        "IsZeroSpendLabel",
        out / "02_R3_cryosleep_by_iszerospend_stacked.png",
        title="Stage C (Rule): CryoSleep by IsZeroSpend",
        subtitle="Stacked counts for rule consistency inspection",
        note=(
            "Used as structural support for the CryoSleep spend rule; IsZeroSpend retains a Missing bucket "
            "for partial spend rows instead of collapsing them into zero-spend."
        ),
    )

    _, group_fill_counts = fill_group_consistent_categories(
        build_group_features_single_split(typed_test.copy()),
        ["HomePlanet", "VIP", "Destination"],
    )
    _plot_bar(
        pd.Series(group_fill_counts),
        out / "02_R4_group_fill_candidates.png",
        title="Stage C (Rule): Group-consistent fill candidates",
        subtitle="Test-only exploratory support for group consistency rules",
        note="Counts show how many missing values can be supported by single-value within-group evidence in test only.",
        xlabel="Field",
        annotate_pct=False,
    )

    raw_age = pd.to_numeric(typed_test["Age"], errors="coerce")
    _, out_mask = _normalize_age_series(typed_test["Age"])
    out_mask = pd.Series(out_mask).astype(bool)
    age_diag = pd.Series(
        {
            "ValidRange": int((raw_age.notna() & ~out_mask).sum()),
            "OutOfRange": int(out_mask.sum()),
            "Missing": int(raw_age.isna().sum()),
        }
    )
    _plot_bar(
        age_diag,
        out / "02_R5_age_out_of_range_diagnostic.png",
        title="Stage C (Rule): Age out-of-range diagnostic",
        subtitle="Diagnostic counts before any train-fit filling",
        note="OutOfRange means Age < 0 or Age > 100 according to _normalize_age_series.",
        xlabel="Age Diagnostic Bucket",
        annotate_pct=True,
    )


def generate_stage_d(common_test: pd.DataFrame) -> None:
    out = OUTPUT_DIRS["D"]
    _plot_count_series(
        common_test["AgeGroup"],
        out / "03_A1_agegroup_distribution.png",
        title="Stage D (Post-common): AgeGroup distribution",
        subtitle="Shared preprocessing output",
        note="AgeGroup is created after train-fit age handling in the shared pipeline.",
        order=AGE_GROUP_ORDER,
        sort_desc=False,
    )
    for column, filename in [
        ("IsChild", "03_A2_ischild_distribution.png"),
        ("IsSenior", "03_A2_issenior_distribution.png"),
    ]:
        _plot_count_series(
            common_test[column],
            out / filename,
            title=f"Stage D (Post-common): {column} distribution",
            subtitle="Shared preprocessing output",
            note="Binary age-derived feature after common preprocessing.",
            order=["0", "1"],
            sort_desc=False,
        )

    _plot_count_series(
        common_test["CabinNumBin"],
        out / "03_B1_cabinnumbin_distribution.png",
        title="Stage D (Post-common): CabinNumBin distribution",
        subtitle="Train-fit statistic: bin edges are fit on train only",
        note="This plot checks how test rows fall into train-fit cabin number bins.",
        sort_desc=True,
    )
    missing_bucket = pd.Series(
        {
            "CabinBin_Missing": int(_coerce_string(common_test["CabinNumBin"]).eq("CabinBin_Missing").sum()),
            "OtherBins": int(_coerce_string(common_test["CabinNumBin"]).ne("CabinBin_Missing").sum()),
        }
    )
    _plot_bar(
        missing_bucket,
        out / "03_B2_cabinnumbin_missing_bucket.png",
        title="Stage D (Post-common): CabinBin_Missing bucket share",
        subtitle="Train-fit statistic: bin edges are fit on train only",
        note="Used to verify whether the missing cabin-number bucket remains a minority bucket.",
        xlabel="CabinNumBin Bucket Type",
        annotate_pct=True,
    )
    _plot_hist(
        common_test["SurnameFreq"],
        out / "03_B3_surnamefreq_hist.png",
        title="Stage D (Post-common): SurnameFreq distribution",
        subtitle="Train-fit frequency; unseen test surnames map to 0",
        note="SurnameFreq is fit on train surnames only and then applied to test.",
        xlabel="SurnameFreq",
        bins=25,
        kde=False,
    )

    _plot_bar(
        _top_other_counts(common_test["DeckSide"], top_n=10),
        out / "03_I1_deckside_top_distribution.png",
        title="Stage D (Post-common): DeckSide top distribution",
        subtitle="Interaction categorical feature after common preprocessing",
        note="Top 10 categories shown; remaining categories are merged into Other.",
        xlabel="DeckSide",
        annotate_pct=True,
        rotate=25,
    )
    _plot_bar(
        _top_other_counts(common_test["HomePlanetDestination"], top_n=10),
        out / "03_I2_homeplanet_destination_top_distribution.png",
        title="Stage D (Post-common): HomePlanetDestination top distribution",
        subtitle="Interaction categorical feature after common preprocessing",
        note="Top 10 categories shown; remaining categories are merged into Other.",
        xlabel="HomePlanetDestination",
        annotate_pct=True,
        rotate=25,
    )

    common_missing = common_test.isna().sum()
    has_post_common_missing = int(common_missing.sum()) > 0
    if not has_post_common_missing:
        common_missing = pd.Series({"AllCheckedColumns": 0})
    else:
        common_missing = common_missing[common_missing > 0].sort_values(ascending=False)
    _plot_bar(
        common_missing,
        out / "03_V1_post_common_missing_bar.png",
        title="Stage D (Post-common): Final missing-value audit",
        subtitle="Expected result for common_test is zero missing in model-input fields",
        note="Any non-zero bar here would indicate a preprocessing pipeline issue.",
        xlabel="Field",
        annotate_pct=False,
        rotate=25,
    )
    if has_post_common_missing:
        missing_summary = ", ".join(f"{column}={int(count)}" for column, count in common_missing.items())
        raise RuntimeError(
            "Post-common validation failed: common_test still contains missing values. "
            f"V1 plot has been saved for diagnosis. Details: {missing_summary}"
        )


def generate_stage_e(common_test: pd.DataFrame) -> None:
    out = OUTPUT_DIRS["E"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    total_spend = pd.to_numeric(common_test["TotalSpend"], errors="coerce").clip(lower=0)
    sns.histplot(total_spend, bins=35, kde=False, ax=axes[0], color="#4C78A8")
    axes[0].set_title("Raw TotalSpend")
    axes[0].set_xlabel("TotalSpend")
    sns.histplot(np.log1p(total_spend), bins=35, kde=False, ax=axes[1], color="#F58518")
    axes[1].set_title("log1p(TotalSpend)")
    axes[1].set_xlabel("log1p(TotalSpend)")
    _finalize(
        fig,
        "Stage E (Model-branch): TotalSpend before vs after log1p",
        "LR / XGBoost-aligned transform check",
        "Both views come from post-common TotalSpend before model-branch encoding.",
        out / "04_P1_totalspend_before_after_log1p.png",
    )

    spend_compare_cols = [column for column in LOG1P_FEATURE_COLUMNS if column != "TotalSpend"]
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    flat_axes = axes.flatten()
    for idx, column in enumerate(spend_compare_cols):
        raw_vals = pd.to_numeric(common_test[column], errors="coerce").clip(lower=0)
        sns.histplot(raw_vals, bins=25, kde=False, ax=flat_axes[idx * 2], color="#4C78A8")
        flat_axes[idx * 2].set_title(f"{column} (raw)")
        flat_axes[idx * 2].set_xlabel(column)
        sns.histplot(np.log1p(raw_vals), bins=25, kde=False, ax=flat_axes[idx * 2 + 1], color="#F58518")
        flat_axes[idx * 2 + 1].set_title(f"{column} (log1p)")
        flat_axes[idx * 2 + 1].set_xlabel(f"log1p({column})")
    _finalize(
        fig,
        "Stage E (Model-branch): Spend columns before vs after log1p",
        "Review of LR / XGBoost-aligned numeric transforms",
        f"Columns reviewed here: {', '.join(spend_compare_cols)}. TotalSpend is handled separately in P1.",
        out / "04_P2_spend_columns_before_after_log1p.png",
    )

    cardinality_cols = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "Deck",
        "Side",
        "AgeGroup",
        "CabinNumBin",
        "DeckSide",
        "HomePlanetDestination",
        "Surname",
    ]
    cardinality = pd.Series(
        {col: int(_coerce_string(common_test[col]).nunique(dropna=False)) for col in cardinality_cols}
    ).sort_values(ascending=False)
    _plot_bar(
        cardinality,
        out / "04_P3_categorical_cardinality_bar.png",
        title="Stage E (Model-branch): Categorical cardinality review",
        subtitle="One-hot expansion and category-modeling risk check",
        note="Surname is included as an extra CatBoost-focused high-cardinality field.",
        xlabel="Categorical Field",
        annotate_pct=False,
        rotate=25,
    )

    surname_freq = _coerce_string(common_test["Surname"]).value_counts()
    rare_buckets = pd.Series(
        {
            "freq=1": int((surname_freq == 1).sum()),
            "freq=2": int((surname_freq == 2).sum()),
            "freq=3-5": int(surname_freq.between(3, 5).sum()),
            "freq=6-10": int(surname_freq.between(6, 10).sum()),
            "freq>10": int((surname_freq > 10).sum()),
        }
    )
    _plot_bar(
        rare_buckets,
        out / "04_P4_surname_rare_category_diagnostic.png",
        title="Stage E (Model-branch): Surname rare-category diagnostic",
        subtitle="Category frequency layering for CatBoost input inspection",
        note="Bars count how many distinct surname categories fall into each frequency bucket.",
        xlabel="Surname Frequency Bucket",
        annotate_pct=True,
    )


def main() -> None:
    _setup_theme()
    _ensure_dirs()
    data = _prepare_data(PROJECT_ROOT)
    generate_stage_a(data["raw_test"], data["typed_test"])
    generate_stage_b(data["typed_test"], data["stage_b"])
    generate_stage_c(data["typed_test"], data["stage_b"])
    generate_stage_d(data["common_test"])
    generate_stage_e(data["common_test"])
    print(f"EDA plots generated under: {EDA_ROOT}")


if __name__ == "__main__":
    main()
