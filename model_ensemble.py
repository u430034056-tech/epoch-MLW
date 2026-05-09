"""Multi-model probability ensemble for Spaceship Titanic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import model_lightgbm1 as lgbm1


MODEL_NAME = "multi_model_probability_ensemble"
DEFAULT_BASE_MODELS = ("lightgbm", "extra_trees")
DEFAULT_WEIGHT_STEP = 0.05
DEFAULT_RANDOM_STATE = lgbm1.DEFAULT_RANDOM_STATE
SKLEARN_DROPPED_CATEGORICAL_FEATURES = {"GroupSurname"}


def _project_root(project_root: str | Path | None = None) -> Path:
    return Path(project_root) if project_root is not None else Path(__file__).resolve().parent


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)


def _score_probabilities(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold_values: Sequence[float] | None,
) -> dict[str, Any]:
    clipped = _clip_probabilities(probabilities)
    threshold, tuned_accuracy = lgbm1._find_best_threshold(
        y_true,
        clipped,
        threshold_values=threshold_values,
    )
    return {
        "accuracy_at_0_5": float(accuracy_score(y_true, clipped >= 0.5)),
        "accuracy": float(tuned_accuracy),
        "threshold": float(threshold),
        "logloss": float(log_loss(y_true, clipped)),
    }


def _model_feature_columns(bundle: dict[str, Any]) -> tuple[list[str], list[str]]:
    X_train, _, _, categorical_features, _, _ = lgbm1._get_bundle_views(bundle)
    categorical = [column for column in categorical_features if column not in SKLEARN_DROPPED_CATEGORICAL_FEATURES]
    numeric = [column for column in X_train.columns if column not in categorical]
    numeric = [column for column in numeric if column not in SKLEARN_DROPPED_CATEGORICAL_FEATURES]
    return numeric, categorical


def _one_hot_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=True)


def _make_extra_trees_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    random_state: int,
    quick: bool = False,
) -> Pipeline:
    n_estimators = 100 if quick else 450
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", list(numeric_features)),
            ("cat", _one_hot_encoder(), list(categorical_features)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _make_sklearn_model(
    model_name: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    random_state: int,
    quick: bool,
) -> Pipeline:
    if model_name == "extra_trees":
        return _make_extra_trees_pipeline(numeric_features, categorical_features, random_state, quick)
    raise ValueError(f"Unsupported sklearn ensemble model: {model_name}")


def _validate_base_models(base_models: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(name) for name in base_models)
    supported = set(DEFAULT_BASE_MODELS)
    unsupported = sorted(set(selected) - supported)
    if unsupported:
        raise ValueError(f"Unsupported base models: {unsupported}")
    if not selected:
        raise ValueError("At least one base model is required.")
    return selected


def _integer_weight_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]

    results: list[tuple[int, ...]] = []
    for value in range(total + 1):
        for suffix in _integer_weight_compositions(total - value, parts - 1):
            results.append((value, *suffix))
    return results


def _search_ensemble_weights(
    y_true: pd.Series,
    base_oof_probabilities: dict[str, np.ndarray],
    threshold_values: Sequence[float] | None,
    weight_step: float,
) -> dict[str, Any]:
    model_names = list(base_oof_probabilities)
    if len(model_names) == 1:
        only_model = model_names[0]
        probabilities = base_oof_probabilities[only_model]
        score = _score_probabilities(y_true, probabilities, threshold_values)
        return {
            "weights": {only_model: 1.0},
            "probabilities": probabilities,
            **score,
        }

    scale = int(round(1.0 / weight_step))
    if not np.isclose(scale * weight_step, 1.0):
        raise ValueError("weight_step must divide 1.0 exactly, for example 0.05 or 0.1.")

    best: dict[str, Any] | None = None
    probability_matrix = np.vstack([base_oof_probabilities[name] for name in model_names])
    for integer_weights in _integer_weight_compositions(scale, len(model_names)):
        weights = np.asarray(integer_weights, dtype=float) / scale
        if weights.sum() <= 0:
            continue
        probabilities = np.average(probability_matrix, axis=0, weights=weights)
        score = _score_probabilities(y_true, probabilities, threshold_values)
        candidate = {
            "weights": {name: float(weight) for name, weight in zip(model_names, weights)},
            "probabilities": probabilities,
            **score,
        }
        if best is None:
            best = candidate
            continue
        candidate_key = (candidate["accuracy"], -candidate["logloss"], candidate["accuracy_at_0_5"])
        best_key = (best["accuracy"], -best["logloss"], best["accuracy_at_0_5"])
        if candidate_key > best_key:
            best = candidate

    if best is None:
        raise RuntimeError("No ensemble weight candidate was evaluated.")
    return best


def _fit_lightgbm_fold(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_features: Sequence[str],
    model_params: dict[str, Any],
    seeds: Sequence[int],
    early_stopping_rounds: int,
) -> tuple[np.ndarray, list[int]]:
    fold_probabilities = np.zeros(len(X_valid), dtype=float)
    best_iterations: list[int] = []
    for seed in seeds:
        model = LGBMClassifier(**lgbm1._build_seeded_params(model_params, seed))
        model.fit(
            X_fit,
            y_fit,
            categorical_feature=list(categorical_features),
            eval_set=[(X_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=[
                early_stopping(early_stopping_rounds, verbose=False),
                log_evaluation(0),
            ],
        )
        fold_probabilities += model.predict_proba(X_valid)[:, 1] / len(seeds)
        best_iterations.append(int(model.best_iteration_ or model.n_estimators))
    return fold_probabilities, best_iterations


def _fit_lightgbm_full(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    categorical_features: Sequence[str],
    model_params: dict[str, Any],
    seeds: Sequence[int],
    final_n_estimators: int,
) -> tuple[list[LGBMClassifier], np.ndarray]:
    final_params = dict(model_params)
    final_params["n_estimators"] = int(final_n_estimators)
    models: list[LGBMClassifier] = []
    test_probabilities = np.zeros(len(X_test), dtype=float)
    for seed in seeds:
        model = LGBMClassifier(**lgbm1._build_seeded_params(final_params, seed))
        model.fit(
            X_train,
            y_train,
            categorical_feature=list(categorical_features),
        )
        models.append(model)
        test_probabilities += model.predict_proba(X_test)[:, 1] / len(seeds)
    return models, test_probabilities


def _fit_sklearn_model(
    estimator: Pipeline,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_predict: pd.DataFrame,
) -> tuple[Pipeline, np.ndarray]:
    model = clone(estimator)
    model.fit(X_fit, y_fit)
    probabilities = model.predict_proba(X_predict)[:, 1]
    return model, probabilities


def train_ensemble(
    project_root: str | Path | None = None,
    base_models: Sequence[str] = DEFAULT_BASE_MODELS,
    lgbm_model_params: dict[str, Any] | None = None,
    lgbm_seeds: Sequence[int] = lgbm1.DEFAULT_ENSEMBLE_SEEDS,
    n_splits: int = 5,
    early_stopping_rounds: int = 250,
    threshold_values: Sequence[float] | None = None,
    weight_step: float = DEFAULT_WEIGHT_STEP,
    cv_group_feature_mode: str = lgbm1.DEFAULT_CV_GROUP_FEATURE_MODE,
    final_group_feature_mode: str = lgbm1.DEFAULT_FINAL_GROUP_FEATURE_MODE,
    extra_feature_mode: str = lgbm1.DEFAULT_EXTRA_FEATURE_MODE,
    quiet_preprocessing: bool = True,
    quick: bool = False,
) -> dict[str, Any]:
    """Train a fold-local multi-model probability ensemble."""
    selected_base_models = _validate_base_models(base_models)
    root = _project_root(project_root)
    train_df, test_df = lgbm1.load_raw_project_data(project_root=root)
    y_all = train_df["Transported"].astype(int).copy()

    lgbm_params = dict(lgbm1.DEFAULT_TUNED_MODEL_PARAMS)
    if quick:
        lgbm_params["n_estimators"] = int(lgbm_params.get("n_estimators", 140))
    if lgbm_model_params is not None:
        lgbm_params.update(lgbm_model_params)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    base_oof_probabilities = {
        model_name: np.zeros(len(train_df), dtype=float)
        for model_name in selected_base_models
    }
    fold_summaries: list[dict[str, Any]] = []
    lgbm_best_iterations: list[int] = []

    for fold_index, (fit_index, valid_index) in enumerate(splitter.split(train_df, y_all), start=1):
        fit_df = train_df.iloc[fit_index].copy()
        valid_df = train_df.iloc[valid_index].copy()
        bundle = lgbm1._build_lightgbm_bundle_from_raw(
            fit_df,
            valid_df,
            group_feature_mode=cv_group_feature_mode,
            extra_feature_mode=extra_feature_mode,
            quiet=quiet_preprocessing,
        )
        X_fit, y_fit, X_valid, categorical_features, _, _ = lgbm1._get_bundle_views(bundle)
        y_valid = valid_df["Transported"].astype(int).copy()
        numeric_features, categorical_features_for_sklearn = _model_feature_columns(bundle)

        fold_summary: dict[str, Any] = {"fold": fold_index, "base_models": {}}

        if "lightgbm" in selected_base_models:
            probabilities, best_iterations = _fit_lightgbm_fold(
                X_fit=X_fit,
                y_fit=y_fit,
                X_valid=X_valid,
                y_valid=y_valid,
                categorical_features=categorical_features,
                model_params=lgbm_params,
                seeds=lgbm_seeds,
                early_stopping_rounds=early_stopping_rounds,
            )
            base_oof_probabilities["lightgbm"][valid_index] = probabilities
            lgbm_best_iterations.extend(best_iterations)
            fold_summary["base_models"]["lightgbm"] = _score_probabilities(
                y_valid,
                probabilities,
                threshold_values,
            )

        for model_name in selected_base_models:
            if model_name == "lightgbm":
                continue
            estimator = _make_sklearn_model(
                model_name,
                numeric_features,
                categorical_features_for_sklearn,
                random_state=DEFAULT_RANDOM_STATE + fold_index,
                quick=quick,
            )
            _, probabilities = _fit_sklearn_model(estimator, X_fit, y_fit, X_valid)
            base_oof_probabilities[model_name][valid_index] = probabilities
            fold_summary["base_models"][model_name] = _score_probabilities(
                y_valid,
                probabilities,
                threshold_values,
            )

        fold_summaries.append(fold_summary)

    base_model_scores = {
        model_name: _score_probabilities(y_all, probabilities, threshold_values)
        for model_name, probabilities in base_oof_probabilities.items()
    }
    ensemble_search = _search_ensemble_weights(
        y_all,
        base_oof_probabilities,
        threshold_values=threshold_values,
        weight_step=weight_step,
    )
    ensemble_probabilities = np.asarray(ensemble_search.pop("probabilities"), dtype=float)

    final_bundle = lgbm1._build_lightgbm_bundle_from_raw(
        train_df=train_df,
        test_df=test_df,
        group_feature_mode=final_group_feature_mode,
        extra_feature_mode=extra_feature_mode,
        quiet=False,
    )
    X_train, y_train, X_test, categorical_features, train_ids, test_ids = lgbm1._get_bundle_views(final_bundle)
    numeric_features, categorical_features_for_sklearn = _model_feature_columns(final_bundle)
    final_base_models: dict[str, Any] = {}
    test_base_probabilities: dict[str, np.ndarray] = {}

    if "lightgbm" in selected_base_models:
        final_n_estimators = max(int(round(np.mean(lgbm_best_iterations) if lgbm_best_iterations else 300)), 100)
        final_models, test_probabilities = _fit_lightgbm_full(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            categorical_features=categorical_features,
            model_params=lgbm_params,
            seeds=lgbm_seeds,
            final_n_estimators=final_n_estimators,
        )
        final_base_models["lightgbm"] = final_models
        test_base_probabilities["lightgbm"] = test_probabilities
    else:
        final_n_estimators = None

    for model_name in selected_base_models:
        if model_name == "lightgbm":
            continue
        estimator = _make_sklearn_model(
            model_name,
            numeric_features,
            categorical_features_for_sklearn,
            random_state=DEFAULT_RANDOM_STATE,
            quick=quick,
        )
        model, test_probabilities = _fit_sklearn_model(estimator, X_train, y_train, X_test)
        final_base_models[model_name] = model
        test_base_probabilities[model_name] = test_probabilities

    weights = dict(ensemble_search["weights"])
    ensemble_test_probabilities = np.zeros(len(X_test), dtype=float)
    for model_name, weight in weights.items():
        ensemble_test_probabilities += float(weight) * test_base_probabilities[model_name]

    category_levels = lgbm1._collect_category_levels(X_train, X_test, categorical_features)
    return {
        "model_name": MODEL_NAME,
        "training_mode": "fold_local_multi_model_oof_weight_search",
        "base_model_names": list(selected_base_models),
        "models": final_base_models,
        "best_weights": weights,
        "cv_accuracy": float(ensemble_search["accuracy"]),
        "cv_accuracy_at_0_5": float(ensemble_search["accuracy_at_0_5"]),
        "cv_logloss": float(ensemble_search["logloss"]),
        "threshold": float(ensemble_search["threshold"]),
        "base_model_scores": base_model_scores,
        "base_oof_probabilities": base_oof_probabilities,
        "oof_probabilities": ensemble_probabilities,
        "test_base_probabilities": test_base_probabilities,
        "test_probabilities": ensemble_test_probabilities,
        "feature_names": X_train.columns.tolist(),
        "categorical_feature_names": list(categorical_features),
        "category_levels": category_levels,
        "feature_count": int(X_train.shape[1]),
        "n_splits": int(n_splits),
        "lgbm_best_iteration_mean": float(np.mean(lgbm_best_iterations)) if lgbm_best_iterations else None,
        "lgbm_final_n_estimators": final_n_estimators,
        "lgbm_params": lgbm_params,
        "lgbm_seeds": [int(seed) for seed in lgbm_seeds],
        "weight_step": float(weight_step),
        "cv_group_feature_mode": cv_group_feature_mode,
        "final_group_feature_mode": final_group_feature_mode,
        "extra_feature_mode": extra_feature_mode,
        "X_test": X_test,
        "test_ids": test_ids,
        "train_ids": train_ids,
        "bundle": final_bundle,
        "fold_summaries": fold_summaries,
    }


def predict_test_set(
    model_artifact: dict[str, Any],
    threshold: float | None = None,
    return_proba: bool = False,
) -> pd.DataFrame:
    """Build competition test predictions from cached ensemble test probabilities."""
    test_ids = model_artifact.get("test_ids")
    probability_values = model_artifact.get("test_probabilities")
    if test_ids is None:
        raise ValueError("test_ids are missing from the ensemble artifact.")
    if probability_values is None:
        raise ValueError("test_probabilities are missing from the ensemble artifact.")

    probability_values = np.asarray(probability_values, dtype=float)
    effective_threshold = model_artifact.get("threshold", 0.5) if threshold is None else threshold
    result = pd.DataFrame(
        {
            "PassengerId": test_ids,
            "Transported": (probability_values >= float(effective_threshold)).astype(bool),
        }
    )
    if return_proba:
        result["TransportedProbability"] = probability_values
    return result


def build_submission(
    model_artifact: dict[str, Any],
    output_path: str | Path = "artifacts/submission_ensemble.csv",
    threshold: float | None = None,
) -> Path:
    submission_df = predict_test_set(model_artifact, threshold=threshold, return_proba=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    return output_path


def save_model(model_artifact: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, output_path)
    return output_path


def load_model(model_path: str | Path) -> dict[str, Any]:
    return joblib.load(Path(model_path))


def save_report(model_artifact: dict[str, Any], output_path: str | Path) -> Path:
    report = {
        "model_name": model_artifact["model_name"],
        "training_mode": model_artifact["training_mode"],
        "base_model_names": model_artifact["base_model_names"],
        "best_weights": model_artifact["best_weights"],
        "cv_accuracy": model_artifact["cv_accuracy"],
        "cv_accuracy_at_0_5": model_artifact["cv_accuracy_at_0_5"],
        "cv_logloss": model_artifact["cv_logloss"],
        "threshold": model_artifact["threshold"],
        "base_model_scores": model_artifact["base_model_scores"],
        "feature_count": model_artifact["feature_count"],
        "lgbm_best_iteration_mean": model_artifact["lgbm_best_iteration_mean"],
        "lgbm_final_n_estimators": model_artifact["lgbm_final_n_estimators"],
        "weight_step": model_artifact["weight_step"],
        "cv_group_feature_mode": model_artifact["cv_group_feature_mode"],
        "final_group_feature_mode": model_artifact["final_group_feature_mode"],
        "extra_feature_mode": model_artifact["extra_feature_mode"],
        "test_positive_rate": float(predict_test_set(model_artifact)["Transported"].astype(bool).mean()),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-model Spaceship Titanic probability ensemble.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--early-stopping-rounds", type=int, default=250)
    parser.add_argument("--lgbm-seeds", default="42,52,62")
    parser.add_argument("--weight-step", type=float, default=DEFAULT_WEIGHT_STEP)
    parser.add_argument("--base-models", default=",".join(DEFAULT_BASE_MODELS))
    parser.add_argument("--output-model", default="artifacts/ensemble_model.joblib")
    parser.add_argument("--output-submission", default="artifacts/submission_ensemble.csv")
    parser.add_argument("--output-report", default="artifacts/ensemble_report.json")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.n_splits = 2
        args.early_stopping_rounds = 20
        args.lgbm_seeds = "42"
    return args


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def _parse_model_names(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    return _validate_base_models(values)


if __name__ == "__main__":
    cli_args = parse_args()
    result = train_ensemble(
        base_models=_parse_model_names(cli_args.base_models),
        lgbm_seeds=_parse_int_tuple(cli_args.lgbm_seeds),
        n_splits=cli_args.n_splits,
        early_stopping_rounds=cli_args.early_stopping_rounds,
        weight_step=cli_args.weight_step,
        quick=cli_args.quick,
    )
    model_path = save_model(result, cli_args.output_model)
    submission_path = build_submission(result, cli_args.output_submission)
    report_path = save_report(result, cli_args.output_report)

    print("Cross-validated accuracy @ 0.5:", result["cv_accuracy_at_0_5"])
    print("Cross-validated accuracy @ tuned threshold:", result["cv_accuracy"])
    print("Cross-validated logloss:", result["cv_logloss"])
    print("Best threshold:", result["threshold"])
    print("Best weights:", result["best_weights"])
    print("Base model scores:", result["base_model_scores"])
    print("Saved model to:", model_path)
    print("Saved submission to:", submission_path)
    print("Saved report to:", report_path)
