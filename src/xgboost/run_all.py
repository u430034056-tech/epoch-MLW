"""End-to-end entry point.

Running ``python -m src.xgboost.run_all`` executes the full A0 → A6 pipeline:

* A0 : legacy bundle + StratifiedKFold       → reproduces the current team baseline
* A1 : legacy bundle + StratifiedGroupKFold  → the honest number behind 0.80804
* A2 : common features + native category     → first real improvement
* A3 : A2 + fold-aware SurnameFreq / CabinNumBin + fold-safe target encoding
* A4 : A3 + Optuna-tuned hyper-parameters
* A5 : A4 + threshold scan + CryoSleep rule post-processing
* A6 : A5 × 5 seeds averaged

All artefacts (OOF csv, test probabilities, feature importance, figures,
submissions) are written under ``reports/xgboost/``.

The script deliberately stays linear and verbose so it doubles as an audit
trail: the printed summary + ``reports/xgboost/logs/ablation.csv`` fully
explains how each +Δ on the leaderboard was earned.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .train import StageConfig, run_cv, save_report, ablation_stages
from .postprocess import scan_threshold, apply_cryosleep_rule, diagnose_rule, make_submission
from .tune import run_optuna
from .ensemble import run_multi_seed
from . import evaluate
from . import data as data_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("xgboost.run_all")


def _log_cv_summary(tag: str, report) -> None:
    fold_df = pd.DataFrame(report.fold_scores)
    log.info(
        "[%s] OOF acc=%.4f logloss=%.4f auc=%.4f | fold acc mean=%.4f std=%.4f | mean best_iter=%.0f | elapsed=%.1fs",
        tag,
        report.oof_acc,
        report.oof_logloss,
        report.oof_auc,
        float(fold_df["acc"].mean()),
        float(fold_df["acc"].std(ddof=1)),
        float(pd.Series(report.best_iterations).mean()) if report.best_iterations else float("nan"),
        report.elapsed_seconds,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the full XGBoost ablation + tuning pipeline")
    parser.add_argument("--optuna-trials", type=int, default=config.OPTUNA_TRIALS,
                        help="Number of Optuna trials (default: %(default)s)")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Skip A4 (Optuna) and reuse STRONG_PARAMS for A5/A6")
    parser.add_argument("--load-best-params", action="store_true",
                        help="Load Optuna best params from logs/best_params.json instead of re-tuning")
    parser.add_argument("--skip-ensemble", action="store_true",
                        help="Skip A6 (multi-seed bagging)")
    parser.add_argument("--stages", nargs="*", default=None,
                        help="Optional subset of stage names to run (A0 A1 A2 A3 A4 A5 A6)")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="Override seed pool for A6")
    args = parser.parse_args(argv)

    run_set = set(args.stages) if args.stages else None
    ablation_rows: list[dict] = []

    log.info("Config: %s", config.describe())
    common = data_module.load_common()
    X_train_preview, _, _, _ = data_module.build_xgb_features(common)
    schema = data_module.summarise_schema(X_train_preview)
    log.info("Feature schema: %s", json.dumps(schema, default=str))
    (config.LOGS_DIR / "feature_schema.json").write_text(json.dumps(schema, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------
    # A0 → A3 canonical stages
    # ------------------------------------------------------------------
    reports = {}
    for stage in ablation_stages():
        if run_set and stage.name not in run_set:
            continue
        log.info("Running stage %s (%s)", stage.name, stage.description)
        rep = run_cv(stage, verbose=False)
        reports[stage.name] = rep
        _log_cv_summary(stage.name, rep)
        save_report(rep, config.LOGS_DIR / stage.name, tag=stage.name)
        row = rep.summary_row()
        row["stage_description"] = stage.description
        ablation_rows.append(row)

    # ------------------------------------------------------------------
    # A4 : Optuna on top of A3
    # ------------------------------------------------------------------
    best_params = None
    if args.load_best_params and (run_set is None or "A4" in run_set):
        params_file = config.LOGS_DIR / "best_params.json"
        if params_file.exists():
            data = json.loads(params_file.read_text(encoding="utf-8"))
            best_params = data.get("best_params")
            log.info("Loaded best params from %s (OOF acc=%.4f)", params_file, data.get("best_value", float("nan")))
    if best_params is None and not args.skip_tuning and (run_set is None or "A4" in run_set):
        log.info("Running Optuna with %d trials (seed=%d) ...", args.optuna_trials, config.RANDOM_SEED)
        base_stage = next(s for s in ablation_stages() if s.name == "A3")
        study_result = run_optuna(base_stage, n_trials=args.optuna_trials)
        best_params = study_result["best_params"]
        log.info("Optuna best OOF acc=%.4f params=%s", study_result["best_value"], best_params)

    if best_params is not None and (run_set is None or "A4" in run_set):
        a4 = StageConfig(
            name="A4",
            group_aware_cv=True,
            fold_aware_surname=True,
            fold_aware_cabin_bin=True,
            target_encode_cols=config.TARGET_ENCODE_COLUMNS,
            use_common_features=True,
            params_override=best_params,
            use_early_stopping=True,
            description="A3 + Optuna-tuned parameters",
        )
        log.info("Running stage %s (%s)", a4.name, a4.description)
        rep = run_cv(a4, verbose=False)
        reports["A4"] = rep
        _log_cv_summary("A4", rep)
        save_report(rep, config.LOGS_DIR / "A4", tag="A4")
        row = rep.summary_row()
        row["stage_description"] = a4.description
        ablation_rows.append(row)
    else:
        a4 = None

    # ------------------------------------------------------------------
    # A5 : threshold scan + audit the CryoSleep rule (disabled by default)
    # ------------------------------------------------------------------
    best_single_key = "A4" if "A4" in reports else ("A3" if "A3" in reports else max(reports) if reports else None)
    if (run_set is None or "A5" in run_set) and best_single_key is not None:
        best_rep = reports[best_single_key]
        raw_train_full = common.train.reset_index(drop=True)
        raw_test_full = common.test.reset_index(drop=True)

        # Audit the naive rule on OOF so the report can quote exact numbers.
        rule_audit = diagnose_rule(
            best_rep.oof_proba, best_rep.oof_true, raw_train_full, boost=0.05, penalty=0.10,
        )
        log.info(
            "[A5][rule audit] flipped=%d correct=%d wrong=%d Δacc=%+0.4f → rule left DISABLED",
            rule_audit["flipped"], rule_audit["correct_flip"], rule_audit["wrong_flip"], rule_audit["delta_acc"],
        )
        (config.LOGS_DIR / "A5_rule_audit.json").write_text(json.dumps(rule_audit, indent=2), encoding="utf-8")

        # Pure threshold scan (no rule)
        thr = scan_threshold(best_rep.oof_true, best_rep.oof_proba)
        thr.scan.to_csv(config.LOGS_DIR / "A5_threshold_scan.csv", index=False)
        log.info("[A5] best_threshold=%.3f best_oof_acc=%.4f", thr.best_threshold, thr.best_accuracy)

        sub = make_submission(best_rep.test_proba, best_rep.passenger_id_test, thr.best_threshold)
        sub_path = config.SUBMISSIONS_DIR / f"submission_A5_{best_single_key}.csv"
        sub.to_csv(sub_path, index=False)
        ablation_rows.append(
            dict(
                stage="A5",
                n_folds=best_rep.n_folds,
                oof_acc=thr.best_accuracy,
                oof_logloss=best_rep.oof_logloss,
                oof_auc=best_rep.oof_auc,
                mean_best_iter=float(np.mean(best_rep.best_iterations)) if best_rep.best_iterations else float("nan"),
                elapsed_seconds=best_rep.elapsed_seconds,
                description=f"{best_single_key} + threshold scan (best={thr.best_threshold:.3f})",
                stage_description=(
                    f"{best_single_key} + threshold scan ({thr.best_threshold:.3f}); "
                    f"rule audit Δacc={rule_audit['delta_acc']:+0.4f} [disabled]"
                ),
            )
        )
        reports["A5"] = best_rep
        reports["A5_threshold"] = thr
    else:
        sub_path = None
        thr = None

    # ------------------------------------------------------------------
    # A7 : stacking-style blend of A4_plain + A4_OOFte
    # (discovered via tmp/blend_a4_variants.py: +0.0009 OOF on top of A4)
    # ------------------------------------------------------------------
    if best_params is not None and (run_set is None or "A7" in run_set):
        log.info("Running A7 (blend A4_plain + A4_OOFte) ...")
        a7_plain = StageConfig(
            name="A7_plain",
            group_aware_cv=True,
            fold_aware_surname=True,
            fold_aware_cabin_bin=True,
            target_encode_cols=config.TARGET_ENCODE_COLUMNS,
            use_oof_target_encoding=False,
            use_common_features=True,
            params_override=best_params,
            use_early_stopping=True,
            description="A7 member: plain target encoding (tuned)",
        )
        a7_ooft = StageConfig(
            name="A7_OOFte",
            group_aware_cv=True,
            fold_aware_surname=True,
            fold_aware_cabin_bin=True,
            target_encode_cols=config.TARGET_ENCODE_COLUMNS,
            use_oof_target_encoding=True,
            use_common_features=True,
            params_override=best_params,
            use_early_stopping=True,
            description="A7 member: OOF target encoding (tuned)",
        )
        rep_plain = run_cv(a7_plain, verbose=False) if "A4" not in reports else reports["A4"]
        rep_oof = run_cv(a7_ooft, verbose=False)
        _log_cv_summary("A7_plain", rep_plain)
        _log_cv_summary("A7_OOFte", rep_oof)

        # Choose blend weight by argmax OOF acc over a small grid
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

        best_blend = None
        for w in np.arange(0.2, 0.81, 0.05):
            blend = w * rep_plain.oof_proba + (1 - w) * rep_oof.oof_proba
            scan = scan_threshold(rep_plain.oof_true, blend)
            if best_blend is None or scan.best_accuracy > best_blend[1]:
                best_blend = (float(w), scan.best_accuracy, scan.best_threshold, blend)
        w_best, acc_best, thr_best, blend_oof = best_blend
        blend_test = w_best * rep_plain.test_proba + (1 - w_best) * rep_oof.test_proba
        ll_blend = log_loss(rep_plain.oof_true, blend_oof.clip(1e-6, 1 - 1e-6))
        auc_blend = roc_auc_score(rep_plain.oof_true, blend_oof)
        log.info(
            "[A7] w=%.2f threshold=%.3f OOF acc=%.4f logloss=%.4f auc=%.4f",
            w_best, thr_best, acc_best, ll_blend, auc_blend,
        )
        sub = make_submission(blend_test, rep_plain.passenger_id_test, thr_best)
        sub_path_blend = config.SUBMISSIONS_DIR / "submission_A7_blend.csv"
        sub.to_csv(sub_path_blend, index=False)
        pd.DataFrame({
            "y_true": rep_plain.oof_true,
            "y_proba_plain": rep_plain.oof_proba,
            "y_proba_ooft": rep_oof.oof_proba,
            "y_proba_blend": blend_oof,
        }).to_csv(config.LOGS_DIR / "A7_oof.csv", index=False)
        pd.DataFrame({
            "PassengerId": rep_plain.passenger_id_test,
            "y_proba_plain": rep_plain.test_proba,
            "y_proba_ooft": rep_oof.test_proba,
            "y_proba_blend": blend_test,
        }).to_csv(config.LOGS_DIR / "A7_test_proba.csv", index=False)
        ablation_rows.append(
            dict(
                stage="A7",
                n_folds=rep_plain.n_folds,
                oof_acc=acc_best,
                oof_logloss=ll_blend,
                oof_auc=auc_blend,
                mean_best_iter=float(np.mean(rep_plain.best_iterations + rep_oof.best_iterations)),
                elapsed_seconds=float(rep_plain.elapsed_seconds + rep_oof.elapsed_seconds),
                description=f"Blend(plain×{w_best:.2f} + OOFte×{1 - w_best:.2f}) @ t={thr_best:.3f}",
                stage_description=(
                    f"Blend(plain×{w_best:.2f} + OOFte×{1-w_best:.2f}) @ t={thr_best:.3f} — "
                    f"plain OOF={rep_plain.oof_acc:.4f} OOFte OOF={rep_oof.oof_acc:.4f}"
                ),
            )
        )
        reports["A7"] = dict(
            rep_plain=rep_plain,
            rep_oof=rep_oof,
            w=w_best,
            threshold=thr_best,
            blend_oof=blend_oof,
            blend_test=blend_test,
            oof_acc=acc_best,
        )

    # ------------------------------------------------------------------
    # A6 : multi-seed bagging on the tuned stage
    # ------------------------------------------------------------------
    if not args.skip_ensemble and (run_set is None or "A6" in run_set):
        log.info("Running A6 (multi-seed bagging) ...")
        base_stage = StageConfig(
            name="A6",
            group_aware_cv=True,
            fold_aware_surname=True,
            fold_aware_cabin_bin=True,
            target_encode_cols=config.TARGET_ENCODE_COLUMNS,
            use_common_features=True,
            params_override=best_params or {},
            use_early_stopping=True,
            description="A4 × multi-seed bagging",
        )
        ens = run_multi_seed(base_stage, seeds=args.seeds)
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

        per_seed_oof_acc = [
            float(accuracy_score(m.oof_true, (m.oof_proba >= 0.5).astype(int))) for m in ens.members
        ]
        log.info("[A6] per-seed OOF acc: %s", [round(a, 4) for a in per_seed_oof_acc])

        thr_ens = scan_threshold(ens.oof_true, ens.oof_proba)
        ens_acc = thr_ens.best_accuracy
        ens_ll = log_loss(ens.oof_true, ens.oof_proba.clip(1e-6, 1 - 1e-6))
        ens_auc = roc_auc_score(ens.oof_true, ens.oof_proba)
        log.info(
            "[A6] seeds=%s ensemble OOF acc=%.4f logloss=%.4f auc=%.4f threshold=%.3f",
            ens.seeds_used, ens_acc, ens_ll, ens_auc, thr_ens.best_threshold,
        )
        sub = make_submission(ens.test_proba, ens.passenger_id_test, thr_ens.best_threshold)
        sub_path_ens = config.SUBMISSIONS_DIR / "submission_A6_ensemble.csv"
        sub.to_csv(sub_path_ens, index=False)
        ablation_rows.append(
            dict(
                stage="A6",
                n_folds=ens.members[0].n_folds,
                oof_acc=ens_acc,
                oof_logloss=ens_ll,
                oof_auc=ens_auc,
                mean_best_iter=float(np.mean(ens.members[0].best_iterations)) if ens.members[0].best_iterations else float("nan"),
                elapsed_seconds=float(sum(m.elapsed_seconds for m in ens.members)),
                description=f"{len(ens.seeds_used)}-seed bagging + threshold={thr_ens.best_threshold:.3f}",
                stage_description=(
                    f"{len(ens.seeds_used)}-seed bagging (per-seed OOF acc min/max = "
                    f"{min(per_seed_oof_acc):.4f}/{max(per_seed_oof_acc):.4f}) + threshold={thr_ens.best_threshold:.3f}"
                ),
            )
        )
        pd.DataFrame({"y_true": ens.oof_true, "y_proba": ens.oof_proba}).to_csv(
            config.LOGS_DIR / "A6_oof.csv", index=False
        )
        pd.DataFrame(
            {"PassengerId": ens.passenger_id_test, "y_proba": ens.test_proba}
        ).to_csv(config.LOGS_DIR / "A6_test_proba.csv", index=False)
        (config.LOGS_DIR / "A6_per_seed_oof.json").write_text(
            json.dumps(dict(zip([int(s) for s in ens.seeds_used], per_seed_oof_acc)), indent=2),
            encoding="utf-8",
        )
        reports["A6"] = ens

    # ------------------------------------------------------------------
    # Final artefacts: ablation table + plots
    # ------------------------------------------------------------------
    ablation_df = pd.DataFrame(ablation_rows)
    # Annotate Δ vs A1 (the honest baseline)
    baseline_row = ablation_df.loc[ablation_df["stage"] == "A1"]
    if not baseline_row.empty:
        a1_acc = float(baseline_row["oof_acc"].iloc[0])
        ablation_df["delta_vs_A1"] = (ablation_df["oof_acc"] - a1_acc).round(4)
        ablation_df["delta_vs_A0"] = (
            ablation_df["oof_acc"] - float(ablation_df.loc[ablation_df["stage"] == "A0", "oof_acc"].iloc[0])
        ).round(4) if (ablation_df["stage"] == "A0").any() else float("nan")
    ablation_df.to_csv(config.LOGS_DIR / "ablation.csv", index=False)
    log.info("Ablation table:\n%s", ablation_df.to_string(index=False))

    # Canonical "best" submission (the model we trust the most)
    if "A7" in reports:
        a7 = reports["A7"]
        best_sub = make_submission(a7["blend_test"], a7["rep_plain"].passenger_id_test, a7["threshold"])
        (config.SUBMISSIONS_DIR / "submission_best.csv").write_text(
            best_sub.to_csv(index=False), encoding="utf-8"
        )
        log.info(
            "submission_best.csv = A7 blend (w=%.2f, threshold=%.3f, OOF=%.4f)",
            a7["w"], a7["threshold"], a7["oof_acc"],
        )
    else:
        best_stage_name = (
            "A4" if "A4" in reports else ("A3" if "A3" in reports else best_single_key or next(iter(reports), None))
        )
        if best_stage_name and best_stage_name in reports and hasattr(reports[best_stage_name], "passenger_id_test"):
            best_rep = reports[best_stage_name]
            best_threshold = reports["A5_threshold"].best_threshold if "A5_threshold" in reports else 0.5
            best_sub = make_submission(best_rep.test_proba, best_rep.passenger_id_test, best_threshold)
            (config.SUBMISSIONS_DIR / "submission_best.csv").write_text(
                best_sub.to_csv(index=False), encoding="utf-8"
            )
            log.info(
                "submission_best.csv (stage=%s, threshold=%.3f)",
                best_stage_name, best_threshold,
            )

    if "A3" in reports or "A4" in reports:
        chosen_key = "A4" if "A4" in reports else "A3"
        chosen = reports[chosen_key]
        evaluate.plot_roc_pr(chosen.oof_true, chosen.oof_proba, tag=chosen_key)
        evaluate.plot_confusion_matrix(chosen.oof_true, chosen.oof_proba, threshold=0.5, tag=chosen_key)
        evaluate.plot_feature_importance(chosen.feature_names, chosen.importance_gain, tag=chosen_key, importance_type="gain")
        evaluate.plot_feature_importance(chosen.feature_names, chosen.importance_weight, tag=chosen_key, importance_type="weight")
        evaluate.plot_learning_curve(chosen.fold_scores, tag=chosen_key)
        if "A5_threshold" in reports:
            evaluate.plot_threshold_scan(reports["A5_threshold"].scan, reports["A5_threshold"].best_threshold, tag="A5")
            evaluate.plot_confusion_matrix(chosen.oof_true, chosen.oof_proba, threshold=reports["A5_threshold"].best_threshold, tag="A5")

        # SHAP explainability on the tuned model (fold 0)
        try:
            params_for_shap = dict(best_params or config.STRONG_PARAMS)
            shap_path = evaluate.plot_shap_summary(
                params=params_for_shap,
                tag=chosen_key,
                target_encode_cols=config.TARGET_ENCODE_COLUMNS,
            )
            if shap_path:
                log.info("Saved SHAP summary to %s", shap_path)
        except Exception as ex:  # pragma: no cover - plotting helper
            log.warning("SHAP plot skipped: %s", ex)

    evaluate.plot_ablation(ablation_df)
    log.info("Done. Reports at %s", config.REPORTS_DIR)


if __name__ == "__main__":
    main()
