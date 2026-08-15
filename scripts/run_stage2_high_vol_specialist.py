from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from app.training.stage2_conditioned_target_research import target_specs
from app.training.stage2_high_vol_specialist_research import (
    Stage2HighVolSpecialistResearch,
)
from database.stage2_signal_data_repository import Stage2SignalDataRepository
from scripts.run_stage2_conditioned_megasearch import (
    build_master,
    columns_for_groups,
    dataset,
    load_training_cutoff,
)


TICKER = "SPY"
TARGET_WINDOW = 90
TARGET_MULTIPLIER = 0.700
OUTER_SPLITS = 3
INNER_SPLITS = 3
RANDOM_STATE = 42
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_BLOCK_LENGTH = 20
REGIME_QUANTILE = 2.0 / 3.0
EXPECTED_REGIME_FEATURE = "realized_volatility_20"
FALLBACK_WINNER_GROUPS = ("breadth", "calendar", "interaction_consensus")
EXPERIMENT_DIRECTORY = Path("experiments")
TREE_PROGRESS_PATH = EXPERIMENT_DIRECTORY / "stage2_return_architecture_tree_v1_progress.json"
TREE_STORAGE_URL = "sqlite:///experiments/optuna_stage2_return_architecture_tree_v1.db"
DEVELOPMENT_SCREEN_PREFIX = "stage2_regime_development_screen_v1_"
EXPERIMENT_NAME = "stage2_high_vol_specialist_v1"


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def latest_verified_winner_groups() -> tuple[str, ...]:
    paths = sorted(
        EXPERIMENT_DIRECTORY.glob("stage2_90d_k700_verified_v1_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        return FALLBACK_WINNER_GROUPS
    frame = pd.read_csv(paths[0])
    if frame.empty or "candidate_name" not in frame.columns:
        return FALLBACK_WINNER_GROUPS
    name = str(frame.iloc[0]["candidate_name"])
    if not name or name == "base_only":
        return FALLBACK_WINNER_GROUPS
    return tuple(sorted(part for part in name.split("+") if part))


def load_nested_binary_predictions() -> dict[int, dict]:
    if not TREE_PROGRESS_PATH.exists():
        raise FileNotFoundError(
            f"{TREE_PROGRESS_PATH} is required. Run the completed Stage-2 return "
            "architecture tree research first."
        )
    with TREE_PROGRESS_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    rows = [
        row
        for row in payload.get("rows", [])
        if row.get("architecture") == "xgboost_binary_winner"
    ]
    by_fold = {int(row["outer_fold"]): row for row in rows}
    missing = sorted(set(range(1, OUTER_SPLITS + 1)) - set(by_fold))
    if missing:
        raise RuntimeError(
            "Nested XGBoost binary predictions are incomplete for folds: "
            + ", ".join(map(str, missing))
        )
    return by_fold


def primary_target_spec():
    matches = [
        spec
        for spec in target_specs()
        if spec.volatility_window == TARGET_WINDOW
        and abs(spec.threshold_multiplier - TARGET_MULTIPLIER) < 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeError("Could not resolve the locked 90d x 0.700 target spec.")
    return matches[0]


def latest_supported_regime_candidate() -> dict:
    candidates: list[tuple[float, Path, pd.DataFrame]] = []
    for path in EXPERIMENT_DIRECTORY.glob(f"{DEVELOPMENT_SCREEN_PREFIX}*.csv"):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        required = {
            "feature",
            "development_regime_candidate",
            "best_regime",
            "best_regime_auc",
            "best_regime_auc_lower_95",
        }
        if not required.issubset(frame.columns):
            continue
        candidates.append((path.stat().st_mtime, path, frame))
    if not candidates:
        raise FileNotFoundError(
            "A completed stage2_regime_development_screen_v1 summary CSV is required."
        )
    _, path, frame = max(candidates, key=lambda item: item[0])
    flag = frame["development_regime_candidate"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    supported = frame.loc[flag].copy()
    if len(supported) != 1:
        raise RuntimeError(
            "This first specialist experiment requires exactly one development-supported "
            f"regime candidate; found {len(supported)} in {path}."
        )
    row = supported.iloc[0].to_dict()
    if str(row["feature"]) != EXPECTED_REGIME_FEATURE:
        raise RuntimeError(
            "The supported regime candidate changed. Review the development screen before "
            "running a different specialist architecture."
        )
    if str(row["best_regime"]).upper() != "HIGH":
        raise RuntimeError(
            "The supported realized-volatility regime is no longer HIGH. Review before "
            "running this specialist architecture."
        )
    return {
        "source_path": path,
        "feature": str(row["feature"]),
        "best_regime": str(row["best_regime"]).upper(),
        "best_regime_auc": float(row["best_regime_auc"]),
        "best_regime_auc_lower_95": float(row["best_regime_auc_lower_95"]),
    }


def load_fold_parameters(outer_fold: int) -> dict:
    study_name = f"xgboost_binary_winner_outer_{outer_fold}"
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=TREE_STORAGE_URL,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the existing development-only Optuna study {study_name!r}. "
            "No new tuning is allowed in this experiment."
        ) from exc
    if study.best_trial is None:
        raise RuntimeError(f"Optuna study {study_name!r} has no best trial.")
    return {
        "study_name": study_name,
        "best_trial": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "parameters": dict(study.best_trial.params),
    }


def build_development_data(
    regime_feature: str,
) -> tuple[pd.DataFrame, tuple[str, ...], list[str], pd.Timestamp]:
    cutoff = load_training_cutoff()
    raw = Stage2SignalDataRepository().get_training_data(ticker=TICKER)
    master = build_master(raw, primary_target_spec(), cutoff)
    winner_groups = latest_verified_winner_groups()
    winner_features = columns_for_groups(winner_groups)
    if regime_feature not in winner_features:
        raise RuntimeError(
            f"{regime_feature} is not part of the locked Stage-2 winner feature set."
        )
    data = dataset(master, winner_features)
    return (
        data.sort_values("target_date").reset_index(drop=True),
        winner_groups,
        winner_features,
        cutoff,
    )


def print_fold_results(frame: pd.DataFrame) -> None:
    print()
    print("NESTED DEVELOPMENT HIGH-VOL SPECIALIST RESULTS")
    columns = [
        "outer_fold",
        "training_high_rows",
        "test_high_rows",
        "test_high_coverage",
        "universal_auc",
        "specialist_auc",
        "delta_auc",
        "universal_balanced_accuracy",
        "specialist_balanced_accuracy",
        "universal_macro_f1",
        "specialist_macro_f1",
    ]
    print(frame[columns].round(4).to_string(index=False))


def print_summary(summary: dict) -> None:
    print()
    print("POOLED HIGH-VOLATILITY DEVELOPMENT OOF")
    print(f"High-regime MOVE rows: {summary['high_regime_rows']}")
    print(f"UP share: {summary['high_regime_up_share']:.4f}")
    print(f"Universal XGBoost AUC: {summary['universal_high_regime_auc']:.4f}")
    print(f"High-vol specialist AUC: {summary['specialist_high_regime_auc']:.4f}")
    print(f"Delta AUC: {summary['delta_auc']:+.4f}")
    print(
        "Specialist bootstrap 95% AUC CI: "
        f"[{summary['specialist_auc_lower_95']:.4f}, "
        f"{summary['specialist_auc_upper_95']:.4f}]"
    )
    print(
        "Delta bootstrap 95% CI: "
        f"[{summary['delta_auc_lower_95']:+.4f}, "
        f"{summary['delta_auc_upper_95']:+.4f}]"
    )
    print(
        "Probability delta > 0: "
        f"{summary['probability_delta_positive'] * 100.0:.2f}%"
    )
    print(
        "Fold delta AUC mean / std: "
        f"{summary['fold_delta_auc_mean']:+.4f} / "
        f"{summary['fold_delta_auc_std']:.4f}"
    )
    print(
        "Folds improved: "
        f"{summary['folds_improved']}/{summary['fold_count']}"
    )
    print()
    print("PREDEFINED ARCHITECTURE GATES")
    for gate, passed in summary["gates"].items():
        print(f"- {gate}: {'PASS' if passed else 'FAIL'}")
    print(
        "OVERALL HIGH-VOL SPECIALIST GATE: "
        f"{'PASS' if summary['architecture_gate_pass'] else 'FAIL'}"
    )


def main():
    EXPERIMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    candidate = latest_supported_regime_candidate()
    data, winner_groups, winner_features, cutoff = build_development_data(
        candidate["feature"]
    )
    saved_by_fold = load_nested_binary_predictions()
    research = Stage2HighVolSpecialistResearch(
        feature_columns=winner_features,
        regime_feature=candidate["feature"],
        regime_quantile=REGIME_QUANTILE,
        inner_splits=INNER_SPLITS,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_block_length=BOOTSTRAP_BLOCK_LENGTH,
        random_state=RANDOM_STATE,
    )

    print("=" * 78)
    print("STAGE-2 HIGH-VOLATILITY SPECIALIST RESEARCH V1")
    print("=" * 78)
    print(f"Target: {TARGET_WINDOW}d x {TARGET_MULTIPLIER:.3f}")
    print("Regime feature:", candidate["feature"])
    print("Regime definition: HIGH = above training-only 66.7th percentile")
    print("Development-screen source:", candidate["source_path"])
    print(
        "Screen-supported HIGH AUC / lower 95%: "
        f"{candidate['best_regime_auc']:.4f} / "
        f"{candidate['best_regime_auc_lower_95']:.4f}"
    )
    print("Winner feature groups:", list(winner_groups))
    print("Winner model features:", len(winner_features))
    print("Development cutoff:", cutoff.date())
    print()
    print(
        "DEVELOPMENT ONLY: this experiment tests one pre-specified regime architecture. "
        "It does not search new features or tune new hyperparameters."
    )
    print(
        "Each specialist reuses the XGBoost hyperparameters already selected by that "
        "development outer fold's original Optuna study."
    )
    print(
        "The regime threshold is calculated from each outer fold's earlier training "
        "MOVE rows only. Outer validation and the final held-out test are NOT loaded."
    )

    splitter = TimeSeriesSplit(n_splits=OUTER_SPLITS)
    fold_rows: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []
    threshold_oof_rows: list[dict] = []
    parameter_rows: list[dict] = []

    for outer_fold, (train_index, test_index) in enumerate(
        splitter.split(data), start=1
    ):
        print()
        print(f"outer fold {outer_fold}/{OUTER_SPLITS}")
        outer_train = data.iloc[train_index].reset_index(drop=True)
        outer_test = data.iloc[test_index].reset_index(drop=True)
        parameter_source = load_fold_parameters(outer_fold)
        result, predictions, threshold_rows = research.evaluate_fold(
            outer_fold=outer_fold,
            outer_train=outer_train,
            outer_test=outer_test,
            universal_saved=saved_by_fold[outer_fold],
            parameters=parameter_source["parameters"],
        )
        fold_rows.append(result)
        prediction_parts.append(predictions)
        threshold_oof_rows.extend(threshold_rows)
        parameter_rows.append(
            {
                "outer_fold": outer_fold,
                "study_name": parameter_source["study_name"],
                "best_trial": parameter_source["best_trial"],
                "best_value": parameter_source["best_value"],
                "parameters_json": json.dumps(parameter_source["parameters"], sort_keys=True),
            }
        )
        print(
            f"HIGH rows {result['test_high_rows']} | universal AUC "
            f"{result['universal_auc']:.4f} | specialist AUC "
            f"{result['specialist_auc']:.4f} | delta "
            f"{result['delta_auc']:+.4f}"
        )

    fold_frame = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        "target_date"
    ).reset_index(drop=True)
    threshold_frame = pd.DataFrame(threshold_oof_rows)
    parameter_frame = pd.DataFrame(parameter_rows)
    summary = research.summarize(predictions, fold_frame)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    paths = {
        "fold_results": EXPERIMENT_DIRECTORY / f"{EXPERIMENT_NAME}_fold_results_{timestamp}.csv",
        "predictions": EXPERIMENT_DIRECTORY / f"{EXPERIMENT_NAME}_predictions_{timestamp}.csv",
        "threshold_oof": EXPERIMENT_DIRECTORY / f"{EXPERIMENT_NAME}_threshold_oof_{timestamp}.csv",
        "parameters": EXPERIMENT_DIRECTORY / f"{EXPERIMENT_NAME}_parameters_{timestamp}.csv",
        "experiment": EXPERIMENT_DIRECTORY / f"{EXPERIMENT_NAME}_{timestamp}.json",
    }
    fold_frame.to_csv(paths["fold_results"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    threshold_frame.to_csv(paths["threshold_oof"], index=False)
    parameter_frame.to_csv(paths["parameters"], index=False)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": EXPERIMENT_NAME,
        "target": {
            "volatility_window": TARGET_WINDOW,
            "threshold_multiplier": TARGET_MULTIPLIER,
        },
        "development_cutoff": cutoff,
        "regime": {
            "feature": candidate["feature"],
            "state": candidate["best_regime"],
            "quantile": REGIME_QUANTILE,
            "development_screen_source": candidate["source_path"],
            "development_screen_auc": candidate["best_regime_auc"],
            "development_screen_auc_lower_95": candidate[
                "best_regime_auc_lower_95"
            ],
        },
        "winner_groups": list(winner_groups),
        "winner_feature_count": len(winner_features),
        "model_policy": {
            "universal_control": "Existing saved nested OOF XGBoost binary predictions",
            "specialist_training_rows": (
                "MOVE rows above the outer-fold training-only 66.7th percentile of "
                "realized_volatility_20"
            ),
            "specialist_parameters": (
                "Reuse the original development outer-fold XGBoost binary Optuna winner; "
                "no new hyperparameter optimization"
            ),
            "specialist_probability_threshold": (
                "Selected from inner walk-forward HIGH-regime OOF predictions using "
                "inner-training-only regime thresholds"
            ),
        },
        "statistics": {
            "outer_splits": OUTER_SPLITS,
            "inner_splits": INNER_SPLITS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_block_length": BOOTSTRAP_BLOCK_LENGTH,
            "architecture_gate": (
                "Specialist pooled HIGH-regime AUC > universal; specialist moving-block "
                "bootstrap lower 95% AUC > 0.50; paired delta moving-block bootstrap lower "
                "95% > 0; specialist improves at least 2 of 3 outer folds"
            ),
        },
        "summary": summary,
        "methodology": {
            "new_feature_search": False,
            "new_hyperparameter_search": False,
            "outer_validation_loaded": False,
            "held_out_test_loaded": False,
            "purpose": (
                "Determine whether the development-supported HIGH realized-volatility "
                "state warrants a separately trained Stage-2 XGBoost specialist before "
                "building a regime router."
            ),
        },
        "outputs": paths,
    }
    with paths["experiment"].open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=json_default)

    print_fold_results(fold_frame)
    print_summary(summary)
    print()
    print("=" * 78)
    print("STAGE-2 HIGH-VOLATILITY SPECIALIST RESEARCH COMPLETE")
    print("=" * 78)
    print("Fold results:", paths["fold_results"])
    print("Predictions:", paths["predictions"])
    print("Threshold OOF diagnostics:", paths["threshold_oof"])
    print("Parameter sources:", paths["parameters"])
    print("Experiment:", paths["experiment"])
    print("Outer validation was NOT evaluated.")
    print("Held-out test set was NOT evaluated.")
    print()
    if summary["architecture_gate_pass"]:
        print(
            "NEXT DECISION RULE: PASS. A high-volatility specialist has development-only "
            "evidence of incremental value. The next experiment may build a calibrated "
            "universal/specialist regime router, still inside development data."
        )
    else:
        print(
            "NEXT DECISION RULE: FAIL. Do not build a high-volatility specialist router. "
            "The regime predicts when the universal model works, but separate specialist "
            "training did not demonstrate stable incremental value."
        )


if __name__ == "__main__":
    main()
