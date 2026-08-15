from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from app.training.stage2_conditioned_target_research import target_specs
from app.training.stage2_selective_prediction_research import (
    Stage2SelectivePredictionResearch,
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
REGIME_FEATURE = "realized_volatility_20"
REGIME_STATE = "HIGH"
REGIME_QUANTILE = 2.0 / 3.0
OUTER_SPLITS = 3
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_BLOCK_LENGTH = 20
RANDOM_STATE = 42
FALLBACK_WINNER_GROUPS = ("breadth", "calendar", "interaction_consensus")
EXPERIMENT_DIRECTORY = Path("experiments")
TREE_PROGRESS_PATH = EXPERIMENT_DIRECTORY / "stage2_return_architecture_tree_v1_progress.json"
EXPERIMENT_NAME = "stage2_selective_prediction_gate_v1"


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
            "architecture tree research before this experiment."
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
            "Nested XGBoost binary development predictions are incomplete for folds: "
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


def build_development_data() -> tuple[pd.DataFrame, tuple[str, ...], list[str], pd.Timestamp]:
    cutoff = load_training_cutoff()
    raw = Stage2SignalDataRepository().get_training_data(ticker=TICKER)
    master = build_master(raw, primary_target_spec(), cutoff)
    winner_groups = latest_verified_winner_groups()
    winner_features = columns_for_groups(winner_groups)
    if REGIME_FEATURE not in winner_features:
        raise RuntimeError(
            f"{REGIME_FEATURE} is not part of the locked Stage-2 winner feature set."
        )
    data = dataset(master, winner_features)
    return (
        data.sort_values("target_date").reset_index(drop=True),
        winner_groups,
        winner_features,
        cutoff,
    )


def reconstruct_selective_oof(
    data: pd.DataFrame,
    saved_by_fold: dict[int, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = TimeSeriesSplit(n_splits=OUTER_SPLITS)
    prediction_parts: list[pd.DataFrame] = []
    threshold_rows: list[dict] = []

    for outer_fold, (train_index, test_index) in enumerate(
        splitter.split(data),
        start=1,
    ):
        outer_train = data.iloc[train_index].reset_index(drop=True)
        outer_test = data.iloc[test_index].reset_index(drop=True)
        train_move = outer_train.loc[
            outer_train["direction"].astype(str) != "FLAT"
        ].reset_index(drop=True)
        test_move = outer_test.loc[
            outer_test["direction"].astype(str) != "FLAT"
        ].reset_index(drop=True)
        saved = saved_by_fold[outer_fold]

        saved_dates = pd.DatetimeIndex(pd.to_datetime(saved["target_dates"]))
        expected_dates = pd.DatetimeIndex(pd.to_datetime(test_move["target_date"]))
        if not saved_dates.equals(expected_dates):
            raise ValueError(
                f"Fold {outer_fold} saved nested predictions no longer align with "
                "the reconstructed development split."
            )

        actual = np.asarray(saved["actual"], dtype=np.int64)
        expected_actual = (
            test_move["direction"].astype(str) == "UP"
        ).astype(int).to_numpy()
        if not np.array_equal(actual, expected_actual):
            raise ValueError(
                f"Fold {outer_fold} saved actual labels do not match the locked target."
            )

        training_regime_values = pd.to_numeric(
            train_move[REGIME_FEATURE],
            errors="coerce",
        )
        training_regime_values = training_regime_values[
            np.isfinite(training_regime_values)
        ]
        if training_regime_values.empty:
            raise ValueError(
                f"Fold {outer_fold} has no finite training values for {REGIME_FEATURE}."
            )
        q33 = float(training_regime_values.quantile(1.0 / 3.0))
        q67 = float(training_regime_values.quantile(REGIME_QUANTILE))

        test_values = pd.to_numeric(
            test_move[REGIME_FEATURE],
            errors="coerce",
        )
        regime = pd.Series("UNKNOWN", index=test_move.index, dtype="object")
        valid = test_values.notna() & np.isfinite(test_values)
        regime.loc[valid & (test_values < q33)] = "LOW"
        regime.loc[valid & (test_values >= q33) & (test_values <= q67)] = "MID"
        regime.loc[valid & (test_values > q67)] = "HIGH"

        scores = np.asarray(saved["score"], dtype=np.float64)
        decision_threshold = float(saved["decision_threshold"])
        predictions = test_move[
            [
                "feature_date",
                "target_date",
                "future_log_return",
                "direction",
                REGIME_FEATURE,
            ]
        ].copy()
        predictions["outer_fold"] = int(outer_fold)
        predictions["actual_up"] = actual
        predictions["score"] = scores
        predictions["saved_decision_threshold"] = decision_threshold
        predictions["predicted_up"] = (scores >= decision_threshold).astype(int)
        predictions["regime"] = regime.to_numpy()

        threshold_rows.append(
            {
                "outer_fold": int(outer_fold),
                "training_move_rows": int(len(train_move)),
                "test_move_rows": int(len(test_move)),
                "q33": q33,
                "q67": q67,
                "test_high_rows": int((regime == "HIGH").sum()),
                "test_high_coverage": float((regime == "HIGH").mean()),
                "train_start": pd.Timestamp(train_move["target_date"].min()),
                "train_end": pd.Timestamp(train_move["target_date"].max()),
                "test_start": pd.Timestamp(test_move["target_date"].min()),
                "test_end": pd.Timestamp(test_move["target_date"].max()),
            }
        )
        prediction_parts.append(predictions)

    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    ).sort_values("target_date").reset_index(drop=True)
    if predictions["regime"].eq("UNKNOWN").any():
        unknown = int(predictions["regime"].eq("UNKNOWN").sum())
        raise ValueError(
            f"{unknown} OOF rows could not be assigned to a realized-volatility regime."
        )
    return predictions, pd.DataFrame(threshold_rows)


def print_results(fold_frame: pd.DataFrame, summary: dict) -> None:
    print()
    print("NESTED DEVELOPMENT SELECTIVE-PREDICTION RESULTS")
    print(
        fold_frame[
            [
                "outer_fold",
                "move_rows",
                "accepted_rows",
                "accepted_coverage",
                "full_auc",
                "accepted_auc",
                "abstained_auc",
                "accepted_balanced_accuracy",
                "accepted_macro_f1",
                "accepted_magnitude_weighted_sign_accuracy",
            ]
        ].round(4).to_string(index=False)
    )

    print()
    print("POOLED DEVELOPMENT SELECTIVE POLICY")
    print(f"MOVE rows: {summary['move_rows']}")
    print(f"Accepted HIGH-vol rows: {summary['accepted_rows']}")
    print(f"Abstained LOW/MID-vol rows: {summary['abstained_rows']}")
    print(f"Directional prediction coverage: {summary['accepted_coverage']:.2%}")
    print(f"Universal full-coverage AUC: {summary['full_auc']:.4f}")
    print(f"Accepted HIGH-vol AUC: {summary['accepted_auc']:.4f}")
    print(
        "Accepted moving-block bootstrap 95% AUC CI: "
        f"[{summary['accepted_auc_lower_95']:.4f}, "
        f"{summary['accepted_auc_upper_95']:.4f}]"
    )
    print(
        "Probability accepted AUC > 0.50: "
        f"{summary['accepted_probability_auc_above_0_50']:.2%}"
    )
    print(f"Abstained LOW/MID-vol AUC: {summary['abstained_auc']:.4f}")
    print(
        "Abstained moving-block bootstrap 95% AUC CI: "
        f"[{summary['abstained_auc_lower_95']:.4f}, "
        f"{summary['abstained_auc_upper_95']:.4f}]"
    )
    print(
        "Accepted AUC lift vs full coverage: "
        f"{summary['accepted_auc_lift_vs_full']:+.4f}"
    )
    print(
        "Accepted balanced accuracy / macro F1: "
        f"{summary['accepted_balanced_accuracy']:.4f} / "
        f"{summary['accepted_macro_f1']:.4f}"
    )
    print(
        "Accepted magnitude-weighted sign accuracy: "
        f"{summary['accepted_magnitude_weighted_sign_accuracy']:.4f}"
    )

    print()
    print("PREDEFINED SELECTIVE-PREDICTION GATES")
    for name, passed in summary["gates"].items():
        if name == "overall_selective_prediction_gate":
            continue
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(
        "  OVERALL SELECTIVE-PREDICTION GATE: "
        f"{'PASS' if summary['gates']['overall_selective_prediction_gate'] else 'FAIL'}"
    )


def main():
    EXPERIMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    saved_by_fold = load_nested_binary_predictions()
    data, winner_groups, winner_features, cutoff = build_development_data()
    predictions, thresholds = reconstruct_selective_oof(
        data=data,
        saved_by_fold=saved_by_fold,
    )

    research = Stage2SelectivePredictionResearch(
        accepted_regime=REGIME_STATE,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_block_length=BOOTSTRAP_BLOCK_LENGTH,
        random_state=RANDOM_STATE,
    )
    fold_frame, summary = research.evaluate(predictions)

    print("=" * 88)
    print("STAGE-2 SELECTIVE PREDICTION GATE V1")
    print("=" * 88)
    print(f"Target: {TARGET_WINDOW}d x {TARGET_MULTIPLIER:.3f}")
    print(f"Regime feature: {REGIME_FEATURE}")
    print(
        "Policy: universal XGBoost predicts UP/DOWN only when "
        f"{REGIME_FEATURE} is above the outer-fold training-only "
        "66.7th percentile; otherwise ABSTAIN."
    )
    print(f"Winner feature groups: {list(winner_groups)}")
    print(f"Winner model features: {len(winner_features)}")
    print(f"Development cutoff: {cutoff.date()}")
    print()
    print(
        "DEVELOPMENT ONLY: no model is trained or tuned. The universal XGBoost "
        "predictions are the existing nested OOF predictions."
    )
    print(
        "The realized-volatility threshold is recalculated from each outer fold's "
        "earlier training MOVE rows only."
    )
    print(
        "This experiment does NOT claim LOW/MID regimes are harmful; it tests whether "
        "UP/DOWN prediction is sufficiently supported in the pre-specified HIGH regime."
    )
    print(
        "Outer validation and the final held-out test are NOT loaded or evaluated."
    )

    print_results(fold_frame, summary)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    paths = {
        "fold_results": (
            EXPERIMENT_DIRECTORY
            / f"{EXPERIMENT_NAME}_fold_results_{timestamp}.csv"
        ),
        "predictions": (
            EXPERIMENT_DIRECTORY
            / f"{EXPERIMENT_NAME}_predictions_{timestamp}.csv"
        ),
        "thresholds": (
            EXPERIMENT_DIRECTORY
            / f"{EXPERIMENT_NAME}_thresholds_{timestamp}.csv"
        ),
        "experiment": (
            EXPERIMENT_DIRECTORY
            / f"{EXPERIMENT_NAME}_{timestamp}.json"
        ),
    }
    fold_frame.to_csv(paths["fold_results"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    thresholds.to_csv(paths["thresholds"], index=False)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": EXPERIMENT_NAME,
        "target": {
            "volatility_window": TARGET_WINDOW,
            "threshold_multiplier": TARGET_MULTIPLIER,
        },
        "development_cutoff": cutoff,
        "winner_groups": list(winner_groups),
        "winner_feature_count": len(winner_features),
        "selective_policy": {
            "model": "existing nested OOF xgboost_binary_winner",
            "regime_feature": REGIME_FEATURE,
            "accepted_regime": REGIME_STATE,
            "threshold_policy": (
                "outer-fold training MOVE 66.7th percentile; prediction only above q67"
            ),
            "abstained_regimes": ["LOW", "MID"],
        },
        "statistics": {
            "outer_splits": OUTER_SPLITS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_block_length": BOOTSTRAP_BLOCK_LENGTH,
            "gates": (
                "Accepted pooled AUC >= 0.55; accepted moving-block bootstrap lower "
                "95% AUC > 0.50; accepted AUC >= 0.55 in at least 2 of 3 outer folds."
            ),
        },
        "summary": summary,
        "methodology": {
            "new_model_training": False,
            "new_feature_search": False,
            "new_hyperparameter_search": False,
            "outer_validation_loaded": False,
            "held_out_test_loaded": False,
            "selection_note": (
                "realized_volatility_20 HIGH was the sole development-supported regime "
                "candidate from the prior FDR-controlled regime screen. This experiment "
                "formalizes the selective-prediction policy on development OOF only."
            ),
        },
        "outputs": paths,
    }
    with paths["experiment"].open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=json_default)

    print()
    print(f"Fold results: {paths['fold_results']}")
    print(f"Predictions: {paths['predictions']}")
    print(f"Thresholds: {paths['thresholds']}")
    print(f"Experiment: {paths['experiment']}")
    print("Outer validation was NOT evaluated.")
    print("Held-out test set was NOT evaluated.")
    print()
    if summary["gates"]["overall_selective_prediction_gate"]:
        print(
            "NEXT DECISION RULE: PASS. Lock the universal Stage-2 XGBoost as the "
            "direction model and the HIGH realized-volatility state as a selective "
            "UP/DOWN eligibility gate. Do not train a regime specialist. Next evaluate "
            "the locked Stage1 -> Stage2 selective hierarchy without changing this policy."
        )
    else:
        print(
            "NEXT DECISION RULE: FAIL. Do not add a regime-based abstention policy. "
            "Retain the universal Stage-2 model as the research baseline and do not "
            "infer a deployment regime from the consumed outer-validation period."
        )


if __name__ == "__main__":
    main()
