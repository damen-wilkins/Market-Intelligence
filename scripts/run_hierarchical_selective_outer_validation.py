from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from app.training.hierarchical_selective_outer_validation_evaluator import (
    HierarchicalSelectiveGateConfig,
    HierarchicalSelectiveOuterValidationEvaluator,
)
from app.training.stage2_outer_validation_gate import (
    ValidationPeriods,
    split_development_and_outer_validation,
)
from app.training.volatility_direction_label_builder import (
    VolatilityDirectionLabelBuilder,
)
from database.direction_training_data_repository import (
    DirectionTrainingDataRepository,
)
from database.stage2_signal_data_repository import (
    Stage2SignalDataRepository,
)
from scripts.run_stage1_long_history_optimization import (
    train_stage1_fold,
)
from scripts.run_stage1_target_optimization import (
    STAGE1_FEATURE_COLUMNS,
    TARGET_STATE_FEATURE,
    build_common_feature_frame,
)
from scripts.run_stage2_outer_validation_gate import (
    RANDOM_STATE as STAGE2_RANDOM_STATE,
    build_research_data as build_stage2_research_data,
    feature_columns as stage2_feature_columns,
    load_reference_periods,
)


TICKER = "SPY"
TARGET_NAME = "flat_90d_k700"
TARGET_WINDOW = 90
TARGET_MULTIPLIER = 0.700
REGIME_FEATURE = "realized_volatility_20"
REGIME_QUANTILE = 2.0 / 3.0
EXPERIMENT_DIRECTORY = Path("experiments")
STAGE1_PROGRESS_PATH = (
    EXPERIMENT_DIRECTORY
    / "stage1_target_optimization_v1_progress.json"
)
STAGE2_OUTER_VALIDATION_PATTERN = (
    "stage2_outer_validation_gate_v1_*.json"
)
EXPERIMENT_NAME = "hierarchical_selective_outer_validation_v1"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_BLOCK_LENGTH = 20
STABILITY_BLOCKS = 3
STAGE1_MIN_AUC = 0.60
STAGE2_MIN_AUC = 0.55
MINIMUM_SELECTIVE_COVERAGE = 0.60
SCORE_REPRODUCTION_ATOL = 1e-7
SCORE_REPRODUCTION_RTOL = 1e-6


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
    raise TypeError(
        f"Cannot serialize {type(value).__name__}."
    )


def load_locked_stage1() -> dict:
    if not STAGE1_PROGRESS_PATH.exists():
        raise FileNotFoundError(
            f"{STAGE1_PROGRESS_PATH} is required."
        )
    with STAGE1_PROGRESS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        rows = json.load(file)

    matches = [
        row
        for row in rows
        if str(row.get("target_name")) == TARGET_NAME
        and int(row.get("volatility_window")) == TARGET_WINDOW
        and abs(
            float(row.get("threshold_multiplier"))
            - TARGET_MULTIPLIER
        ) < 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one locked Stage-1 result for {TARGET_NAME}; "
            f"found {len(matches)}."
        )
    row = matches[0]
    if not isinstance(row.get("parameters"), dict):
        raise ValueError(
            "Locked Stage-1 result does not contain parameters."
        )
    if row.get("optimized_decision_threshold") is None:
        raise ValueError(
            "Locked Stage-1 result does not contain the optimized decision threshold."
        )
    return row


def latest_stage2_outer_validation_experiment() -> tuple[Path, dict]:
    paths = sorted(
        EXPERIMENT_DIRECTORY.glob(
            STAGE2_OUTER_VALIDATION_PATTERN
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        raise FileNotFoundError(
            "No Stage-2 outer-validation gate JSON was found."
        )
    path = paths[0]
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return path, payload


def load_locked_stage2(
    periods: ValidationPeriods,
) -> tuple[Path, dict, dict, Path]:
    experiment_path, payload = latest_stage2_outer_validation_experiment()

    target = payload.get("target", {})
    if int(target.get("volatility_window", -1)) != TARGET_WINDOW:
        raise ValueError(
            "Stage-2 outer-validation target window does not match the locked target."
        )
    if abs(
        float(target.get("threshold_multiplier", -1.0))
        - TARGET_MULTIPLIER
    ) > 1e-12:
        raise ValueError(
            "Stage-2 outer-validation multiplier does not match the locked target."
        )

    payload_periods = payload.get("periods", {})
    if pd.Timestamp(payload_periods.get("training_end")) != periods.training_end:
        raise ValueError(
            "Stage-2 outer-validation training boundary does not match."
        )
    if pd.Timestamp(payload_periods.get("validation_start")) != periods.validation_start:
        raise ValueError(
            "Stage-2 outer-validation start does not match."
        )
    if pd.Timestamp(payload_periods.get("validation_end")) != periods.validation_end:
        raise ValueError(
            "Stage-2 outer-validation end does not match."
        )

    result = payload.get("results", {}).get(
        "xgboost_binary_winner"
    )
    if not isinstance(result, dict):
        raise ValueError(
            "Locked xgboost_binary_winner outer-validation result was not found."
        )
    parameters = result.get("parameters")
    metrics = result.get("metrics")
    prediction_path = Path(result.get("prediction_path", ""))
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError(
            "Locked Stage-2 result is missing parameters."
        )
    if not isinstance(metrics, dict):
        raise ValueError(
            "Locked Stage-2 result is missing metrics."
        )
    if metrics.get("decision_threshold") is None:
        raise ValueError(
            "Locked Stage-2 result is missing the decision threshold."
        )
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Saved Stage-2 validation predictions were not found: {prediction_path}"
        )

    return experiment_path, parameters, metrics, prediction_path


def build_stage1_data(
    periods: ValidationPeriods,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = DirectionTrainingDataRepository().get_training_data(
        ticker=TICKER,
        include_breadth=False,
        include_cross_asset=False,
    )
    raw = raw.copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw = raw.loc[
        raw["trade_date"] <= periods.validation_end
    ].reset_index(drop=True)

    features = build_common_feature_frame(raw)
    labels = VolatilityDirectionLabelBuilder(
        volatility_window=TARGET_WINDOW,
        threshold_multiplier=TARGET_MULTIPLIER,
    ).build(
        raw[
            [
                "trade_date",
                "close",
            ]
        ].copy()
    )

    master = (
        features.merge(
            labels,
            on="feature_date",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("target_date")
        .reset_index(drop=True)
    )
    master[TARGET_STATE_FEATURE] = (
        master["rolling_volatility"].astype(float)
    )
    master = master.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=STAGE1_FEATURE_COLUMNS,
    ).reset_index(drop=True)

    keep = [
        "feature_date",
        "target_date",
        *STAGE1_FEATURE_COLUMNS,
        "future_log_return",
        "rolling_volatility",
        "threshold",
        "direction",
    ]
    data = master[
        keep
    ].sort_values(
        "target_date"
    ).reset_index(
        drop=True
    )
    return split_development_and_outer_validation(
        data,
        periods,
    )


def fit_stage1_outer_validation(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    locked_stage1: dict,
) -> pd.DataFrame:
    result = train_stage1_fold(
        fold_train=development,
        fold_validation=validation,
        feature_columns=STAGE1_FEATURE_COLUMNS,
        parameters=dict(locked_stage1["parameters"]),
        seed=424242,
    )
    result_dates = pd.DatetimeIndex(
        pd.to_datetime(result["target_dates"])
    )
    expected_dates = pd.DatetimeIndex(
        pd.to_datetime(validation["target_date"])
    )
    if not result_dates.equals(expected_dates):
        raise ValueError(
            "Stage-1 outer-validation predictions do not align to the locked "
            "outer-validation dates."
        )

    frame = validation[
        [
            "feature_date",
            "target_date",
            "future_log_return",
            "direction",
        ]
    ].copy()
    frame = frame.rename(
        columns={
            "direction": "actual_direction",
        }
    )
    frame["stage1_move_probability"] = np.asarray(
        result["move_probabilities"],
        dtype=np.float64,
    )

    expected_actual = (
        frame["actual_direction"].astype(str) != "FLAT"
    ).astype(int).to_numpy()
    if not np.array_equal(
        expected_actual,
        np.asarray(result["actual"], dtype=np.int64),
    ):
        raise ValueError(
            "Stage-1 returned labels do not match the locked 90d x 0.700 target."
        )
    return frame


def fit_stage2_all_validation(
    periods: ValidationPeriods,
    parameters: dict,
    decision_threshold: float,
    saved_prediction_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    columns = stage2_feature_columns()
    raw = Stage2SignalDataRepository().get_training_data(
        ticker=TICKER
    )
    research = build_stage2_research_data(
        raw_data=raw,
        periods=periods,
        columns=columns,
    )
    development, validation = split_development_and_outer_validation(
        research,
        periods,
    )
    development_move = development.loc[
        development["direction"].astype(str) != "FLAT"
    ].reset_index(drop=True)

    if REGIME_FEATURE not in development.columns:
        raise ValueError(
            f"{REGIME_FEATURE} is not present in the locked Stage-2 feature set."
        )

    regime_threshold = float(
        pd.to_numeric(
            development_move[REGIME_FEATURE],
            errors="raise",
        ).quantile(
            REGIME_QUANTILE
        )
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=STAGE2_RANDOM_STATE + 900000,
        n_jobs=-1,
        **parameters,
    )
    model.fit(
        development_move[columns],
        (
            development_move["direction"].astype(str)
            == "UP"
        ).astype(int),
    )

    all_scores = model.predict_proba(
        validation[columns]
    )[:, 1].astype(
        np.float64
    )

    frame = validation[
        [
            "feature_date",
            "target_date",
            "future_log_return",
            "direction",
            REGIME_FEATURE,
        ]
    ].copy()
    frame = frame.rename(
        columns={
            "direction": "stage2_actual_direction",
        }
    )
    frame["stage2_up_score"] = all_scores
    frame["high_volatility_regime"] = (
        frame[REGIME_FEATURE].astype(float)
        > regime_threshold
    )

    maximum_score_difference = verify_stage2_reproduction(
        all_validation=frame,
        saved_prediction_path=saved_prediction_path,
    )
    return (
        development,
        frame,
        regime_threshold,
        maximum_score_difference,
    )


def verify_stage2_reproduction(
    all_validation: pd.DataFrame,
    saved_prediction_path: Path,
) -> float:
    saved = pd.read_csv(
        saved_prediction_path,
        parse_dates=["target_date"],
    )
    saved = saved.sort_values(
        "target_date"
    ).reset_index(
        drop=True
    )

    actual_move = all_validation.loc[
        all_validation["stage2_actual_direction"].astype(str)
        != "FLAT",
        [
            "target_date",
            "stage2_up_score",
        ],
    ].copy()
    actual_move = actual_move.sort_values(
        "target_date"
    ).reset_index(
        drop=True
    )

    saved_dates = pd.DatetimeIndex(
        pd.to_datetime(saved["target_date"])
    )
    reproduced_dates = pd.DatetimeIndex(
        pd.to_datetime(actual_move["target_date"])
    )
    if not saved_dates.equals(reproduced_dates):
        raise ValueError(
            "Refit Stage-2 MOVE dates do not reproduce the saved outer-validation "
            "prediction dates."
        )

    saved_scores = saved["score"].to_numpy(
        dtype=np.float64
    )
    reproduced_scores = actual_move[
        "stage2_up_score"
    ].to_numpy(
        dtype=np.float64
    )
    maximum_difference = float(
        np.max(
            np.abs(
                saved_scores
                - reproduced_scores
            )
        )
    )
    scores_match = np.allclose(
        saved_scores,
        reproduced_scores,
        rtol=SCORE_REPRODUCTION_RTOL,
        atol=SCORE_REPRODUCTION_ATOL,
    )
    if not scores_match:
        raise ValueError(
            "Refit Stage-2 model failed the saved-score reproduction check. "
            f"Maximum absolute score difference={maximum_difference:.12g}, "
            f"atol={SCORE_REPRODUCTION_ATOL:.12g}, "
            f"rtol={SCORE_REPRODUCTION_RTOL:.12g}."
        )

    if "predicted_up" in saved.columns:
        saved_predictions = saved["predicted_up"].astype(int).to_numpy()
        saved_threshold = (
            float(saved["decision_threshold"].iloc[0])
            if "decision_threshold" in saved.columns
            else None
        )
        if saved_threshold is not None:
            reproduced_predictions = (
                reproduced_scores >= saved_threshold
            ).astype(np.int64)
            if not np.array_equal(
                saved_predictions,
                reproduced_predictions,
            ):
                raise ValueError(
                    "Refit Stage-2 scores are numerically close, but the saved "
                    "UP/DOWN decisions are not reproduced exactly."
                )

    return maximum_difference


def align_hierarchy(
    stage1: pd.DataFrame,
    stage2: pd.DataFrame,
) -> pd.DataFrame:
    merged = stage1.merge(
        stage2[
            [
                "target_date",
                "stage2_actual_direction",
                "stage2_up_score",
                REGIME_FEATURE,
                "high_volatility_regime",
            ]
        ],
        on="target_date",
        how="inner",
        validate="one_to_one",
    ).sort_values(
        "target_date"
    ).reset_index(
        drop=True
    )

    mismatch = (
        merged["actual_direction"].astype(str)
        != merged["stage2_actual_direction"].astype(str)
    )
    if mismatch.any():
        raise ValueError(
            "Stage-1 and Stage-2 target directions disagree after alignment."
        )
    merged = merged.drop(
        columns=["stage2_actual_direction"]
    )
    if merged.empty:
        raise ValueError(
            "No aligned outer-validation rows were available for the hierarchy."
        )
    return merged


def print_metric_set(
    title: str,
    metrics: dict,
) -> None:
    print(title)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(
        "  Balanced accuracy:",
        f"{metrics['balanced_accuracy']:.4f}",
    )
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(
        "  DOWN / FLAT / UP F1:",
        " / ".join(
            f"{metrics['per_class'][name]['f1']:.4f}"
            for name in ("DOWN", "FLAT", "UP")
        ),
    )


def main():
    EXPERIMENT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    periods = load_reference_periods()
    locked_stage1 = load_locked_stage1()
    (
        stage2_experiment_path,
        stage2_parameters,
        stage2_metrics,
        stage2_saved_prediction_path,
    ) = load_locked_stage2(
        periods
    )

    stage1_development, stage1_validation = (
        build_stage1_data(
            periods
        )
    )

    print("=" * 88)
    print("LOCKED STAGE1 -> STAGE2 SELECTIVE HIERARCHY OUTER VALIDATION V1")
    print("=" * 88)
    print(f"Target: {TARGET_WINDOW}d x {TARGET_MULTIPLIER:.3f}")
    print(
        "Stage 1: xLSTM FLAT vs MOVE | locked development parameters and "
        "locked OOF decision threshold."
    )
    print(
        "Stage 2: universal XGBoost UP vs DOWN | exact parameters and threshold "
        "from the already-consumed Stage-2 outer-validation gate."
    )
    print(
        "Eligibility: Stage-2 direction only when realized_volatility_20 is above "
        "the full-development MOVE 66.7th percentile; otherwise ABSTAIN."
    )
    print(
        "Outer validation:",
        periods.validation_start.date(),
        "->",
        periods.validation_end.date(),
    )
    print(
        "Held-out test begins AFTER",
        periods.validation_end.date(),
        "and is NOT evaluated.",
    )
    print()
    print(
        "IMPORTANT: this outer-validation period was already consumed during Stage-2 "
        "research. This run is a locked end-to-end characterization only."
    )
    print(
        "No target, feature, threshold, regime rule, hyperparameter, or architecture "
        "may be changed because of this result."
    )
    print()

    print("Fitting locked Stage 1 on all development rows...")
    stage1_predictions = fit_stage1_outer_validation(
        development=stage1_development,
        validation=stage1_validation,
        locked_stage1=locked_stage1,
    )

    print("Fitting locked Stage 2 on all development MOVE rows...")
    (
        stage2_development,
        stage2_predictions,
        regime_threshold,
        maximum_score_difference,
    ) = fit_stage2_all_validation(
        periods=periods,
        parameters=stage2_parameters,
        decision_threshold=float(
            stage2_metrics["decision_threshold"]
        ),
        saved_prediction_path=stage2_saved_prediction_path,
    )
    print(
        "Stage-2 saved-score reproduction check: PASS "
        f"(max abs difference {maximum_score_difference:.3e})"
    )

    hierarchy = align_hierarchy(
        stage1=stage1_predictions,
        stage2=stage2_predictions,
    )
    print(
        "Aligned hierarchy rows:",
        len(hierarchy),
    )
    print(
        "Full-development HIGH-volatility threshold:",
        f"{regime_threshold:.8f}",
    )

    evaluator = HierarchicalSelectiveOuterValidationEvaluator(
        HierarchicalSelectiveGateConfig(
            stage1_min_auc=STAGE1_MIN_AUC,
            stage2_min_auc=STAGE2_MIN_AUC,
            minimum_selective_coverage=MINIMUM_SELECTIVE_COVERAGE,
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
            bootstrap_block_length=BOOTSTRAP_BLOCK_LENGTH,
            stability_blocks=STABILITY_BLOCKS,
            random_state=42,
        )
    )
    result = evaluator.evaluate(
        dataframe=hierarchy,
        stage1_threshold=float(
            locked_stage1[
                "optimized_decision_threshold"
            ]
        ),
        stage2_threshold=float(
            stage2_metrics["decision_threshold"]
        ),
    )

    print()
    print("STAGE 1 OUTER VALIDATION")
    print(
        "ROC AUC:",
        f"{result['stage1']['roc_auc']:.4f}",
    )
    print(
        "Moving-block bootstrap 95% AUC CI:",
        f"[{result['stage1']['bootstrap_auc_lower_95']:.4f}, "
        f"{result['stage1']['bootstrap_auc_upper_95']:.4f}]",
    )
    print(
        "Balanced accuracy:",
        f"{result['stage1']['balanced_accuracy']:.4f}",
    )
    print(
        "Macro F1:",
        f"{result['stage1']['macro_f1']:.4f}",
    )
    print(
        "FLAT F1 / MOVE F1:",
        f"{result['stage1']['per_class']['FLAT']['f1']:.4f} / "
        f"{result['stage1']['per_class']['MOVE']['f1']:.4f}",
    )

    print()
    print("SELECTIVE STAGE 2 ON ROUTED TRUE-MOVE ROWS")
    stage2_selective = result[
        "stage2_selective_true_move"
    ]
    print(
        "Rows:",
        stage2_selective["rows"],
    )
    print(
        "ROC AUC:",
        f"{stage2_selective['roc_auc']:.4f}",
    )
    print(
        "Moving-block bootstrap 95% AUC CI:",
        f"[{stage2_selective['bootstrap_auc_lower_95']:.4f}, "
        f"{stage2_selective['bootstrap_auc_upper_95']:.4f}]",
    )
    print(
        "Balanced accuracy / Macro F1:",
        f"{stage2_selective['balanced_accuracy']:.4f} / "
        f"{stage2_selective['macro_f1']:.4f}",
    )
    print(
        "Magnitude-weighted sign accuracy:",
        f"{stage2_selective['magnitude_weighted_sign_accuracy']:.4f}",
    )

    print()
    print("ROUTING")
    routing = result["routing"]
    print(
        "Stage-1 FLAT outputs:",
        f"{routing['stage1_predicted_flat_rows']} "
        f"({routing['stage1_flat_output_rate']:.2%})",
    )
    print(
        "Stage-1 MOVE outputs:",
        routing["stage1_predicted_move_rows"],
    )
    print(
        "Accepted directional outputs:",
        f"{routing['accepted_direction_rows']} "
        f"({routing['directional_prediction_coverage']:.2%} of all rows)",
    )
    print(
        "Abstained rows:",
        f"{routing['abstained_rows']} "
        f"({routing['abstention_rate']:.2%})",
    )
    print(
        "Total non-abstained hierarchy coverage:",
        f"{routing['selective_total_coverage']:.2%}",
    )
    print(
        "Accepted directional route MOVE purity:",
        f"{routing['accepted_route_move_purity']:.2%}",
    )
    print(
        "Accepted directional end-to-end accuracy:",
        f"{routing['accepted_route_end_to_end_direction_accuracy']:.4f}",
    )

    print()
    print_metric_set(
        "UNIVERSAL HIERARCHY - FULL COVERAGE",
        result["universal_hierarchy"],
    )
    print()
    print_metric_set(
        "SELECTIVE HIERARCHY - NON-ABSTAINED ROWS",
        result["selective_hierarchy"],
    )

    block_frame = pd.DataFrame(
        result["stability_blocks"]
    )
    print()
    print("CHRONOLOGICAL END-TO-END STABILITY")
    print(
        block_frame[
            [
                "block",
                "start",
                "end",
                "rows",
                "selective_coverage",
                "accepted_direction_rows",
                "accepted_true_move_rows",
                "accepted_true_move_auc",
                "universal_balanced_accuracy",
                "selective_balanced_accuracy",
                "universal_macro_f1",
                "selective_macro_f1",
            ]
        ].round(4).to_string(
            index=False
        )
    )

    print()
    print("PREDEFINED LOCKED-HIERARCHY CONFIRMATION GATES")
    for name, passed in result["gates"].items():
        if name == "overall_locked_hierarchy_gate":
            continue
        print(
            f"- {name}:",
            "PASS" if passed else "FAIL",
        )
    print(
        "  OVERALL LOCKED HIERARCHY GATE:",
        (
            "PASS"
            if result["gates"][
                "overall_locked_hierarchy_gate"
            ]
            else "FAIL"
        ),
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    prediction_path = (
        EXPERIMENT_DIRECTORY
        / f"{EXPERIMENT_NAME}_predictions_{timestamp}.csv"
    )
    block_path = (
        EXPERIMENT_DIRECTORY
        / f"{EXPERIMENT_NAME}_blocks_{timestamp}.csv"
    )
    experiment_path = (
        EXPERIMENT_DIRECTORY
        / f"{EXPERIMENT_NAME}_{timestamp}.json"
    )

    result["predictions"].to_csv(
        prediction_path,
        index=False,
    )
    block_frame.to_csv(
        block_path,
        index=False,
    )

    payload = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment_name": EXPERIMENT_NAME,
        "target": {
            "name": TARGET_NAME,
            "volatility_window": TARGET_WINDOW,
            "threshold_multiplier": TARGET_MULTIPLIER,
        },
        "periods": {
            "training_end": periods.training_end,
            "validation_start": periods.validation_start,
            "validation_end": periods.validation_end,
        },
        "stage1": {
            "source": str(
                STAGE1_PROGRESS_PATH
            ),
            "parameters": locked_stage1[
                "parameters"
            ],
            "decision_threshold": float(
                locked_stage1[
                    "optimized_decision_threshold"
                ]
            ),
            "development_auc": float(
                locked_stage1[
                    "optimized_roc_auc"
                ]
            ),
            "development_bootstrap_lower_95": float(
                locked_stage1[
                    "bootstrap_auc_lower_95"
                ]
            ),
        },
        "stage2": {
            "source": str(
                stage2_experiment_path
            ),
            "saved_prediction_source": str(
                stage2_saved_prediction_path
            ),
            "parameters": stage2_parameters,
            "decision_threshold": float(
                stage2_metrics[
                    "decision_threshold"
                ]
            ),
            "saved_score_reproduction_max_abs_difference": (
                maximum_score_difference
            ),
        },
        "selective_policy": {
            "regime_feature": REGIME_FEATURE,
            "regime_quantile": REGIME_QUANTILE,
            "full_development_move_threshold": regime_threshold,
            "rule": (
                "If Stage1 predicts FLAT -> FLAT. "
                "If Stage1 predicts MOVE and realized_volatility_20 > "
                "full-development MOVE q66.7 -> Stage2 UP/DOWN. "
                "Otherwise -> ABSTAIN."
            ),
        },
        "confirmation_gates": {
            "stage1_min_auc": STAGE1_MIN_AUC,
            "stage2_min_auc": STAGE2_MIN_AUC,
            "minimum_selective_coverage": MINIMUM_SELECTIVE_COVERAGE,
            "stage1_bootstrap_lower_must_exceed": 0.50,
            "stage2_bootstrap_lower_must_exceed": 0.50,
            "selective_balanced_accuracy_must_exceed_universal": True,
        },
        "results": {
            key: value
            for key, value in result.items()
            if key != "predictions"
        },
        "methodology": {
            "target_search": False,
            "feature_search": False,
            "hyperparameter_search": False,
            "threshold_search": False,
            "regime_search": False,
            "outer_validation_already_consumed": True,
            "outer_validation_used_for_locked_end_to_end_characterization": True,
            "held_out_test_evaluated": False,
            "validation_result_may_change_locked_policy": False,
        },
        "outputs": {
            "predictions": prediction_path,
            "stability_blocks": block_path,
            "experiment": experiment_path,
        },
    }
    with experiment_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            default=json_default,
        )

    print()
    print("Predictions:", prediction_path)
    print("Stability blocks:", block_path)
    print("Experiment:", experiment_path)
    print("Held-out test set was NOT evaluated.")
    print()
    if result["gates"]["overall_locked_hierarchy_gate"]:
        print(
            "NEXT DECISION RULE: PASS. The locked hierarchy has cleared the "
            "predefined outer-validation characterization gates. Do not change it. "
            "The next statistically valid evaluation is the untouched final held-out test."
        )
    else:
        print(
            "NEXT DECISION RULE: FAIL. Do NOT tune against this consumed validation "
            "period and do NOT evaluate the held-out test. Record the failure and return "
            "to development-only research if a materially new architecture is justified."
        )


if __name__ == "__main__":
    main()
