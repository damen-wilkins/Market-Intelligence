from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from app.training.stage2_outer_validation_gate import (
    ValidationPeriods,
    split_development_and_outer_validation,
)
from app.training.stage2_regime_diagnostic import Stage2RegimeDiagnostic
from app.training.stage2_wide_feature_builder import Stage2WideFeatureBuilder
from app.training.volatility_direction_label_builder import (
    VolatilityDirectionLabelBuilder,
)
from database.stage2_signal_data_repository import Stage2SignalDataRepository


TICKER = "SPY"
TARGET_WINDOW = 90
TARGET_MULTIPLIER = 0.700
TARGET_STATE_FEATURE = "target_rolling_volatility"
REFERENCE_MODEL_PATH = Path("models/xlstm_hierarchical_direction.pt")
EXPERIMENT_DIRECTORY = Path("experiments")
PREDICTION_GLOB = "stage2_outer_validation_xgboost_binary_winner_*.csv"
EXPERIMENT_NAME = "stage2_regime_diagnostic_v1"
BLOCK_COUNT = 3
MINIMUM_AUC_ROWS = 20
TOP_CONTRASTS_TO_PRINT = 15


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


def load_reference_periods() -> ValidationPeriods:
    if not REFERENCE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "models/xlstm_hierarchical_direction.pt is required to preserve "
            "the original train/validation/test boundaries."
        )
    try:
        package = torch.load(
            REFERENCE_MODEL_PATH,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        package = torch.load(REFERENCE_MODEL_PATH, map_location="cpu")
    metadata = package.get("metadata", {})
    training_period = metadata.get("training_period", {})
    validation_period = metadata.get("validation_period", {})
    required = {
        "training_end": training_period.get("end"),
        "validation_start": validation_period.get("start"),
        "validation_end": validation_period.get("end"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Reference model is missing period metadata: " + ", ".join(missing)
        )
    return ValidationPeriods(
        training_end=pd.Timestamp(required["training_end"]),
        validation_start=pd.Timestamp(required["validation_start"]),
        validation_end=pd.Timestamp(required["validation_end"]),
    )


def latest_binary_outer_validation_predictions() -> Path:
    candidates = sorted(
        EXPERIMENT_DIRECTORY.glob(PREDICTION_GLOB),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            "No saved Stage-2 XGBoost binary outer-validation prediction file was "
            "found. Run scripts.run_stage2_outer_validation_gate first."
        )
    return candidates[-1]


def load_predictions(path: Path, periods: ValidationPeriods) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    required = {
        "target_date",
        "actual_direction",
        "actual_up",
        "score",
        "predicted_up",
        "actual_future_log_return",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(
            f"Prediction file {path} is missing columns: " + ", ".join(missing)
        )
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    predictions = predictions.sort_values("target_date").reset_index(drop=True)
    if predictions["target_date"].duplicated().any():
        raise ValueError("Saved outer-validation predictions contain duplicate dates.")
    if predictions["target_date"].min() < periods.validation_start:
        raise ValueError("Prediction file begins before the locked validation period.")
    if predictions["target_date"].max() > periods.validation_end:
        raise ValueError(
            "Prediction file crosses beyond the locked validation period into the "
            "held-out test. Aborting."
        )
    if not predictions["actual_direction"].isin(["UP", "DOWN"]).all():
        raise ValueError("Stage-2 prediction file must contain MOVE days only.")
    return predictions


def build_regime_master(
    raw_data: pd.DataFrame,
    periods: ValidationPeriods,
    diagnostic: Stage2RegimeDiagnostic,
) -> pd.DataFrame:
    raw = raw_data.copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw = raw.loc[raw["trade_date"] <= periods.validation_end].reset_index(drop=True)
    features = Stage2WideFeatureBuilder().build_library(raw)
    labels = VolatilityDirectionLabelBuilder(
        volatility_window=TARGET_WINDOW,
        threshold_multiplier=TARGET_MULTIPLIER,
    ).build(raw[["trade_date", "close"]].copy())
    master = (
        features.rename(columns={"trade_date": "feature_date"})
        .merge(labels, on="feature_date", how="inner", validate="one_to_one")
        .sort_values("target_date")
        .reset_index(drop=True)
    )
    master[TARGET_STATE_FEATURE] = master["rolling_volatility"].astype(float)
    diagnostic.validate_feature_columns(master)
    keep = [
        "feature_date",
        "target_date",
        "direction",
        "future_log_return",
        "rolling_volatility",
        "threshold",
        *diagnostic.feature_columns,
    ]
    keep = list(dict.fromkeys(keep))
    return master[keep].replace([np.inf, -np.inf], np.nan)


def enrich_validation_predictions(
    predictions: pd.DataFrame,
    validation: pd.DataFrame,
    diagnostic: Stage2RegimeDiagnostic,
) -> pd.DataFrame:
    columns = [
        "feature_date",
        "target_date",
        "direction",
        *diagnostic.feature_columns,
    ]
    columns = list(dict.fromkeys(columns))
    enriched = predictions.merge(
        validation[columns],
        on="target_date",
        how="left",
        validate="one_to_one",
    )
    if enriched["feature_date"].isna().any():
        missing_dates = enriched.loc[
            enriched["feature_date"].isna(), "target_date"
        ].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(
            "Could not align saved predictions to regime features for dates: "
            + ", ".join(missing_dates[:10])
        )
    disagreement = enriched["actual_direction"].astype(str) != enriched["direction"].astype(str)
    if disagreement.any():
        raise ValueError(
            "Saved predictions do not align with the locked 90d x 0.700 target."
        )
    enriched = diagnostic.assign_chronological_blocks(enriched)
    enriched["correct_prediction"] = (
        enriched["predicted_up"].astype(int) == enriched["actual_up"].astype(int)
    ).astype(int)
    enriched["probability_distance_from_0_5"] = (
        enriched["score"].astype(float) - 0.5
    ).abs()
    return enriched


def print_block_diagnostics(blocks: pd.DataFrame) -> None:
    print()
    print("MODEL PERFORMANCE BY LOCKED CHRONOLOGICAL BLOCK")
    print(
        blocks[
            [
                "validation_block",
                "start",
                "end",
                "rows",
                "up_share",
                "roc_auc",
                "sign_accuracy",
                "magnitude_weighted_sign_accuracy",
                "mean_score",
            ]
        ].round(4).to_string(index=False)
    )


def print_top_contrasts(contrasts: pd.DataFrame) -> None:
    print()
    print("LARGEST BLOCK 2 VS BLOCK 1 REGIME DIFFERENCES")
    if contrasts.empty:
        print("No valid regime contrasts were produced.")
        return
    display = contrasts.head(TOP_CONTRASTS_TO_PRINT)[
        [
            "family",
            "feature",
            "block1_mean_z_vs_development",
            "block2_mean_z_vs_development",
            "block2_minus_block1_mean_in_dev_std",
        ]
    ].copy()
    print(display.round(4).to_string(index=False))


def print_regime_auc_extremes(regime_auc: pd.DataFrame) -> None:
    print()
    print("VALIDATION AUC BY DEVELOPMENT-DEFINED REGIME TERTILE")
    valid = regime_auc.dropna(subset=["roc_auc"]).copy()
    if valid.empty:
        print("No regime bucket contained enough rows and both classes for AUC.")
        return
    pivot = valid.pivot_table(
        index=["family", "feature"],
        columns="regime",
        values="roc_auc",
        aggfunc="first",
    ).reset_index()
    for column in ("LOW", "MID", "HIGH"):
        if column not in pivot.columns:
            pivot[column] = np.nan
    pivot["auc_range"] = pivot[["LOW", "MID", "HIGH"]].max(axis=1) - pivot[
        ["LOW", "MID", "HIGH"]
    ].min(axis=1)
    pivot = pivot.sort_values("auc_range", ascending=False).head(TOP_CONTRASTS_TO_PRINT)
    print(
        pivot[["family", "feature", "LOW", "MID", "HIGH", "auc_range"]]
        .round(4)
        .to_string(index=False)
    )


def main():
    EXPERIMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    periods = load_reference_periods()
    diagnostic = Stage2RegimeDiagnostic(
        block_count=BLOCK_COUNT,
        minimum_auc_rows=MINIMUM_AUC_ROWS,
    )
    prediction_path = latest_binary_outer_validation_predictions()
    predictions = load_predictions(prediction_path, periods)

    print("=" * 78)
    print("STAGE-2 REGIME DIAGNOSTIC V1")
    print("=" * 78)
    print(f"Target: {TARGET_WINDOW}d x {TARGET_MULTIPLIER:.3f}")
    print("Source predictions:", prediction_path)
    print(
        "Outer validation:",
        periods.validation_start.date(),
        "->",
        periods.validation_end.date(),
    )
    print("Held-out test begins AFTER", periods.validation_end.date(), "and is NOT evaluated.")
    print()
    print(
        "DIAGNOSTIC ONLY: no model fitting, no Optuna search, no feature selection, "
        "and no validation-derived regime thresholds."
    )
    print(
        "LOW/MID/HIGH regime thresholds are defined from development MOVE rows only; "
        "outer validation is used only to describe conditional performance."
    )

    raw = Stage2SignalDataRepository().get_training_data(ticker=TICKER)
    master = build_regime_master(raw, periods, diagnostic)
    development, validation = split_development_and_outer_validation(master, periods)
    development_move = development.loc[
        development["direction"] != "FLAT"
    ].reset_index(drop=True)
    validation_move = validation.loc[
        validation["direction"] != "FLAT"
    ].reset_index(drop=True)

    enriched = enrich_validation_predictions(
        predictions=predictions,
        validation=validation_move,
        diagnostic=diagnostic,
    )
    if len(enriched) != len(predictions):
        raise ValueError("Prediction/regime merge changed outer-validation row count.")

    block_diagnostics = diagnostic.block_model_diagnostics(enriched)
    block_profiles = diagnostic.block_feature_profiles(
        development_move=development_move,
        enriched_validation=enriched,
    )
    block_contrasts = diagnostic.block_feature_contrasts(block_profiles)
    tertiles = diagnostic.development_tertiles(development_move)
    regime_auc = diagnostic.validation_auc_by_development_tertile(
        enriched_validation=enriched,
        tertiles=tertiles,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    paths = {
        "enriched_validation": (
            EXPERIMENT_DIRECTORY
            / f"{EXPERIMENT_NAME}_enriched_validation_{timestamp}.csv"
        ),
        "block_diagnostics": (
            EXPERIMENT_DIRECTORY
            / f"{EXPERIMENT_NAME}_block_diagnostics_{timestamp}.csv"
        ),
        "block_profiles": (
            EXPERIMENT_DIRECTORY
            / f"{EXPERIMENT_NAME}_block_profiles_{timestamp}.csv"
        ),
        "block_contrasts": (
            EXPERIMENT_DIRECTORY
            / f"{EXPERIMENT_NAME}_block_contrasts_{timestamp}.csv"
        ),
        "development_tertiles": (
            EXPERIMENT_DIRECTORY
            / f"{EXPERIMENT_NAME}_development_tertiles_{timestamp}.csv"
        ),
        "regime_auc": (
            EXPERIMENT_DIRECTORY
            / f"{EXPERIMENT_NAME}_regime_auc_{timestamp}.csv"
        ),
        "experiment": (
            EXPERIMENT_DIRECTORY / f"{EXPERIMENT_NAME}_{timestamp}.json"
        ),
    }

    enriched.to_csv(paths["enriched_validation"], index=False)
    block_diagnostics.to_csv(paths["block_diagnostics"], index=False)
    block_profiles.to_csv(paths["block_profiles"], index=False)
    block_contrasts.to_csv(paths["block_contrasts"], index=False)
    tertiles.to_csv(paths["development_tertiles"], index=False)
    regime_auc.to_csv(paths["regime_auc"], index=False)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": EXPERIMENT_NAME,
        "target": {
            "volatility_window": TARGET_WINDOW,
            "threshold_multiplier": TARGET_MULTIPLIER,
        },
        "periods": {
            "training_end": periods.training_end,
            "validation_start": periods.validation_start,
            "validation_end": periods.validation_end,
        },
        "source_prediction_path": prediction_path,
        "prediction_rows": int(len(predictions)),
        "development_move_rows": int(len(development_move)),
        "outer_validation_move_rows": int(len(enriched)),
        "regime_features": [
            {"family": spec.family, "feature": spec.feature}
            for spec in diagnostic.feature_specs
        ],
        "methodology": {
            "purpose": (
                "Diagnose temporal instability in the consumed Stage-2 outer-validation "
                "period without changing the locked model or target."
            ),
            "chronological_blocks": BLOCK_COUNT,
            "regime_threshold_policy": (
                "Tertile thresholds are calculated from development-period true MOVE "
                "rows only. No outer-validation values are used to set thresholds."
            ),
            "contrast_policy": (
                "Block 2 vs Block 1 standardized mean differences are descriptive only. "
                "They may generate hypotheses but may not directly define production "
                "regimes or select model features."
            ),
            "minimum_rows_for_conditional_auc": MINIMUM_AUC_ROWS,
            "model_refit": False,
            "hyperparameter_search": False,
            "feature_search": False,
            "validation_threshold_tuning": False,
        },
        "outputs": paths,
        "held_out_test_evaluated": False,
    }
    with paths["experiment"].open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=json_default)

    print_block_diagnostics(block_diagnostics)
    print_top_contrasts(block_contrasts)
    print_regime_auc_extremes(regime_auc)

    print()
    print("=" * 78)
    print("REGIME DIAGNOSTIC COMPLETE")
    print("=" * 78)
    print("Regime features:", len(diagnostic.feature_columns))
    print("Development MOVE rows:", len(development_move))
    print("Outer-validation MOVE rows:", len(enriched))
    print("Experiment:", paths["experiment"])
    print("Block diagnostics:", paths["block_diagnostics"])
    print("Block contrasts:", paths["block_contrasts"])
    print("Conditional regime AUC:", paths["regime_auc"])
    print("Held-out test set was NOT evaluated.")
    print()
    print(
        "NEXT DECISION RULE: use these results only to formulate candidate regime "
        "definitions. Any regime model, threshold, feature subset, or abstention rule "
        "must be developed and selected inside the pre-validation development period."
    )


if __name__ == "__main__":
    main()
