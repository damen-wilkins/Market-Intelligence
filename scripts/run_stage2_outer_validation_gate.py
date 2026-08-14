from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

from app.training.hierarchical_xlstm_parameter_selector import (
    HierarchicalXLSTMParameterSelector,
)
from app.training.stage2_outer_validation_gate import (
    ValidationPeriods,
    chronological_auc_blocks,
    classification_metrics,
    moving_block_bootstrap_auc_ci,
    parameter_signature,
    split_development_and_outer_validation,
)
from app.training.stage2_wide_feature_builder import Stage2WideFeatureBuilder
from app.training.volatility_direction_label_builder import (
    VolatilityDirectionLabelBuilder,
)
from database.stage2_signal_data_repository import Stage2SignalDataRepository


TICKER = "SPY"
TARGET_WINDOW = 90
TARGET_MULTIPLIER = 0.700
TARGET_STATE_FEATURE = "target_rolling_volatility"
RAW_TARGET = "future_log_return"
NORMALIZED_TARGET = "future_return_vol_units"
WINNER_GROUPS = ("breadth", "calendar", "interaction_consensus")
REFERENCE_MODEL_PATH = Path("models/xlstm_hierarchical_direction.pt")
EXPERIMENT_DIRECTORY = Path("experiments")
OPTUNA_STORAGE_URL = "sqlite:///experiments/optuna_stage2_return_architecture_tree_v1.db"
ARCHITECTURES = {
    "xgboost_binary_winner": {
        "task": "classification",
        "target_column": RAW_TARGET,
    },
    "xgboost_regression_volnorm_winner": {
        "task": "regression",
        "target_column": NORMALIZED_TARGET,
    },
}
INNER_SPLITS = 3
RANDOM_STATE = 42
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_BLOCK_LENGTH = 20
STABILITY_BLOCKS = 3
PRIMARY_PROMOTION_AUC = 0.55


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


def feature_columns() -> list[str]:
    columns = Stage2WideFeatureBuilder.columns_for_groups(WINNER_GROUPS)
    columns.append(TARGET_STATE_FEATURE)
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate feature columns detected.")
    return columns


def build_research_data(
    raw_data: pd.DataFrame,
    periods: ValidationPeriods,
    columns: list[str],
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
    master[NORMALIZED_TARGET] = (
        master[RAW_TARGET].astype(float)
        / master["rolling_volatility"].astype(float)
    )
    master = master.replace([np.inf, -np.inf], np.nan)
    data = master.dropna(subset=[*columns, NORMALIZED_TARGET]).copy()
    keep = [
        "feature_date",
        "target_date",
        *columns,
        RAW_TARGET,
        NORMALIZED_TARGET,
        "rolling_volatility",
        "threshold",
        "direction",
    ]
    return data[keep].sort_values("target_date").reset_index(drop=True)


def load_presearched_parameter_sets(architecture_name: str) -> list[dict]:
    candidates: dict[tuple, dict] = {}
    for fold_number in range(1, 4):
        study_name = f"{architecture_name}_outer_{fold_number}"
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=OPTUNA_STORAGE_URL,
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Required prior Optuna study '{study_name}' was not found. "
                "Run the completed Stage-2 return architecture search first."
            ) from exc
        parameters = dict(study.best_trial.params)
        candidates[parameter_signature(parameters)] = {
            "parameters": parameters,
            "source_study": study_name,
            "source_inner_cv_auc": float(study.best_value),
        }
    return list(candidates.values())


def evaluate_parameters_cv(
    development: pd.DataFrame,
    columns: list[str],
    task: str,
    target_column: str,
    parameters: dict,
    random_state: int,
) -> dict:
    move = development.loc[development["direction"] != "FLAT"].reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=INNER_SPLITS)
    aucs: list[float] = []
    for train_index, validation_index in splitter.split(move):
        X_train = move.iloc[train_index][columns]
        X_validation = move.iloc[validation_index][columns]
        actual = (
            move.iloc[validation_index]["direction"] == "UP"
        ).astype(int).to_numpy()
        if task == "classification":
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
                **parameters,
            )
            model.fit(
                X_train,
                (move.iloc[train_index]["direction"] == "UP").astype(int),
            )
            score = model.predict_proba(X_validation)[:, 1]
        else:
            model = XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
                **parameters,
            )
            model.fit(
                X_train,
                move.iloc[train_index][target_column].astype(float),
            )
            score = model.predict(X_validation)
        aucs.append(float(roc_auc_score(actual, score)))
    return {
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs, ddof=0)),
        "fold_auc": aucs,
    }


def elect_frozen_parameters(
    architecture_name: str,
    development: pd.DataFrame,
    columns: list[str],
    task: str,
    target_column: str,
) -> dict:
    candidates = load_presearched_parameter_sets(architecture_name)
    evaluated = []
    for candidate_number, candidate in enumerate(candidates, start=1):
        metrics = evaluate_parameters_cv(
            development=development,
            columns=columns,
            task=task,
            target_column=target_column,
            parameters=candidate["parameters"],
            random_state=RANDOM_STATE + candidate_number * 100,
        )
        evaluated.append({**candidate, **metrics})
    ranked = sorted(
        evaluated,
        key=lambda row: (-row["cv_auc_mean"], row["cv_auc_std"]),
    )
    return {
        "selected": ranked[0],
        "evaluated_existing_parameter_sets": ranked,
    }


def classification_oof_threshold(
    development: pd.DataFrame,
    columns: list[str],
    parameters: dict,
) -> float:
    move = development.loc[development["direction"] != "FLAT"].reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=INNER_SPLITS)
    actual_batches: list[np.ndarray] = []
    score_batches: list[np.ndarray] = []
    for train_index, validation_index in splitter.split(move):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **parameters,
        )
        model.fit(
            move.iloc[train_index][columns],
            (move.iloc[train_index]["direction"] == "UP").astype(int),
        )
        actual_batches.append(
            (move.iloc[validation_index]["direction"] == "UP")
            .astype(int)
            .to_numpy()
        )
        score_batches.append(
            model.predict_proba(move.iloc[validation_index][columns])[:, 1]
        )
    actual = np.concatenate(actual_batches)
    score = np.concatenate(score_batches)
    threshold = HierarchicalXLSTMParameterSelector.select_probability_threshold(
        actual,
        score,
    )["threshold"]
    return float(threshold)


def evaluate_outer_validation(
    architecture_name: str,
    development: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    task: str,
    target_column: str,
    election: dict,
) -> tuple[dict, pd.DataFrame]:
    parameters = election["selected"]["parameters"]
    train_move = development.loc[development["direction"] != "FLAT"].reset_index(drop=True)
    validation_move = validation.loc[validation["direction"] != "FLAT"].reset_index(drop=True)
    actual = (validation_move["direction"] == "UP").astype(int).to_numpy()
    weights = np.abs(validation_move[RAW_TARGET].astype(float).to_numpy())
    if task == "classification":
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=RANDOM_STATE + 900000,
            n_jobs=-1,
            **parameters,
        )
        model.fit(
            train_move[columns],
            (train_move["direction"] == "UP").astype(int),
        )
        score = model.predict_proba(validation_move[columns])[:, 1]
        threshold = classification_oof_threshold(development, columns, parameters)
        predicted_raw = None
    else:
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE + 900000,
            n_jobs=-1,
            **parameters,
        )
        model.fit(train_move[columns], train_move[target_column].astype(float))
        score = model.predict(validation_move[columns])
        threshold = 0.0
        predicted_raw = (
            score * validation_move["rolling_volatility"].astype(float).to_numpy()
            if target_column == NORMALIZED_TARGET
            else score
        )
    metrics = classification_metrics(
        actual=actual,
        score=score,
        threshold=threshold,
        weights=weights,
    )
    bootstrap = moving_block_bootstrap_auc_ci(
        actual=actual,
        score=score,
        resamples=BOOTSTRAP_RESAMPLES,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        random_state=RANDOM_STATE,
    )
    blocks = chronological_auc_blocks(
        actual=actual,
        score=score,
        dates=validation_move["target_date"],
        block_count=STABILITY_BLOCKS,
    )
    metrics.update(
        {
            "architecture": architecture_name,
            "move_rows": int(len(validation_move)),
            "up_share": float(actual.mean()),
            "decision_threshold": float(threshold),
            "bootstrap_auc_lower_95": float(bootstrap["lower_95"]),
            "bootstrap_auc_upper_95": float(bootstrap["upper_95"]),
            "bootstrap_probability_auc_above_0_50": float(
                bootstrap["probability_auc_above_0_50"]
            ),
            "development_parameter_cv_auc": float(election["selected"]["cv_auc_mean"]),
            "development_parameter_cv_auc_std": float(election["selected"]["cv_auc_std"]),
            "parameter_source_study": election["selected"]["source_study"],
            "stability_block_auc_mean": float(
                np.nanmean([row["roc_auc"] for row in blocks])
            ),
            "stability_block_auc_std": float(
                np.nanstd([row["roc_auc"] for row in blocks], ddof=0)
            ),
            "primary_auc_gate_0_55": bool(metrics["roc_auc"] >= PRIMARY_PROMOTION_AUC),
            "bootstrap_lower_above_0_50": bool(bootstrap["lower_95"] > 0.50),
        }
    )
    predictions = pd.DataFrame(
        {
            "target_date": pd.to_datetime(validation_move["target_date"]),
            "actual_direction": validation_move["direction"].astype(str).to_numpy(),
            "actual_up": actual,
            "score": score,
            "predicted_up": (np.asarray(score) >= threshold).astype(int),
            "actual_future_log_return": validation_move[RAW_TARGET].astype(float).to_numpy(),
        }
    )
    if predicted_raw is not None:
        predictions["predicted_future_log_return"] = predicted_raw
    return {
        "metrics": metrics,
        "blocks": blocks,
        "parameters": parameters,
        "parameter_election": election,
    }, predictions


def main():
    EXPERIMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    periods = load_reference_periods()
    columns = feature_columns()
    raw = Stage2SignalDataRepository().get_training_data(ticker=TICKER)
    research_data = build_research_data(raw, periods, columns)
    development, validation = split_development_and_outer_validation(
        research_data,
        periods,
    )
    print("=" * 78)
    print("STAGE-2 ONE-SHOT OUTER VALIDATION GATE")
    print("=" * 78)
    print(f"Target: {TARGET_WINDOW}d x {TARGET_MULTIPLIER:.3f}")
    print("Feature groups:", list(WINNER_GROUPS))
    print(
        "Development:",
        pd.Timestamp(development["target_date"].min()).date(),
        "->",
        pd.Timestamp(development["target_date"].max()).date(),
    )
    print(
        "Outer validation:",
        pd.Timestamp(validation["target_date"].min()).date(),
        "->",
        pd.Timestamp(validation["target_date"].max()).date(),
    )
    print("Held-out test begins AFTER", periods.validation_end.date(), "and is NOT evaluated.")
    print()
    print(
        "No new Optuna search is performed. Only the parameter sets already selected "
        "inside the completed nested development experiment are eligible."
    )

    all_results: dict[str, dict] = {}
    summary_rows: list[dict] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    for architecture_name, spec in ARCHITECTURES.items():
        print()
        print("-" * 78)
        print(architecture_name)
        print("-" * 78)
        election = elect_frozen_parameters(
            architecture_name=architecture_name,
            development=development,
            columns=columns,
            task=spec["task"],
            target_column=spec["target_column"],
        )
        selected = election["selected"]
        print(
            "Frozen parameter set:",
            selected["source_study"],
            f"| train-only CV AUC {selected['cv_auc_mean']:.4f}",
        )
        result, predictions = evaluate_outer_validation(
            architecture_name=architecture_name,
            development=development,
            validation=validation,
            columns=columns,
            task=spec["task"],
            target_column=spec["target_column"],
            election=election,
        )
        metrics = result["metrics"]
        prediction_path = (
            EXPERIMENT_DIRECTORY
            / f"stage2_outer_validation_{architecture_name}_{timestamp}.csv"
        )
        predictions.to_csv(prediction_path, index=False)
        result["prediction_path"] = str(prediction_path)
        all_results[architecture_name] = result
        summary_rows.append(metrics)
        print(f"Outer-validation AUC: {metrics['roc_auc']:.4f}")
        print(
            "Bootstrap 95% AUC CI:",
            f"[{metrics['bootstrap_auc_lower_95']:.4f}, "
            f"{metrics['bootstrap_auc_upper_95']:.4f}]",
        )
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(
            "Magnitude-weighted sign accuracy:",
            f"{metrics['magnitude_weighted_sign_accuracy']:.4f}",
        )
        print(
            "3-block AUC:",
            ", ".join(
                "nan" if np.isnan(row["roc_auc"]) else f"{row['roc_auc']:.4f}"
                for row in result["blocks"]
            ),
        )
        print(
            "AUC >= 0.55 gate:",
            "PASS" if metrics["primary_auc_gate_0_55"] else "FAIL",
        )
        print(
            "Bootstrap lower bound > 0.50:",
            "PASS" if metrics["bootstrap_lower_above_0_50"] else "FAIL",
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["roc_auc", "bootstrap_auc_lower_95"],
        ascending=[False, False],
    )
    summary_path = EXPERIMENT_DIRECTORY / f"stage2_outer_validation_gate_v1_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    experiment_path = EXPERIMENT_DIRECTORY / f"stage2_outer_validation_gate_v1_{timestamp}.json"
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": {
            "volatility_window": TARGET_WINDOW,
            "threshold_multiplier": TARGET_MULTIPLIER,
        },
        "feature_groups": list(WINNER_GROUPS),
        "feature_count": len(columns),
        "periods": {
            "training_end": periods.training_end,
            "validation_start": periods.validation_start,
            "validation_end": periods.validation_end,
        },
        "selection_policy": (
            "No new hyperparameter search. Re-evaluate only the three parameter sets "
            "already selected by the nested development folds, elect one using "
            "development-only TimeSeriesSplit, fit on all development MOVE rows, then "
            "evaluate the locked outer-validation period once."
        ),
        "results": all_results,
        "summary_path": str(summary_path),
        "held_out_test_evaluated": False,
    }
    with experiment_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=json_default)

    print()
    print("=" * 78)
    print("OUTER VALIDATION FINAL RANKING")
    print("=" * 78)
    print(
        summary[
            [
                "architecture",
                "roc_auc",
                "bootstrap_auc_lower_95",
                "bootstrap_auc_upper_95",
                "balanced_accuracy",
                "macro_f1",
                "magnitude_weighted_sign_accuracy",
                "stability_block_auc_mean",
                "stability_block_auc_std",
                "primary_auc_gate_0_55",
                "bootstrap_lower_above_0_50",
            ]
        ].round(4).to_string(index=False)
    )
    print()
    print("Summary:", summary_path)
    print("Experiment:", experiment_path)
    print("Held-out test set was NOT evaluated.")
    print(
        "IMPORTANT: this outer-validation period has now been consumed as a one-shot "
        "gate. Do not tune features, targets, thresholds, or hyperparameters against "
        "these validation results."
    )


if __name__ == "__main__":
    main()
