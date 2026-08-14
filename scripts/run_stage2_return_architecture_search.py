import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

from app.training.hierarchical_xlstm_parameter_selector import (
    HierarchicalXLSTMParameterSelector,
)
from app.training.stage2_conditioned_target_research import (
    moving_block_bootstrap_auc_delta,
    target_specs,
)
from scripts.run_stage2_conditioned_megasearch import (
    TARGET_STATE_FEATURE,
    base_columns,
    build_master,
    columns_for_groups,
    dataset,
    holdout_metrics,
    load_training_cutoff,
)
from app.training.stage2_return_architecture_research import (
    Stage2ReturnSequencePreprocessor,
    TorchReturnTrainer,
    XLSTMReturnParameterSelector,
    auc_orientation,
    normal_up_probability,
    regression_direction_metrics,
    verify_stage2_orientation,
    xgboost_parameter_search,
)
from app.training.xlstm_return_model import XLSTMReturnModel
from database.stage2_signal_data_repository import Stage2SignalDataRepository


TICKER = "SPY"
RANDOM_STATE = 42
OUTER_SPLITS = 3
INNER_SPLITS = 3
DEEP_TRIALS = 30
TREE_TRIALS = 75
MAX_EPOCHS = 80
PATIENCE = 8
BOOTSTRAP_RESAMPLES = 1500
BOOTSTRAP_BLOCK_LENGTH = 20
EXPERIMENT_DIRECTORY = Path("experiments")
DEEP_PROGRESS = EXPERIMENT_DIRECTORY / "stage2_return_architecture_deep_v1_progress.json"
TREE_PROGRESS = EXPERIMENT_DIRECTORY / "stage2_return_architecture_tree_v1_progress.json"
DEEP_STORAGE_URL = "sqlite:///experiments/optuna_stage2_return_architecture_deep_v1.db"
TREE_STORAGE_URL = "sqlite:///experiments/optuna_stage2_return_architecture_tree_v1.db"
FALLBACK_WINNER_GROUPS = ("breadth", "calendar", "interaction_consensus")
NORMALIZED_TARGET = "future_return_vol_units"
RAW_TARGET = "future_log_return"


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


def load_progress(path: Path) -> dict:
    if not path.exists():
        return {"rows": []}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=json_default)
    temp.replace(path)


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


def add_normalized_target(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    rolling = result["rolling_volatility"].astype(float)
    result[NORMALIZED_TARGET] = result[RAW_TARGET].astype(float) / rolling
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=[NORMALIZED_TARGET]).reset_index(drop=True)


def study_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value)


def outer_splits(data: pd.DataFrame):
    splitter = TimeSeriesSplit(n_splits=OUTER_SPLITS)
    for fold_number, (train_index, test_index) in enumerate(splitter.split(data), start=1):
        yield (
            fold_number,
            data.iloc[train_index].reset_index(drop=True),
            data.iloc[test_index].reset_index(drop=True),
        )


def actual_for_dates(data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    lookup = data.set_index(pd.to_datetime(data["target_date"]))
    selected = lookup.loc[dates].copy()
    if len(selected) != len(dates):
        raise ValueError("Could not align architecture predictions to target dates.")
    return selected


def classification_extra_metrics(actual_frame: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict:
    actual = (actual_frame["direction"].astype(str) == "UP").astype(int).to_numpy()
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    weights = np.abs(actual_frame[RAW_TARGET].astype(float).to_numpy())
    weighted_accuracy = float(np.average(predicted == actual, weights=weights)) if weights.sum() > 0 else float(np.mean(predicted == actual))
    return {
        "sign_accuracy": float(np.mean(predicted == actual)),
        "magnitude_weighted_sign_accuracy": weighted_accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
    }


def xlstm_binary_fold(
    architecture_name: str,
    outer_fold: int,
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    selector = HierarchicalXLSTMParameterSelector(
        feature_columns=feature_columns,
        task="direction",
        n_splits=INNER_SPLITS,
        n_trials=DEEP_TRIALS,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        random_state=RANDOM_STATE + outer_fold * 100,
        objective_metric="roc_auc",
        study_name=study_token(f"{architecture_name}_outer_{outer_fold}"),
        storage_url=DEEP_STORAGE_URL,
    )
    selection = selector.select_best_parameters(outer_train)
    result = holdout_metrics(
        development=outer_train,
        verification=outer_test,
        feature_columns=feature_columns,
        selection=selection,
        seed=RANDOM_STATE + 500000 + outer_fold,
    )
    actual_frame = actual_for_dates(outer_test, result["target_dates"])
    extra = classification_extra_metrics(
        actual_frame,
        result["probabilities"],
        float(selection["decision_threshold"]),
    )
    return {
        "architecture": architecture_name,
        "outer_fold": outer_fold,
        "cv_auc": float(selection["threshold_oof_roc_auc"]),
        "cv_auc_fold_std": float(selection["threshold_oof_roc_auc_fold_std"]),
        "decision_threshold": float(selection["decision_threshold"]),
        "verification_auc": float(result["roc_auc"]),
        "verification_inverted_auc": float(1.0 - result["roc_auc"]),
        "balanced_accuracy": extra["balanced_accuracy"],
        "macro_f1": extra["macro_f1"],
        "sign_accuracy": extra["sign_accuracy"],
        "magnitude_weighted_sign_accuracy": extra["magnitude_weighted_sign_accuracy"],
        "move_rows": int(result["move_rows"]),
        "actual": result["actual"].tolist(),
        "score": np.asarray(result["probabilities"], dtype=float).tolist(),
        "target_dates": [pd.Timestamp(value).isoformat() for value in result["target_dates"]],
    }


def build_return_model(parameters: dict, feature_count: int, output_mode: str) -> XLSTMReturnModel:
    return XLSTMReturnModel(
        input_size=feature_count,
        context_length=int(parameters["sequence_length"]),
        embedding_dim=int(parameters["embedding_dim"]),
        num_blocks=int(parameters["num_blocks"]),
        num_heads=int(parameters["num_heads"]),
        conv1d_kernel_size=int(parameters["conv1d_kernel_size"]),
        qkv_proj_blocksize=int(parameters["qkv_proj_blocksize"]),
        proj_factor=float(parameters["proj_factor"]),
        dropout=float(parameters["dropout"]),
        output_mode=output_mode,
    )


def xlstm_return_fold(
    architecture_name: str,
    outer_fold: int,
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    output_mode: str,
) -> dict:
    selector = XLSTMReturnParameterSelector(
        feature_columns=feature_columns,
        target_column=target_column,
        output_mode=output_mode,
        n_splits=INNER_SPLITS,
        n_trials=DEEP_TRIALS,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        random_state=RANDOM_STATE + outer_fold * 100,
        study_name=study_token(f"{architecture_name}_outer_{outer_fold}"),
        storage_url=DEEP_STORAGE_URL,
    )
    selection = selector.select(outer_train)
    parameters = selection["parameters"]
    preprocessor = Stage2ReturnSequencePreprocessor(
        feature_columns,
        int(parameters["sequence_length"]),
    ).fit(outer_train, target_column)
    training_sequences = preprocessor.build_training_sequences(outer_train, target_column)
    test_sequences = preprocessor.build_inference_sequences(outer_train, outer_test, target_column)
    loss_name = "gaussian_nll" if output_mode == "gaussian" else "huber"
    trainer = TorchReturnTrainer(
        learning_rate=float(parameters["learning_rate"]),
        batch_size=int(parameters["batch_size"]),
        max_epochs=int(parameters["epochs"]),
        patience=PATIENCE,
        weight_decay=float(parameters["weight_decay"]),
        gradient_clip=float(parameters["gradient_clip"]),
        loss_name=loss_name,
        seed=RANDOM_STATE + 600000 + outer_fold,
    )
    model = build_return_model(parameters, len(feature_columns), output_mode)
    model = trainer.fit_fixed_epochs(
        model,
        training_sequences["X"],
        training_sequences["y"],
        int(parameters["epochs"]),
    )
    prediction = trainer.predict(model, test_sequences["X"])
    predicted_target = preprocessor.inverse_target(prediction["mean"])
    actual_frame = actual_for_dates(outer_test, test_sequences["target_dates"])
    if target_column == NORMALIZED_TARGET:
        predicted_raw = predicted_target * actual_frame["rolling_volatility"].astype(float).to_numpy()
    else:
        predicted_raw = predicted_target
    if output_mode == "gaussian":
        predicted_scale = prediction["scale"] * preprocessor.target_scale
        direction_score = normal_up_probability(predicted_target, predicted_scale)
    else:
        direction_score = predicted_target
    metrics = regression_direction_metrics(
        actual_return=actual_frame[RAW_TARGET].astype(float).to_numpy(),
        predicted_return=predicted_raw,
        direction_score=direction_score,
    )
    actual_direction = (actual_frame["direction"].astype(str) == "UP").astype(int).to_numpy()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "architecture": architecture_name,
        "outer_fold": outer_fold,
        "cv_auc": float(selection["cv_auc"]),
        "cv_auc_fold_std": float(selection["cv_auc_fold_std"]),
        "decision_threshold": 0.0,
        "verification_auc": float(metrics["direct_auc"]),
        "verification_inverted_auc": float(metrics["inverted_auc"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "sign_accuracy": float(metrics["sign_accuracy"]),
        "magnitude_weighted_sign_accuracy": float(metrics["magnitude_weighted_sign_accuracy"]),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "return_correlation": float(metrics["return_correlation"]),
        "move_rows": int(len(actual_direction)),
        "actual": actual_direction.tolist(),
        "score": np.asarray(direction_score, dtype=float).tolist(),
        "target_dates": [pd.Timestamp(value).isoformat() for value in test_sequences["target_dates"]],
    }


def xgb_oof_threshold(training_data: pd.DataFrame, feature_columns: list[str], parameters: dict) -> float:
    move = training_data.loc[training_data["direction"] != "FLAT"].reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=INNER_SPLITS)
    actual_batches, score_batches = [], []
    for train_index, validation_index in splitter.split(move):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **parameters,
        )
        y_train = (move.iloc[train_index]["direction"] == "UP").astype(int)
        model.fit(move.iloc[train_index][feature_columns], y_train)
        actual_batches.append((move.iloc[validation_index]["direction"] == "UP").astype(int).to_numpy())
        score_batches.append(model.predict_proba(move.iloc[validation_index][feature_columns])[:, 1])
    actual = np.concatenate(actual_batches)
    score = np.concatenate(score_batches)
    return float(HierarchicalXLSTMParameterSelector.select_probability_threshold(actual, score)["threshold"])


def xgboost_fold(
    architecture_name: str,
    outer_fold: int,
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    feature_columns: list[str],
    task: str,
    target_column: str,
) -> dict:
    selection = xgboost_parameter_search(
        training_data=outer_train,
        feature_columns=feature_columns,
        target_column=target_column,
        task=task,
        n_splits=INNER_SPLITS,
        n_trials=TREE_TRIALS,
        random_state=RANDOM_STATE + outer_fold * 100,
        study_name=study_token(f"{architecture_name}_outer_{outer_fold}"),
        storage_url=TREE_STORAGE_URL,
    )
    parameters = selection["parameters"]
    train_move = outer_train.loc[outer_train["direction"] != "FLAT"].reset_index(drop=True)
    test_move = outer_test.loc[outer_test["direction"] != "FLAT"].reset_index(drop=True)
    actual = (test_move["direction"] == "UP").astype(int).to_numpy()
    if task == "classification":
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=RANDOM_STATE + 700000 + outer_fold,
            n_jobs=-1,
            **parameters,
        )
        model.fit(
            train_move[feature_columns],
            (train_move["direction"] == "UP").astype(int),
        )
        score = model.predict_proba(test_move[feature_columns])[:, 1]
        threshold = xgb_oof_threshold(outer_train, feature_columns, parameters)
        predicted = (score >= threshold).astype(int)
        weights = np.abs(test_move[RAW_TARGET].astype(float).to_numpy())
        result = {
            "verification_auc": float(roc_auc_score(actual, score)),
            "verification_inverted_auc": float(roc_auc_score(actual, -score)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
            "sign_accuracy": float(np.mean(predicted == actual)),
            "magnitude_weighted_sign_accuracy": float(np.average(predicted == actual, weights=weights)),
            "decision_threshold": threshold,
        }
    else:
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE + 700000 + outer_fold,
            n_jobs=-1,
            **parameters,
        )
        model.fit(train_move[feature_columns], train_move[target_column].astype(float))
        predicted_target = model.predict(test_move[feature_columns])
        if target_column == NORMALIZED_TARGET:
            predicted_raw = predicted_target * test_move["rolling_volatility"].astype(float).to_numpy()
        else:
            predicted_raw = predicted_target
        metrics = regression_direction_metrics(
            actual_return=test_move[RAW_TARGET].astype(float).to_numpy(),
            predicted_return=predicted_raw,
            direction_score=predicted_target,
        )
        score = predicted_target
        result = {
            "verification_auc": float(metrics["direct_auc"]),
            "verification_inverted_auc": float(metrics["inverted_auc"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "sign_accuracy": float(metrics["sign_accuracy"]),
            "magnitude_weighted_sign_accuracy": float(metrics["magnitude_weighted_sign_accuracy"]),
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "return_correlation": float(metrics["return_correlation"]),
            "decision_threshold": 0.0,
        }
    return {
        "architecture": architecture_name,
        "outer_fold": outer_fold,
        "cv_auc": float(selection["cv_auc"]),
        "cv_auc_fold_std": float(selection["cv_auc_fold_std"]),
        **result,
        "move_rows": int(len(actual)),
        "actual": actual.tolist(),
        "score": np.asarray(score, dtype=float).tolist(),
        "target_dates": [pd.Timestamp(value).isoformat() for value in test_move["target_date"]],
    }


def architecture_specs(family: str):
    deep = [
        ("xlstm_binary_base", "binary", "base", None, None),
        ("xlstm_binary_winner", "binary", "winner", None, None),
        ("xlstm_huber_raw_winner", "return", "winner", RAW_TARGET, "point"),
        ("xlstm_huber_volnorm_winner", "return", "winner", NORMALIZED_TARGET, "point"),
        ("xlstm_gaussian_volnorm_winner", "return", "winner", NORMALIZED_TARGET, "gaussian"),
    ]
    tree = [
        ("xgboost_binary_winner", "xgb", "winner", RAW_TARGET, "classification"),
        ("xgboost_regression_volnorm_winner", "xgb", "winner", NORMALIZED_TARGET, "regression"),
    ]
    if family == "deep":
        return deep
    if family == "tree":
        return tree
    return deep + tree


def run_family(family: str, data: pd.DataFrame, base_features: list[str], winner_features: list[str]):
    progress_path = DEEP_PROGRESS if family == "deep" else TREE_PROGRESS
    progress = load_progress(progress_path)
    completed = {(row["architecture"], int(row["outer_fold"])): row for row in progress.get("rows", [])}
    specs = architecture_specs(family)
    for architecture_name, kind, scope, target_column, mode in specs:
        features = base_features if scope == "base" else winner_features
        print()
        print("=" * 78)
        print(architecture_name)
        print("=" * 78)
        for fold_number, outer_train, outer_test in outer_splits(data):
            key = (architecture_name, fold_number)
            if key in completed:
                row = completed[key]
                print(f"outer fold {fold_number}/{OUTER_SPLITS} -- completed AUC {row['verification_auc']:.4f}")
                continue
            print(f"outer fold {fold_number}/{OUTER_SPLITS}")
            if kind == "binary":
                row = xlstm_binary_fold(
                    architecture_name,
                    fold_number,
                    outer_train,
                    outer_test,
                    features,
                )
            elif kind == "return":
                row = xlstm_return_fold(
                    architecture_name,
                    fold_number,
                    outer_train,
                    outer_test,
                    features,
                    target_column,
                    mode,
                )
            else:
                row = xgboost_fold(
                    architecture_name,
                    fold_number,
                    outer_train,
                    outer_test,
                    features,
                    mode,
                    target_column,
                )
            completed[key] = row
            save_progress(progress_path, {"rows": list(completed.values())})
            print(
                f"AUC {row['verification_auc']:.4f} | inverted {row['verification_inverted_auc']:.4f} | "
                f"balanced {row['balanced_accuracy']:.4f} | macro F1 {row['macro_f1']:.4f}"
            )


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["architecture"], []).append(row)
    control_rows = grouped.get("xlstm_binary_winner", [])
    if len(control_rows) != OUTER_SPLITS:
        raise RuntimeError("xlstm_binary_winner must be complete before final reporting.")
    control_actual = np.concatenate([np.asarray(row["actual"], dtype=np.int64) for row in control_rows])
    control_score = np.concatenate([np.asarray(row["score"], dtype=np.float64) for row in control_rows])
    control_dates = [date for row in control_rows for date in row["target_dates"]]
    summaries = []
    for architecture, architecture_rows in grouped.items():
        if len(architecture_rows) != OUTER_SPLITS:
            continue
        architecture_rows = sorted(architecture_rows, key=lambda row: int(row["outer_fold"]))
        actual = np.concatenate([np.asarray(row["actual"], dtype=np.int64) for row in architecture_rows])
        score = np.concatenate([np.asarray(row["score"], dtype=np.float64) for row in architecture_rows])
        dates = [date for row in architecture_rows for date in row["target_dates"]]
        if dates != control_dates or not np.array_equal(actual, control_actual):
            raise ValueError(f"{architecture} is not aligned to the binary xLSTM control.")
        orientation = auc_orientation(actual, score)
        if architecture == "xlstm_binary_winner":
            bootstrap = {
                "delta_auc": 0.0,
                "lower_95": 0.0,
                "upper_95": 0.0,
                "probability_delta_positive": 0.5,
            }
        else:
            bootstrap = moving_block_bootstrap_auc_delta(
                actual=actual,
                candidate_probabilities=score,
                baseline_probabilities=control_score,
                resamples=BOOTSTRAP_RESAMPLES,
                block_length=BOOTSTRAP_BLOCK_LENGTH,
                random_state=RANDOM_STATE,
            )
        summary = {
            "architecture": architecture,
            "nested_oof_auc": float(orientation["direct_auc"]),
            "nested_oof_inverted_auc": float(orientation["inverted_auc"]),
            "outer_fold_auc_mean": float(np.mean([row["verification_auc"] for row in architecture_rows])),
            "outer_fold_auc_std": float(np.std([row["verification_auc"] for row in architecture_rows], ddof=0)),
            "inner_cv_auc_mean": float(np.mean([row["cv_auc"] for row in architecture_rows])),
            "delta_auc_vs_xlstm_binary_winner": float(bootstrap["delta_auc"]),
            "delta_auc_lower_95": float(bootstrap["lower_95"]),
            "delta_auc_upper_95": float(bootstrap["upper_95"]),
            "probability_delta_positive": float(bootstrap["probability_delta_positive"]),
            "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in architecture_rows])),
            "macro_f1_mean": float(np.mean([row["macro_f1"] for row in architecture_rows])),
            "sign_accuracy_mean": float(np.mean([row["sign_accuracy"] for row in architecture_rows])),
            "magnitude_weighted_sign_accuracy_mean": float(np.mean([row["magnitude_weighted_sign_accuracy"] for row in architecture_rows])),
            "move_rows": int(len(actual)),
        }
        optional = ["mae", "rmse", "return_correlation"]
        for key in optional:
            values = [float(row[key]) for row in architecture_rows if key in row]
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda row: (
            -row["nested_oof_auc"],
            row["outer_fold_auc_std"],
        ),
    )


def report():
    deep = load_progress(DEEP_PROGRESS).get("rows", [])
    tree = load_progress(TREE_PROGRESS).get("rows", [])
    summaries = summarize_rows([*deep, *tree])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    summary_path = EXPERIMENT_DIRECTORY / f"stage2_return_architecture_nested_v1_{timestamp}.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print()
    print("=" * 78)
    print("STAGE-2 RETURN/DISTRIBUTION ARCHITECTURE SEARCH - FINAL RANKING")
    print("=" * 78)
    columns = [
        "architecture",
        "nested_oof_auc",
        "nested_oof_inverted_auc",
        "outer_fold_auc_mean",
        "outer_fold_auc_std",
        "delta_auc_vs_xlstm_binary_winner",
        "delta_auc_lower_95",
        "delta_auc_upper_95",
        "probability_delta_positive",
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "magnitude_weighted_sign_accuracy_mean",
    ]
    frame = pd.DataFrame(summaries)
    print(frame[columns].round(4).to_string(index=False))
    print()
    print("Summary:", summary_path)
    print("Deep progress:", DEEP_PROGRESS)
    print("Tree progress:", TREE_PROGRESS)
    print("Outer validation was NOT evaluated.")
    print("Held-out test set was NOT evaluated.")
    print(
        "This is nested walk-forward DEVELOPMENT research. The feature contract was "
        "selected in earlier development research, so absolute AUC still requires a "
        "fresh outer-validation gate before promotion."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["deep", "tree", "all", "report"], default="all")
    args = parser.parse_args()
    EXPERIMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if args.family == "report":
        report()
        return

    cutoff = load_training_cutoff()
    primary = next(spec for spec in target_specs() if spec.role == "primary")
    raw = Stage2SignalDataRepository().get_training_data(ticker=TICKER)
    master = build_master(raw, primary, cutoff)
    verify_stage2_orientation(dataset(master, base_columns()))
    winner_groups = latest_verified_winner_groups()
    winner_features = columns_for_groups(winner_groups)
    base_features = base_columns()
    research_data = add_normalized_target(dataset(master, winner_features))

    print("=" * 78)
    print("STAGE-2 RETURN/DISTRIBUTION ARCHITECTURE SEARCH V1")
    print("=" * 78)
    print("Target: 90d x 0.700")
    print("Winner feature groups:", list(winner_groups))
    print("Rows:", len(research_data))
    print("Dates:", pd.Timestamp(research_data["target_date"].min()).date(), "->", pd.Timestamp(research_data["target_date"].max()).date())
    print("Nested walk-forward outer folds:", OUTER_SPLITS)
    print("Inner tuning folds:", INNER_SPLITS)
    print("Deep Optuna trials per architecture/fold:", DEEP_TRIALS)
    print("Tree Optuna trials per architecture/fold:", TREE_TRIALS)
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Outer validation and held-out test are NOT evaluated.")

    if args.family in {"deep", "all"}:
        run_family("deep", research_data, base_features, winner_features)
    if args.family in {"tree", "all"}:
        run_family("tree", research_data, base_features, winner_features)
    if args.family == "all":
        report()


if __name__ == "__main__":
    main()
