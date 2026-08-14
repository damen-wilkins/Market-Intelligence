from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ValidationPeriods:
    training_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def split_development_and_outer_validation(
    dataframe: pd.DataFrame,
    periods: ValidationPeriods,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "target_date" not in dataframe.columns:
        raise ValueError("Dataframe must contain target_date.")
    data = dataframe.copy()
    data["target_date"] = pd.to_datetime(data["target_date"])
    data = data.sort_values("target_date").reset_index(drop=True)
    development = data.loc[
        data["target_date"] <= periods.training_end
    ].reset_index(drop=True)
    validation = data.loc[
        (data["target_date"] >= periods.validation_start)
        & (data["target_date"] <= periods.validation_end)
    ].reset_index(drop=True)
    if development.empty:
        raise ValueError("Development split is empty.")
    if validation.empty:
        raise ValueError("Outer-validation split is empty.")
    if development["target_date"].max() >= validation["target_date"].min():
        raise ValueError("Development and outer-validation periods overlap.")
    if validation["target_date"].max() > periods.validation_end:
        raise ValueError("Outer-validation split crossed into the held-out test period.")
    return development, validation


def classification_metrics(
    actual: np.ndarray,
    score: np.ndarray,
    threshold: float,
    weights: np.ndarray | None = None,
) -> dict:
    actual = np.asarray(actual, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if len(actual) != len(score):
        raise ValueError("Actual and score arrays must have equal length.")
    if len(np.unique(actual)) < 2:
        raise ValueError("Both UP and DOWN classes are required for evaluation.")
    predicted = (score >= float(threshold)).astype(np.int64)
    if weights is None:
        weighted_accuracy = float(np.mean(predicted == actual))
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != len(actual):
            raise ValueError("Weights must align with actual values.")
        weighted_accuracy = (
            float(np.average(predicted == actual, weights=weights))
            if float(weights.sum()) > 0.0
            else float(np.mean(predicted == actual))
        )
    return {
        "roc_auc": float(roc_auc_score(actual, score)),
        "inverted_roc_auc": float(roc_auc_score(actual, -score)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "sign_accuracy": float(np.mean(predicted == actual)),
        "magnitude_weighted_sign_accuracy": weighted_accuracy,
    }


def moving_block_bootstrap_auc_ci(
    actual: np.ndarray,
    score: np.ndarray,
    resamples: int = 2000,
    block_length: int = 20,
    random_state: int = 42,
) -> dict:
    actual = np.asarray(actual, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if len(actual) != len(score):
        raise ValueError("Actual and score arrays must have equal length.")
    if resamples <= 0 or block_length <= 0:
        raise ValueError("resamples and block_length must be positive.")
    if len(np.unique(actual)) < 2:
        raise ValueError("Both classes are required for AUC bootstrap.")
    point_auc = float(roc_auc_score(actual, score))
    rng = np.random.default_rng(random_state)
    n = len(actual)
    starts = np.arange(max(1, n - block_length + 1))
    values: list[float] = []
    for _ in range(int(resamples)):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.choice(starts))
            indices.extend(range(start, min(start + block_length, n)))
        sample_index = np.asarray(indices[:n], dtype=np.int64)
        sample_actual = actual[sample_index]
        if len(np.unique(sample_actual)) < 2:
            continue
        values.append(float(roc_auc_score(sample_actual, score[sample_index])))
    if not values:
        return {
            "auc": point_auc,
            "lower_95": point_auc,
            "upper_95": point_auc,
            "probability_auc_above_0_50": float(point_auc > 0.50),
            "valid_resamples": 0,
        }
    values_array = np.asarray(values, dtype=np.float64)
    return {
        "auc": point_auc,
        "lower_95": float(np.quantile(values_array, 0.025)),
        "upper_95": float(np.quantile(values_array, 0.975)),
        "probability_auc_above_0_50": float(np.mean(values_array > 0.50)),
        "valid_resamples": int(len(values_array)),
    }


def chronological_auc_blocks(
    actual: np.ndarray,
    score: np.ndarray,
    dates: pd.Series | pd.DatetimeIndex,
    block_count: int = 3,
) -> list[dict]:
    actual = np.asarray(actual, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    if not (len(actual) == len(score) == len(dates)):
        raise ValueError("Actual, score, and dates must align.")
    if block_count <= 0:
        raise ValueError("block_count must be positive.")
    indices = np.array_split(np.arange(len(actual)), block_count)
    rows: list[dict] = []
    for block_number, block_index in enumerate(indices, start=1):
        if len(block_index) == 0:
            continue
        block_actual = actual[block_index]
        block_score = score[block_index]
        auc = (
            float(roc_auc_score(block_actual, block_score))
            if len(np.unique(block_actual)) == 2
            else float("nan")
        )
        rows.append(
            {
                "block": block_number,
                "start": dates[block_index[0]],
                "end": dates[block_index[-1]],
                "rows": int(len(block_index)),
                "up_share": float(block_actual.mean()),
                "roc_auc": auc,
            }
        )
    return rows


def parameter_signature(parameters: dict) -> tuple:
    return tuple(sorted((str(key), value) for key, value in parameters.items()))
