from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class TargetCandidate:
    name: str
    volatility_window: int
    threshold_multiplier: float
    description: str


def build_target_candidates() -> list[TargetCandidate]:
    return [
        TargetCandidate(
            name="current_20d_k015",
            volatility_window=20,
            threshold_multiplier=0.15,
            description="Current benchmark target, roughly 15% FLAT.",
        ),
        TargetCandidate(
            name="flat20_20d_k020",
            volatility_window=20,
            threshold_multiplier=0.20,
            description="Approximately 20% FLAT on the training-period sensitivity study.",
        ),
        TargetCandidate(
            name="flat30_20d_k030",
            volatility_window=20,
            threshold_multiplier=0.30,
            description="Intermediate 20-day target around the 30% FLAT region.",
        ),
        TargetCandidate(
            name="flat30_40d_k030",
            volatility_window=40,
            threshold_multiplier=0.30,
            description="Approximately 30% FLAT with a slower volatility regime estimate.",
        ),
        TargetCandidate(
            name="flat40_20d_k045",
            volatility_window=20,
            threshold_multiplier=0.45,
            description="Approximately 40% FLAT on the training-period sensitivity study.",
        ),
        TargetCandidate(
            name="research_20d_k050",
            volatility_window=20,
            threshold_multiplier=0.50,
            description="Research-reference 0.5 x rolling-volatility neutral zone.",
        ),
    ]


def align_datasets_on_common_target_dates(
    datasets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    if not datasets:
        raise ValueError("At least one dataset is required.")

    common_dates = None

    for name, dataframe in datasets.items():
        if dataframe.empty:
            raise ValueError(f"Dataset {name} is empty.")

        if "target_date" not in dataframe.columns:
            raise ValueError(
                f"Dataset {name} does not contain target_date."
            )

        target_dates = pd.DatetimeIndex(
            pd.to_datetime(dataframe["target_date"])
        )

        if target_dates.duplicated().any():
            raise ValueError(
                f"Dataset {name} contains duplicate target dates."
            )

        date_set = set(target_dates)

        if common_dates is None:
            common_dates = date_set
        else:
            common_dates &= date_set

    if not common_dates:
        raise ValueError(
            "The candidate datasets do not share any common target dates."
        )

    common_index = pd.DatetimeIndex(
        sorted(common_dates)
    )

    aligned = {}

    for name, dataframe in datasets.items():
        current = dataframe.copy()
        current["target_date"] = pd.to_datetime(
            current["target_date"]
        )
        current = (
            current[
                current["target_date"].isin(common_index)
            ]
            .sort_values("target_date")
            .reset_index(drop=True)
        )

        if len(current) != len(common_index):
            raise ValueError(
                f"Dataset {name} could not be aligned one-to-one on common dates."
            )

        aligned[name] = current

    reference_dates = pd.DatetimeIndex(
        aligned[next(iter(aligned))]["target_date"]
    )

    for name, dataframe in aligned.items():
        dates = pd.DatetimeIndex(dataframe["target_date"])

        if not dates.equals(reference_dates):
            raise ValueError(
                f"Dataset {name} is not chronologically aligned."
            )

    return aligned


def target_distribution(
    dataframe: pd.DataFrame,
) -> dict:
    if "direction" not in dataframe.columns:
        raise ValueError("Dataframe does not contain direction.")

    counts = (
        dataframe["direction"]
        .value_counts()
        .reindex(
            ["DOWN", "FLAT", "UP"],
            fill_value=0,
        )
    )

    total = int(counts.sum())

    if total == 0:
        raise ValueError("Target distribution cannot be calculated on zero rows.")

    return {
        "rows": total,
        "down_count": int(counts["DOWN"]),
        "flat_count": int(counts["FLAT"]),
        "up_count": int(counts["UP"]),
        "down_share": float(counts["DOWN"] / total),
        "flat_share": float(counts["FLAT"] / total),
        "up_share": float(counts["UP"] / total),
    }


def binary_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    negative_name: str,
    positive_name: str,
) -> dict:
    actual = np.asarray(actual, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Binary labels must be one-dimensional.")

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted labels must contain the same rows.")

    if len(actual) == 0:
        raise ValueError("Binary metrics cannot be calculated on zero rows.")

    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=[0, 1],
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": {
            negative_name: {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            positive_name: {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        ).tolist(),
    }


def three_class_metrics(
    actual_directions: np.ndarray,
    predicted_directions: np.ndarray,
) -> dict:
    mapping = {
        "DOWN": 0,
        "FLAT": 1,
        "UP": 2,
    }

    actual = np.asarray(
        [mapping[str(value)] for value in actual_directions],
        dtype=np.int64,
    )
    predicted = np.asarray(
        [mapping[str(value)] for value in predicted_directions],
        dtype=np.int64,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=[0, 1, 2],
        zero_division=0,
    )

    class_names = ["DOWN", "FLAT", "UP"]

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=[0, 1, 2],
        ).tolist(),
    }


def compose_hierarchical_predictions(
    move_probabilities: np.ndarray,
    up_probabilities: np.ndarray,
    move_threshold: float,
    up_threshold: float,
) -> np.ndarray:
    move_probabilities = np.asarray(
        move_probabilities,
        dtype=np.float64,
    )
    up_probabilities = np.asarray(
        up_probabilities,
        dtype=np.float64,
    )

    if move_probabilities.ndim != 1 or up_probabilities.ndim != 1:
        raise ValueError("Probability arrays must be one-dimensional.")

    if len(move_probabilities) != len(up_probabilities):
        raise ValueError("Stage probabilities must contain the same rows.")

    if not 0.0 <= move_threshold <= 1.0:
        raise ValueError("MOVE threshold must be between zero and one.")

    if not 0.0 <= up_threshold <= 1.0:
        raise ValueError("UP threshold must be between zero and one.")

    predicted = np.full(
        len(move_probabilities),
        "FLAT",
        dtype=object,
    )

    move_mask = move_probabilities >= move_threshold
    up_mask = up_probabilities >= up_threshold

    predicted[move_mask & ~up_mask] = "DOWN"
    predicted[move_mask & up_mask] = "UP"

    return predicted


def load_latest_hierarchical_experiment(
    experiment_directory: str | Path = "experiments",
) -> tuple[Path, dict]:
    experiment_directory = Path(experiment_directory)
    matches = []

    for path in experiment_directory.glob(
        "xlstm_hierarchical_direction_*.json"
    ):
        with path.open("r", encoding="utf-8") as file:
            experiment = json.load(file)

        if experiment.get("experiment_name") != "xlstm_hierarchical_direction":
            continue

        parameters = experiment.get("parameters", {})

        if "stage1" not in parameters or "stage2" not in parameters:
            continue

        matches.append(
            (
                str(experiment.get("timestamp_utc", "")),
                path,
                experiment,
            )
        )

    if not matches:
        raise FileNotFoundError(
            "No completed hierarchical xLSTM experiment was found in experiments/."
        )

    matches.sort(key=lambda item: item[0])
    _, path, experiment = matches[-1]

    return path, experiment
