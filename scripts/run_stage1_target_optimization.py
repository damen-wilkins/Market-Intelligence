from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from app.training.direction_feature_builder import DirectionFeatureBuilder
from app.training.experiment_tracker import ExperimentTracker
from app.training.hierarchical_stage_feature_research import (
    binary_probability_metrics,
)
from app.training.hierarchical_target_feature_research import (
    binary_metrics,
)
from app.training.hierarchical_xlstm_parameter_selector import (
    HierarchicalXLSTMParameterSelector,
)
from app.training.stage1_target_optimization import (
    Stage1TargetCandidate,
    add_neighborhood_statistics,
    build_target_grid,
    moving_block_bootstrap_auc,
    select_target_shortlist,
    target_stability_statistics,
)
from app.training.volatility_direction_label_builder import (
    VolatilityDirectionLabelBuilder,
)
from database.direction_training_data_repository import (
    DirectionTrainingDataRepository,
)
from scripts.run_stage1_long_history_optimization import (
    evaluate_candidate,
    train_stage1_fold,
)


TICKER = "SPY"
CV_SPLITS = 3
RANDOM_STATE = 42
BASE_FEATURE_COLUMNS = list(
    DirectionFeatureBuilder.BASE_FEATURE_COLUMNS
)
TARGET_STATE_FEATURE = "target_rolling_volatility"
STAGE1_FEATURE_COLUMNS = [
    *BASE_FEATURE_COLUMNS,
    TARGET_STATE_FEATURE,
]

REFERENCE_HIERARCHICAL_MODEL_PATH = Path(
    "models/xlstm_hierarchical_direction.pt"
)

EXPERIMENT_DIRECTORY = Path("experiments")
EXPERIMENT_NAME = "xlstm_stage1_target_optimization_v1"
MODEL_NAME = "xlstm_stage1_target_optimization_v1"
SCREENING_CHECKPOINT_PATH = (
    EXPERIMENT_DIRECTORY
    / "stage1_target_optimization_v1_screening_checkpoint.json"
)
OPTIMIZATION_PROGRESS_PATH = (
    EXPERIMENT_DIRECTORY
    / "stage1_target_optimization_v1_progress.json"
)
OPTUNA_STORAGE_URL = (
    "sqlite:///experiments/optuna_stage1_target_optimization_v1.db"
)

SCREENING_SHORTLIST_SIZE = 16
SCREENING_MAX_PER_WINDOW = 2
OPTUNA_TRIALS_PER_TARGET = 50
MAX_SELECTION_EPOCHS = 80
OPTUNA_PATIENCE = 10
BOOTSTRAP_RESAMPLES = 500
BOOTSTRAP_BLOCK_LENGTH = 20


class JsonStore:
    @staticmethod
    def load(
        path: Path,
        default,
    ):
        if not path.exists():
            return default
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    @staticmethod
    def save(
        path: Path,
        payload,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        temporary = path.with_suffix(
            path.suffix + ".tmp"
        )
        with temporary.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
                default=JsonStore._default,
            )
        temporary.replace(path)

    @staticmethod
    def _default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable."
        )


def parameter_signature(
    parameters: dict,
) -> str:
    payload = json.dumps(
        parameters,
        sort_keys=True,
    ).encode(
        "utf-8"
    )
    return hashlib.sha1(
        payload
    ).hexdigest()


def load_reference_contract() -> tuple[pd.Timestamp, dict]:
    if not REFERENCE_HIERARCHICAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            "The original hierarchical xLSTM artifact was not found. "
            "It is required to preserve the original train/validation boundary."
        )

    try:
        package = torch.load(
            REFERENCE_HIERARCHICAL_MODEL_PATH,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        package = torch.load(
            REFERENCE_HIERARCHICAL_MODEL_PATH,
            map_location="cpu",
        )

    metadata = package.get(
        "metadata",
        {}
    )
    training_period = metadata.get(
        "training_period",
        {}
    )
    stage1_selection = metadata.get(
        "stage1_selection",
        {}
    )
    end_date = training_period.get(
        "end"
    )
    parameters = stage1_selection.get(
        "parameters"
    )

    if end_date is None:
        raise ValueError(
            "Original hierarchical artifact does not contain the training end date."
        )
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError(
            "Original hierarchical artifact does not contain Stage-1 parameters."
        )

    return (
        pd.Timestamp(end_date),
        dict(parameters),
    )


def build_common_feature_frame(
    raw_data: pd.DataFrame,
) -> pd.DataFrame:
    features = DirectionFeatureBuilder(
        feature_scope="base"
    ).build(
        raw_data
    )

    return features.rename(
        columns={
            "trade_date": "feature_date",
        }
    )


def build_target_dataset(
    candidate: Stage1TargetCandidate,
    raw_data: pd.DataFrame,
    base_features: pd.DataFrame,
    common_target_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    labels = VolatilityDirectionLabelBuilder(
        volatility_window=(
            candidate.volatility_window
        ),
        threshold_multiplier=(
            candidate.threshold_multiplier
        ),
    ).build(
        raw_data[
            [
                "trade_date",
                "close",
            ]
        ].copy()
    )

    dataset = base_features.merge(
        labels,
        on="feature_date",
        how="inner",
        validate="one_to_one",
    )

    dataset[TARGET_STATE_FEATURE] = dataset[
        "rolling_volatility"
    ].astype(float)

    dataset = dataset.loc[
        dataset["target_date"].isin(
            common_target_dates
        )
    ].sort_values(
        "target_date"
    ).reset_index(
        drop=True
    )

    if len(dataset) != len(
        common_target_dates
    ):
        raise ValueError(
            f"Target {candidate.name} did not align to all common training dates."
        )

    observed_dates = pd.DatetimeIndex(
        pd.to_datetime(
            dataset["target_date"]
        )
    )
    if not observed_dates.equals(
        common_target_dates
    ):
        raise ValueError(
            f"Target {candidate.name} has misaligned target dates."
        )

    return dataset[
        [
            "feature_date",
            "target_date",
            *STAGE1_FEATURE_COLUMNS,
            "future_log_return",
            "rolling_volatility",
            "threshold",
            "direction",
        ]
    ]


def resolve_common_target_dates(
    raw_data: pd.DataFrame,
    base_features: pd.DataFrame,
    training_end_date: pd.Timestamp,
    target_grid: list[Stage1TargetCandidate],
) -> pd.DatetimeIndex:
    maximum_window = max(
        candidate.volatility_window
        for candidate in target_grid
    )

    anchor_labels = VolatilityDirectionLabelBuilder(
        volatility_window=maximum_window,
        threshold_multiplier=0.10,
    ).build(
        raw_data[
            [
                "trade_date",
                "close",
            ]
        ].copy()
    )

    anchor = base_features[
        [
            "feature_date",
        ]
    ].merge(
        anchor_labels[
            [
                "feature_date",
                "target_date",
            ]
        ],
        on="feature_date",
        how="inner",
        validate="one_to_one",
    )

    anchor["target_date"] = pd.to_datetime(
        anchor["target_date"]
    )
    anchor = anchor.loc[
        anchor["target_date"]
        <= training_end_date
    ].sort_values(
        "target_date"
    )

    dates = pd.DatetimeIndex(
        anchor["target_date"]
    )

    if dates.empty:
        raise ValueError(
            "No common target-optimization training dates were found."
        )
    if dates.duplicated().any():
        raise ValueError(
            "Common target-optimization dates contain duplicates."
        )

    return dates


def screening_metadata(
    training_end_date: pd.Timestamp,
    common_target_dates: pd.DatetimeIndex,
    screening_parameters: dict,
    target_grid: list[Stage1TargetCandidate],
) -> dict:
    return {
        "version": 1,
        "training_end_date": training_end_date.strftime(
            "%Y-%m-%d"
        ),
        "common_start_date": pd.Timestamp(
            common_target_dates.min()
        ).strftime(
            "%Y-%m-%d"
        ),
        "common_end_date": pd.Timestamp(
            common_target_dates.max()
        ).strftime(
            "%Y-%m-%d"
        ),
        "common_rows": int(
            len(common_target_dates)
        ),
        "target_count": int(
            len(target_grid)
        ),
        "screening_parameter_signature": parameter_signature(
            screening_parameters
        ),
        "cv_splits": CV_SPLITS,
    }


def load_screening_checkpoint(
    expected_metadata: dict,
) -> dict[str, dict]:
    payload = JsonStore.load(
        SCREENING_CHECKPOINT_PATH,
        None,
    )

    if payload is None:
        return {}

    if payload.get(
        "metadata"
    ) != expected_metadata:
        raise ValueError(
            "Existing Stage-1 target screening checkpoint was created with "
            "different settings. Remove it before starting a new search."
        )

    return {
        row["target_name"]: row
        for row in payload.get(
            "results",
            [],
        )
    }


def save_screening_checkpoint(
    metadata: dict,
    results: dict[str, dict],
) -> None:
    JsonStore.save(
        SCREENING_CHECKPOINT_PATH,
        {
            "metadata": metadata,
            "results": list(
                results.values()
            ),
        },
    )


def run_screening(
    raw_data: pd.DataFrame,
    base_features: pd.DataFrame,
    common_target_dates: pd.DatetimeIndex,
    target_grid: list[Stage1TargetCandidate],
    screening_parameters: dict,
    training_end_date: pd.Timestamp,
) -> pd.DataFrame:
    metadata = screening_metadata(
        training_end_date=training_end_date,
        common_target_dates=common_target_dates,
        screening_parameters=screening_parameters,
        target_grid=target_grid,
    )
    completed = load_screening_checkpoint(
        metadata
    )

    print()
    print(
        "=" * 72
    )
    print(
        "PHASE 1 - EXHAUSTIVE FLAT-TARGET SCREENING"
    )
    print(
        "=" * 72
    )
    print(
        f"Targets: {len(target_grid)}"
    )
    print(
        f"Common rows: {len(common_target_dates)}"
    )
    print(
        "Every target uses the exact same dates, base features plus its "
        "own causal rolling-volatility state, and one fixed Stage-1 parameter set."
    )

    for index, candidate in enumerate(
        target_grid,
        start=1,
    ):
        if candidate.name in completed:
            print(
                f"[{index}/{len(target_grid)}] {candidate.name} -- already completed"
            )
            continue

        print()
        print(
            f"[{index}/{len(target_grid)}] {candidate.name}"
        )
        print(
            f"  target: {candidate.volatility_window}d x "
            f"{candidate.threshold_multiplier:.3f}"
        )

        dataset = build_target_dataset(
            candidate=candidate,
            raw_data=raw_data,
            base_features=base_features,
            common_target_dates=common_target_dates,
        )
        stability = target_stability_statistics(
            dataset
        )

        print(
            "  DOWN / FLAT / UP: "
            f"{stability['down_share'] * 100:.1f}% / "
            f"{stability['flat_share'] * 100:.1f}% / "
            f"{stability['up_share'] * 100:.1f}%"
        )

        metrics = evaluate_candidate(
            training_data=dataset,
            feature_columns=STAGE1_FEATURE_COLUMNS,
            parameters=screening_parameters,
        )

        row = {
            "target_name": candidate.name,
            "volatility_window": int(
                candidate.volatility_window
            ),
            "threshold_multiplier": float(
                candidate.threshold_multiplier
            ),
            **stability,
            **metrics,
        }
        completed[
            candidate.name
        ] = row

        print(
            "  ROC AUC:",
            round(
                row["stage1_roc_auc"],
                4,
            ),
        )
        print(
            "  Balanced accuracy:",
            round(
                row[
                    "stage1_balanced_accuracy"
                ],
                4,
            ),
        )
        print(
            "  FLAT F1:",
            round(
                row["stage1_flat_f1"],
                4,
            ),
        )
        print(
            "  AUC fold std:",
            round(
                row[
                    "stage1_roc_auc_fold_std"
                ],
                4,
            ),
        )

        save_screening_checkpoint(
            metadata=metadata,
            results=completed,
        )

    summary = pd.DataFrame(
        completed.values()
    )
    summary = add_neighborhood_statistics(
        summary
    )
    summary = summary.sort_values(
        [
            "stage1_roc_auc",
            "stage1_roc_auc_fold_std",
            "stage1_flat_f1",
        ],
        ascending=[
            False,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return summary


def study_name(
    candidate: Stage1TargetCandidate,
) -> str:
    return (
        "stage1_target_v1_"
        f"{candidate.volatility_window}d_"
        f"k{int(round(candidate.threshold_multiplier * 1000)):03d}"
    )


def optimize_target(
    candidate: Stage1TargetCandidate,
    dataset: pd.DataFrame,
) -> dict:
    selector = HierarchicalXLSTMParameterSelector(
        feature_columns=STAGE1_FEATURE_COLUMNS,
        task="move",
        n_splits=CV_SPLITS,
        n_trials=OPTUNA_TRIALS_PER_TARGET,
        max_epochs=MAX_SELECTION_EPOCHS,
        patience=OPTUNA_PATIENCE,
        random_state=RANDOM_STATE,
        objective_metric="roc_auc",
        study_name=study_name(
            candidate
        ),
        storage_url=OPTUNA_STORAGE_URL,
    )

    return selector.select_best_parameters(
        training_data=dataset
    )


def collect_oof_probabilities(
    training_data: pd.DataFrame,
    parameters: dict,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = TimeSeriesSplit(
        n_splits=CV_SPLITS
    )
    actual_batches = []
    probability_batches = []

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        splitter.split(
            training_data
        ),
        start=1,
    ):
        fold_train = (
            training_data.iloc[
                train_indices
            ].reset_index(
                drop=True
            )
        )
        fold_validation = (
            training_data.iloc[
                validation_indices
            ].reset_index(
                drop=True
            )
        )

        batch = train_stage1_fold(
            fold_train=fold_train,
            fold_validation=(
                fold_validation
            ),
            feature_columns=(
                STAGE1_FEATURE_COLUMNS
            ),
            parameters=parameters,
            seed=(
                RANDOM_STATE
                + 20000
                + fold_number
            ),
        )
        actual_batches.append(
            batch["actual"]
        )
        probability_batches.append(
            batch[
                "move_probabilities"
            ]
        )

    return (
        np.concatenate(
            actual_batches
        ).astype(
            np.int64
        ),
        np.concatenate(
            probability_batches
        ).astype(
            np.float64
        ),
    )


def optimized_progress() -> dict[str, dict]:
    payload = JsonStore.load(
        OPTIMIZATION_PROGRESS_PATH,
        [],
    )
    if not isinstance(payload, list):
        raise ValueError(
            "Stage-1 target optimization progress must contain a list."
        )
    return {
        row["target_name"]: row
        for row in payload
    }


def save_optimized_progress(
    completed: dict[str, dict],
) -> None:
    JsonStore.save(
        OPTIMIZATION_PROGRESS_PATH,
        list(
            completed.values()
        ),
    )


def run_optimization(
    shortlist: pd.DataFrame,
    raw_data: pd.DataFrame,
    base_features: pd.DataFrame,
    common_target_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    completed = optimized_progress()

    print()
    print(
        "=" * 72
    )
    print(
        "PHASE 2 - FULL xLSTM OPTUNA RETUNING OF ROBUST TARGETS"
    )
    print(
        "=" * 72
    )
    print(
        f"Shortlisted targets: {len(shortlist)}"
    )
    print(
        f"Optuna trials per target: {OPTUNA_TRIALS_PER_TARGET}"
    )
    print(
        "SQLite storage is enabled. Interrupted Optuna targets resume."
    )

    for index, row in shortlist.iterrows():
        target_name = str(
            row["target_name"]
        )
        if target_name in completed:
            print(
                f"[{index + 1}/{len(shortlist)}] {target_name} -- already optimized"
            )
            continue

        candidate = Stage1TargetCandidate(
            name=target_name,
            volatility_window=int(
                row[
                    "volatility_window"
                ]
            ),
            threshold_multiplier=float(
                row[
                    "threshold_multiplier"
                ]
            ),
        )

        dataset = build_target_dataset(
            candidate=candidate,
            raw_data=raw_data,
            base_features=base_features,
            common_target_dates=common_target_dates,
        )

        print()
        print(
            f"[{index + 1}/{len(shortlist)}] {candidate.name}"
        )
        print(
            f"  target: {candidate.volatility_window}d x "
            f"{candidate.threshold_multiplier:.3f}"
        )
        print(
            f"  FLAT share: {float(row['flat_share']) * 100:.1f}%"
        )

        selection = optimize_target(
            candidate=candidate,
            dataset=dataset,
        )
        parameters = dict(
            selection["parameters"]
        )

        metrics = evaluate_candidate(
            training_data=dataset,
            feature_columns=STAGE1_FEATURE_COLUMNS,
            parameters=parameters,
        )

        actual, probabilities = (
            collect_oof_probabilities(
                training_data=dataset,
                parameters=parameters,
            )
        )
        probability_metrics = binary_probability_metrics(
            actual=actual,
            positive_probabilities=probabilities,
        )
        threshold_result = (
            HierarchicalXLSTMParameterSelector
            .select_probability_threshold(
                actual=actual,
                positive_probabilities=probabilities,
            )
        )
        predicted = (
            probabilities
            >= float(
                threshold_result[
                    "threshold"
                ]
            )
        ).astype(
            np.int64
        )
        class_metrics = binary_metrics(
            actual=actual,
            predicted=predicted,
            negative_name="FLAT",
            positive_name="MOVE",
        )
        bootstrap = moving_block_bootstrap_auc(
            actual=actual,
            probabilities=probabilities,
            block_length=BOOTSTRAP_BLOCK_LENGTH,
            n_resamples=BOOTSTRAP_RESAMPLES,
            random_state=(
                RANDOM_STATE
                + int(
                    candidate.volatility_window
                )
                * 1000
                + int(
                    round(
                        candidate.threshold_multiplier
                        * 1000
                    )
                )
            ),
        )

        result = {
            "target_name": candidate.name,
            "volatility_window": int(
                candidate.volatility_window
            ),
            "threshold_multiplier": float(
                candidate.threshold_multiplier
            ),
            "flat_share": float(
                row["flat_share"]
            ),
            "down_share": float(
                row["down_share"]
            ),
            "up_share": float(
                row["up_share"]
            ),
            "median_flat_boundary_percent": float(
                row[
                    "median_flat_boundary_percent"
                ]
            ),
            "flat_share_block_range": float(
                row[
                    "flat_share_block_range"
                ]
            ),
            "screening_roc_auc": float(
                row["stage1_roc_auc"]
            ),
            "screening_neighbor_roc_auc_mean": float(
                row[
                    "neighbor_roc_auc_mean"
                ]
            ),
            "screening_neighbor_roc_auc_min": float(
                row[
                    "neighbor_roc_auc_min"
                ]
            ),
            "selection": selection,
            "parameters": parameters,
            "optimized_roc_auc": float(
                probability_metrics[
                    "roc_auc"
                ]
            ),
            "optimized_average_precision": float(
                probability_metrics[
                    "average_precision"
                ]
            ),
            "optimized_brier_score": float(
                probability_metrics[
                    "brier_score"
                ]
            ),
            "optimized_balanced_accuracy": float(
                class_metrics[
                    "balanced_accuracy"
                ]
            ),
            "optimized_macro_f1": float(
                class_metrics[
                    "macro_f1"
                ]
            ),
            "optimized_flat_precision": float(
                class_metrics[
                    "per_class"
                ]["FLAT"]["precision"]
            ),
            "optimized_flat_recall": float(
                class_metrics[
                    "per_class"
                ]["FLAT"]["recall"]
            ),
            "optimized_flat_f1": float(
                class_metrics[
                    "per_class"
                ]["FLAT"]["f1"]
            ),
            "optimized_move_f1": float(
                class_metrics[
                    "per_class"
                ]["MOVE"]["f1"]
            ),
            "optimized_decision_threshold": float(
                threshold_result[
                    "threshold"
                ]
            ),
            "optimized_roc_auc_fold_std": float(
                metrics[
                    "stage1_roc_auc_fold_std"
                ]
            ),
            "bootstrap_auc_lower_95": float(
                bootstrap[
                    "lower_95"
                ]
            ),
            "bootstrap_auc_upper_95": float(
                bootstrap[
                    "upper_95"
                ]
            ),
            "bootstrap_auc_std": float(
                bootstrap[
                    "bootstrap_std"
                ]
            ),
            "bootstrap_valid_resamples": int(
                bootstrap[
                    "valid_resamples"
                ]
            ),
        }

        completed[
            candidate.name
        ] = result
        save_optimized_progress(
            completed
        )

        print(
            "  optimized ROC AUC:",
            round(
                result[
                    "optimized_roc_auc"
                ],
                4,
            ),
        )
        print(
            "  95% moving-block bootstrap AUC CI:",
            f"[{result['bootstrap_auc_lower_95']:.4f}, "
            f"{result['bootstrap_auc_upper_95']:.4f}]",
        )
        print(
            "  balanced accuracy:",
            round(
                result[
                    "optimized_balanced_accuracy"
                ],
                4,
            ),
        )
        print(
            "  FLAT F1:",
            round(
                result[
                    "optimized_flat_f1"
                ],
                4,
            ),
        )
        print(
            "  fold std:",
            round(
                result[
                    "optimized_roc_auc_fold_std"
                ],
                4,
            ),
        )

    rows = [
        completed[
            str(row["target_name"])
        ]
        for _, row in shortlist.iterrows()
        if str(
            row["target_name"]
        ) in completed
    ]

    return pd.DataFrame(rows)


def flattened_optimized_summary(
    optimized: pd.DataFrame,
) -> pd.DataFrame:
    if optimized.empty:
        return optimized

    rows = []
    for _, row in optimized.iterrows():
        parameters = row["parameters"]
        rows.append(
            {
                "target_name": row[
                    "target_name"
                ],
                "volatility_window": row[
                    "volatility_window"
                ],
                "threshold_multiplier": row[
                    "threshold_multiplier"
                ],
                "flat_share": row[
                    "flat_share"
                ],
                "median_flat_boundary_percent": row[
                    "median_flat_boundary_percent"
                ],
                "flat_share_block_range": row[
                    "flat_share_block_range"
                ],
                "screening_roc_auc": row[
                    "screening_roc_auc"
                ],
                "screening_neighbor_roc_auc_mean": row[
                    "screening_neighbor_roc_auc_mean"
                ],
                "optimized_roc_auc": row[
                    "optimized_roc_auc"
                ],
                "bootstrap_auc_lower_95": row[
                    "bootstrap_auc_lower_95"
                ],
                "bootstrap_auc_upper_95": row[
                    "bootstrap_auc_upper_95"
                ],
                "optimized_roc_auc_fold_std": row[
                    "optimized_roc_auc_fold_std"
                ],
                "optimized_balanced_accuracy": row[
                    "optimized_balanced_accuracy"
                ],
                "optimized_flat_precision": row[
                    "optimized_flat_precision"
                ],
                "optimized_flat_recall": row[
                    "optimized_flat_recall"
                ],
                "optimized_flat_f1": row[
                    "optimized_flat_f1"
                ],
                "optimized_move_f1": row[
                    "optimized_move_f1"
                ],
                "decision_threshold": row[
                    "optimized_decision_threshold"
                ],
                "sequence_length": parameters[
                    "sequence_length"
                ],
                "embedding_dim": parameters[
                    "embedding_dim"
                ],
                "num_blocks": parameters[
                    "num_blocks"
                ],
                "num_heads": parameters[
                    "num_heads"
                ],
                "epochs": parameters[
                    "epochs"
                ],
            }
        )

    summary = pd.DataFrame(rows)

    return summary.sort_values(
        [
            "bootstrap_auc_lower_95",
            "optimized_roc_auc",
            "optimized_balanced_accuracy",
            "optimized_flat_f1",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )


def select_recommended_target(
    optimized_summary: pd.DataFrame,
) -> pd.Series:
    robust = optimized_summary.loc[
        optimized_summary[
            "flat_share"
        ].between(
            0.25,
            0.55,
            inclusive="both",
        )
        & (
            optimized_summary[
                "flat_share_block_range"
            ]
            <= 0.15
        )
        & (
            optimized_summary[
                "optimized_roc_auc_fold_std"
            ]
            <= 0.03
        )
    ].copy()

    if robust.empty:
        robust = optimized_summary.copy()

    robust = robust.sort_values(
        [
            "bootstrap_auc_lower_95",
            "optimized_roc_auc",
            "screening_neighbor_roc_auc_mean",
            "optimized_balanced_accuracy",
            "optimized_flat_f1",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
        ],
    )

    return robust.iloc[0]


def save_outputs(
    screening_summary: pd.DataFrame,
    shortlist: pd.DataFrame,
    optimized_summary: pd.DataFrame,
    recommended: pd.Series,
    training_end_date: pd.Timestamp,
    common_target_dates: pd.DatetimeIndex,
    screening_parameters: dict,
) -> tuple[Path, Path, Path, Path]:
    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    screening_path = (
        EXPERIMENT_DIRECTORY
        / f"stage1_target_screening_v1_{timestamp}.csv"
    )
    shortlist_path = (
        EXPERIMENT_DIRECTORY
        / f"stage1_target_shortlist_v1_{timestamp}.csv"
    )
    optimized_path = (
        EXPERIMENT_DIRECTORY
        / f"stage1_target_optimized_v1_{timestamp}.csv"
    )

    screening_summary.to_csv(
        screening_path,
        index=False,
    )
    shortlist.to_csv(
        shortlist_path,
        index=False,
    )
    optimized_summary.to_csv(
        optimized_path,
        index=False,
    )

    experiment_path = ExperimentTracker(
        str(
            EXPERIMENT_DIRECTORY
        )
    ).save(
        experiment_name=EXPERIMENT_NAME,
        model_name=MODEL_NAME,
        parameters={
            "ticker": TICKER,
            "training_end_date": training_end_date.strftime(
                "%Y-%m-%d"
            ),
            "common_training_start": pd.Timestamp(
                common_target_dates.min()
            ).strftime(
                "%Y-%m-%d"
            ),
            "common_training_end": pd.Timestamp(
                common_target_dates.max()
            ).strftime(
                "%Y-%m-%d"
            ),
            "common_training_rows": int(
                len(common_target_dates)
            ),
            "feature_columns": STAGE1_FEATURE_COLUMNS,
            "target_grid": {
                "volatility_windows": [
                    5,
                    10,
                    15,
                    20,
                    30,
                    40,
                    60,
                    90,
                ],
                "threshold_multiplier_start": 0.10,
                "threshold_multiplier_end": 0.80,
                "threshold_multiplier_step": 0.025,
                "target_count": 232,
            },
            "screening_parameters": screening_parameters,
            "screening_shortlist_size": SCREENING_SHORTLIST_SIZE,
            "optuna_trials_per_target": OPTUNA_TRIALS_PER_TARGET,
            "optuna_storage_url": OPTUNA_STORAGE_URL,
            "cv_splits": CV_SPLITS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_block_length": BOOTSTRAP_BLOCK_LENGTH,
            "outer_validation_evaluated": False,
            "held_out_test_evaluated": False,
        },
        metrics={
            "recommended_target": recommended.to_dict(),
            "optimized_targets": optimized_summary.to_dict(
                orient="records"
            ),
        },
        features=STAGE1_FEATURE_COLUMNS,
    )

    return (
        screening_path,
        shortlist_path,
        optimized_path,
        experiment_path,
    )


def main() -> None:
    EXPERIMENT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading base SPY/VIX/VVIX data..."
    )
    raw_data = DirectionTrainingDataRepository().get_training_data(
        ticker=TICKER,
        include_breadth=False,
        include_cross_asset=False,
    )
    base_features = build_common_feature_frame(
        raw_data
    )

    (
        training_end_date,
        screening_parameters,
    ) = load_reference_contract()
    target_grid = build_target_grid()
    common_target_dates = resolve_common_target_dates(
        raw_data=raw_data,
        base_features=base_features,
        training_end_date=training_end_date,
        target_grid=target_grid,
    )

    print(
        f"Raw rows: {len(raw_data)}"
    )
    print(
        f"Base feature rows: {len(base_features)}"
    )
    print(
        "Locked training cutoff:",
        training_end_date.strftime(
            "%Y-%m-%d"
        ),
    )
    print(
        "Common target-search sample:",
        f"{len(common_target_dates)} rows",
        f"({pd.Timestamp(common_target_dates.min()).date()} -> "
        f"{pd.Timestamp(common_target_dates.max()).date()})",
    )
    print(
        "Dense target grid: 8 volatility windows x 29 multipliers = "
        f"{len(target_grid)} definitions"
    )
    print(
        f"Stage-1 feature count: {len(STAGE1_FEATURE_COLUMNS)} "
        "(21 base + target-window rolling volatility)"
    )
    print(
        "The original hierarchical split boundary is preserved exactly; "
        "outer validation and held-out test remain untouched."
    )

    screening_summary = run_screening(
        raw_data=raw_data,
        base_features=base_features,
        common_target_dates=common_target_dates,
        target_grid=target_grid,
        screening_parameters=screening_parameters,
        training_end_date=training_end_date,
    )

    shortlist = select_target_shortlist(
        screening_summary,
        shortlist_size=SCREENING_SHORTLIST_SIZE,
        max_per_window=SCREENING_MAX_PER_WINDOW,
    )

    print()
    print(
        "=" * 72
    )
    print(
        "ROBUST TARGET SHORTLIST"
    )
    print(
        "=" * 72
    )
    print(
        shortlist[
            [
                "target_name",
                "volatility_window",
                "threshold_multiplier",
                "flat_share",
                "stage1_roc_auc",
                "neighbor_roc_auc_mean",
                "stage1_roc_auc_fold_std",
                "stage1_balanced_accuracy",
                "stage1_flat_f1",
                "flat_share_block_range",
            ]
        ].to_string(
            index=False
        )
    )

    optimized = run_optimization(
        shortlist=shortlist,
        raw_data=raw_data,
        base_features=base_features,
        common_target_dates=common_target_dates,
    )
    optimized_summary = flattened_optimized_summary(
        optimized
    )
    recommended = select_recommended_target(
        optimized_summary
    )

    (
        screening_path,
        shortlist_path,
        optimized_path,
        experiment_path,
    ) = save_outputs(
        screening_summary=screening_summary,
        shortlist=shortlist,
        optimized_summary=optimized_summary,
        recommended=recommended,
        training_end_date=training_end_date,
        common_target_dates=common_target_dates,
        screening_parameters=screening_parameters,
    )

    print()
    print(
        "=" * 72
    )
    print(
        "STAGE-1 FLAT TARGET OPTIMIZATION - FINAL RANKING"
    )
    print(
        "=" * 72
    )
    print(
        optimized_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "RECOMMENDED DEVELOPMENT TARGET"
    )
    print(
        "  window:",
        int(
            recommended[
                "volatility_window"
            ]
        ),
    )
    print(
        "  k:",
        round(
            float(
                recommended[
                    "threshold_multiplier"
                ]
            ),
            3,
        ),
    )
    print(
        "  FLAT share:",
        f"{float(recommended['flat_share']) * 100:.1f}%",
    )
    print(
        "  optimized ROC AUC:",
        round(
            float(
                recommended[
                    "optimized_roc_auc"
                ]
            ),
            4,
        ),
    )
    print(
        "  bootstrap 95% AUC CI:",
        f"[{float(recommended['bootstrap_auc_lower_95']):.4f}, "
        f"{float(recommended['bootstrap_auc_upper_95']):.4f}]",
    )
    print(
        "  balanced accuracy:",
        round(
            float(
                recommended[
                    "optimized_balanced_accuracy"
                ]
            ),
            4,
        ),
    )
    print(
        "  FLAT F1:",
        round(
            float(
                recommended[
                    "optimized_flat_f1"
                ]
            ),
            4,
        ),
    )
    print()
    print(
        "This is a development recommendation, not a final test result."
    )
    print(
        "Outer validation was NOT evaluated."
    )
    print(
        "Held-out test set was NOT evaluated."
    )
    print()
    print(
        "Screening summary:",
        screening_path,
    )
    print(
        "Shortlist:",
        shortlist_path,
    )
    print(
        "Optimized targets:",
        optimized_path,
    )
    print(
        "Experiment:",
        experiment_path,
    )
    print(
        "Screening checkpoint:",
        SCREENING_CHECKPOINT_PATH,
    )
    print(
        "Optimization progress:",
        OPTIMIZATION_PROGRESS_PATH,
    )
    print(
        "Optuna DB:",
        OPTUNA_STORAGE_URL,
    )


if __name__ == "__main__":
    main()
