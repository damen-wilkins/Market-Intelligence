from datetime import datetime, timezone
from itertools import combinations
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit

from app.training.classification_evaluator import (
    ClassificationEvaluator,
)
from app.training.classification_sequence_preprocessor import (
    ClassificationSequencePreprocessor,
)
from app.training.date_aware_data_splitter import (
    DateAwareDataSplitter,
)
from app.training.direction_dataset_builder import (
    DirectionDatasetBuilder,
)
from app.training.direction_feature_builder import (
    DirectionFeatureBuilder,
)
from app.training.experiment_tracker import (
    ExperimentTracker,
)
from app.training.torch_classification_predictor import (
    TorchClassificationPredictor,
)
from app.training.torch_classification_trainer import (
    TorchClassificationTrainer,
)
from app.training.torch_classifier_serializer import (
    TorchClassifierSerializer,
)
from app.training.torch_reproducibility import (
    TorchReproducibility,
)
from app.training.xlstm_classifier_model import (
    XLSTMClassifier,
)
from database.direction_training_data_repository import (
    DirectionTrainingDataRepository,
)


TICKER = "SPY"
HORIZON = "1_day"

REFERENCE_EXPERIMENT_NAME = (
    "xlstm_direction_classifier"
)

REFERENCE_MODEL_NAME = (
    "xlstm_direction_v1"
)

MODEL_NAME = (
    "xlstm_cross_asset_group_selection"
)

MODEL_VERSION = (
    "xlstm_cross_asset_group_v1"
)

EXPERIMENT_DIRECTORY = Path(
    "experiments"
)

MODEL_PATH = Path(
    "models/"
    "xlstm_direction_classifier_cross_asset_selected.pt"
)

VALIDATION_OUTPUT_PATH = Path(
    "models/"
    "xlstm_direction_cross_asset_selected_"
    "validation_predictions.csv"
)

CV_SPLITS = 3
RANDOM_STATE = 42
TRAINER_PATIENCE = 8

TARGET_VOLATILITY_WINDOW = 20
TARGET_THRESHOLD_MULTIPLIER = 0.15


def format_date(
    value,
) -> str:
    return pd.Timestamp(
        value
    ).strftime(
        "%Y-%m-%d"
    )


def class_name(
    class_index: int,
) -> str:
    return (
        ClassificationSequencePreprocessor
        .REVERSE_CLASS_MAPPING[
            int(
                class_index
            )
        ]
    )


def load_reference_experiment() -> tuple[
    Path,
    dict,
]:
    matches = []

    for path in (
        EXPERIMENT_DIRECTORY.glob(
            "xlstm_direction_classifier_*.json"
        )
    ):
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            experiment = json.load(
                file
            )

        if (
            experiment.get(
                "experiment_name"
            )
            != REFERENCE_EXPERIMENT_NAME
        ):
            continue

        if (
            experiment.get(
                "model_name"
            )
            != REFERENCE_MODEL_NAME
        ):
            continue

        matches.append(
            (
                str(
                    experiment.get(
                        "timestamp_utc",
                        "",
                    )
                ),
                path,
                experiment,
            )
        )

    if not matches:
        raise FileNotFoundError(
            "No original xLSTM "
            "reference experiment was found."
        )

    matches.sort(
        key=lambda item: item[0]
    )

    _, path, experiment = (
        matches[-1]
    )

    return path, experiment


def validate_reference_experiment(
    experiment: dict,
) -> None:
    required_parameters = {
        "sequence_length",
        "embedding_dim",
        "num_blocks",
        "num_heads",
        "conv1d_kernel_size",
        "qkv_proj_blocksize",
        "proj_factor",
        "dropout",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "gradient_clip",
        "loss_name",
        "focal_gamma",
        "epochs",
    }

    parameters = experiment.get(
        "parameters",
        {},
    )

    missing_parameters = (
        required_parameters
        - set(
            parameters
        )
    )

    if missing_parameters:
        raise ValueError(
            "Reference xLSTM experiment "
            "is missing parameters: "
            f"{sorted(missing_parameters)}"
        )

    reference_features = list(
        experiment.get(
            "features",
            [],
        )
    )

    expected_features = list(
        DirectionFeatureBuilder
        .BASE_FEATURE_COLUMNS
    )

    if (
        reference_features
        != expected_features
    ):
        raise ValueError(
            "Reference xLSTM feature contract "
            "does not match "
            "DirectionFeatureBuilder."
            "BASE_FEATURE_COLUMNS."
        )


def build_group_candidates() -> list[
    dict
]:
    group_names = list(
        DirectionFeatureBuilder
        .CROSS_ASSET_FEATURE_GROUPS
    )

    candidates = []

    for group_count in range(
        len(
            group_names
        )
        + 1
    ):
        for selected_groups in combinations(
            group_names,
            group_count,
        ):
            features = []

            for group_name in (
                selected_groups
            ):
                features.extend(
                    DirectionFeatureBuilder
                    .CROSS_ASSET_FEATURE_GROUPS[
                        group_name
                    ]
                )

            candidates.append(
                {
                    "candidate": (
                        "baseline"
                        if not selected_groups
                        else "+".join(
                            selected_groups
                        )
                    ),
                    "groups": list(
                        selected_groups
                    ),
                    "cross_asset_features": (
                        features
                    ),
                }
            )

    return candidates


def build_model_config(
    parameters: dict,
    feature_count: int,
) -> dict:
    return {
        "input_size": (
            feature_count
        ),
        "context_length": int(
            parameters[
                "sequence_length"
            ]
        ),
        "embedding_dim": int(
            parameters[
                "embedding_dim"
            ]
        ),
        "num_blocks": int(
            parameters[
                "num_blocks"
            ]
        ),
        "num_heads": int(
            parameters[
                "num_heads"
            ]
        ),
        "conv1d_kernel_size": int(
            parameters[
                "conv1d_kernel_size"
            ]
        ),
        "qkv_proj_blocksize": int(
            parameters[
                "qkv_proj_blocksize"
            ]
        ),
        "proj_factor": float(
            parameters[
                "proj_factor"
            ]
        ),
        "dropout": float(
            parameters[
                "dropout"
            ]
        ),
        "num_classes": 3,
    }


def build_trainer(
    parameters: dict,
    seed: int,
) -> TorchClassificationTrainer:
    return (
        TorchClassificationTrainer(
            learning_rate=float(
                parameters[
                    "learning_rate"
                ]
            ),
            batch_size=int(
                parameters[
                    "batch_size"
                ]
            ),
            max_epochs=int(
                parameters[
                    "epochs"
                ]
            ),
            patience=(
                TRAINER_PATIENCE
            ),
            loss_name=str(
                parameters[
                    "loss_name"
                ]
            ),
            focal_gamma=float(
                parameters[
                    "focal_gamma"
                ]
            ),
            weight_decay=float(
                parameters[
                    "weight_decay"
                ]
            ),
            gradient_clip=float(
                parameters[
                    "gradient_clip"
                ]
            ),
            seed=seed,
            deterministic=True,
        )
    )


def cleanup_cuda() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_candidate_cv(
    training_data: pd.DataFrame,
    parameters: dict,
    candidate: dict,
) -> dict:
    feature_columns = [
        *DirectionFeatureBuilder
        .BASE_FEATURE_COLUMNS,
        *candidate[
            "cross_asset_features"
        ],
    ]

    splitter = TimeSeriesSplit(
        n_splits=CV_SPLITS
    )

    fold_metrics = []

    for (
        fold_number,
        (
            train_indices,
            validation_indices,
        ),
    ) in enumerate(
        splitter.split(
            training_data
        ),
        start=1,
    ):
        fold_train = (
            training_data
            .iloc[
                train_indices
            ]
            .reset_index(
                drop=True
            )
        )

        fold_validation = (
            training_data
            .iloc[
                validation_indices
            ]
            .reset_index(
                drop=True
            )
        )

        preprocessor = (
            ClassificationSequencePreprocessor(
                feature_columns=(
                    feature_columns
                ),
                sequence_length=int(
                    parameters[
                        "sequence_length"
                    ]
                ),
            )
        )

        preprocessor.fit(
            fold_train
        )

        training_sequences = (
            preprocessor
            .build_training_sequences(
                fold_train
            )
        )

        validation_sequences = (
            preprocessor
            .build_inference_sequences(
                history=fold_train,
                dataframe=(
                    fold_validation
                ),
            )
        )

        fold_seed = (
            RANDOM_STATE
            + fold_number
        )

        TorchReproducibility.configure(
            seed=fold_seed,
            deterministic=True,
        )

        model = XLSTMClassifier(
            **build_model_config(
                parameters,
                len(
                    feature_columns
                ),
            )
        )

        trainer = build_trainer(
            parameters,
            fold_seed,
        )

        training_result = (
            trainer.fit_fixed_epochs(
                model=model,
                X_train=(
                    training_sequences[
                        "X"
                    ]
                ),
                y_train=(
                    training_sequences[
                        "y"
                    ]
                ),
                epochs=int(
                    parameters[
                        "epochs"
                    ]
                ),
            )
        )

        prediction_result = (
            TorchClassificationPredictor(
                batch_size=int(
                    parameters[
                        "batch_size"
                    ]
                )
            )
            .predict(
                model=(
                    training_result[
                        "model"
                    ]
                ),
                X=(
                    validation_sequences[
                        "X"
                    ]
                ),
            )
        )

        metrics = (
            ClassificationEvaluator()
            .evaluate(
                actual=(
                    validation_sequences[
                        "y"
                    ]
                ),
                predicted=(
                    prediction_result[
                        "predictions"
                    ]
                ),
            )
        )

        fold_metrics.append(
            {
                "fold": (
                    fold_number
                ),
                "macro_f1": float(
                    metrics[
                        "macro_f1"
                    ]
                ),
                "balanced_accuracy": float(
                    metrics[
                        "balanced_accuracy"
                    ]
                ),
                "accuracy": float(
                    metrics[
                        "accuracy"
                    ]
                ),
                "down_f1": float(
                    metrics[
                        "per_class"
                    ][
                        "DOWN"
                    ][
                        "f1"
                    ]
                ),
                "flat_f1": float(
                    metrics[
                        "per_class"
                    ][
                        "FLAT"
                    ][
                        "f1"
                    ]
                ),
                "up_f1": float(
                    metrics[
                        "per_class"
                    ][
                        "UP"
                    ][
                        "f1"
                    ]
                ),
            }
        )

        del model
        del training_result
        del prediction_result
        del training_sequences
        del validation_sequences
        del preprocessor

        cleanup_cuda()

    def mean_metric(
        metric_name: str,
    ) -> float:
        return float(
            np.mean(
                [
                    fold[
                        metric_name
                    ]
                    for fold
                    in fold_metrics
                ]
            )
        )

    result = dict(
        candidate
    )

    result.update(
        {
            "feature_count": len(
                feature_columns
            ),
            "cv_macro_f1": (
                mean_metric(
                    "macro_f1"
                )
            ),
            "cv_macro_f1_std": float(
                np.std(
                    [
                        fold[
                            "macro_f1"
                        ]
                        for fold
                        in fold_metrics
                    ],
                    ddof=0,
                )
            ),
            "cv_balanced_accuracy": (
                mean_metric(
                    "balanced_accuracy"
                )
            ),
            "cv_accuracy": (
                mean_metric(
                    "accuracy"
                )
            ),
            "cv_down_f1": (
                mean_metric(
                    "down_f1"
                )
            ),
            "cv_flat_f1": (
                mean_metric(
                    "flat_f1"
                )
            ),
            "cv_up_f1": (
                mean_metric(
                    "up_f1"
                )
            ),
            "fold_metrics": (
                fold_metrics
            ),
        }
    )

    return result


def apply_baseline_comparison(
    results: list[dict],
) -> dict:
    baseline = next(
        result
        for result
        in results
        if not result[
            "groups"
        ]
    )

    baseline_fold_scores = [
        fold[
            "macro_f1"
        ]
        for fold
        in baseline[
            "fold_metrics"
        ]
    ]

    for result in results:
        result[
            "delta_macro_f1_vs_baseline"
        ] = float(
            result[
                "cv_macro_f1"
            ]
            - baseline[
                "cv_macro_f1"
            ]
        )

        result_fold_scores = [
            fold[
                "macro_f1"
            ]
            for fold
            in result[
                "fold_metrics"
            ]
        ]

        result[
            "folds_beating_baseline"
        ] = int(
            sum(
                result_score
                > baseline_score
                for (
                    result_score,
                    baseline_score,
                )
                in zip(
                    result_fold_scores,
                    baseline_fold_scores,
                )
            )
        )

        result[
            "passes_selection_gate"
        ] = bool(
            result[
                "groups"
            ]
            and result[
                "delta_macro_f1_vs_baseline"
            ]
            > 0.0
            and result[
                "folds_beating_baseline"
            ]
            >= 2
        )

    return baseline


def select_best_candidate(
    results: list[dict],
) -> dict | None:
    eligible = [
        result
        for result
        in results
        if result[
            "passes_selection_gate"
        ]
    ]

    if not eligible:
        return None

    return sorted(
        eligible,
        key=lambda result: (
            -result[
                "cv_macro_f1"
            ],
            len(
                result[
                    "cross_asset_features"
                ]
            ),
            result[
                "cv_macro_f1_std"
            ],
        ),
    )[0]


def fit_final_candidate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    parameters: dict,
    feature_columns: list[str],
) -> dict:
    preprocessor = (
        ClassificationSequencePreprocessor(
            feature_columns=(
                feature_columns
            ),
            sequence_length=int(
                parameters[
                    "sequence_length"
                ]
            ),
        )
    )

    preprocessor.fit(
        train
    )

    training_sequences = (
        preprocessor
        .build_training_sequences(
            train
        )
    )

    validation_sequences = (
        preprocessor
        .build_inference_sequences(
            history=train,
            dataframe=validation,
        )
    )

    TorchReproducibility.configure(
        seed=RANDOM_STATE,
        deterministic=True,
    )

    model_config = (
        build_model_config(
            parameters,
            len(
                feature_columns
            ),
        )
    )

    model = XLSTMClassifier(
        **model_config
    )

    training_result = (
        build_trainer(
            parameters,
            RANDOM_STATE,
        )
        .fit_fixed_epochs(
            model=model,
            X_train=(
                training_sequences[
                    "X"
                ]
            ),
            y_train=(
                training_sequences[
                    "y"
                ]
            ),
            epochs=int(
                parameters[
                    "epochs"
                ]
            ),
        )
    )

    prediction_result = (
        TorchClassificationPredictor(
            batch_size=int(
                parameters[
                    "batch_size"
                ]
            )
        )
        .predict(
            model=(
                training_result[
                    "model"
                ]
            ),
            X=(
                validation_sequences[
                    "X"
                ]
            ),
        )
    )

    metrics = (
        ClassificationEvaluator()
        .evaluate(
            actual=(
                validation_sequences[
                    "y"
                ]
            ),
            predicted=(
                prediction_result[
                    "predictions"
                ]
            ),
        )
    )

    return {
        "model": (
            training_result[
                "model"
            ]
        ),
        "model_config": (
            model_config
        ),
        "preprocessor": (
            preprocessor
        ),
        "class_weights": (
            training_result[
                "class_weights"
            ]
        ),
        "validation_sequences": (
            validation_sequences
        ),
        "prediction_result": (
            prediction_result
        ),
        "metrics": (
            metrics
        ),
    }


def build_validation_output(
    validation: pd.DataFrame,
    baseline_result: dict,
    selected_result: dict,
) -> pd.DataFrame:
    baseline_sequences = (
        baseline_result[
            "validation_sequences"
        ]
    )

    selected_sequences = (
        selected_result[
            "validation_sequences"
        ]
    )

    baseline_dates = (
        pd.DatetimeIndex(
            baseline_sequences[
                "target_dates"
            ]
        )
    )

    selected_dates = (
        pd.DatetimeIndex(
            selected_sequences[
                "target_dates"
            ]
        )
    )

    if not baseline_dates.equals(
        selected_dates
    ):
        raise ValueError(
            "Baseline and selected "
            "validation target dates "
            "do not align."
        )

    if not np.array_equal(
        baseline_sequences[
            "y"
        ],
        selected_sequences[
            "y"
        ],
    ):
        raise ValueError(
            "Baseline and selected "
            "validation labels "
            "do not align."
        )

    aligned_validation = (
        validation
        .set_index(
            "target_date"
        )
        .loc[
            selected_dates
        ]
    )

    baseline_predictions = (
        baseline_result[
            "prediction_result"
        ][
            "predictions"
        ]
    )

    selected_predictions = (
        selected_result[
            "prediction_result"
        ][
            "predictions"
        ]
    )

    baseline_probabilities = (
        baseline_result[
            "prediction_result"
        ][
            "probabilities"
        ]
    )

    selected_probabilities = (
        selected_result[
            "prediction_result"
        ][
            "probabilities"
        ]
    )

    return pd.DataFrame(
        {
            "feature_date": (
                selected_sequences[
                    "feature_dates"
                ]
            ),
            "target_date": (
                selected_dates
            ),
            "actual_direction": [
                class_name(
                    value
                )
                for value
                in selected_sequences[
                    "y"
                ]
            ],
            "baseline_predicted_direction": [
                class_name(
                    value
                )
                for value
                in baseline_predictions
            ],
            "selected_predicted_direction": [
                class_name(
                    value
                )
                for value
                in selected_predictions
            ],
            "baseline_prob_down": (
                baseline_probabilities[
                    :,
                    0,
                ]
            ),
            "baseline_prob_flat": (
                baseline_probabilities[
                    :,
                    1,
                ]
            ),
            "baseline_prob_up": (
                baseline_probabilities[
                    :,
                    2,
                ]
            ),
            "selected_prob_down": (
                selected_probabilities[
                    :,
                    0,
                ]
            ),
            "selected_prob_flat": (
                selected_probabilities[
                    :,
                    1,
                ]
            ),
            "selected_prob_up": (
                selected_probabilities[
                    :,
                    2,
                ]
            ),
            "future_log_return": (
                aligned_validation[
                    "future_log_return"
                ].to_numpy()
            ),
            "rolling_volatility": (
                aligned_validation[
                    "rolling_volatility"
                ].to_numpy()
            ),
            "threshold": (
                aligned_validation[
                    "threshold"
                ].to_numpy()
            ),
        }
    )


def build_summary_dataframe(
    results: list[dict],
) -> pd.DataFrame:
    rows = []

    for result in results:
        rows.append(
            {
                "candidate": (
                    result[
                        "candidate"
                    ]
                ),
                "groups": "|".join(
                    result[
                        "groups"
                    ]
                ),
                "cross_asset_features": (
                    "|".join(
                        result[
                            "cross_asset_features"
                        ]
                    )
                ),
                "feature_count": (
                    result[
                        "feature_count"
                    ]
                ),
                "cv_macro_f1": (
                    result[
                        "cv_macro_f1"
                    ]
                ),
                "cv_macro_f1_std": (
                    result[
                        "cv_macro_f1_std"
                    ]
                ),
                "delta_macro_f1_vs_baseline": (
                    result[
                        "delta_macro_f1_vs_baseline"
                    ]
                ),
                "folds_beating_baseline": (
                    result[
                        "folds_beating_baseline"
                    ]
                ),
                "passes_selection_gate": (
                    result[
                        "passes_selection_gate"
                    ]
                ),
                "cv_balanced_accuracy": (
                    result[
                        "cv_balanced_accuracy"
                    ]
                ),
                "cv_accuracy": (
                    result[
                        "cv_accuracy"
                    ]
                ),
                "cv_down_f1": (
                    result[
                        "cv_down_f1"
                    ]
                ),
                "cv_flat_f1": (
                    result[
                        "cv_flat_f1"
                    ]
                ),
                "cv_up_f1": (
                    result[
                        "cv_up_f1"
                    ]
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "cv_macro_f1",
                "feature_count",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def print_cv_summary(
    summary: pd.DataFrame,
) -> None:
    display = summary[
        [
            "candidate",
            "cv_macro_f1",
            "delta_macro_f1_vs_baseline",
            "folds_beating_baseline",
            "cv_down_f1",
            "cv_flat_f1",
            "cv_up_f1",
        ]
    ].copy()

    numeric_columns = [
        "cv_macro_f1",
        "delta_macro_f1_vs_baseline",
        "cv_down_f1",
        "cv_flat_f1",
        "cv_up_f1",
    ]

    display[
        numeric_columns
    ] = (
        display[
            numeric_columns
        ]
        .round(4)
    )

    print()
    print(
        "=============================================="
    )
    print(
        "xLSTM CROSS-ASSET GROUP CV RESULTS"
    )
    print(
        "=============================================="
    )

    print(
        display.to_string(
            index=False
        )
    )


def print_validation_comparison(
    baseline_metrics: dict,
    selected_metrics: dict,
) -> None:
    print()
    print(
        "=============================================="
    )
    print(
        "OUTER VALIDATION COMPARISON"
    )
    print(
        "=============================================="
    )

    print(
        "Baseline Macro F1:",
        round(
            baseline_metrics[
                "macro_f1"
            ],
            4,
        ),
    )

    print(
        "Selected Macro F1:",
        round(
            selected_metrics[
                "macro_f1"
            ],
            4,
        ),
    )

    print(
        "Macro F1 delta:",
        round(
            selected_metrics[
                "macro_f1"
            ]
            - baseline_metrics[
                "macro_f1"
            ],
            4,
        ),
    )

    print(
        "Baseline Balanced Accuracy:",
        round(
            baseline_metrics[
                "balanced_accuracy"
            ],
            4,
        ),
    )

    print(
        "Selected Balanced Accuracy:",
        round(
            selected_metrics[
                "balanced_accuracy"
            ],
            4,
        ),
    )

    print(
        "Baseline Accuracy:",
        round(
            baseline_metrics[
                "accuracy"
            ],
            4,
        ),
    )

    print(
        "Selected Accuracy:",
        round(
            selected_metrics[
                "accuracy"
            ],
            4,
        ),
    )

    print()
    print(
        "Per-class F1:"
    )

    for label in (
        "DOWN",
        "FLAT",
        "UP",
    ):
        baseline_f1 = (
            baseline_metrics[
                "per_class"
            ][
                label
            ][
                "f1"
            ]
        )

        selected_f1 = (
            selected_metrics[
                "per_class"
            ][
                label
            ][
                "f1"
            ]
        )

        print(
            f"{label}: "
            f"baseline={baseline_f1:.4f}, "
            f"selected={selected_f1:.4f}, "
            f"delta={selected_f1 - baseline_f1:+.4f}"
        )


def save_no_selection_experiment(
    reference_path: Path,
    parameters: dict,
    baseline: dict,
    results: list[dict],
) -> Path:
    return (
        ExperimentTracker(
            str(
                EXPERIMENT_DIRECTORY
            )
        )
        .save(
            experiment_name=(
                MODEL_NAME
            ),
            model_name=(
                MODEL_VERSION
            ),
            parameters={
                "reference_experiment": (
                    str(
                        reference_path
                    )
                ),
                "reference_parameters": (
                    parameters
                ),
                "cv_splits": (
                    CV_SPLITS
                ),
                "selection_gate": (
                    "Positive mean Macro F1 "
                    "delta versus controlled "
                    "baseline and improvement "
                    "in at least 2 of 3 folds."
                ),
            },
            metrics={
                "controlled_baseline": (
                    baseline
                ),
                "candidates": (
                    results
                ),
                "selected_candidate": (
                    None
                ),
            },
            features=list(
                DirectionFeatureBuilder
                .CROSS_ASSET_FEATURE_COLUMNS
            ),
        )
    )


def main():
    (
        reference_path,
        reference_experiment,
    ) = (
        load_reference_experiment()
    )

    validate_reference_experiment(
        reference_experiment
    )

    parameters = dict(
        reference_experiment[
            "parameters"
        ]
    )

    print(
        "Reference experiment:",
        reference_path,
    )

    print(
        "Reference xLSTM parameters "
        "are locked."
    )

    print(
        "No Optuna retuning will occur "
        "during cross-asset selection."
    )

    print()
    print(
        "Loading direct-classification dataset..."
    )

    raw_data = (
        DirectionTrainingDataRepository()
        .get_training_data(
            ticker=TICKER
        )
    )

    dataset = (
        DirectionDatasetBuilder()
        .build(
            raw_data
        )
    )

    train, validation, _ = (
        DateAwareDataSplitter()
        .split(
            dataset,
            date_column="target_date",
        )
    )

    train = (
        train
        .sort_values(
            "target_date"
        )
        .reset_index(
            drop=True
        )
    )

    validation = (
        validation
        .sort_values(
            "target_date"
        )
        .reset_index(
            drop=True
        )
    )

    candidates = (
        build_group_candidates()
    )

    print(
        f"Training rows: {len(train)}"
    )

    print(
        f"Validation rows: {len(validation)}"
    )

    print(
        "Cross-asset group combinations:",
        len(
            candidates
        ),
    )

    print(
        "Held-out test set will not "
        "be evaluated."
    )

    results = []

    for (
        candidate_number,
        candidate,
    ) in enumerate(
        candidates,
        start=1,
    ):
        print()

        print(
            f"[{candidate_number}/"
            f"{len(candidates)}] "
            f"Evaluating "
            f"{candidate['candidate']}"
        )

        result = (
            evaluate_candidate_cv(
                training_data=(
                    train
                ),
                parameters=(
                    parameters
                ),
                candidate=(
                    candidate
                ),
            )
        )

        results.append(
            result
        )

        print(
            "CV Macro F1:",
            round(
                result[
                    "cv_macro_f1"
                ],
                4,
            ),
        )

    controlled_baseline = (
        apply_baseline_comparison(
            results
        )
    )

    selected_candidate = (
        select_best_candidate(
            results
        )
    )

    summary = (
        build_summary_dataframe(
            results
        )
    )

    print_cv_summary(
        summary
    )

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )

    EXPERIMENT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        EXPERIMENT_DIRECTORY
        / (
            "xlstm_cross_asset_group_cv_"
            f"{timestamp}.csv"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    if selected_candidate is None:
        experiment_path = (
            save_no_selection_experiment(
                reference_path=(
                    reference_path
                ),
                parameters=(
                    parameters
                ),
                baseline=(
                    controlled_baseline
                ),
                results=(
                    results
                ),
            )
        )

        print()
        print(
            "No cross-asset group combination "
            "passed the train-only selection gate."
        )

        print(
            "Outer validation was NOT "
            "re-evaluated."
        )

        print(
            "Held-out test set was NOT evaluated."
        )

        print(
            "CV summary:",
            summary_path,
        )

        print(
            "Experiment:",
            experiment_path,
        )

        return

    selected_features = [
        *DirectionFeatureBuilder
        .BASE_FEATURE_COLUMNS,
        *selected_candidate[
            "cross_asset_features"
        ],
    ]

    print()
    print(
        "Selected cross-asset groups:"
    )

    for group_name in (
        selected_candidate[
            "groups"
        ]
    ):
        print(
            f"- {group_name}"
        )

    print()
    print(
        "Training controlled baseline "
        "on full training data..."
    )

    baseline_final = (
        fit_final_candidate(
            train=train,
            validation=validation,
            parameters=parameters,
            feature_columns=list(
                DirectionFeatureBuilder
                .BASE_FEATURE_COLUMNS
            ),
        )
    )

    baseline_model = (
        baseline_final.pop(
            "model"
        )
    )

    del baseline_model

    cleanup_cuda()

    print(
        "Training selected cross-asset "
        "candidate on full training data..."
    )

    selected_final = (
        fit_final_candidate(
            train=train,
            validation=validation,
            parameters=parameters,
            feature_columns=(
                selected_features
            ),
        )
    )

    baseline_metrics = (
        baseline_final[
            "metrics"
        ]
    )

    selected_metrics = (
        selected_final[
            "metrics"
        ]
    )

    print_validation_comparison(
        baseline_metrics,
        selected_metrics,
    )

    validation_output = (
        build_validation_output(
            validation=validation,
            baseline_result=(
                baseline_final
            ),
            selected_result=(
                selected_final
            ),
        )
    )

    VALIDATION_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_output.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
    )

    metadata = {
        "model_version": (
            MODEL_VERSION
        ),
        "trained_at_utc": (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        ),
        "ticker": TICKER,
        "horizon": HORIZON,
        "architecture": (
            "Official NX-AI "
            "xLSTMBlockStack using "
            "mLSTM blocks"
        ),
        "reference_experiment": (
            str(
                reference_path
            )
        ),
        "reference_parameters": (
            parameters
        ),
        "feature_selection": {
            "method": (
                "All 16 combinations of "
                "four cross-asset feature "
                "groups using fixed reference "
                "xLSTM parameters and "
                "train-only 3-fold "
                "walk-forward CV."
            ),
            "selection_gate": (
                "Positive mean Macro F1 delta "
                "versus controlled baseline "
                "and improvement in at least "
                "2 of 3 folds."
            ),
            "selected_groups": (
                selected_candidate[
                    "groups"
                ]
            ),
            "selected_cross_asset_features": (
                selected_candidate[
                    "cross_asset_features"
                ]
            ),
            "selected_cv_result": (
                selected_candidate
            ),
        },
        "features": (
            selected_features
        ),
        "feature_count": len(
            selected_features
        ),
        "target": {
            "classes": [
                "DOWN",
                "FLAT",
                "UP",
            ],
            "volatility_window": (
                TARGET_VOLATILITY_WINDOW
            ),
            "threshold_multiplier": (
                TARGET_THRESHOLD_MULTIPLIER
            ),
        },
        "training_period": {
            "start": format_date(
                train[
                    "target_date"
                ].min()
            ),
            "end": format_date(
                train[
                    "target_date"
                ].max()
            ),
        },
        "validation_period": {
            "start": format_date(
                validation[
                    "target_date"
                ].min()
            ),
            "end": format_date(
                validation[
                    "target_date"
                ].max()
            ),
        },
        "controlled_baseline_validation_metrics": (
            baseline_metrics
        ),
        "selected_validation_metrics": (
            selected_metrics
        ),
        "class_weights": (
            selected_final[
                "class_weights"
            ].tolist()
        ),
        "experiment_notes": (
            "Cross-asset group selection "
            "was performed only on the "
            "training period. The original "
            "xLSTM hyperparameters were "
            "locked. The controlled baseline "
            "and candidate use the same "
            "common cross-asset sample. "
            "Outer validation was used only "
            "after group selection. Held-out "
            "test data was not evaluated."
        ),
    }

    model_path = (
        TorchClassifierSerializer
        .save(
            model=(
                selected_final[
                    "model"
                ]
            ),
            model_type=(
                "xlstm"
            ),
            model_config=(
                selected_final[
                    "model_config"
                ]
            ),
            preprocessor=(
                selected_final[
                    "preprocessor"
                ]
            ),
            metadata=(
                metadata
            ),
            filepath=str(
                MODEL_PATH
            ),
        )
    )

    experiment_path = (
        ExperimentTracker(
            str(
                EXPERIMENT_DIRECTORY
            )
        )
        .save(
            experiment_name=(
                MODEL_NAME
            ),
            model_name=(
                MODEL_VERSION
            ),
            parameters={
                "reference_experiment": (
                    str(
                        reference_path
                    )
                ),
                "reference_parameters": (
                    parameters
                ),
                "cv_splits": (
                    CV_SPLITS
                ),
                "selected_groups": (
                    selected_candidate[
                        "groups"
                    ]
                ),
                "selected_cross_asset_features": (
                    selected_candidate[
                        "cross_asset_features"
                    ]
                ),
            },
            metrics={
                "controlled_baseline_cv": (
                    controlled_baseline
                ),
                "candidates": (
                    results
                ),
                "selected_candidate_cv": (
                    selected_candidate
                ),
                "controlled_baseline_validation": (
                    baseline_metrics
                ),
                "selected_validation": (
                    selected_metrics
                ),
                "validation_macro_f1_delta": float(
                    selected_metrics[
                        "macro_f1"
                    ]
                    - baseline_metrics[
                        "macro_f1"
                    ]
                ),
            },
            features=(
                selected_features
            ),
        )
    )

    print()
    print(
        "Selected model:",
        model_path,
    )

    print(
        "Validation predictions:",
        VALIDATION_OUTPUT_PATH,
    )

    print(
        "CV summary:",
        summary_path,
    )

    print(
        "Experiment:",
        experiment_path,
    )

    print()
    print(
        "Held-out test set was NOT evaluated."
    )


if __name__ == "__main__":
    main()