from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.training.classification_evaluator import ClassificationEvaluator
from app.training.date_aware_data_splitter import DateAwareDataSplitter
from app.training.direction_dataset_builder import DirectionDatasetBuilder
from app.training.direction_feature_builder import DirectionFeatureBuilder
from app.training.experiment_tracker import ExperimentTracker
from app.training.hierarchical_direction_evaluator import (
    HierarchicalDirectionEvaluator,
)
from app.training.hierarchical_direction_predictor import (
    HierarchicalDirectionPredictor,
)
from app.training.hierarchical_sequence_preprocessor import (
    HierarchicalSequencePreprocessor,
)
from app.training.hierarchical_xlstm_parameter_selector import (
    HierarchicalXLSTMParameterSelector,
)
from app.training.hierarchical_xlstm_serializer import (
    HierarchicalXLSTMSerializer,
)
from app.training.torch_classification_trainer import (
    TorchClassificationTrainer,
)
from app.training.torch_reproducibility import TorchReproducibility
from app.training.xlstm_classifier_model import XLSTMClassifier
from database.direction_training_data_repository import (
    DirectionTrainingDataRepository,
)


TICKER = "SPY"
HORIZON = "1_day"

MODEL_NAME = "xlstm_hierarchical_direction"
MODEL_VERSION = "xlstm_hierarchical_direction_v1"

MODEL_PATH = Path(
    "models/xlstm_hierarchical_direction.pt"
)

VALIDATION_OUTPUT_PATH = Path(
    "models/xlstm_hierarchical_direction_validation_predictions.csv"
)

EXPERIMENT_DIRECTORY = Path(
    "experiments"
)

REFERENCE_EXPERIMENT_NAME = "xlstm_direction_classifier"
REFERENCE_MODEL_NAME = "xlstm_direction_v1"

OPTUNA_TRIALS = 20
OPTUNA_SPLITS = 3
MAX_SELECTION_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 8
RANDOM_STATE = 42

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


def load_reference_experiment() -> tuple[
    Path | None,
    dict | None,
]:
    matches = []

    for path in EXPERIMENT_DIRECTORY.glob(
        "xlstm_direction_classifier_*.json"
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
        return None, None

    matches.sort(
        key=lambda item: item[0]
    )

    _, path, experiment = matches[-1]

    return path, experiment


def build_model_config(
    parameters: dict,
    feature_count: int,
) -> dict:
    return {
        "input_size": int(
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
        "num_classes": 2,
    }


def build_trainer(
    parameters: dict,
    seed: int,
) -> TorchClassificationTrainer:
    return TorchClassificationTrainer(
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
        patience=EARLY_STOPPING_PATIENCE,
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
        seed=int(
            seed
        ),
        deterministic=True,
        num_classes=2,
    )


def fit_final_stage(
    train: pd.DataFrame,
    feature_columns: list[str],
    task: str,
    parameters: dict,
    seed: int,
) -> dict:
    preprocessor = HierarchicalSequencePreprocessor(
        feature_columns=feature_columns,
        sequence_length=int(
            parameters[
                "sequence_length"
            ]
        ),
    )

    preprocessor.fit(
        train
    )

    training_sequences = (
        preprocessor
        .build_training_sequences(
            dataframe=train,
            task=task,
        )
    )

    TorchReproducibility.configure(
        seed=seed,
        deterministic=True,
    )

    model_config = build_model_config(
        parameters=parameters,
        feature_count=len(
            feature_columns
        ),
    )

    model = XLSTMClassifier(
        **model_config
    )

    training_result = (
        build_trainer(
            parameters=parameters,
            seed=seed,
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

    return {
        "model": training_result[
            "model"
        ],
        "model_config": model_config,
        "preprocessor": preprocessor,
        "class_weights": training_result[
            "class_weights"
        ],
        "training_rows": int(
            len(
                training_sequences[
                    "y"
                ]
            )
        ),
    }


def build_majority_baseline(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict:
    mapping = {
        "DOWN": 0,
        "FLAT": 1,
        "UP": 2,
    }

    training_labels = np.asarray(
        [
            mapping[
                str(value)
            ]
            for value in train[
                "direction"
            ]
        ],
        dtype=np.int64,
    )

    validation_labels = np.asarray(
        [
            mapping[
                str(value)
            ]
            for value in validation[
                "direction"
            ]
        ],
        dtype=np.int64,
    )

    majority_class = int(
        np.bincount(
            training_labels,
            minlength=3,
        ).argmax()
    )

    predictions = np.full(
        len(
            validation_labels
        ),
        fill_value=majority_class,
        dtype=np.int64,
    )

    return ClassificationEvaluator().evaluate(
        actual=validation_labels,
        predicted=predictions,
    )


def main():
    print(
        "Loading base direct-classification dataset..."
    )

    repository = DirectionTrainingDataRepository()

    raw_data = repository.get_training_data(
        ticker=TICKER,
        include_breadth=False,
        include_cross_asset=False,
    )

    feature_builder = DirectionFeatureBuilder(
        feature_scope="base"
    )

    dataset = DirectionDatasetBuilder(
        feature_builder=feature_builder
    ).build(
        raw_data
    )

    train, validation, _ = DateAwareDataSplitter().split(
        dataset,
        date_column="target_date",
    )

    feature_columns = list(
        feature_builder.feature_columns
    )

    print(
        f"Feature count: {len(feature_columns)}"
    )
    print(
        f"Dataset rows: {len(dataset)}"
    )
    print(
        f"Training rows: {len(train)}"
    )
    print(
        f"Validation rows: {len(validation)}"
    )
    print(
        "Training period:",
        format_date(
            train[
                "target_date"
            ].min()
        ),
        "->",
        format_date(
            train[
                "target_date"
            ].max()
        ),
    )
    print(
        "Validation period:",
        format_date(
            validation[
                "target_date"
            ].min()
        ),
        "->",
        format_date(
            validation[
                "target_date"
            ].max()
        ),
    )
    print(
        "Held-out test set will NOT be evaluated."
    )

    print()
    print(
        "Selecting Stage 1 parameters: MOVE vs FLAT..."
    )

    stage1_selection = (
        HierarchicalXLSTMParameterSelector(
            feature_columns=feature_columns,
            task="move",
            n_splits=OPTUNA_SPLITS,
            n_trials=OPTUNA_TRIALS,
            max_epochs=MAX_SELECTION_EPOCHS,
            patience=EARLY_STOPPING_PATIENCE,
            random_state=RANDOM_STATE,
        )
        .select_best_parameters(
            training_data=train
        )
    )

    print()
    print(
        "Stage 1 selection complete."
    )
    print(
        "CV Macro F1:",
        round(
            stage1_selection[
                "cv_macro_f1"
            ],
            4,
        ),
    )
    print(
        "OOF threshold Macro F1:",
        round(
            stage1_selection[
                "threshold_oof_macro_f1"
            ],
            4,
        ),
    )
    print(
        "Selected MOVE threshold:",
        round(
            stage1_selection[
                "decision_threshold"
            ],
            4,
        ),
    )
    print(
        "Selected parameters:",
        stage1_selection[
            "parameters"
        ],
    )

    print()
    print(
        "Selecting Stage 2 parameters: UP vs DOWN on true MOVE rows..."
    )

    stage2_selection = (
        HierarchicalXLSTMParameterSelector(
            feature_columns=feature_columns,
            task="direction",
            n_splits=OPTUNA_SPLITS,
            n_trials=OPTUNA_TRIALS,
            max_epochs=MAX_SELECTION_EPOCHS,
            patience=EARLY_STOPPING_PATIENCE,
            random_state=(
                RANDOM_STATE
                + 10000
            ),
        )
        .select_best_parameters(
            training_data=train
        )
    )

    print()
    print(
        "Stage 2 selection complete."
    )
    print(
        "CV Macro F1:",
        round(
            stage2_selection[
                "cv_macro_f1"
            ],
            4,
        ),
    )
    print(
        "OOF threshold Macro F1:",
        round(
            stage2_selection[
                "threshold_oof_macro_f1"
            ],
            4,
        ),
    )
    print(
        "Selected UP threshold:",
        round(
            stage2_selection[
                "decision_threshold"
            ],
            4,
        ),
    )
    print(
        "Selected parameters:",
        stage2_selection[
            "parameters"
        ],
    )

    print()
    print(
        "Training final Stage 1 model on full training data..."
    )

    stage1_final = fit_final_stage(
        train=train,
        feature_columns=feature_columns,
        task="move",
        parameters=stage1_selection[
            "parameters"
        ],
        seed=RANDOM_STATE,
    )

    print(
        "Training final Stage 2 model on true MOVE targets "
        "with complete market-day context..."
    )

    stage2_final = fit_final_stage(
        train=train,
        feature_columns=feature_columns,
        task="direction",
        parameters=stage2_selection[
            "parameters"
        ],
        seed=(
            RANDOM_STATE
            + 1
        ),
    )

    predictor = HierarchicalDirectionPredictor(
        stage1_model=stage1_final[
            "model"
        ],
        stage1_preprocessor=stage1_final[
            "preprocessor"
        ],
        stage1_move_threshold=stage1_selection[
            "decision_threshold"
        ],
        stage2_model=stage2_final[
            "model"
        ],
        stage2_preprocessor=stage2_final[
            "preprocessor"
        ],
        stage2_up_threshold=stage2_selection[
            "decision_threshold"
        ],
        batch_size=256,
    )

    prediction_result = predictor.predict(
        history=train,
        dataframe=validation,
    )

    metrics = HierarchicalDirectionEvaluator().evaluate(
        prediction_result
    )

    majority_baseline = build_majority_baseline(
        train=train,
        validation=validation,
    )

    reference_path, reference_experiment = (
        load_reference_experiment()
    )

    reference_validation_metrics = None

    if reference_experiment is not None:
        reference_validation_metrics = (
            reference_experiment
            .get(
                "metrics",
                {},
            )
            .get(
                "validation"
            )
        )

    validation_lookup = validation.set_index(
        "target_date"
    )

    target_dates = pd.DatetimeIndex(
        prediction_result[
            "target_dates"
        ]
    )

    aligned_validation = validation_lookup.loc[
        target_dates
    ]

    validation_output = pd.DataFrame(
        {
            "feature_date": prediction_result[
                "feature_dates"
            ],
            "target_date": target_dates,
            "actual_direction": prediction_result[
                "actual_directions"
            ],
            "stage1_move_probability": prediction_result[
                "stage1_move_probability"
            ],
            "stage1_predicted_move": prediction_result[
                "stage1_predicted_move"
            ],
            "stage2_up_probability": prediction_result[
                "stage2_up_probability"
            ],
            "stage2_predicted_up": prediction_result[
                "stage2_predicted_up"
            ],
            "predicted_direction": prediction_result[
                "final_predicted_directions"
            ],
            "future_log_return": aligned_validation[
                "future_log_return"
            ].to_numpy(),
            "rolling_volatility": aligned_validation[
                "rolling_volatility"
            ].to_numpy(),
            "threshold": aligned_validation[
                "threshold"
            ].to_numpy(),
        }
    )

    VALIDATION_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_output.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
    )

    trained_at = datetime.now(
        timezone.utc
    ).isoformat()

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": trained_at,
        "ticker": TICKER,
        "horizon": HORIZON,
        "architecture": (
            "Two-stage hierarchical xLSTM: "
            "Stage 1 FLAT vs MOVE, Stage 2 DOWN vs UP."
        ),
        "target": {
            "classes": [
                "DOWN",
                "FLAT",
                "UP",
            ],
            "volatility_window": TARGET_VOLATILITY_WINDOW,
            "threshold_multiplier": TARGET_THRESHOLD_MULTIPLIER,
        },
        "features": feature_columns,
        "feature_count": len(
            feature_columns
        ),
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
        "stage1_selection": stage1_selection,
        "stage2_selection": stage2_selection,
        "validation_metrics": metrics,
        "majority_baseline": majority_baseline,
        "reference_experiment": (
            str(
                reference_path
            )
            if reference_path is not None
            else None
        ),
        "reference_validation_metrics": (
            reference_validation_metrics
        ),
        "stage1_training_rows": stage1_final[
            "training_rows"
        ],
        "stage2_training_rows": stage2_final[
            "training_rows"
        ],
        "stage1_class_weights": stage1_final[
            "class_weights"
        ].tolist(),
        "stage2_class_weights": stage2_final[
            "class_weights"
        ].tolist(),
        "experiment_notes": (
            "Target definition remains locked at the existing "
            "20-day realized-volatility threshold with multiplier 0.15. "
            "Both stages use only the original 21-feature base contract. "
            "Each stage performs independent train-only walk-forward Optuna "
            "selection. Decision thresholds are selected from train-only "
            "out-of-fold probabilities. Stage 2 trains only on true MOVE "
            "targets while retaining every market day inside sequence context. "
            "Outer validation is used only after both stages and thresholds "
            "are locked. Held-out test data is not evaluated."
        ),
    }

    model_path = HierarchicalXLSTMSerializer.save(
        stage1_model=stage1_final[
            "model"
        ],
        stage1_model_config=stage1_final[
            "model_config"
        ],
        stage1_preprocessor=stage1_final[
            "preprocessor"
        ],
        stage1_threshold=stage1_selection[
            "decision_threshold"
        ],
        stage2_model=stage2_final[
            "model"
        ],
        stage2_model_config=stage2_final[
            "model_config"
        ],
        stage2_preprocessor=stage2_final[
            "preprocessor"
        ],
        stage2_threshold=stage2_selection[
            "decision_threshold"
        ],
        metadata=metadata,
        filepath=str(
            MODEL_PATH
        ),
    )

    experiment_path = ExperimentTracker(
        str(
            EXPERIMENT_DIRECTORY
        )
    ).save(
        experiment_name=MODEL_NAME,
        model_name=MODEL_VERSION,
        parameters={
            "stage1": stage1_selection,
            "stage2": stage2_selection,
        },
        metrics={
            "validation": metrics,
            "majority_baseline": majority_baseline,
            "reference_validation": (
                reference_validation_metrics
            ),
        },
        features=feature_columns,
    )

    print()
    print(
        "============================================="
    )
    print(
        "HIERARCHICAL xLSTM VALIDATION RESULT"
    )
    print(
        "============================================="
    )

    print()
    print(
        "Stage 1: MOVE vs FLAT"
    )
    print(
        "Macro F1:",
        round(
            metrics[
                "stage1_move_vs_flat"
            ][
                "macro_f1"
            ],
            4,
        ),
    )
    print(
        "Balanced Accuracy:",
        round(
            metrics[
                "stage1_move_vs_flat"
            ][
                "balanced_accuracy"
            ],
            4,
        ),
    )
    print(
        "FLAT F1:",
        round(
            metrics[
                "stage1_move_vs_flat"
            ][
                "per_class"
            ][
                "FLAT"
            ][
                "f1"
            ],
            4,
        ),
    )
    print(
        "MOVE F1:",
        round(
            metrics[
                "stage1_move_vs_flat"
            ][
                "per_class"
            ][
                "MOVE"
            ][
                "f1"
            ],
            4,
        ),
    )

    print()
    print(
        "Stage 2: UP vs DOWN on actual MOVE rows"
    )
    print(
        "Macro F1:",
        round(
            metrics[
                "stage2_up_vs_down_oracle"
            ][
                "macro_f1"
            ],
            4,
        ),
    )
    print(
        "Balanced Accuracy:",
        round(
            metrics[
                "stage2_up_vs_down_oracle"
            ][
                "balanced_accuracy"
            ],
            4,
        ),
    )

    print()
    print(
        "End-to-end DOWN / FLAT / UP"
    )
    print(
        "Macro F1:",
        round(
            metrics[
                "end_to_end"
            ][
                "macro_f1"
            ],
            4,
        ),
    )
    print(
        "Balanced Accuracy:",
        round(
            metrics[
                "end_to_end"
            ][
                "balanced_accuracy"
            ],
            4,
        ),
    )
    print(
        "Accuracy:",
        round(
            metrics[
                "end_to_end"
            ][
                "accuracy"
            ],
            4,
        ),
    )

    for class_label, class_metrics in (
        metrics[
            "end_to_end"
        ][
            "per_class"
        ].items()
    ):
        print(
            f"{class_label}: "
            f"precision={class_metrics['precision']:.4f}, "
            f"recall={class_metrics['recall']:.4f}, "
            f"f1={class_metrics['f1']:.4f}, "
            f"support={class_metrics['support']}"
        )

    print()
    print(
        "End-to-end confusion matrix:"
    )

    for row in metrics[
        "end_to_end"
    ][
        "confusion_matrix"
    ]:
        print(
            row
        )

    print()
    print(
        "Majority baseline Macro F1:",
        round(
            majority_baseline[
                "macro_f1"
            ],
            4,
        ),
    )

    if reference_validation_metrics is not None:
        print(
            "Original 3-class xLSTM Macro F1:",
            round(
                reference_validation_metrics[
                    "macro_f1"
                ],
                4,
            ),
        )
        print(
            "Hierarchical delta vs original xLSTM:",
            round(
                metrics[
                    "end_to_end"
                ][
                    "macro_f1"
                ]
                - reference_validation_metrics[
                    "macro_f1"
                ],
                4,
            ),
        )

    print()
    print(
        "Stage 1 MOVE threshold:",
        round(
            stage1_selection[
                "decision_threshold"
            ],
            4,
        ),
    )
    print(
        "Stage 2 UP threshold:",
        round(
            stage2_selection[
                "decision_threshold"
            ],
            4,
        ),
    )

    print()
    print(
        "Model:",
        model_path,
    )
    print(
        "Validation predictions:",
        VALIDATION_OUTPUT_PATH,
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
