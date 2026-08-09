from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.training.torch_reproducibility import (
    TorchReproducibility,
)
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
from app.training.lstm_classifier_model import (
    LSTMClassifier,
)
from app.training.lstm_classifier_parameter_selector import (
    LSTMClassifierParameterSelector,
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
from database.direction_training_data_repository import (
    DirectionTrainingDataRepository,
)


TICKER = "SPY"

MODEL_NAME = "lstm_direction_classifier"
MODEL_VERSION = "lstm_direction_v1"

MODEL_PATH = Path(
    "models/lstm_direction_classifier.pt"
)

VALIDATION_OUTPUT_PATH = Path(
    "models/lstm_direction_validation_predictions.csv"
)

EXPERIMENT_DIRECTORY = "experiments"

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


def main():
    print(
        "Loading direct-classification dataset..."
    )

    repository = (
        DirectionTrainingDataRepository()
    )

    raw_data = (
        repository.get_training_data(
            ticker=TICKER
        )
    )

    dataset = (
        DirectionDatasetBuilder()
        .build(
            raw_data
        )
    )

    splitter = (
        DateAwareDataSplitter()
    )

    train, validation, _ = (
        splitter.split(
            dataset,
            date_column="target_date",
        )
    )

    feature_columns = list(
        DirectionFeatureBuilder.FEATURE_COLUMNS
    )

    print(
        f"Training rows: {len(train)}"
    )

    print(
        f"Validation rows: {len(validation)}"
    )

    print(
        "Starting train-only walk-forward "
        "Optuna search..."
    )

    selector = (
        LSTMClassifierParameterSelector(
            feature_columns=feature_columns,
            n_splits=OPTUNA_SPLITS,
            n_trials=OPTUNA_TRIALS,
            max_epochs=(
                MAX_SELECTION_EPOCHS
            ),
            patience=(
                EARLY_STOPPING_PATIENCE
            ),
            random_state=RANDOM_STATE,
        )
    )

    selection = (
        selector.select_best_parameters(
            training_data=train
        )
    )

    parameters = dict(
        selection[
            "parameters"
        ]
    )

    print()
    print(
        "Parameter selection complete."
    )
    print(
        "CV Macro F1:",
        round(
            selection[
                "cv_macro_f1"
            ],
            4,
        ),
    )
    print(
        "CV Macro F1 Std:",
        round(
            selection[
                "cv_macro_f1_std"
            ],
            4,
        ),
    )
    print(
        "Fold Macro F1:",
        [
            round(
                value,
                4,
            )
            for value in (
                selection[
                    "fold_macro_f1"
                ]
            )
        ],
    )
    print(
        "Selected parameters:",
        parameters,
    )

    sequence_length = int(
        parameters[
            "sequence_length"
        ]
    )

    preprocessor = (
        ClassificationSequencePreprocessor(
            feature_columns=(
                feature_columns
            ),
            sequence_length=(
                sequence_length
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

    model_config = {
        "input_size": len(
            feature_columns
        ),
        "hidden_size": int(
            parameters[
                "hidden_size"
            ]
        ),
        "num_layers": int(
            parameters[
                "num_layers"
            ]
        ),
        "dropout": float(
            parameters[
                "dropout"
            ]
        ),
        "num_classes": 3,
    }

    model = LSTMClassifier(
        **model_config
    )

    trainer = (
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
                EARLY_STOPPING_PATIENCE
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
            seed=RANDOM_STATE,
            deterministic=True,
        )
    )

    print()
    print(
        "Training final LSTM candidate "
        "on training data..."
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

    predictor = (
        TorchClassificationPredictor(
            batch_size=int(
                parameters[
                    "batch_size"
                ]
            )
        )
    )

    prediction_result = (
        predictor.predict(
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

    actual = (
        validation_sequences[
            "y"
        ]
    )

    predicted = (
        prediction_result[
            "predictions"
        ]
    )

    probabilities = (
        prediction_result[
            "probabilities"
        ]
    )

    evaluator = (
        ClassificationEvaluator()
    )

    validation_metrics = (
        evaluator.evaluate(
            actual=actual,
            predicted=predicted,
        )
    )

    training_majority_class = int(
        np.bincount(
            training_sequences[
                "y"
            ],
            minlength=3,
        ).argmax()
    )

    baseline_predictions = (
        np.full(
            shape=len(
                actual
            ),
            fill_value=(
                training_majority_class
            ),
            dtype=np.int64,
        )
    )

    baseline_metrics = (
        evaluator.evaluate(
            actual=actual,
            predicted=(
                baseline_predictions
            ),
        )
    )

    validation_lookup = (
        validation
        .set_index(
            "target_date"
        )
    )

    target_dates = pd.to_datetime(
        validation_sequences[
            "target_dates"
        ]
    )

    aligned_validation = (
        validation_lookup.loc[
            target_dates
        ]
    )

    validation_output = pd.DataFrame(
        {
            "feature_date": (
                validation_sequences[
                    "feature_dates"
                ]
            ),
            "target_date": (
                target_dates
            ),
            "actual_direction": [
                class_name(
                    value
                )
                for value in actual
            ],
            "predicted_direction": [
                class_name(
                    value
                )
                for value in predicted
            ],
            "prob_down": (
                probabilities[
                    :,
                    0,
                ]
            ),
            "prob_flat": (
                probabilities[
                    :,
                    1,
                ]
            ),
            "prob_up": (
                probabilities[
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
        "model_version": (
            MODEL_VERSION
        ),
        "trained_at_utc": (
            trained_at
        ),
        "ticker": TICKER,
        "horizon": "1_day",
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
        "selection": selection,
        "validation_metrics": (
            validation_metrics
        ),
        "baseline_metrics": (
            baseline_metrics
        ),
        "class_weights": (
            training_result[
                "class_weights"
            ].tolist()
        ),
        "experiment_notes": (
            "Direct next-day SPY "
            "UP/FLAT/DOWN classifier. "
            "Hyperparameters selected "
            "using train-only walk-forward "
            "cross-validation. Outer "
            "validation used once for "
            "candidate evaluation. "
            "Held-out test not evaluated."
        ),
    }

    model_path = (
        TorchClassifierSerializer.save(
            model=(
                training_result[
                    "model"
                ]
            ),
            model_type="lstm",
            model_config=(
                model_config
            ),
            preprocessor=(
                preprocessor
            ),
            metadata=metadata,
            filepath=str(
                MODEL_PATH
            ),
        )
    )

    experiment_path = (
        ExperimentTracker(
            EXPERIMENT_DIRECTORY
        ).save(
            experiment_name=(
                MODEL_NAME
            ),
            model_name=(
                MODEL_VERSION
            ),
            parameters=(
                parameters
            ),
            metrics={
                "selection": {
                    "cv_macro_f1": (
                        selection[
                            "cv_macro_f1"
                        ]
                    ),
                    "cv_macro_f1_std": (
                        selection[
                            "cv_macro_f1_std"
                        ]
                    ),
                    "fold_macro_f1": (
                        selection[
                            "fold_macro_f1"
                        ]
                    ),
                },
                "validation": (
                    validation_metrics
                ),
                "training_majority_baseline": (
                    baseline_metrics
                ),
            },
            features=(
                feature_columns
            ),
        )
    )

    print()
    print(
        "================================="
    )
    print(
        "LSTM DIRECT CLASSIFIER RESULT"
    )
    print(
        "================================="
    )

    print(
        "CV Macro F1:",
        round(
            selection[
                "cv_macro_f1"
            ],
            4,
        ),
    )

    print(
        "Validation Macro F1:",
        round(
            validation_metrics[
                "macro_f1"
            ],
            4,
        ),
    )

    print(
        "Validation Balanced Accuracy:",
        round(
            validation_metrics[
                "balanced_accuracy"
            ],
            4,
        ),
    )

    print(
        "Validation Accuracy:",
        round(
            validation_metrics[
                "accuracy"
            ],
            4,
        ),
    )

    print(
        "Majority Baseline Macro F1:",
        round(
            baseline_metrics[
                "macro_f1"
            ],
            4,
        ),
    )

    print()
    print(
        "Per-class metrics:"
    )

    for class_label, metrics in (
        validation_metrics[
            "per_class"
        ].items()
    ):
        print(
            f"{class_label}: "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"support={metrics['support']}"
        )

    print()
    print(
        "Confusion matrix:"
    )

    for row in (
        validation_metrics[
            "confusion_matrix"
        ]
    ):
        print(
            row
        )

    print()
    print(
        "Selected loss:",
        parameters[
            "loss_name"
        ],
    )

    print(
        "Selected sequence length:",
        parameters[
            "sequence_length"
        ],
    )

    print(
        "Selected epochs:",
        parameters[
            "epochs"
        ],
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