from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.training.date_aware_data_splitter import DateAwareDataSplitter
from app.training.direction_classifier import DirectionClassifier
from app.training.experiment_tracker import ExperimentTracker
from app.training.feature_contract import (
    REFERENCE_DATE_MACRO_FEATURE_COLUMNS,
    RESIDUAL_MACRO_FEATURE_COLUMNS,
    RESIDUAL_MODEL_FEATURE_COLUMNS,
)
from app.training.lstm_parameter_selector import LSTMParameterSelector
from app.training.lstm_predictor import LSTMPredictor
from app.training.lstm_residual_model import LSTMResidualModel
from app.training.lstm_trainer import LSTMTrainer
from app.training.model_comparison_evaluator import ModelComparisonEvaluator
from app.training.residual_dataset_builder import ResidualDatasetBuilder
from app.training.residual_forecast_corrector import ResidualForecastCorrector
from app.training.residual_sequence_preprocessor import (
    ResidualSequencePreprocessor,
)
from app.training.sarimax_forecast_loader import SarimaxForecastLoader
from app.training.torch_model_serializer import TorchModelSerializer
from app.training.xgboost_evaluator import XGBoostEvaluator
from database.training_data_repository import TrainingDataRepository


TICKER = "SPY"
MODEL_NAME = "lstm_residual"
MODEL_VERSION = "lstm_residual_v1"
SARIMAX_FORECAST_PATH = Path("models/sarimax_forecasts.csv")
XGBOOST_VALIDATION_PATH = Path(
    "models/xgboost_validation_predictions.csv"
)
VALIDATION_OUTPUT_PATH = Path(
    "models/lstm_validation_predictions.csv"
)
OPTUNA_TRIALS = 30
OPTUNA_SPLITS = 3
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
RANDOM_STATE = 42


def load_xgboost_validation_predictions() -> pd.DataFrame:
    if not XGBOOST_VALIDATION_PATH.exists():
        raise FileNotFoundError(
            "XGBoost validation artifact was not found. Run "
            "python -m scripts.train_xgboost before training LSTM."
        )

    dataframe = pd.read_csv(
        XGBOOST_VALIDATION_PATH,
        parse_dates=["trade_date"],
    )
    required_columns = {
        "trade_date",
        "actual_log_return",
        "sarimax_prediction",
        "corrected_prediction",
    }
    missing_columns = required_columns - set(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "XGBoost validation artifact is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe["trade_date"].duplicated().any():
        raise ValueError(
            "XGBoost validation artifact contains duplicate dates."
        )

    if dataframe[
        list(required_columns - {"trade_date"})
    ].isna().any().any():
        raise ValueError(
            "XGBoost validation artifact contains missing values."
        )

    return dataframe[
        [
            "trade_date",
            "actual_log_return",
            "sarimax_prediction",
            "corrected_prediction",
        ]
    ].rename(
        columns={
            "actual_log_return": "xgboost_actual_log_return",
            "sarimax_prediction": "xgboost_sarimax_prediction",
            "corrected_prediction": "xgboost_corrected_prediction",
        }
    ).sort_values(
        "trade_date"
    ).reset_index(drop=True)


def validate_split(
    dataframe: pd.DataFrame,
    split_name: str,
) -> None:
    if dataframe.empty:
        raise ValueError(
            f"The residual {split_name} split is empty."
        )

    observed_splits = set(
        dataframe["data_split"].unique()
    )

    if observed_splits != {split_name}:
        raise ValueError(
            f"The residual {split_name} split contains labels: "
            f"{sorted(observed_splits)}"
        )


def format_date(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def main():
    repository = TrainingDataRepository()
    splitter = DateAwareDataSplitter()

    features = repository.get_training_data(
        ticker=TICKER,
        macro_feature_names=RESIDUAL_MACRO_FEATURE_COLUMNS,
    )
    features["trade_date"] = pd.to_datetime(
        features["trade_date"]
    )

    forecasts = SarimaxForecastLoader().load(
        SARIMAX_FORECAST_PATH
    )
    residual_dataset = ResidualDatasetBuilder().build(
        features=features,
        forecasts=forecasts,
    )

    train, validation, residual_test = splitter.split(
        dataframe=residual_dataset,
        reference_dates=features["trade_date"],
    )
    _, _, held_out_test = splitter.split(
        dataframe=features
    )

    validate_split(train, "train")
    validate_split(validation, "validation")

    if not residual_test.empty:
        raise ValueError(
            "The development residual artifact must not contain "
            "test observations."
        )

    feature_columns = list(
        RESIDUAL_MODEL_FEATURE_COLUMNS
    )

    selection = LSTMParameterSelector(
        n_splits=OPTUNA_SPLITS,
        n_trials=OPTUNA_TRIALS,
        max_epochs=MAX_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        random_state=RANDOM_STATE,
    ).select_best_parameters(
        training_data=train,
        feature_columns=feature_columns,
    )
    parameters = dict(selection["parameters"])

    sequence_length = int(
        parameters["sequence_length"]
    )
    preprocessor = ResidualSequencePreprocessor(
        sequence_length=sequence_length
    ).fit(
        dataframe=train,
        feature_columns=feature_columns,
    )
    training_sequences = (
        preprocessor.build_training_sequences(
            train
        )
    )
    validation_sequences = (
        preprocessor.build_inference_sequences(
            history=train,
            dataframe=validation,
        )
    )

    trainer = LSTMTrainer(
        random_state=RANDOM_STATE
    )
    model = LSTMResidualModel(
        input_size=len(
            preprocessor.feature_columns
        ),
        hidden_size=int(parameters["hidden_size"]),
        num_layers=int(parameters["num_layers"]),
        dropout=float(parameters["dropout"]),
    )
    training_result = trainer.train(
        model=model,
        training_data=training_sequences,
        validation_data=None,
        epochs=int(parameters["epochs"]),
        batch_size=int(parameters["batch_size"]),
        learning_rate=float(
            parameters["learning_rate"]
        ),
        weight_decay=float(
            parameters["weight_decay"]
        ),
        gradient_clip=float(
            parameters["gradient_clip"]
        ),
    )

    validation_results = LSTMPredictor().predict(
        model=training_result["model"],
        dataset=validation_sequences,
        preprocessor=preprocessor,
        batch_size=int(parameters["batch_size"]),
    )
    validation_results["actual_log_return"] = (
        validation_results["sarimax_prediction"]
        + validation_results["sarimax_residual"]
    )
    validation_results["lstm_corrected_prediction"] = (
        ResidualForecastCorrector().apply(
            sarimax_predictions=validation_results[
                "sarimax_prediction"
            ],
            predicted_residuals=validation_results[
                "predicted_residual"
            ],
        ).to_numpy()
    )

    xgboost_validation = (
        load_xgboost_validation_predictions()
    )
    validation_results = validation_results.merge(
        xgboost_validation,
        on="trade_date",
        how="inner",
        validate="one_to_one",
    )

    if len(validation_results) != len(
        validation_sequences
    ):
        raise ValueError(
            "LSTM and XGBoost validation dates do not align."
        )

    if not np.allclose(
        validation_results["actual_log_return"],
        validation_results[
            "xgboost_actual_log_return"
        ],
    ):
        raise ValueError(
            "LSTM and XGBoost actual returns do not align."
        )

    if not np.allclose(
        validation_results["sarimax_prediction"],
        validation_results[
            "xgboost_sarimax_prediction"
        ],
    ):
        raise ValueError(
            "LSTM and XGBoost SARIMAX predictions do not align."
        )

    training_actual = (
        train["sarimax_prediction"]
        + train["sarimax_residual"]
    )
    direction_classifier = DirectionClassifier().fit(
        training_actual
    )
    evaluator = XGBoostEvaluator()

    regression_metrics = {
        "sarimax": evaluator.evaluate(
            actual=validation_results[
                "actual_log_return"
            ],
            predicted=validation_results[
                "sarimax_prediction"
            ],
        ),
        "xgboost": evaluator.evaluate(
            actual=validation_results[
                "actual_log_return"
            ],
            predicted=validation_results[
                "xgboost_corrected_prediction"
            ],
        ),
        "lstm": evaluator.evaluate(
            actual=validation_results[
                "actual_log_return"
            ],
            predicted=validation_results[
                "lstm_corrected_prediction"
            ],
        ),
    }

    actual_labels = direction_classifier.classify(
        validation_results["actual_log_return"]
    )
    classification_metrics = (
        ModelComparisonEvaluator().evaluate(
            actual_labels=actual_labels,
            model_labels={
                "sarimax": direction_classifier.classify(
                    validation_results[
                        "sarimax_prediction"
                    ]
                ),
                "xgboost": direction_classifier.classify(
                    validation_results[
                        "xgboost_corrected_prediction"
                    ]
                ),
                "lstm": direction_classifier.classify(
                    validation_results[
                        "lstm_corrected_prediction"
                    ]
                ),
            },
        )
    )

    metrics = {
        "regression_comparison": regression_metrics,
        "classification_comparison": (
            classification_metrics
        ),
        "direction_thresholds": (
            direction_classifier.get_state()
        ),
        "parameter_selection": {
            "cv_mse": selection["cv_mse"],
            "best_trial_number": selection[
                "best_trial_number"
            ],
            "fold_best_epochs": selection[
                "fold_best_epochs"
            ],
            "completed_trials": selection[
                "completed_trials"
            ],
        },
        "observations": {
            "train_rows": len(train),
            "train_sequences": len(
                training_sequences
            ),
            "validation_rows": len(validation),
            "validation_sequences": len(
                validation_sequences
            ),
            "test_held_out": len(held_out_test),
        },
        "date_ranges": {
            "train_start": format_date(
                train["trade_date"].min()
            ),
            "train_end": format_date(
                train["trade_date"].max()
            ),
            "validation_start": format_date(
                validation_results[
                    "trade_date"
                ].min()
            ),
            "validation_end": format_date(
                validation_results[
                    "trade_date"
                ].max()
            ),
        },
    }

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "ticker": TICKER,
        "training_date_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "target": "sarimax_residual",
        "parameters": parameters,
        "base_features": feature_columns,
        "sequence_features": (
            preprocessor.feature_columns
        ),
        "macro_features": list(
            RESIDUAL_MACRO_FEATURE_COLUMNS
        ),
        "excluded_reference_date_macro_features": list(
            REFERENCE_DATE_MACRO_FEATURE_COLUMNS
        ),
        "uses_realized_residual_history": True,
        "forecast_mode": "one_step_walk_forward",
        "metrics": metrics,
        "test_predictions_generated": False,
    }

    model_path = TorchModelSerializer().save(
        model=training_result["model"],
        preprocessor=preprocessor,
        metadata=metadata,
        filename=MODEL_NAME,
    )
    experiment_path = ExperimentTracker(
        "experiments"
    ).save(
        experiment_name=MODEL_NAME,
        model_name="LSTM",
        parameters=parameters,
        metrics=metrics,
        features=preprocessor.feature_columns,
    )

    VALIDATION_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    validation_results.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
    )

    print("=" * 60)
    print("LSTM RESIDUAL PIPELINE COMPLETE")
    print("=" * 60)
    print()
    print(f"Model       : {model_path}")
    print(f"Experiment  : {experiment_path}")
    print(f"Predictions : {VALIDATION_OUTPUT_PATH}")
    print()

    for model_name, model_metrics in (
        classification_metrics.items()
    ):
        print(model_name.upper())
        for metric, value in model_metrics.items():
            print(f"{metric:<12}{value:.4f}")
        print()


if __name__ == "__main__":
    main()
