from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from app.training.date_aware_data_splitter import DateAwareDataSplitter
from app.training.direction_classifier import DirectionClassifier
from app.training.experiment_tracker import ExperimentTracker
from app.training.feature_contract import (
    REFERENCE_DATE_MACRO_FEATURE_COLUMNS,
    RESIDUAL_MACRO_FEATURE_COLUMNS,
    RESIDUAL_MODEL_FEATURE_COLUMNS,
    require_columns,
)
from app.training.model_comparison_evaluator import ModelComparisonEvaluator
from app.training.model_serializer import ModelSerializer
from app.training.residual_dataset_builder import ResidualDatasetBuilder
from app.training.residual_forecast_corrector import ResidualForecastCorrector
from app.training.sarimax_forecast_loader import SarimaxForecastLoader
from app.training.xgboost_evaluator import XGBoostEvaluator
from app.training.xgboost_parameter_selector import XGBoostParameterSelector
from app.training.xgboost_predictor import XGBoostPredictor
from app.training.xgboost_trainer import XGBoostTrainer
from database.training_data_repository import TrainingDataRepository


TICKER = "SPY"
MODEL_NAME = "xgboost_residual"
MODEL_VERSION = "xgboost_residual_v2_feature_contract"
SARIMAX_FORECAST_PATH = Path("models/sarimax_forecasts.csv")
VALIDATION_OUTPUT_PATH = Path(
    "models/xgboost_validation_predictions.csv"
)


def validate_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    feature_columns = list(
        RESIDUAL_MODEL_FEATURE_COLUMNS
    )

    require_columns(
        dataframe,
        feature_columns,
        "Residual dataset",
    )

    invalid_columns = [
        column
        for column in feature_columns
        if not is_numeric_dtype(dataframe[column])
    ]

    if invalid_columns:
        raise ValueError(
            "XGBoost predictors must be numeric. Invalid columns: "
            f"{invalid_columns}"
        )

    return feature_columns


def validate_residual_split(
    dataframe: pd.DataFrame,
    split_name: str,
) -> None:
    if dataframe.empty:
        raise ValueError(
            f"The residual {split_name} split is empty."
        )

    if dataframe["trade_date"].duplicated().any():
        raise ValueError(
            f"The residual {split_name} split contains duplicate dates."
        )

    observed_splits = set(
        dataframe["data_split"].unique()
    )

    if observed_splits != {split_name}:
        raise ValueError(
            f"The residual {split_name} split contains artifact labels: "
            f"{sorted(observed_splits)}"
        )


def format_date(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def main():
    repository = TrainingDataRepository()
    serializer = ModelSerializer()
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

    validate_residual_split(train, "train")
    validate_residual_split(validation, "validation")

    if not residual_test.empty:
        raise ValueError(
            "The development residual artifact must not contain "
            "test observations."
        )

    feature_columns = validate_feature_columns(
        residual_dataset
    )

    X_train = train[feature_columns].copy()
    y_train = train["sarimax_residual"].copy()
    X_validation = validation[feature_columns].copy()

    training_actual = (
        train["sarimax_prediction"]
        + train["sarimax_residual"]
    )
    direction_classifier = DirectionClassifier().fit(
        training_actual
    )

    parameters = (
        XGBoostParameterSelector()
        .select_best_parameters(
            X=X_train,
            y=y_train,
        )
    )

    model = XGBoostTrainer().train(
        X_train=X_train,
        y_train=y_train,
        parameters=parameters,
    )

    predicted_residuals = XGBoostPredictor().predict(
        model=model,
        dataset=X_validation,
    )

    validation_results = validation[
        [
            "trade_date",
            "sarimax_prediction",
            "sarimax_residual",
        ]
    ].copy()
    validation_results["predicted_residual"] = (
        predicted_residuals.to_numpy()
    )
    validation_results["actual_log_return"] = (
        validation_results["sarimax_prediction"]
        + validation_results["sarimax_residual"]
    )
    validation_results["corrected_prediction"] = (
        ResidualForecastCorrector().apply(
            sarimax_predictions=validation_results[
                "sarimax_prediction"
            ],
            predicted_residuals=validation_results[
                "predicted_residual"
            ],
        ).to_numpy()
    )

    evaluator = XGBoostEvaluator()
    sarimax_regression_metrics = evaluator.evaluate(
        actual=validation_results[
            "actual_log_return"
        ],
        predicted=validation_results[
            "sarimax_prediction"
        ],
    )
    xgboost_regression_metrics = evaluator.evaluate(
        actual=validation_results[
            "actual_log_return"
        ],
        predicted=validation_results[
            "corrected_prediction"
        ],
    )

    actual_labels = direction_classifier.classify(
        validation_results["actual_log_return"]
    )
    comparison_metrics = (
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
                        "corrected_prediction"
                    ]
                ),
            },
        )
    )

    metrics = {
        "sarimax_regression": sarimax_regression_metrics,
        "xgboost_regression": xgboost_regression_metrics,
        "classification_comparison": comparison_metrics,
        "direction_thresholds": (
            direction_classifier.get_state()
        ),
        "observations": {
            "train": len(train),
            "validation": len(validation),
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
                validation["trade_date"].min()
            ),
            "validation_end": format_date(
                validation["trade_date"].max()
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
        "features": feature_columns,
        "macro_features": list(
            RESIDUAL_MACRO_FEATURE_COLUMNS
        ),
        "excluded_reference_date_macro_features": list(
            REFERENCE_DATE_MACRO_FEATURE_COLUMNS
        ),
        "metrics": metrics,
        "test_predictions_generated": False,
    }

    model_path = serializer.save(
        model=model,
        metadata=metadata,
        filename=MODEL_NAME,
    )

    experiment_path = ExperimentTracker(
        "experiments"
    ).save(
        experiment_name=MODEL_NAME,
        model_name="XGBoost",
        parameters=parameters,
        metrics=metrics,
        features=feature_columns,
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
    print("XGBOOST RESIDUAL PIPELINE COMPLETE")
    print("=" * 60)
    print()
    print(f"Model       : {model_path}")
    print(f"Experiment  : {experiment_path}")
    print(f"Predictions : {VALIDATION_OUTPUT_PATH}")
    print()

    for model_name, model_metrics in comparison_metrics.items():
        print(model_name.upper())
        for metric, value in model_metrics.items():
            print(f"{metric:<12}{value:.4f}")
        print()


if __name__ == "__main__":
    main()
