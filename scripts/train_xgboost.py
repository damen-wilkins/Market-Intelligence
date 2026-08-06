from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from app.training.date_aware_data_splitter import DateAwareDataSplitter
from app.training.experiment_tracker import ExperimentTracker
from app.training.model_comparison_evaluator import ModelComparisonEvaluator
from app.training.model_serializer import ModelSerializer
from app.training.residual_dataset_builder import ResidualDatasetBuilder
from app.training.residual_forecast_corrector import ResidualForecastCorrector
from app.training.xgboost_evaluator import XGBoostEvaluator
from app.training.xgboost_parameter_selector import XGBoostParameterSelector
from app.training.xgboost_predictor import XGBoostPredictor
from app.training.xgboost_trainer import XGBoostTrainer
from database.training_data_repository import TrainingDataRepository


TICKER = "SPY"
MODEL_NAME = "xgboost_residual"
SARIMAX_FORECAST_PATH = Path("models/sarimax_forecasts.csv")
VALIDATION_OUTPUT_PATH = Path(
    "models/xgboost_validation_predictions.csv"
)


def load_sarimax_forecasts() -> pd.DataFrame:
    if not SARIMAX_FORECAST_PATH.exists():
        raise FileNotFoundError(
            "SARIMAX forecast artifact was not found: "
            f"{SARIMAX_FORECAST_PATH}"
        )

    forecasts = pd.read_csv(
        SARIMAX_FORECAST_PATH,
        parse_dates=["trade_date"],
    )

    required_columns = {
        "trade_date",
        "actual_log_return",
        "sarimax_prediction",
        "is_out_of_sample",
    }

    missing_columns = required_columns - set(forecasts.columns)

    if missing_columns:
        raise ValueError(
            "SARIMAX forecast artifact is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if forecasts["trade_date"].duplicated().any():
        raise ValueError(
            "SARIMAX forecast artifact contains duplicate trade dates."
        )

    out_of_sample = (
        forecasts["is_out_of_sample"]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    if out_of_sample.isna().any():
        raise ValueError(
            "SARIMAX is_out_of_sample contains invalid values."
        )

    if not out_of_sample.all():
        raise ValueError(
            "Residual learning requires out-of-sample SARIMAX "
            "predictions for every row."
        )

    forecasts["is_out_of_sample"] = out_of_sample

    return forecasts.sort_values(
        "trade_date"
    ).reset_index(drop=True)


def get_feature_columns(
    residual_dataset: pd.DataFrame,
) -> list[str]:
    excluded_columns = {
        "trade_date",
        "sarimax_residual",
    }

    feature_columns = [
        column
        for column in residual_dataset.columns
        if column not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError(
            "No XGBoost predictor columns are available."
        )

    invalid_columns = [
        column
        for column in feature_columns
        if not is_numeric_dtype(residual_dataset[column])
    ]

    if invalid_columns:
        raise ValueError(
            "XGBoost predictors must be numeric. Invalid columns: "
            f"{sorted(invalid_columns)}"
        )

    return feature_columns


def calculate_direction_thresholds(
    actual_returns: pd.Series,
) -> tuple[float, float]:
    lower_threshold = actual_returns[
        actual_returns < 0
    ].median()

    upper_threshold = actual_returns[
        actual_returns > 0
    ].median()

    if pd.isna(lower_threshold) or pd.isna(upper_threshold):
        raise ValueError(
            "Training data cannot produce valid direction thresholds."
        )

    return float(lower_threshold), float(upper_threshold)


def classify_direction(
    values: pd.Series,
    lower_threshold: float,
    upper_threshold: float,
) -> pd.Series:
    labels = pd.Series(
        "FLAT",
        index=values.index,
        dtype="object",
    )

    labels.loc[values < lower_threshold] = "DOWN"
    labels.loc[values > upper_threshold] = "UP"

    return labels


def validate_split(
    dataframe: pd.DataFrame,
    split_name: str,
) -> None:
    if dataframe.empty:
        raise ValueError(
            f"The residual {split_name} split is empty."
        )

    if dataframe["trade_date"].duplicated().any():
        raise ValueError(
            f"The residual {split_name} split contains "
            "duplicate trade dates."
        )


def format_date(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def main():
    repository = TrainingDataRepository()
    serializer = ModelSerializer()
    splitter = DateAwareDataSplitter()

    features = repository.get_training_data(TICKER)
    features["trade_date"] = pd.to_datetime(
        features["trade_date"]
    )

    forecasts = load_sarimax_forecasts()

    residual_dataset = ResidualDatasetBuilder().build(
        features=features,
        forecasts=forecasts,
    )

    train, validation, test = splitter.split(
        dataframe=residual_dataset,
        reference_dates=features["trade_date"],
    )

    validate_split(train, "training")
    validate_split(validation, "validation")

    feature_columns = get_feature_columns(
        residual_dataset
    )

    X_train = train[feature_columns].copy()
    y_train = train["sarimax_residual"].copy()

    X_validation = validation[
        feature_columns
    ].copy()

    training_actual = (
        train["sarimax_prediction"]
        + train["sarimax_residual"]
    )

    validation_actual = (
        validation["sarimax_prediction"]
        + validation["sarimax_residual"]
    )

    lower_threshold, upper_threshold = (
        calculate_direction_thresholds(
            training_actual
        )
    )

    parameter_selector = XGBoostParameterSelector()

    parameters = (
        parameter_selector.select_best_parameters(
            X=X_train,
            y=y_train,
        )
    )

    model = XGBoostTrainer().train(
        X_train=X_train,
        y_train=y_train,
        parameters=parameters,
    )
    print("\nTRAINING DATA")
    print("----------------")
    print(X_train.shape)
    print()

    print("TARGET")
    print("----------------")
    print(y_train.describe())
    print()

    print("FEATURE VARIANCE")
    print("----------------")
    print(X_train.var().sort_values().head(20))
    print()

    print("FEATURE IMPORTANCE")
    print("----------------")
    print(model.feature_importances_)
    print()

    print("BOOSTER SPLITS")
    print("----------------")
    print(model.get_booster().get_score())
    print()
    predicted_residuals = XGBoostPredictor().predict(
        model=model,
        dataset=X_validation,
    )

    predicted_residual_frame = pd.DataFrame(
        {
            "trade_date": validation[
                "trade_date"
            ].to_numpy(),
            "predicted_residual": (
                predicted_residuals.to_numpy()
            ),
        }
    )

    validation_results = validation[
        [
            "trade_date",
            "sarimax_prediction",
            "sarimax_residual",
        ]
    ].merge(
        predicted_residual_frame,
        on="trade_date",
        how="inner",
        validate="one_to_one",
    )

    if len(validation_results) != len(validation):
        raise ValueError(
            "XGBoost predictions did not align with every "
            "validation trade date."
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

    comparison_metrics = (
        ModelComparisonEvaluator().evaluate(
            actual_labels=classify_direction(
                validation_results[
                    "actual_log_return"
                ],
                lower_threshold,
                upper_threshold,
            ),
            sarimax_labels=classify_direction(
                validation_results[
                    "sarimax_prediction"
                ],
                lower_threshold,
                upper_threshold,
            ),
            xgboost_labels=classify_direction(
                validation_results[
                    "corrected_prediction"
                ],
                lower_threshold,
                upper_threshold,
            ),
        )
    )

    metrics = {
        "sarimax_regression": (
            sarimax_regression_metrics
        ),
        "xgboost_regression": (
            xgboost_regression_metrics
        ),
        "classification_comparison": (
            comparison_metrics
        ),
        "direction_thresholds": {
            "lower": lower_threshold,
            "upper": upper_threshold,
        },
        "observations": {
            "train": len(train),
            "validation": len(validation),
            "test_held_out": len(test),
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
        "ticker": TICKER,
        "target": "sarimax_residual",
        "parameters": parameters,
        "features": feature_columns,
        "metrics": metrics,
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
    print("SARIMAX")
    for metric, value in comparison_metrics[
        "sarimax"
    ].items():
        print(f"{metric:<12}{value:.4f}")

    print()
    print("XGBOOST")
    for metric, value in comparison_metrics[
        "xgboost"
    ].items():
        print(f"{metric:<12}{value:.4f}")


if __name__ == "__main__":
    main()