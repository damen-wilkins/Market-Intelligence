from datetime import datetime, timezone
from pathlib import Path
import warnings

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.training.date_aware_data_splitter import DateAwareDataSplitter
from app.training.model_serializer import ModelSerializer
from app.training.sarimax_evaluator import SarimaxEvaluator
from app.training.sarimax_predictor import SarimaxPredictor
from app.training.sarimax_trainer import SarimaxTrainer
from database.training_data_repository import TrainingDataRepository


TICKER = "SPY"
MODEL_NAME = "sarimax"
MODEL_VERSION = "sarimax_baseline_v1"
OOF_SPLITS = 5
FORECAST_OUTPUT_PATH = Path("models/sarimax_forecasts.csv")


def get_exogenous_columns(dataset: pd.DataFrame) -> list[str]:
    excluded_columns = {
        "ticker",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "daily_return",
        "log_return",
        "return_1d",
        "return_1w",
        "return_1m",
        "return_1y",
        "label_1d",
        "label_1w",
        "label_1m",
        "label_1y",
    }

    exogenous_columns = [
        column
        for column in dataset.columns
        if column not in excluded_columns
    ]

    invalid_columns = [
        column
        for column in exogenous_columns
        if not is_numeric_dtype(dataset[column])
    ]

    if invalid_columns:
        raise ValueError(
            "SARIMAX exogenous features must be numeric. "
            f"Invalid columns: {sorted(invalid_columns)}"
        )

    return exogenous_columns


def prepare_model_data(
    dataset: pd.DataFrame,
    exogenous_columns: list[str],
) -> pd.DataFrame:
    model_data = dataset.copy()

    model_data["trade_date"] = pd.to_datetime(
        model_data["trade_date"]
    )

    if model_data["trade_date"].isna().any():
        raise ValueError(
            "Training data contains invalid trade dates."
        )

    if model_data["trade_date"].duplicated().any():
        raise ValueError(
            "Training data contains duplicate trade dates."
        )

    model_data = model_data.sort_values(
        "trade_date"
    ).reset_index(drop=True)

    if exogenous_columns:
        model_data[exogenous_columns] = model_data[
            exogenous_columns
        ].shift(1)

    model_data = model_data.dropna(
        subset=[
            "log_return",
            *exogenous_columns,
        ]
    ).reset_index(drop=True)

    if model_data.empty:
        raise ValueError(
            "No valid observations remain after preparing "
            "SARIMAX data."
        )

    return model_data


def get_exogenous_data(
    dataframe: pd.DataFrame,
    exogenous_columns: list[str],
) -> pd.DataFrame | None:
    if not exogenous_columns:
        return None

    return dataframe[exogenous_columns].copy()


def fit_fixed_sarimax(
    dataframe: pd.DataFrame,
    exogenous_columns: list[str],
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=ConvergenceWarning,
        )

        model = SARIMAX(
            endog=dataframe["log_return"],
            exog=get_exogenous_data(
                dataframe,
                exogenous_columns,
            ),
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        return model.fit(disp=False)


def build_forecast_frame(
    dataframe: pd.DataFrame,
    predictions: pd.Series,
    data_split: str,
) -> pd.DataFrame:
    if len(dataframe) != len(predictions):
        raise ValueError(
            f"SARIMAX {data_split} predictions do not match "
            "the number of target observations."
        )

    return pd.DataFrame(
        {
            "trade_date": dataframe[
                "trade_date"
            ].to_numpy(),
            "actual_log_return": dataframe[
                "log_return"
            ].to_numpy(),
            "sarimax_prediction": predictions.to_numpy(),
            "is_out_of_sample": True,
            "data_split": data_split,
        }
    )


def format_date(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def generate_training_oof_forecasts(
    train: pd.DataFrame,
    exogenous_columns: list[str],
    predictor: SarimaxPredictor,
) -> tuple[pd.DataFrame, list[dict]]:
    if len(train) <= OOF_SPLITS:
        raise ValueError(
            "The SARIMAX training split is too small for "
            f"{OOF_SPLITS} out-of-fold splits."
        )

    splitter = TimeSeriesSplit(
        n_splits=OOF_SPLITS
    )

    forecast_frames = []
    fold_metadata = []

    for fold_number, (
        fit_indices,
        forecast_indices,
    ) in enumerate(
        splitter.split(train),
        start=1,
    ):
        fold_train = train.iloc[
            fit_indices
        ].reset_index(drop=True)

        fold_forecast = train.iloc[
            forecast_indices
        ].reset_index(drop=True)

        try:
            fold_results = SarimaxTrainer().train(
                endog=fold_train["log_return"],
                exog=get_exogenous_data(
                    fold_train,
                    exogenous_columns,
                ),
            )
        except Exception as error:
            raise RuntimeError(
                "SARIMAX parameter selection failed for "
                f"out-of-fold split {fold_number}."
            ) from error

        fold_model = fold_results["model"]

        if (
            fold_model is None
            or fold_results["order"] is None
        ):
            raise RuntimeError(
                f"SARIMAX out-of-fold split {fold_number} "
                "did not produce a valid model."
            )

        predictions = predictor.predict(
            model=fold_model,
            start=len(fold_train),
            end=(
                len(fold_train)
                + len(fold_forecast)
                - 1
            ),
            exog=get_exogenous_data(
                fold_forecast,
                exogenous_columns,
            ),
        )

        forecast_frames.append(
            build_forecast_frame(
                dataframe=fold_forecast,
                predictions=predictions,
                data_split="train",
            )
        )

        fold_metadata.append(
            {
                "fold": fold_number,
                "order": fold_results["order"],
                "seasonal_order": fold_results[
                    "seasonal_order"
                ],
                "aicc": fold_results["aicc"],
                "fit_start": format_date(
                    fold_train[
                        "trade_date"
                    ].min()
                ),
                "fit_end": format_date(
                    fold_train[
                        "trade_date"
                    ].max()
                ),
                "forecast_start": format_date(
                    fold_forecast[
                        "trade_date"
                    ].min()
                ),
                "forecast_end": format_date(
                    fold_forecast[
                        "trade_date"
                    ].max()
                ),
            }
        )

    forecasts = pd.concat(
        forecast_frames,
        ignore_index=True,
    )

    if forecasts["trade_date"].duplicated().any():
        raise ValueError(
            "SARIMAX out-of-fold forecasts contain "
            "duplicate dates."
        )

    return forecasts, fold_metadata


def generate_holdout_forecasts(
    model,
    fit_observations: int,
    holdout: pd.DataFrame,
    exogenous_columns: list[str],
    data_split: str,
    predictor: SarimaxPredictor,
) -> pd.DataFrame:
    predictions = predictor.predict(
        model=model,
        start=fit_observations,
        end=(
            fit_observations
            + len(holdout)
            - 1
        ),
        exog=get_exogenous_data(
            holdout,
            exogenous_columns,
        ),
    )

    return build_forecast_frame(
        dataframe=holdout,
        predictions=predictions,
        data_split=data_split,
    )


def main():
    repository = TrainingDataRepository()
    splitter = DateAwareDataSplitter()
    trainer = SarimaxTrainer()
    predictor = SarimaxPredictor()
    evaluator = SarimaxEvaluator()
    serializer = ModelSerializer()

    raw_dataset = repository.get_training_data(
        TICKER
    )

    raw_dataset["trade_date"] = pd.to_datetime(
        raw_dataset["trade_date"]
    )

    exogenous_columns = get_exogenous_columns(
        raw_dataset
    )

    dataset = prepare_model_data(
        dataset=raw_dataset,
        exogenous_columns=exogenous_columns,
    )

    train, validation, test = splitter.split(
        dataframe=dataset,
        reference_dates=raw_dataset[
            "trade_date"
        ],
    )

    training_results = trainer.train(
        endog=train["log_return"],
        exog=get_exogenous_data(
            train,
            exogenous_columns,
        ),
    )

    model = training_results["model"]
    order = training_results["order"]

    seasonal_order = training_results[
        "seasonal_order"
    ]

    if model is None or order is None:
        raise RuntimeError(
            "SARIMAX parameter selection did not "
            "produce a valid model."
        )

    training_forecasts, oof_fold_metadata = (
        generate_training_oof_forecasts(
            train=train,
            exogenous_columns=exogenous_columns,
            predictor=predictor,
        )
    )

    validation_forecasts = (
        generate_holdout_forecasts(
            model=model,
            fit_observations=len(train),
            holdout=validation,
            exogenous_columns=exogenous_columns,
            data_split="validation",
            predictor=predictor,
        )
    )

    train_validation = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    test_model = fit_fixed_sarimax(
        dataframe=train_validation,
        exogenous_columns=exogenous_columns,
        order=order,
        seasonal_order=seasonal_order,
    )

    test_forecasts = generate_holdout_forecasts(
        model=test_model,
        fit_observations=len(train_validation),
        holdout=test,
        exogenous_columns=exogenous_columns,
        data_split="test",
        predictor=predictor,
    )

    forecasts = pd.concat(
        [
            training_forecasts,
            validation_forecasts,
            test_forecasts,
        ],
        ignore_index=True,
    ).sort_values(
        "trade_date"
    ).reset_index(drop=True)

    if forecasts["trade_date"].duplicated().any():
        raise ValueError(
            "SARIMAX forecast artifact contains "
            "duplicate dates."
        )

    if forecasts[
        [
            "actual_log_return",
            "sarimax_prediction",
        ]
    ].isna().any().any():
        raise ValueError(
            "SARIMAX forecast artifact contains "
            "missing values."
        )

    validation_metrics = evaluator.evaluate(
        actual=validation_forecasts[
            "actual_log_return"
        ],
        predicted=validation_forecasts[
            "sarimax_prediction"
        ],
    )

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "ticker": TICKER,
        "training_date_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "target": "log_return",
        "order": order,
        "seasonal_order": seasonal_order,
        "aicc": training_results["aicc"],
        "exogenous_features": (
            exogenous_columns
        ),
        "exogenous_lag_periods": 1,
        "oof_splits": OOF_SPLITS,
        "oof_fold_models": (
            oof_fold_metadata
        ),
        "validation_metrics": (
            validation_metrics
        ),
        "observations": {
            "train": len(train),
            "train_oof": len(
                training_forecasts
            ),
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
                validation[
                    "trade_date"
                ].min()
            ),
            "validation_end": format_date(
                validation[
                    "trade_date"
                ].max()
            ),
            "test_start": format_date(
                test["trade_date"].min()
            ),
            "test_end": format_date(
                test["trade_date"].max()
            ),
        },
    }

    model_path = serializer.save(
        model=model,
        metadata=metadata,
        filename=MODEL_NAME,
    )

    FORECAST_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecasts.to_csv(
        FORECAST_OUTPUT_PATH,
        index=False,
    )

    print("=" * 60)
    print("SARIMAX PIPELINE COMPLETE")
    print("=" * 60)
    print()
    print(f"Model       : {model_path}")
    print(
        f"Forecasts   : "
        f"{FORECAST_OUTPUT_PATH}"
    )
    print(f"Order       : {order}")
    print(f"Seasonal    : {seasonal_order}")
    print(
        f"AICc        : "
        f"{training_results['aicc']:.4f}"
    )
    print()
    print("VALIDATION METRICS")

    for metric, value in (
        validation_metrics.items()
    ):
        print(f"{metric:<24}{value:.6f}")


if __name__ == "__main__":
    main()