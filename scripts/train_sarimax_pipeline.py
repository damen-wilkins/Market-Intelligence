from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from app.training.date_aware_data_splitter import DateAwareDataSplitter
from app.training.feature_contract import (
    REFERENCE_DATE_MACRO_FEATURE_COLUMNS,
    SARIMAX_EXOGENOUS_COLUMNS,
    require_columns,
)
from app.training.model_serializer import ModelSerializer
from app.training.sarimax_evaluator import SarimaxEvaluator
from app.training.sarimax_trainer import SarimaxTrainer
from database.training_data_repository import TrainingDataRepository


TICKER = "SPY"
MODEL_NAME = "sarimax"
MODEL_VERSION = "sarimax_baseline_v2_walk_forward"
OOF_SPLITS = 5
FORECAST_OUTPUT_PATH = Path("models/sarimax_forecasts.csv")


def prepare_model_data(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        dataset,
        [
            "trade_date",
            "log_return",
            *SARIMAX_EXOGENOUS_COLUMNS,
        ],
        "SARIMAX training data",
    )

    model_data = dataset[
        [
            "trade_date",
            "log_return",
            *SARIMAX_EXOGENOUS_COLUMNS,
        ]
    ].copy()

    model_data["trade_date"] = pd.to_datetime(
        model_data["trade_date"]
    )

    if model_data["trade_date"].isna().any():
        raise ValueError(
            "SARIMAX training data contains invalid trade dates."
        )

    if model_data["trade_date"].duplicated().any():
        raise ValueError(
            "SARIMAX training data contains duplicate trade dates."
        )

    model_data = model_data.sort_values(
        "trade_date"
    ).reset_index(drop=True)

    model_data[
        list(SARIMAX_EXOGENOUS_COLUMNS)
    ] = model_data[
        list(SARIMAX_EXOGENOUS_COLUMNS)
    ].shift(1)

    model_data = model_data.dropna(
        subset=[
            "log_return",
            *SARIMAX_EXOGENOUS_COLUMNS,
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
) -> pd.DataFrame:
    return dataframe[
        list(SARIMAX_EXOGENOUS_COLUMNS)
    ].copy()


def generate_walk_forward_forecasts(
    model,
    holdout: pd.DataFrame,
    data_split: str,
) -> pd.DataFrame:
    if holdout.empty:
        raise ValueError(
            f"The SARIMAX {data_split} forecast window is empty."
        )

    current_results = model
    forecast_rows = []

    for position in range(len(holdout)):
        row = holdout.iloc[position]
        exogenous_values = row[
            list(SARIMAX_EXOGENOUS_COLUMNS)
        ].to_numpy(dtype=float).reshape(1, -1)

        prediction = current_results.forecast(
            steps=1,
            exog=exogenous_values,
        )
        prediction_value = float(
            np.asarray(prediction).reshape(-1)[0]
        )
        actual_value = float(row["log_return"])

        forecast_rows.append(
            {
                "trade_date": row["trade_date"],
                "actual_log_return": actual_value,
                "sarimax_prediction": prediction_value,
                "is_out_of_sample": True,
                "data_split": data_split,
                "forecast_mode": "one_step_walk_forward",
            }
        )

        current_results = current_results.extend(
            endog=np.asarray(
                [actual_value],
                dtype=float,
            ),
            exog=exogenous_values,
        )

    return pd.DataFrame(forecast_rows)


def generate_training_oof_forecasts(
    train: pd.DataFrame,
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

        fold_results = SarimaxTrainer().train(
            endog=fold_train["log_return"],
            exog=get_exogenous_data(fold_train),
        )
        fold_model = fold_results["model"]

        if fold_model is None or fold_results["order"] is None:
            raise RuntimeError(
                f"SARIMAX out-of-fold split {fold_number} "
                "did not produce a valid model."
            )

        fold_forecasts = generate_walk_forward_forecasts(
            model=fold_model,
            holdout=fold_forecast,
            data_split="train",
        )
        fold_forecasts["oof_fold"] = fold_number
        forecast_frames.append(fold_forecasts)

        fold_metadata.append(
            {
                "fold": fold_number,
                "order": fold_results["order"],
                "seasonal_order": fold_results[
                    "seasonal_order"
                ],
                "aicc": fold_results["aicc"],
                "fit_start": format_date(
                    fold_train["trade_date"].min()
                ),
                "fit_end": format_date(
                    fold_train["trade_date"].max()
                ),
                "forecast_start": format_date(
                    fold_forecast["trade_date"].min()
                ),
                "forecast_end": format_date(
                    fold_forecast["trade_date"].max()
                ),
                "forecast_mode": "one_step_walk_forward",
            }
        )

    forecasts = pd.concat(
        forecast_frames,
        ignore_index=True,
    )

    if forecasts["trade_date"].duplicated().any():
        raise ValueError(
            "SARIMAX out-of-fold forecasts contain duplicate dates."
        )

    return forecasts, fold_metadata


def format_date(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def main():
    repository = TrainingDataRepository()
    splitter = DateAwareDataSplitter()
    trainer = SarimaxTrainer()
    evaluator = SarimaxEvaluator()
    serializer = ModelSerializer()

    raw_dataset = repository.get_training_data(
        ticker=TICKER,
        macro_feature_names=SARIMAX_EXOGENOUS_COLUMNS,
    )
    raw_dataset["trade_date"] = pd.to_datetime(
        raw_dataset["trade_date"]
    )

    dataset = prepare_model_data(raw_dataset)

    train, validation, test = splitter.split(
        dataframe=dataset,
        reference_dates=raw_dataset["trade_date"],
    )

    training_results = trainer.train(
        endog=train["log_return"],
        exog=get_exogenous_data(train),
    )
    model = training_results["model"]
    order = training_results["order"]
    seasonal_order = training_results[
        "seasonal_order"
    ]

    if model is None or order is None:
        raise RuntimeError(
            "SARIMAX parameter selection did not produce "
            "a valid model."
        )

    training_forecasts, oof_fold_metadata = (
        generate_training_oof_forecasts(train)
    )

    validation_forecasts = generate_walk_forward_forecasts(
        model=model,
        holdout=validation,
        data_split="validation",
    )
    validation_forecasts["oof_fold"] = pd.NA

    forecasts = pd.concat(
        [
            training_forecasts,
            validation_forecasts,
        ],
        ignore_index=True,
    ).sort_values(
        "trade_date"
    ).reset_index(drop=True)

    if forecasts["trade_date"].duplicated().any():
        raise ValueError(
            "SARIMAX forecast artifact contains duplicate dates."
        )

    if forecasts[
        [
            "actual_log_return",
            "sarimax_prediction",
        ]
    ].isna().any().any():
        raise ValueError(
            "SARIMAX forecast artifact contains missing values."
        )

    if (forecasts["data_split"] == "test").any():
        raise ValueError(
            "The development SARIMAX artifact must not contain "
            "test predictions."
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
        "exogenous_features": list(
            SARIMAX_EXOGENOUS_COLUMNS
        ),
        "excluded_reference_date_macro_features": list(
            REFERENCE_DATE_MACRO_FEATURE_COLUMNS
        ),
        "exogenous_lag_periods": 1,
        "forecast_mode": "one_step_walk_forward",
        "oof_splits": OOF_SPLITS,
        "oof_fold_models": oof_fold_metadata,
        "validation_metrics": validation_metrics,
        "observations": {
            "train": len(train),
            "train_oof": len(training_forecasts),
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
            "test_start": format_date(
                test["trade_date"].min()
            ),
            "test_end": format_date(
                test["trade_date"].max()
            ),
        },
        "test_predictions_generated": False,
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
    print(f"Forecasts   : {FORECAST_OUTPUT_PATH}")
    print(f"Order       : {order}")
    print(f"Seasonal    : {seasonal_order}")
    print(f"AICc        : {training_results['aicc']:.4f}")
    print()
    print("VALIDATION METRICS")

    for metric, value in validation_metrics.items():
        print(f"{metric:<24}{value:.6f}")


if __name__ == "__main__":
    main()
