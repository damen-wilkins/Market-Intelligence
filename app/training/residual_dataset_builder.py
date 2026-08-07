import pandas as pd

from app.training.feature_contract import (
    RESIDUAL_BASE_FEATURE_COLUMNS,
    require_columns,
)


class ResidualDatasetBuilder:
    def build(
        self,
        features: pd.DataFrame,
        forecasts: pd.DataFrame,
    ) -> pd.DataFrame:
        require_columns(
            features,
            ["trade_date", *RESIDUAL_BASE_FEATURE_COLUMNS],
            "Feature data",
        )
        require_columns(
            forecasts,
            [
                "trade_date",
                "actual_log_return",
                "sarimax_prediction",
                "is_out_of_sample",
                "data_split",
            ],
            "SARIMAX forecast data",
        )

        feature_data = features[
            ["trade_date", *RESIDUAL_BASE_FEATURE_COLUMNS]
        ].copy()
        forecast_data = forecasts[
            [
                "trade_date",
                "actual_log_return",
                "sarimax_prediction",
                "is_out_of_sample",
                "data_split",
            ]
        ].copy()

        feature_data["trade_date"] = pd.to_datetime(
            feature_data["trade_date"]
        )
        forecast_data["trade_date"] = pd.to_datetime(
            forecast_data["trade_date"]
        )

        if feature_data["trade_date"].isna().any():
            raise ValueError(
                "Feature data contains invalid trade dates."
            )

        if forecast_data["trade_date"].isna().any():
            raise ValueError(
                "SARIMAX forecast data contains invalid trade dates."
            )

        if feature_data["trade_date"].duplicated().any():
            raise ValueError(
                "Feature data contains duplicate trade dates."
            )

        if forecast_data["trade_date"].duplicated().any():
            raise ValueError(
                "SARIMAX forecast data contains duplicate trade dates."
            )

        if not forecast_data["is_out_of_sample"].astype(bool).all():
            raise ValueError(
                "Residual learning requires out-of-sample SARIMAX "
                "predictions for every row."
            )

        feature_data = feature_data.sort_values(
            "trade_date"
        ).reset_index(drop=True)

        feature_data[
            list(RESIDUAL_BASE_FEATURE_COLUMNS)
        ] = feature_data[
            list(RESIDUAL_BASE_FEATURE_COLUMNS)
        ].shift(1)

        dataset = forecast_data.merge(
            feature_data,
            on="trade_date",
            how="left",
            validate="one_to_one",
        )

        required_model_columns = [
            *RESIDUAL_BASE_FEATURE_COLUMNS,
            "actual_log_return",
            "sarimax_prediction",
        ]

        invalid_rows = dataset[
            required_model_columns
        ].isna().any(axis=1)

        if invalid_rows.any():
            invalid_dates = dataset.loc[
                invalid_rows,
                "trade_date",
            ].dt.strftime("%Y-%m-%d").tolist()

            raise ValueError(
                "Residual dataset contains missing model inputs on "
                f"forecast dates: {invalid_dates[:10]}"
            )

        dataset["sarimax_residual"] = (
            dataset["actual_log_return"]
            - dataset["sarimax_prediction"]
        )

        dataset = dataset.drop(
            columns=[
                "actual_log_return",
                "is_out_of_sample",
            ]
        )

        return dataset[
            [
                "trade_date",
                *RESIDUAL_BASE_FEATURE_COLUMNS,
                "sarimax_prediction",
                "sarimax_residual",
                "data_split",
            ]
        ].sort_values(
            "trade_date"
        ).reset_index(drop=True)
