import pandas as pd


class ResidualDatasetBuilder:
    def build(
        self,
        features: pd.DataFrame,
        forecasts: pd.DataFrame,
    ) -> pd.DataFrame:
        required_feature_columns = {"trade_date"}
        required_forecast_columns = {
            "trade_date",
            "actual_log_return",
            "sarimax_prediction",
        }

        missing_feature_columns = (
            required_feature_columns - set(features.columns)
        )
        missing_forecast_columns = (
            required_forecast_columns - set(forecasts.columns)
        )

        if missing_feature_columns:
            raise ValueError(
                "Feature data is missing required columns: "
                f"{sorted(missing_feature_columns)}"
            )

        if missing_forecast_columns:
            raise ValueError(
                "SARIMAX forecast data is missing required columns: "
                f"{sorted(missing_forecast_columns)}"
            )

        feature_data = features.copy()
        forecast_data = forecasts.copy()

        feature_data["trade_date"] = pd.to_datetime(
            feature_data["trade_date"]
        )
        forecast_data["trade_date"] = pd.to_datetime(
            forecast_data["trade_date"]
        )

        if feature_data["trade_date"].duplicated().any():
            raise ValueError(
                "Feature data contains duplicate trade dates."
            )

        if forecast_data["trade_date"].duplicated().any():
            raise ValueError(
                "SARIMAX forecast data contains duplicate trade dates."
            )

        excluded_columns = {
            "ticker",
            "trade_date",
            "log_return",
            "return_1d",
            "return_1w",
            "return_1m",
            "return_1y",
            "label_1d",
            "label_1w",
            "label_1m",
            "label_1y",
            "sarimax_prediction",
            "sarimax_residual",
        }

        predictor_columns = [
            column
            for column in feature_data.columns
            if column not in excluded_columns
        ]

        if not predictor_columns:
            raise ValueError(
                "No predictor columns are available for residual learning."
            )

        feature_data = feature_data.sort_values(
            "trade_date"
        ).reset_index(drop=True)

        feature_data[predictor_columns] = feature_data[
            predictor_columns
        ].shift(1)

        feature_data = feature_data[
            ["trade_date", *predictor_columns]
        ]

        dataset = forecast_data.merge(
            feature_data,
            on="trade_date",
            how="inner",
            validate="one_to_one",
        )

        unmatched_dates = forecast_data.loc[
            ~forecast_data["trade_date"].isin(dataset["trade_date"]),
            "trade_date",
        ]

        if not unmatched_dates.empty:
            raise ValueError(
                "Feature data is missing SARIMAX forecast dates: "
                f"{unmatched_dates.dt.strftime('%Y-%m-%d').tolist()}"
            )

        dataset["sarimax_residual"] = (
            dataset["actual_log_return"]
            - dataset["sarimax_prediction"]
        )

        dataset = dataset.drop(
            columns=["actual_log_return"]
        )

        dataset = dataset.dropna(
            subset=[
                *predictor_columns,
                "sarimax_prediction",
                "sarimax_residual",
            ]
        )

        dataset = dataset.sort_values(
            "trade_date"
        ).reset_index(drop=True)

        return dataset[
            [
                "trade_date",
                *predictor_columns,
                "sarimax_prediction",
                "sarimax_residual",
            ]
        ]