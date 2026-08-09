import numpy as np
import pandas as pd

from app.training.feature_contract import require_columns


class VolatilityDirectionLabelBuilder:
    def __init__(
        self,
        volatility_window: int = 20,
        threshold_multiplier: float = 0.15,
    ):
        if volatility_window <= 1:
            raise ValueError(
                "Volatility window must be greater than one."
            )

        if threshold_multiplier <= 0:
            raise ValueError(
                "Threshold multiplier must be greater than zero."
            )

        self.volatility_window = volatility_window
        self.threshold_multiplier = threshold_multiplier

    def build(
        self,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        require_columns(
            market_data,
            ["trade_date", "close"],
            "Market data",
        )

        data = market_data[
            ["trade_date", "close"]
        ].copy()

        data["trade_date"] = pd.to_datetime(
            data["trade_date"]
        )

        if data["trade_date"].isna().any():
            raise ValueError(
                "Market data contains invalid trade dates."
            )

        if data["trade_date"].duplicated().any():
            raise ValueError(
                "Market data contains duplicate trade dates."
            )

        if data["close"].isna().any():
            raise ValueError(
                "Market data contains missing close prices."
            )

        if (data["close"] <= 0).any():
            raise ValueError(
                "Market data contains non-positive close prices."
            )

        data = data.sort_values(
            "trade_date"
        ).reset_index(drop=True)

        data["log_return"] = np.log(
            data["close"] / data["close"].shift(1)
        )

        data["rolling_volatility"] = (
            data["log_return"]
            .rolling(
                window=self.volatility_window,
                min_periods=self.volatility_window,
            )
            .std()
        )

        data["threshold"] = (
            data["rolling_volatility"]
            * self.threshold_multiplier
        )

        data["target_date"] = data[
            "trade_date"
        ].shift(-1)

        data["future_log_return"] = np.log(
            data["close"].shift(-1)
            / data["close"]
        )

        data["direction"] = "FLAT"

        data.loc[
            data["future_log_return"]
            < -data["threshold"],
            "direction",
        ] = "DOWN"

        data.loc[
            data["future_log_return"]
            > data["threshold"],
            "direction",
        ] = "UP"

        data = data.dropna(
            subset=[
                "rolling_volatility",
                "threshold",
                "target_date",
                "future_log_return",
            ]
        ).reset_index(drop=True)

        data = data.rename(
            columns={
                "trade_date": "feature_date",
            }
        )

        return data[
            [
                "feature_date",
                "target_date",
                "future_log_return",
                "rolling_volatility",
                "threshold",
                "direction",
            ]
        ]