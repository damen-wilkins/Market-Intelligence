import numpy as np
import pandas as pd

from app.training.feature_contract import require_columns


class DirectionFeatureBuilder:
    SECTOR_SYMBOLS = (
        "xlb",
        "xle",
        "xlf",
        "xli",
        "xlk",
        "xlp",
        "xlu",
        "xlv",
        "xly",
    )

    BREADTH_WINDOW = 20

    FEATURE_COLUMNS = [
        "log_return",
        "overnight_gap",
        "intraday_return",
        "high_low_range",
        "log_volume_change",
        "close_vs_sma_10",
        "close_vs_sma_20",
        "close_vs_sma_50",
        "close_vs_ema_20",
        "rsi_14",
        "macd_normalized",
        "macd_signal_normalized",
        "macd_histogram_normalized",
        "bollinger_width",
        "bollinger_position",
        "vix_level",
        "vix_change",
        "vvix_level",
        "vvix_change",
        "vvix_vix_ratio",
        "realized_volatility_20",
        "rsp_relative_return",
        "sector_positive_participation",
        "sector_return_dispersion",
        "sector_average_correlation_20d",
        "sector_volume_breadth",
    ]

    REQUIRED_COLUMNS = [
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_10",
        "sma_20",
        "sma_50",
        "ema_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "bollinger_upper",
        "bollinger_middle",
        "bollinger_lower",
        "log_return",
        "vix_close",
        "vvix_close",
        "rsp_close",
        "xlb_close",
        "xlb_volume",
        "xle_close",
        "xle_volume",
        "xlf_close",
        "xlf_volume",
        "xli_close",
        "xli_volume",
        "xlk_close",
        "xlk_volume",
        "xlp_close",
        "xlp_volume",
        "xlu_close",
        "xlu_volume",
        "xlv_close",
        "xlv_volume",
        "xly_close",
        "xly_volume",
    ]

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        require_columns(
            data,
            self.REQUIRED_COLUMNS,
            "Direction training data",
        )

        result = data.copy()

        result["trade_date"] = pd.to_datetime(
            result["trade_date"]
        )

        result = result.sort_values(
            "trade_date"
        ).reset_index(drop=True)

        if result["trade_date"].duplicated().any():
            raise ValueError(
                "Direction training data contains duplicate trade dates."
            )

        result["overnight_gap"] = np.log(
            result["open"]
            / result["close"].shift(1)
        )

        result["intraday_return"] = np.log(
            result["close"]
            / result["open"]
        )

        result["high_low_range"] = np.log(
            result["high"]
            / result["low"]
        )

        result["log_volume_change"] = np.log(
            result["volume"]
            / result["volume"].shift(1)
        )

        result["close_vs_sma_10"] = (
            result["close"] / result["sma_10"]
        ) - 1.0

        result["close_vs_sma_20"] = (
            result["close"] / result["sma_20"]
        ) - 1.0

        result["close_vs_sma_50"] = (
            result["close"] / result["sma_50"]
        ) - 1.0

        result["close_vs_ema_20"] = (
            result["close"] / result["ema_20"]
        ) - 1.0

        result["macd_normalized"] = (
            result["macd"] / result["close"]
        )

        result["macd_signal_normalized"] = (
            result["macd_signal"] / result["close"]
        )

        result["macd_histogram_normalized"] = (
            result["macd_histogram"] / result["close"]
        )

        result["bollinger_width"] = (
            result["bollinger_upper"]
            - result["bollinger_lower"]
        ) / result["bollinger_middle"]

        bollinger_range = (
            result["bollinger_upper"]
            - result["bollinger_lower"]
        )

        result["bollinger_position"] = (
            result["close"]
            - result["bollinger_lower"]
        ) / bollinger_range

        result["vix_level"] = result["vix_close"]

        result["vix_change"] = np.log(
            result["vix_close"]
            / result["vix_close"].shift(1)
        )

        result["vvix_level"] = result["vvix_close"]

        result["vvix_change"] = np.log(
            result["vvix_close"]
            / result["vvix_close"].shift(1)
        )

        result["vvix_vix_ratio"] = (
            result["vvix_close"]
            / result["vix_close"]
        )

        result["realized_volatility_20"] = (
            result["log_return"]
            .rolling(
                window=20,
                min_periods=20,
            )
            .std()
        )

        self._add_breadth_features(result)

        result = result.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        result = result.dropna(
            subset=self.FEATURE_COLUMNS
        ).reset_index(drop=True)

        return result[
            ["trade_date", *self.FEATURE_COLUMNS]
        ]

    def _add_breadth_features(
        self,
        result: pd.DataFrame,
    ) -> None:
        rsp_return = np.log(
            result["rsp_close"]
            / result["rsp_close"].shift(1)
        )

        result["rsp_relative_return"] = (
            rsp_return - result["log_return"]
        )

        sector_return_columns = []
        signed_relative_volume_columns = []

        for symbol in self.SECTOR_SYMBOLS:
            close_column = f"{symbol}_close"
            volume_column = f"{symbol}_volume"
            return_column = f"{symbol}_return"
            signed_volume_column = (
                f"{symbol}_signed_relative_volume"
            )

            result[return_column] = np.log(
                result[close_column]
                / result[close_column].shift(1)
            )

            sector_return_columns.append(
                return_column
            )

            rolling_average_volume = (
                result[volume_column]
                .rolling(
                    window=self.BREADTH_WINDOW,
                    min_periods=self.BREADTH_WINDOW,
                )
                .mean()
            )

            relative_volume = (
                result[volume_column]
                / rolling_average_volume
            )

            result[signed_volume_column] = (
                np.sign(result[return_column])
                * relative_volume
            )

            signed_relative_volume_columns.append(
                signed_volume_column
            )

        sector_returns = result[
            sector_return_columns
        ]

        result["sector_positive_participation"] = (
            sector_returns.gt(0).sum(axis=1)
            / len(self.SECTOR_SYMBOLS)
        )

        result["sector_return_dispersion"] = (
            sector_returns.std(
                axis=1,
                ddof=0,
            )
        )

        correlation_series = []

        for left_index in range(
            len(sector_return_columns)
        ):
            for right_index in range(
                left_index + 1,
                len(sector_return_columns),
            ):
                left_column = sector_return_columns[
                    left_index
                ]
                right_column = sector_return_columns[
                    right_index
                ]

                pair_correlation = (
                    result[left_column]
                    .rolling(
                        window=self.BREADTH_WINDOW,
                        min_periods=self.BREADTH_WINDOW,
                    )
                    .corr(
                        result[right_column]
                    )
                )

                correlation_series.append(
                    pair_correlation
                )

        result["sector_average_correlation_20d"] = (
            pd.concat(
                correlation_series,
                axis=1,
            )
            .mean(axis=1)
        )

        result["sector_volume_breadth"] = (
            result[
                signed_relative_volume_columns
            ]
            .mean(axis=1)
        )