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
    TREND_SLOPE_LOOKBACK = 5
    DMI_WINDOW = 14

    BASE_FEATURE_COLUMNS = [
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
    ]

    TREND_STATE_FEATURE_COLUMNS = [
        "sma_10_20_spread",
        "sma_20_50_spread",
        "sma_10_50_spread",
        "ma_alignment_score",
        "ma_compression",
        "sma_10_slope_5",
        "sma_20_slope_5",
        "sma_50_slope_5",
        "adx_14",
        "dmi_spread_14",
    ]

    BREADTH_FEATURE_COLUMNS = [
        "rsp_relative_return",
        "sector_positive_participation",
        "sector_return_dispersion",
        "sector_average_correlation_20d",
        "sector_volume_breadth",
    ]

    CROSS_ASSET_FEATURE_GROUPS = {
        "equity_rotation": [
            "qqq_relative_return",
            "iwm_relative_return",
            "dia_relative_return",
        ],
        "rates_duration": [
            "tlt_return",
            "ief_return",
            "tlt_ief_relative_return",
        ],
        "credit_risk": [
            "hyg_return",
            "lqd_return",
            "hyg_lqd_relative_return",
        ],
        "safe_haven": [
            "gld_return",
        ],
    }

    CROSS_ASSET_FEATURE_COLUMNS = [
        feature
        for group_features in CROSS_ASSET_FEATURE_GROUPS.values()
        for feature in group_features
    ]

    FEATURE_COLUMNS = [
        *BASE_FEATURE_COLUMNS,
        *BREADTH_FEATURE_COLUMNS,
        *CROSS_ASSET_FEATURE_COLUMNS,
    ]

    BASE_REQUIRED_COLUMNS = [
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
    ]

    BREADTH_REQUIRED_COLUMNS = [
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

    CROSS_ASSET_REQUIRED_COLUMNS = [
        "qqq_close",
        "iwm_close",
        "dia_close",
        "tlt_close",
        "ief_close",
        "hyg_close",
        "lqd_close",
        "gld_close",
    ]

    REQUIRED_COLUMNS = [
        *BASE_REQUIRED_COLUMNS,
        *BREADTH_REQUIRED_COLUMNS,
        *CROSS_ASSET_REQUIRED_COLUMNS,
    ]

    VALID_FEATURE_SCOPES = {
        "base",
        "base_trend",
        "base_breadth",
        "base_cross_asset",
        "all",
        "all_trend",
    }

    def __init__(
        self,
        feature_scope: str = "all",
    ):
        if feature_scope not in self.VALID_FEATURE_SCOPES:
            raise ValueError(
                "Feature scope must be one of: "
                f"{sorted(self.VALID_FEATURE_SCOPES)}"
            )

        self.feature_scope = feature_scope
        self.feature_columns = self._resolve_feature_columns()
        self.required_columns = self._resolve_required_columns()

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        require_columns(
            data,
            self.required_columns,
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

        self._add_base_features(result)

        if self._includes_trend_state():
            self._add_trend_state_features(result)

        if self._includes_breadth():
            self._add_breadth_features(result)

        if self._includes_cross_asset():
            self._add_cross_asset_features(result)

        result = result.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        result = result.dropna(
            subset=self.feature_columns
        ).reset_index(drop=True)

        return result[
            [
                "trade_date",
                *self.feature_columns,
            ]
        ]

    def _add_base_features(
        self,
        result: pd.DataFrame,
    ) -> None:
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
            result["close"]
            / result["sma_10"]
        ) - 1.0

        result["close_vs_sma_20"] = (
            result["close"]
            / result["sma_20"]
        ) - 1.0

        result["close_vs_sma_50"] = (
            result["close"]
            / result["sma_50"]
        ) - 1.0

        result["close_vs_ema_20"] = (
            result["close"]
            / result["ema_20"]
        ) - 1.0

        result["macd_normalized"] = (
            result["macd"]
            / result["close"]
        )

        result["macd_signal_normalized"] = (
            result["macd_signal"]
            / result["close"]
        )

        result["macd_histogram_normalized"] = (
            result["macd_histogram"]
            / result["close"]
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

    def _add_trend_state_features(
        self,
        result: pd.DataFrame,
    ) -> None:
        result["sma_10_20_spread"] = (
            result["sma_10"]
            / result["sma_20"]
        ) - 1.0

        result["sma_20_50_spread"] = (
            result["sma_20"]
            / result["sma_50"]
        ) - 1.0

        result["sma_10_50_spread"] = (
            result["sma_10"]
            / result["sma_50"]
        ) - 1.0

        bullish_alignment = (
            (result["sma_10"] > result["sma_20"])
            & (result["sma_20"] > result["sma_50"])
        )

        bearish_alignment = (
            (result["sma_10"] < result["sma_20"])
            & (result["sma_20"] < result["sma_50"])
        )

        result["ma_alignment_score"] = np.select(
            [
                bullish_alignment,
                bearish_alignment,
            ],
            [
                1.0,
                -1.0,
            ],
            default=0.0,
        )

        moving_averages = result[
            [
                "sma_10",
                "sma_20",
                "sma_50",
            ]
        ]

        result["ma_compression"] = (
            moving_averages.max(axis=1)
            - moving_averages.min(axis=1)
        ) / result["close"]

        for window in (
            10,
            20,
            50,
        ):
            column = f"sma_{window}"
            result[
                f"sma_{window}_slope_5"
            ] = np.log(
                result[column]
                / result[column].shift(
                    self.TREND_SLOPE_LOOKBACK
                )
            )

        adx, dmi_spread = self._calculate_adx_dmi(
            high=result["high"],
            low=result["low"],
            close=result["close"],
            window=self.DMI_WINDOW,
        )

        result["adx_14"] = adx
        result["dmi_spread_14"] = dmi_spread

    def _add_breadth_features(
        self,
        result: pd.DataFrame,
    ) -> None:
        rsp_return = self._log_return(
            result["rsp_close"]
        )

        result["rsp_relative_return"] = (
            rsp_return
            - result["log_return"]
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

            result[return_column] = self._log_return(
                result[close_column]
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
                np.sign(
                    result[return_column]
                )
                * relative_volume
            )

            signed_relative_volume_columns.append(
                signed_volume_column
            )

        sector_returns = result[
            sector_return_columns
        ]

        result[
            "sector_positive_participation"
        ] = (
            sector_returns.gt(0).sum(axis=1)
            / len(self.SECTOR_SYMBOLS)
        )

        result[
            "sector_return_dispersion"
        ] = sector_returns.std(
            axis=1,
            ddof=0,
        )

        correlation_series = []

        for left_index in range(
            len(sector_return_columns)
        ):
            for right_index in range(
                left_index + 1,
                len(sector_return_columns),
            ):
                left_column = (
                    sector_return_columns[
                        left_index
                    ]
                )
                right_column = (
                    sector_return_columns[
                        right_index
                    ]
                )

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

        result[
            "sector_average_correlation_20d"
        ] = (
            pd.concat(
                correlation_series,
                axis=1,
            )
            .mean(axis=1)
        )

        result[
            "sector_volume_breadth"
        ] = (
            result[
                signed_relative_volume_columns
            ]
            .mean(axis=1)
        )

    def _add_cross_asset_features(
        self,
        result: pd.DataFrame,
    ) -> None:
        qqq_return = self._log_return(
            result["qqq_close"]
        )
        iwm_return = self._log_return(
            result["iwm_close"]
        )
        dia_return = self._log_return(
            result["dia_close"]
        )
        tlt_return = self._log_return(
            result["tlt_close"]
        )
        ief_return = self._log_return(
            result["ief_close"]
        )
        hyg_return = self._log_return(
            result["hyg_close"]
        )
        lqd_return = self._log_return(
            result["lqd_close"]
        )
        gld_return = self._log_return(
            result["gld_close"]
        )

        result["qqq_relative_return"] = (
            qqq_return
            - result["log_return"]
        )

        result["iwm_relative_return"] = (
            iwm_return
            - result["log_return"]
        )

        result["dia_relative_return"] = (
            dia_return
            - result["log_return"]
        )

        result["tlt_return"] = tlt_return
        result["ief_return"] = ief_return
        result["tlt_ief_relative_return"] = (
            tlt_return
            - ief_return
        )

        result["hyg_return"] = hyg_return
        result["lqd_return"] = lqd_return
        result["hyg_lqd_relative_return"] = (
            hyg_return
            - lqd_return
        )

        result["gld_return"] = gld_return

    def _resolve_feature_columns(self) -> list[str]:
        columns = list(
            self.BASE_FEATURE_COLUMNS
        )

        if self._includes_trend_state():
            columns.extend(
                self.TREND_STATE_FEATURE_COLUMNS
            )

        if self._includes_breadth():
            columns.extend(
                self.BREADTH_FEATURE_COLUMNS
            )

        if self._includes_cross_asset():
            columns.extend(
                self.CROSS_ASSET_FEATURE_COLUMNS
            )

        return columns

    def _resolve_required_columns(self) -> list[str]:
        columns = list(
            self.BASE_REQUIRED_COLUMNS
        )

        if self._includes_breadth():
            columns.extend(
                self.BREADTH_REQUIRED_COLUMNS
            )

        if self._includes_cross_asset():
            columns.extend(
                self.CROSS_ASSET_REQUIRED_COLUMNS
            )

        return columns

    def _includes_trend_state(self) -> bool:
        return self.feature_scope in {
            "base_trend",
            "all_trend",
        }

    def _includes_breadth(self) -> bool:
        return self.feature_scope in {
            "base_breadth",
            "all",
            "all_trend",
        }

    def _includes_cross_asset(self) -> bool:
        return self.feature_scope in {
            "base_cross_asset",
            "all",
            "all_trend",
        }

    @staticmethod
    def _calculate_adx_dmi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int,
    ) -> tuple[pd.Series, pd.Series]:
        previous_close = close.shift(1)

        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        upward_move = high.diff()
        downward_move = -low.diff()

        positive_dm = pd.Series(
            np.where(
                (upward_move > downward_move)
                & (upward_move > 0.0),
                upward_move,
                0.0,
            ),
            index=high.index,
            dtype=np.float64,
        )

        negative_dm = pd.Series(
            np.where(
                (downward_move > upward_move)
                & (downward_move > 0.0),
                downward_move,
                0.0,
            ),
            index=high.index,
            dtype=np.float64,
        )

        average_true_range = (
            true_range
            .ewm(
                alpha=1.0 / window,
                adjust=False,
                min_periods=window,
            )
            .mean()
        )

        positive_di = 100.0 * (
            positive_dm
            .ewm(
                alpha=1.0 / window,
                adjust=False,
                min_periods=window,
            )
            .mean()
            / average_true_range
        )

        negative_di = 100.0 * (
            negative_dm
            .ewm(
                alpha=1.0 / window,
                adjust=False,
                min_periods=window,
            )
            .mean()
            / average_true_range
        )

        denominator = (
            positive_di
            + negative_di
        )

        dx = 100.0 * (
            positive_di
            - negative_di
        ).abs() / denominator

        adx = (
            dx
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .ewm(
                alpha=1.0 / window,
                adjust=False,
                min_periods=window,
            )
            .mean()
        )

        dmi_spread = (
            positive_di
            - negative_di
        ) / 100.0

        return adx, dmi_spread

    @staticmethod
    def _log_return(
        series: pd.Series,
    ) -> pd.Series:
        return np.log(
            series
            / series.shift(1)
        )
