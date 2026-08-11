import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from app.training.stage2_wide_feature_builder import Stage2WideFeatureBuilder


class Stage1WideFeatureBuilder:
    BASE_FEATURE_COLUMNS = list(
        Stage2WideFeatureBuilder.BASE_FEATURE_COLUMNS
    )

    FEATURE_GROUPS = {
        **Stage2WideFeatureBuilder.FEATURE_GROUPS,
        "flat_regime_state": [
            "absolute_return_1d",
            "mean_absolute_return_5",
            "mean_absolute_return_10",
            "mean_absolute_return_20",
            "mean_absolute_return_40",
            "max_absolute_return_5",
            "max_absolute_return_20",
            "realized_volatility_ratio_5_20",
            "realized_volatility_ratio_10_40",
            "realized_volatility_ratio_20_40",
            "return_sign_change_rate_10",
            "return_sign_change_rate_20",
            "return_directional_imbalance_10",
            "return_directional_imbalance_20",
            "return_sign_entropy_20",
            "return_autocorrelation_20",
            "return_autocorrelation_40",
            "range_compression_5_20",
            "bollinger_width_change_5",
            "bollinger_width_zscore_20",
            "choppiness_index_14",
            "efficiency_ratio_10",
            "efficiency_ratio_40",
            "vix_change_abs",
            "vix_change_volatility_10",
            "vvix_vix_ratio_change_abs",
            "breadth_extremity",
            "breadth_instability_5",
            "sector_dispersion_ratio_5_20",
            "cross_asset_dispersion_1d",
            "cross_asset_dispersion_5d",
            "implied_realized_gap_abs",
            "futures_cash_divergence_abs",
        ],
    }

    SHORT_HISTORY_GROUPS = set(
        Stage2WideFeatureBuilder.SHORT_HISTORY_GROUPS
    )

    FEATURE_COLUMNS = [
        *BASE_FEATURE_COLUMNS,
        *[
            feature
            for group_features in FEATURE_GROUPS.values()
            for feature in group_features
        ],
    ]

    def __init__(
        self,
        group_names: list[str] | tuple[str, ...] | None = None,
    ):
        if group_names is None:
            group_names = list(self.FEATURE_GROUPS)

        invalid = sorted(
            set(group_names) - set(self.FEATURE_GROUPS)
        )
        if invalid:
            raise ValueError(
                f"Unknown Stage-1 feature groups: {invalid}"
            )

        self.group_names = list(group_names)
        self.feature_columns = self.columns_for_groups(
            self.group_names
        )

    @classmethod
    def columns_for_groups(
        cls,
        group_names: list[str] | tuple[str, ...],
    ) -> list[str]:
        columns = list(cls.BASE_FEATURE_COLUMNS)

        for group_name in group_names:
            columns.extend(
                cls.FEATURE_GROUPS[group_name]
            )

        if len(columns) != len(set(columns)):
            raise ValueError(
                "Stage-1 feature contract contains duplicate columns."
            )

        return columns

    def build(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        result = self.build_library(data)
        result = result.dropna(
            subset=self.feature_columns
        ).reset_index(drop=True)

        return result[
            [
                "trade_date",
                *self.feature_columns,
            ]
        ]

    def build_library(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        stage2_library = (
            Stage2WideFeatureBuilder()
            .build_library(data)
        )

        raw = data.copy()
        raw["trade_date"] = pd.to_datetime(
            raw["trade_date"]
        )
        raw = raw.sort_values(
            "trade_date"
        ).reset_index(drop=True)

        extra = raw.copy()

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                PerformanceWarning,
            )
            self._add_flat_regime_state(extra)

        flat_columns = self.FEATURE_GROUPS[
            "flat_regime_state"
        ]

        extra = extra[
            [
                "trade_date",
                *flat_columns,
            ]
        ]

        result = stage2_library.merge(
            extra,
            on="trade_date",
            how="inner",
            validate="one_to_one",
        )

        result = result.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return result[
            [
                "trade_date",
                *self.FEATURE_COLUMNS,
            ]
        ]

    def _add_flat_regime_state(
        self,
        result: pd.DataFrame,
    ) -> None:
        log_return = np.log(
            result["close"]
            / result["close"].shift(1)
        )
        absolute_return = log_return.abs()

        result["absolute_return_1d"] = absolute_return

        for window in (
            5,
            10,
            20,
            40,
        ):
            result[f"mean_absolute_return_{window}"] = (
                absolute_return
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .mean()
            )

        result["max_absolute_return_5"] = (
            absolute_return
            .rolling(
                window=5,
                min_periods=5,
            )
            .max()
        )
        result["max_absolute_return_20"] = (
            absolute_return
            .rolling(
                window=20,
                min_periods=20,
            )
            .max()
        )

        rv_5 = self._rolling_std(
            log_return,
            5,
        )
        rv_10 = self._rolling_std(
            log_return,
            10,
        )
        rv_20 = self._rolling_std(
            log_return,
            20,
        )
        rv_40 = self._rolling_std(
            log_return,
            40,
        )

        result["realized_volatility_ratio_5_20"] = (
            rv_5 / rv_20
        )
        result["realized_volatility_ratio_10_40"] = (
            rv_10 / rv_40
        )
        result["realized_volatility_ratio_20_40"] = (
            rv_20 / rv_40
        )

        sign = np.sign(log_return)
        sign_change = (
            sign
            .ne(sign.shift(1))
            .astype(np.float64)
        )
        valid_sign_change = (
            sign.ne(0.0)
            & sign.shift(1).ne(0.0)
        )
        sign_change = sign_change.where(
            valid_sign_change,
            np.nan,
        )

        result["return_sign_change_rate_10"] = (
            sign_change
            .rolling(
                window=10,
                min_periods=10,
            )
            .mean()
        )
        result["return_sign_change_rate_20"] = (
            sign_change
            .rolling(
                window=20,
                min_periods=20,
            )
            .mean()
        )

        result["return_directional_imbalance_10"] = (
            sign
            .rolling(
                window=10,
                min_periods=10,
            )
            .mean()
            .abs()
        )
        result["return_directional_imbalance_20"] = (
            sign
            .rolling(
                window=20,
                min_periods=20,
            )
            .mean()
            .abs()
        )

        up_share_20 = (
            log_return.gt(0.0)
            .astype(np.float64)
            .rolling(
                window=20,
                min_periods=20,
            )
            .mean()
        )
        result["return_sign_entropy_20"] = (
            self._binary_entropy(
                up_share_20
            )
        )

        result["return_autocorrelation_20"] = (
            log_return
            .rolling(
                window=20,
                min_periods=20,
            )
            .corr(
                log_return.shift(1)
            )
        )
        result["return_autocorrelation_40"] = (
            log_return
            .rolling(
                window=40,
                min_periods=40,
            )
            .corr(
                log_return.shift(1)
            )
        )

        true_range = self._true_range(result)
        range_5 = (
            true_range
            .rolling(
                window=5,
                min_periods=5,
            )
            .mean()
        )
        range_20 = (
            true_range
            .rolling(
                window=20,
                min_periods=20,
            )
            .mean()
        )
        result["range_compression_5_20"] = (
            range_5 / range_20
        )

        rolling_std_20 = (
            result["close"]
            .rolling(
                window=20,
                min_periods=20,
            )
            .std()
        )
        rolling_mean_20 = (
            result["close"]
            .rolling(
                window=20,
                min_periods=20,
            )
            .mean()
        )
        bollinger_width = (
            4.0
            * rolling_std_20
            / rolling_mean_20
        )
        result["bollinger_width_change_5"] = (
            bollinger_width
            / bollinger_width.shift(5)
            - 1.0
        )
        result["bollinger_width_zscore_20"] = (
            self._rolling_zscore(
                bollinger_width,
                20,
            )
        )

        result["choppiness_index_14"] = (
            self._choppiness_index(
                high=result["high"],
                low=result["low"],
                close=result["close"],
                window=14,
            )
        )
        result["efficiency_ratio_10"] = (
            self._efficiency_ratio(
                result["close"],
                10,
            )
        )
        result["efficiency_ratio_40"] = (
            self._efficiency_ratio(
                result["close"],
                40,
            )
        )

        vix_change = np.log(
            result["vix_close"]
            / result["vix_close"].shift(1)
        )
        result["vix_change_abs"] = (
            vix_change.abs()
        )
        result["vix_change_volatility_10"] = (
            self._rolling_std(
                vix_change,
                10,
            )
        )

        vvix_vix_ratio = (
            result["vvix_close"]
            / result["vix_close"]
        )
        result["vvix_vix_ratio_change_abs"] = (
            np.log(
                vvix_vix_ratio
                / vvix_vix_ratio.shift(1)
            )
            .abs()
        )

        sector_returns = pd.DataFrame(
            {
                symbol: np.log(
                    result[f"{symbol}_close"]
                    / result[f"{symbol}_close"].shift(1)
                )
                for symbol in (
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
            },
            index=result.index,
        )
        participation = (
            sector_returns.gt(0.0)
            .mean(axis=1)
        )
        dispersion = (
            sector_returns.std(
                axis=1,
                ddof=0,
            )
        )

        result["breadth_extremity"] = (
            participation - 0.5
        ).abs()
        result["breadth_instability_5"] = (
            participation
            .rolling(
                window=5,
                min_periods=5,
            )
            .std()
        )
        result["sector_dispersion_ratio_5_20"] = (
            dispersion
            .rolling(
                window=5,
                min_periods=5,
            )
            .mean()
            / dispersion
            .rolling(
                window=20,
                min_periods=20,
            )
            .mean()
        )

        cross_asset_1d = pd.DataFrame(
            {
                "qqq": self._return(
                    result["qqq_close"],
                    1,
                ),
                "iwm": self._return(
                    result["iwm_close"],
                    1,
                ),
                "dia": self._return(
                    result["dia_close"],
                    1,
                ),
                "tlt": self._return(
                    result["tlt_close"],
                    1,
                ),
                "hyg": self._return(
                    result["hyg_close"],
                    1,
                ),
                "gld": self._return(
                    result["gld_close"],
                    1,
                ),
            },
            index=result.index,
        )
        cross_asset_5d = pd.DataFrame(
            {
                "qqq": self._return(
                    result["qqq_close"],
                    5,
                ),
                "iwm": self._return(
                    result["iwm_close"],
                    5,
                ),
                "dia": self._return(
                    result["dia_close"],
                    5,
                ),
                "tlt": self._return(
                    result["tlt_close"],
                    5,
                ),
                "hyg": self._return(
                    result["hyg_close"],
                    5,
                ),
                "gld": self._return(
                    result["gld_close"],
                    5,
                ),
            },
            index=result.index,
        )

        result["cross_asset_dispersion_1d"] = (
            cross_asset_1d.std(
                axis=1,
                ddof=0,
            )
        )
        result["cross_asset_dispersion_5d"] = (
            cross_asset_5d.std(
                axis=1,
                ddof=0,
            )
        )

        implied_daily_volatility = (
            result["vix_close"]
            / 100.0
            / np.sqrt(252.0)
        )
        result["implied_realized_gap_abs"] = (
            implied_daily_volatility
            - rv_20
        ).abs()

        es_return = self._return(
            result["es_close"],
            1,
        )
        spy_return = self._return(
            result["close"],
            1,
        )
        result["futures_cash_divergence_abs"] = (
            es_return - spy_return
        ).abs()

    @staticmethod
    def _return(
        series: pd.Series,
        periods: int,
    ) -> pd.Series:
        return np.log(
            series / series.shift(periods)
        )

    @staticmethod
    def _rolling_std(
        series: pd.Series,
        window: int,
    ) -> pd.Series:
        return series.rolling(
            window=window,
            min_periods=window,
        ).std()

    @staticmethod
    def _rolling_zscore(
        series: pd.Series,
        window: int,
    ) -> pd.Series:
        mean = series.rolling(
            window=window,
            min_periods=window,
        ).mean()
        std = series.rolling(
            window=window,
            min_periods=window,
        ).std()
        return (series - mean) / std

    @staticmethod
    def _true_range(
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        previous_close = dataframe["close"].shift(1)
        ranges = pd.concat(
            [
                dataframe["high"] - dataframe["low"],
                (dataframe["high"] - previous_close).abs(),
                (dataframe["low"] - previous_close).abs(),
            ],
            axis=1,
        )
        return ranges.max(axis=1)

    @classmethod
    def _choppiness_index(
        cls,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int,
    ) -> pd.Series:
        frame = pd.DataFrame(
            {
                "high": high,
                "low": low,
                "close": close,
            }
        )
        true_range = cls._true_range(frame)
        tr_sum = true_range.rolling(
            window=window,
            min_periods=window,
        ).sum()
        high_max = high.rolling(
            window=window,
            min_periods=window,
        ).max()
        low_min = low.rolling(
            window=window,
            min_periods=window,
        ).min()
        price_range = high_max - low_min

        return (
            100.0
            * np.log10(tr_sum / price_range)
            / np.log10(float(window))
        )

    @staticmethod
    def _efficiency_ratio(
        close: pd.Series,
        window: int,
    ) -> pd.Series:
        net_change = (
            close - close.shift(window)
        ).abs()
        path_length = (
            close.diff()
            .abs()
            .rolling(
                window=window,
                min_periods=window,
            )
            .sum()
        )
        return net_change / path_length

    @staticmethod
    def _binary_entropy(
        probability: pd.Series,
    ) -> pd.Series:
        clipped = probability.clip(
            lower=1e-12,
            upper=1.0 - 1e-12,
        )
        return -(
            clipped * np.log(clipped)
            + (1.0 - clipped)
            * np.log(1.0 - clipped)
        )
