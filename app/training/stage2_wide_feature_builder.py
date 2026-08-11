import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from app.training.direction_feature_builder import DirectionFeatureBuilder
from app.training.feature_contract import require_columns


class Stage2WideFeatureBuilder:
    BASE_FEATURE_COLUMNS = list(
        DirectionFeatureBuilder.BASE_FEATURE_COLUMNS
    )

    FEATURE_GROUPS = {
        "technical_dynamics": [
            "return_2d",
            "return_3d",
            "return_5d",
            "return_10d",
            "return_20d",
            "realized_volatility_5",
            "realized_volatility_10",
            "realized_volatility_40",
            "atr_14_normalized",
            "relative_volume_5",
            "relative_volume_20",
            "close_location_value",
            "candle_body_to_range",
            "gap_intraday_interaction",
            "signed_return_streak",
            "higher_high_share_5",
            "lower_low_share_5",
            "stochastic_14",
            "rsi_centered",
            "macd_histogram_change",
            "bollinger_position_change",
        ],
        "trend_direction": [
            *DirectionFeatureBuilder.TREND_STATE_FEATURE_COLUMNS,
            "close_vs_sma_5",
            "sma_5_10_spread",
            "sma_5_slope_5",
            "ma_5_10_20_50_alignment_score",
            "trend_efficiency_20",
            "signed_trend_efficiency_20",
            "distance_from_252d_high",
            "distance_from_252d_low",
        ],
        "volatility_options_core": [
            "vix_change_5",
            "vvix_change_5",
            "vix3m_level",
            "vix3m_change",
            "vix3m_change_5",
            "skew_level",
            "skew_change",
            "skew_change_5",
            "vxn_level",
            "vxn_change",
            "vxn_vix_ratio",
            "vix_vix3m_ratio",
            "implied_realized_spread_20",
            "implied_realized_ratio_20",
        ],
        "volatility_options_short": [
            "vix9d_level",
            "vix9d_change",
            "vix9d_change_5",
            "vix9d_vix_ratio",
            "vix_term_slope",
            "vix_term_inversion",
        ],
        "breadth": [
            *DirectionFeatureBuilder.BREADTH_FEATURE_COLUMNS,
            "rsp_relative_return_5",
            "rsp_relative_return_20",
            "sector_positive_participation_5d",
            "sector_return_dispersion_5d",
            "sector_breadth_acceleration_5",
            "cyclical_defensive_spread_1d",
            "cyclical_defensive_spread_5d",
            "tech_defensive_spread_1d",
            "tech_defensive_spread_5d",
            "sector_correlation_change_5",
        ],
        "equity_rotation": [
            "qqq_relative_return",
            "iwm_relative_return",
            "dia_relative_return",
            "qqq_relative_return_5",
            "iwm_relative_return_5",
            "dia_relative_return_5",
            "iwm_qqq_relative_return_1d",
            "iwm_qqq_relative_return_5d",
            "qqq_dia_relative_return_1d",
            "qqq_dia_relative_return_5d",
        ],
        "rates_credit": [
            "tlt_return",
            "ief_return",
            "tlt_ief_relative_return",
            "hyg_return",
            "lqd_return",
            "hyg_lqd_relative_return",
            "gld_return",
            "tlt_return_5",
            "ief_return_5",
            "tlt_ief_relative_return_5",
            "hyg_return_5",
            "lqd_return_5",
            "hyg_lqd_relative_return_5",
            "gld_return_5",
            "hyg_tlt_risk_on_1d",
            "hyg_tlt_risk_on_5d",
        ],
        "macro_cross_asset": [
            "dxy_return_1d",
            "dxy_return_5d",
            "dxy_return_20d",
            "crude_return_1d",
            "crude_return_5d",
            "crude_return_20d",
            "gold_dollar_relative_return_1d",
            "gold_dollar_relative_return_5d",
        ],
        "futures_core": [
            "es_return_1d",
            "nq_return_1d",
            "es_return_5d",
            "nq_return_5d",
            "nq_es_relative_return_1d",
            "nq_es_relative_return_5d",
            "es_spy_divergence_1d",
            "es_spy_divergence_5d",
            "es_nq_dispersion_1d",
            "es_nq_dispersion_5d",
        ],
        "futures_smallcap": [
            "rty_return_1d",
            "rty_return_5d",
            "rty_es_relative_return_1d",
            "rty_es_relative_return_5d",
            "rty_nq_relative_return_1d",
            "rty_nq_relative_return_5d",
            "futures_threeway_dispersion_1d",
            "futures_threeway_dispersion_5d",
        ],
        "calendar": [
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
            "turn_of_month_flag",
            "month_end_flag",
            "quarter_end_flag",
            "third_friday_flag",
            "third_friday_week_flag",
        ],
        "interaction_consensus": [
            "risk_on_score_20",
            "directional_consensus_20",
            "signal_disagreement_20",
            "consensus_strength_20",
            "breadth_momentum_interaction",
            "volatility_breadth_interaction",
            "credit_volatility_interaction",
            "term_breadth_interaction",
            "smallcap_credit_confirmation",
            "trend_breadth_confirmation",
            "futures_cash_confirmation",
            "gap_volatility_interaction",
        ],
    }

    SHORT_HISTORY_GROUPS = {
        "volatility_options_short",
        "futures_smallcap",
    }

    FEATURE_COLUMNS = [
        *BASE_FEATURE_COLUMNS,
        *[
            feature
            for group_features in FEATURE_GROUPS.values()
            for feature in group_features
        ],
    ]

    REQUIRED_COLUMNS = [
        *DirectionFeatureBuilder.REQUIRED_COLUMNS,
        "vix9d_close",
        "vix3m_close",
        "skew_close",
        "vxn_close",
        "dxy_close",
        "es_close",
        "nq_close",
        "rty_close",
        "cl_close",
    ]

    CYCLICAL_SECTORS = (
        "xly",
        "xli",
        "xlf",
        "xlk",
    )

    DEFENSIVE_SECTORS = (
        "xlp",
        "xlu",
        "xlv",
    )

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
                f"Unknown Stage-2 feature groups: {invalid}"
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
                "Stage-2 feature contract contains duplicate columns."
            )

        return columns

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def build_library(self, data: pd.DataFrame) -> pd.DataFrame:
        require_columns(
            data,
            self.REQUIRED_COLUMNS,
            "Stage-2 wide signal data",
        )

        raw = data.copy()
        raw["trade_date"] = pd.to_datetime(
            raw["trade_date"]
        )
        raw = raw.sort_values(
            "trade_date"
        ).reset_index(drop=True)

        if raw["trade_date"].duplicated().any():
            raise ValueError(
                "Stage-2 wide signal data contains duplicate trade dates."
            )

        core = DirectionFeatureBuilder(
            feature_scope="all_trend"
        ).build(raw)

        extra = raw.copy()

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                PerformanceWarning,
            )
            self._add_technical_dynamics(extra)
            self._add_extended_trend(extra)
            self._add_volatility_options(extra)
            self._add_extended_breadth(extra)
            self._add_equity_rotation(extra)
            self._add_rates_credit(extra)
            self._add_macro_cross_asset(extra)
            self._add_futures_leadership(extra)
            self._add_calendar(extra)
            self._add_interaction_consensus(extra)

        derived_columns = [
            feature
            for group_features in self.FEATURE_GROUPS.values()
            for feature in group_features
            if feature not in core.columns
        ]

        extra = extra[
            [
                "trade_date",
                *derived_columns,
            ]
        ]

        result = core.merge(
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

    def _add_technical_dynamics(
        self,
        result: pd.DataFrame,
    ) -> None:
        for horizon in (
            2,
            3,
            5,
            10,
            20,
        ):
            result[f"return_{horizon}d"] = np.log(
                result["close"]
                / result["close"].shift(horizon)
            )

        for window in (
            5,
            10,
            40,
        ):
            result[f"realized_volatility_{window}"] = (
                result["log_return"]
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .std()
            )

        previous_close = result["close"].shift(1)
        true_range = pd.concat(
            [
                result["high"] - result["low"],
                (result["high"] - previous_close).abs(),
                (result["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        result["atr_14_normalized"] = (
            true_range
            .ewm(
                alpha=1.0 / 14.0,
                adjust=False,
                min_periods=14,
            )
            .mean()
            / result["close"]
        )

        for window in (
            5,
            20,
        ):
            rolling_volume = (
                result["volume"]
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .mean()
            )
            result[f"relative_volume_{window}"] = (
                result["volume"] / rolling_volume
            )

        daily_range = result["high"] - result["low"]

        result["close_location_value"] = (
            (2.0 * result["close"])
            - result["high"]
            - result["low"]
        ) / daily_range

        result["candle_body_to_range"] = (
            result["close"] - result["open"]
        ) / daily_range

        overnight_gap = np.log(
            result["open"]
            / result["close"].shift(1)
        )
        intraday_return = np.log(
            result["close"] / result["open"]
        )
        result["gap_intraday_interaction"] = (
            overnight_gap * intraday_return
        )

        result["signed_return_streak"] = self._signed_streak(
            result["log_return"]
        )

        result["higher_high_share_5"] = (
            result["high"]
            .diff()
            .gt(0.0)
            .rolling(
                window=5,
                min_periods=5,
            )
            .mean()
        )

        result["lower_low_share_5"] = (
            result["low"]
            .diff()
            .lt(0.0)
            .rolling(
                window=5,
                min_periods=5,
            )
            .mean()
        )

        rolling_low = (
            result["low"]
            .rolling(
                window=14,
                min_periods=14,
            )
            .min()
        )
        rolling_high = (
            result["high"]
            .rolling(
                window=14,
                min_periods=14,
            )
            .max()
        )

        result["stochastic_14"] = (
            2.0
            * (
                (result["close"] - rolling_low)
                / (rolling_high - rolling_low)
            )
            - 1.0
        )

        result["rsi_centered"] = (
            result["rsi_14"] - 50.0
        ) / 50.0

        normalized_histogram = (
            result["macd_histogram"]
            / result["close"]
        )
        result["macd_histogram_change"] = (
            normalized_histogram.diff()
        )

        bollinger_position = (
            result["close"]
            - result["bollinger_lower"]
        ) / (
            result["bollinger_upper"]
            - result["bollinger_lower"]
        )
        result["bollinger_position_change"] = (
            bollinger_position.diff()
        )

    def _add_extended_trend(
        self,
        result: pd.DataFrame,
    ) -> None:
        sma_5 = (
            result["close"]
            .rolling(
                window=5,
                min_periods=5,
            )
            .mean()
        )

        result["close_vs_sma_5"] = (
            result["close"] / sma_5
        ) - 1.0
        result["sma_5_10_spread"] = (
            sma_5 / result["sma_10"]
        ) - 1.0
        result["sma_5_slope_5"] = np.log(
            sma_5 / sma_5.shift(5)
        )

        bullish = (
            (sma_5 > result["sma_10"])
            & (result["sma_10"] > result["sma_20"])
            & (result["sma_20"] > result["sma_50"])
        )
        bearish = (
            (sma_5 < result["sma_10"])
            & (result["sma_10"] < result["sma_20"])
            & (result["sma_20"] < result["sma_50"])
        )

        result["ma_5_10_20_50_alignment_score"] = np.select(
            [bullish, bearish],
            [1.0, -1.0],
            default=0.0,
        )

        absolute_return_sum = (
            result["log_return"]
            .abs()
            .rolling(
                window=20,
                min_periods=20,
            )
            .sum()
        )
        net_return = np.log(
            result["close"]
            / result["close"].shift(20)
        )
        result["trend_efficiency_20"] = (
            net_return.abs()
            / absolute_return_sum
        )
        result["signed_trend_efficiency_20"] = (
            np.sign(net_return)
            * result["trend_efficiency_20"]
        )

        rolling_high = (
            result["close"]
            .rolling(
                window=252,
                min_periods=252,
            )
            .max()
        )
        rolling_low = (
            result["close"]
            .rolling(
                window=252,
                min_periods=252,
            )
            .min()
        )

        result["distance_from_252d_high"] = (
            result["close"] / rolling_high
        ) - 1.0
        result["distance_from_252d_low"] = (
            result["close"] / rolling_low
        ) - 1.0

    def _add_volatility_options(
        self,
        result: pd.DataFrame,
    ) -> None:
        result["vix_change_5"] = np.log(
            result["vix_close"]
            / result["vix_close"].shift(5)
        )
        result["vvix_change_5"] = np.log(
            result["vvix_close"]
            / result["vvix_close"].shift(5)
        )

        for raw_column, prefix in (
            ("vix9d_close", "vix9d"),
            ("vix3m_close", "vix3m"),
            ("skew_close", "skew"),
            ("vxn_close", "vxn"),
        ):
            result[f"{prefix}_level"] = result[raw_column]
            result[f"{prefix}_change"] = np.log(
                result[raw_column]
                / result[raw_column].shift(1)
            )

            if prefix in {
                "vix9d",
                "vix3m",
                "skew",
            }:
                result[f"{prefix}_change_5"] = np.log(
                    result[raw_column]
                    / result[raw_column].shift(5)
                )

        result["vxn_vix_ratio"] = (
            result["vxn_close"]
            / result["vix_close"]
        )
        result["vix9d_vix_ratio"] = (
            result["vix9d_close"]
            / result["vix_close"]
        )
        result["vix_vix3m_ratio"] = (
            result["vix_close"]
            / result["vix3m_close"]
        )
        result["vix_term_slope"] = (
            result["vix9d_close"]
            / result["vix3m_close"]
        ) - 1.0
        vix_term_available = (
            result["vix9d_close"].notna()
            & result["vix3m_close"].notna()
        )
        result["vix_term_inversion"] = np.where(
            vix_term_available,
            (
                result["vix9d_close"]
                > result["vix3m_close"]
            ).astype(np.float64),
            np.nan,
        )

        realized_annualized = (
            result["log_return"]
            .rolling(
                window=20,
                min_periods=20,
            )
            .std()
            * np.sqrt(252.0)
        )
        implied_decimal = (
            result["vix_close"] / 100.0
        )
        result["implied_realized_spread_20"] = (
            implied_decimal - realized_annualized
        )
        result["implied_realized_ratio_20"] = (
            implied_decimal / realized_annualized
        )

    def _add_extended_breadth(
        self,
        result: pd.DataFrame,
    ) -> None:
        spy_return_5 = self._return(result["close"], 5)
        spy_return_20 = self._return(result["close"], 20)
        rsp_return_1 = self._return(result["rsp_close"], 1)
        rsp_return_5 = self._return(result["rsp_close"], 5)
        rsp_return_20 = self._return(result["rsp_close"], 20)

        result["rsp_relative_return_5"] = (
            rsp_return_5 - spy_return_5
        )
        result["rsp_relative_return_20"] = (
            rsp_return_20 - spy_return_20
        )

        sector_return_1 = {}
        sector_return_5 = {}

        for symbol in DirectionFeatureBuilder.SECTOR_SYMBOLS:
            close_column = f"{symbol}_close"
            sector_return_1[symbol] = self._return(
                result[close_column],
                1,
            )
            sector_return_5[symbol] = self._return(
                result[close_column],
                5,
            )

        sector_1 = pd.DataFrame(
            sector_return_1,
            index=result.index,
        )
        sector_5 = pd.DataFrame(
            sector_return_5,
            index=result.index,
        )

        participation_1 = (
            sector_1.gt(0.0).mean(axis=1)
        )
        result["sector_positive_participation_5d"] = (
            sector_5.gt(0.0).mean(axis=1)
        )
        result["sector_return_dispersion_5d"] = (
            sector_5.std(
                axis=1,
                ddof=0,
            )
        )
        result["sector_breadth_acceleration_5"] = (
            participation_1
            - participation_1.rolling(
                window=5,
                min_periods=5,
            ).mean()
        )

        cyclical_1 = sector_1[
            list(self.CYCLICAL_SECTORS)
        ].mean(axis=1)
        defensive_1 = sector_1[
            list(self.DEFENSIVE_SECTORS)
        ].mean(axis=1)
        cyclical_5 = sector_5[
            list(self.CYCLICAL_SECTORS)
        ].mean(axis=1)
        defensive_5 = sector_5[
            list(self.DEFENSIVE_SECTORS)
        ].mean(axis=1)

        result["cyclical_defensive_spread_1d"] = (
            cyclical_1 - defensive_1
        )
        result["cyclical_defensive_spread_5d"] = (
            cyclical_5 - defensive_5
        )
        result["tech_defensive_spread_1d"] = (
            sector_1["xlk"] - defensive_1
        )
        result["tech_defensive_spread_5d"] = (
            sector_5["xlk"] - defensive_5
        )

        correlation_series = []
        for left_index, left_column in enumerate(sector_1.columns):
            for right_column in sector_1.columns[left_index + 1 :]:
                correlation_series.append(
                    sector_1[left_column]
                    .rolling(
                        window=20,
                        min_periods=20,
                    )
                    .corr(
                        sector_1[right_column]
                    )
                )

        average_correlation = pd.concat(
            correlation_series,
            axis=1,
        ).mean(axis=1)
        result["sector_correlation_change_5"] = (
            average_correlation
            - average_correlation.shift(5)
        )

        result["_rsp_return_1"] = rsp_return_1
        result["_sector_participation_1"] = participation_1

    def _add_equity_rotation(
        self,
        result: pd.DataFrame,
    ) -> None:
        spy_5 = self._return(result["close"], 5)
        qqq_1 = self._return(result["qqq_close"], 1)
        iwm_1 = self._return(result["iwm_close"], 1)
        dia_1 = self._return(result["dia_close"], 1)
        qqq_5 = self._return(result["qqq_close"], 5)
        iwm_5 = self._return(result["iwm_close"], 5)
        dia_5 = self._return(result["dia_close"], 5)

        result["qqq_relative_return_5"] = qqq_5 - spy_5
        result["iwm_relative_return_5"] = iwm_5 - spy_5
        result["dia_relative_return_5"] = dia_5 - spy_5
        result["iwm_qqq_relative_return_1d"] = iwm_1 - qqq_1
        result["iwm_qqq_relative_return_5d"] = iwm_5 - qqq_5
        result["qqq_dia_relative_return_1d"] = qqq_1 - dia_1
        result["qqq_dia_relative_return_5d"] = qqq_5 - dia_5

    def _add_rates_credit(
        self,
        result: pd.DataFrame,
    ) -> None:
        tlt_1 = self._return(result["tlt_close"], 1)
        ief_1 = self._return(result["ief_close"], 1)
        hyg_1 = self._return(result["hyg_close"], 1)
        lqd_1 = self._return(result["lqd_close"], 1)
        tlt_5 = self._return(result["tlt_close"], 5)
        ief_5 = self._return(result["ief_close"], 5)
        hyg_5 = self._return(result["hyg_close"], 5)
        lqd_5 = self._return(result["lqd_close"], 5)
        gld_5 = self._return(result["gld_close"], 5)

        result["tlt_return_5"] = tlt_5
        result["ief_return_5"] = ief_5
        result["tlt_ief_relative_return_5"] = tlt_5 - ief_5
        result["hyg_return_5"] = hyg_5
        result["lqd_return_5"] = lqd_5
        result["hyg_lqd_relative_return_5"] = hyg_5 - lqd_5
        result["gld_return_5"] = gld_5
        result["hyg_tlt_risk_on_1d"] = hyg_1 - tlt_1
        result["hyg_tlt_risk_on_5d"] = hyg_5 - tlt_5

    def _add_macro_cross_asset(
        self,
        result: pd.DataFrame,
    ) -> None:
        dxy_1 = self._return(result["dxy_close"], 1)
        dxy_5 = self._return(result["dxy_close"], 5)
        crude_1 = self._return(result["cl_close"], 1)
        crude_5 = self._return(result["cl_close"], 5)

        result["dxy_return_1d"] = dxy_1
        result["dxy_return_5d"] = dxy_5
        result["dxy_return_20d"] = self._return(
            result["dxy_close"],
            20,
        )
        result["crude_return_1d"] = crude_1
        result["crude_return_5d"] = crude_5
        result["crude_return_20d"] = self._return(
            result["cl_close"],
            20,
        )

        gld_1 = self._return(result["gld_close"], 1)
        gld_5 = self._return(result["gld_close"], 5)
        result["gold_dollar_relative_return_1d"] = (
            gld_1 - dxy_1
        )
        result["gold_dollar_relative_return_5d"] = (
            gld_5 - dxy_5
        )

    def _add_futures_leadership(
        self,
        result: pd.DataFrame,
    ) -> None:
        spy_1 = self._return(result["close"], 1)
        spy_5 = self._return(result["close"], 5)
        es_1 = self._return(result["es_close"], 1)
        nq_1 = self._return(result["nq_close"], 1)
        rty_1 = self._return(result["rty_close"], 1)
        es_5 = self._return(result["es_close"], 5)
        nq_5 = self._return(result["nq_close"], 5)
        rty_5 = self._return(result["rty_close"], 5)

        result["es_return_1d"] = es_1
        result["nq_return_1d"] = nq_1
        result["es_return_5d"] = es_5
        result["nq_return_5d"] = nq_5
        result["nq_es_relative_return_1d"] = nq_1 - es_1
        result["nq_es_relative_return_5d"] = nq_5 - es_5
        result["es_spy_divergence_1d"] = es_1 - spy_1
        result["es_spy_divergence_5d"] = es_5 - spy_5
        result["es_nq_dispersion_1d"] = pd.concat(
            [es_1, nq_1],
            axis=1,
        ).std(axis=1, ddof=0)
        result["es_nq_dispersion_5d"] = pd.concat(
            [es_5, nq_5],
            axis=1,
        ).std(axis=1, ddof=0)

        result["rty_return_1d"] = rty_1
        result["rty_return_5d"] = rty_5
        result["rty_es_relative_return_1d"] = rty_1 - es_1
        result["rty_es_relative_return_5d"] = rty_5 - es_5
        result["rty_nq_relative_return_1d"] = rty_1 - nq_1
        result["rty_nq_relative_return_5d"] = rty_5 - nq_5

        threeway_1 = pd.concat(
            [es_1, nq_1, rty_1],
            axis=1,
        )
        threeway_5 = pd.concat(
            [es_5, nq_5, rty_5],
            axis=1,
        )
        result["futures_threeway_dispersion_1d"] = (
            threeway_1.std(
                axis=1,
                ddof=0,
                skipna=False,
            )
        )
        result["futures_threeway_dispersion_5d"] = (
            threeway_5.std(
                axis=1,
                ddof=0,
                skipna=False,
            )
        )

    def _add_calendar(
        self,
        result: pd.DataFrame,
    ) -> None:
        dates = pd.to_datetime(result["trade_date"])
        weekday = dates.dt.weekday.astype(np.float64)
        month = dates.dt.month.astype(np.float64)

        result["day_of_week_sin"] = np.sin(
            2.0 * np.pi * weekday / 5.0
        )
        result["day_of_week_cos"] = np.cos(
            2.0 * np.pi * weekday / 5.0
        )
        result["month_sin"] = np.sin(
            2.0 * np.pi * (month - 1.0) / 12.0
        )
        result["month_cos"] = np.cos(
            2.0 * np.pi * (month - 1.0) / 12.0
        )

        period = dates.dt.to_period("M")
        day_from_start = (
            result.groupby(period).cumcount()
            + 1
        )
        day_from_end = (
            result.iloc[::-1]
            .groupby(period.iloc[::-1])
            .cumcount()
            .iloc[::-1]
            + 1
        )

        result["turn_of_month_flag"] = (
            (day_from_start <= 3)
            | (day_from_end <= 3)
        ).astype(np.float64)
        result["month_end_flag"] = (
            day_from_end <= 3
        ).astype(np.float64)
        result["quarter_end_flag"] = (
            month.isin([3.0, 6.0, 9.0, 12.0])
            & (day_from_end <= 3)
        ).astype(np.float64)

        third_friday = (
            (dates.dt.weekday == 4)
            & (dates.dt.day >= 15)
            & (dates.dt.day <= 21)
        )
        result["third_friday_flag"] = (
            third_friday.astype(np.float64)
        )

        third_friday_by_month = {}
        for current_period in period.unique():
            rows = dates[period == current_period]
            fridays = rows[
                (rows.dt.weekday == 4)
                & (rows.dt.day >= 15)
                & (rows.dt.day <= 21)
            ]
            if len(fridays) > 0:
                third_friday_by_month[current_period] = fridays.iloc[0]

        third_friday_week = []
        for current_date, current_period in zip(dates, period):
            third = third_friday_by_month.get(current_period)
            if third is None:
                third_friday_week.append(0.0)
                continue
            delta_days = int((current_date - third).days)
            third_friday_week.append(
                1.0
                if -4 <= delta_days <= 0
                else 0.0
            )

        result["third_friday_week_flag"] = (
            np.asarray(
                third_friday_week,
                dtype=np.float64,
            )
        )

    def _add_interaction_consensus(
        self,
        result: pd.DataFrame,
    ) -> None:
        spy_5 = self._return(result["close"], 5)
        qqq_rel = (
            self._return(result["qqq_close"], 1)
            - self._return(result["close"], 1)
        )
        iwm_rel = (
            self._return(result["iwm_close"], 1)
            - self._return(result["close"], 1)
        )
        rsp_rel = (
            self._return(result["rsp_close"], 1)
            - self._return(result["close"], 1)
        )
        hyg_lqd = (
            self._return(result["hyg_close"], 1)
            - self._return(result["lqd_close"], 1)
        )
        vix_change = self._return(result["vix_close"], 1)
        dxy_change = self._return(result["dxy_close"], 1)
        es_spy = (
            self._return(result["es_close"], 1)
            - self._return(result["close"], 1)
        )

        sector_returns = pd.DataFrame(
            {
                symbol: self._return(
                    result[f"{symbol}_close"],
                    1,
                )
                for symbol in DirectionFeatureBuilder.SECTOR_SYMBOLS
            },
            index=result.index,
        )
        participation = sector_returns.gt(0.0).mean(axis=1)

        source_signals = pd.DataFrame(
            {
                "qqq_rel": qqq_rel,
                "iwm_rel": iwm_rel,
                "rsp_rel": rsp_rel,
                "credit": hyg_lqd,
                "inverse_vix": -vix_change,
                "inverse_dxy": -dxy_change,
                "breadth": participation - 0.5,
                "futures_cash": es_spy,
            },
            index=result.index,
        )

        standardized = source_signals.apply(
            self._rolling_zscore,
            axis=0,
        )

        result["risk_on_score_20"] = standardized.mean(axis=1)
        result["directional_consensus_20"] = (
            np.sign(standardized).mean(axis=1)
        )
        result["signal_disagreement_20"] = standardized.std(
            axis=1,
            ddof=0,
        )
        result["consensus_strength_20"] = standardized.abs().mean(
            axis=1
        )

        result["breadth_momentum_interaction"] = (
            (participation - 0.5) * spy_5
        )
        result["volatility_breadth_interaction"] = (
            (-vix_change) * (participation - 0.5)
        )
        result["credit_volatility_interaction"] = (
            hyg_lqd * (-vix_change)
        )
        result["term_breadth_interaction"] = (
            (
                result["vix_close"]
                / result["vix3m_close"]
                - 1.0
            )
            * (participation - 0.5)
        )
        result["smallcap_credit_confirmation"] = (
            iwm_rel * hyg_lqd
        )

        bullish = (
            (result["sma_10"] > result["sma_20"])
            & (result["sma_20"] > result["sma_50"])
        )
        bearish = (
            (result["sma_10"] < result["sma_20"])
            & (result["sma_20"] < result["sma_50"])
        )
        alignment = np.select(
            [bullish, bearish],
            [1.0, -1.0],
            default=0.0,
        )
        result["trend_breadth_confirmation"] = (
            alignment * (participation - 0.5)
        )
        result["futures_cash_confirmation"] = (
            es_spy * np.sign(spy_5)
        )

        overnight_gap = np.log(
            result["open"]
            / result["close"].shift(1)
        )
        result["gap_volatility_interaction"] = (
            overnight_gap
            * (result["vix_close"] / 100.0)
        )

    @staticmethod
    def _return(
        series: pd.Series,
        periods: int,
    ) -> pd.Series:
        return np.log(
            series / series.shift(periods)
        )

    @staticmethod
    def _rolling_zscore(
        series: pd.Series,
    ) -> pd.Series:
        mean = series.rolling(
            window=20,
            min_periods=20,
        ).mean()
        std = series.rolling(
            window=20,
            min_periods=20,
        ).std()
        return (series - mean) / std

    @staticmethod
    def _signed_streak(
        returns: pd.Series,
    ) -> pd.Series:
        values = returns.to_numpy(
            dtype=np.float64
        )
        output = np.zeros(
            len(values),
            dtype=np.float64,
        )

        streak = 0
        previous_sign = 0

        for index, value in enumerate(values):
            if not np.isfinite(value) or value == 0.0:
                streak = 0
                previous_sign = 0
                output[index] = 0.0
                continue

            current_sign = 1 if value > 0.0 else -1

            if current_sign == previous_sign:
                streak += current_sign
            else:
                streak = current_sign

            previous_sign = current_sign
            output[index] = float(streak)

        return pd.Series(
            output,
            index=returns.index,
            dtype=np.float64,
        )
