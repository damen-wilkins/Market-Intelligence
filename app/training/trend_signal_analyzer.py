import numpy as np
import pandas as pd


class TrendSignalAnalyzer:
    ALIGNMENT_LABELS = {
        -1.0: "BEARISH",
        0.0: "MIXED",
        1.0: "BULLISH",
    }

    def __init__(
        self,
        training_fraction: float = 0.70,
        stability_blocks: int = 5,
    ):
        if not 0.0 < training_fraction < 1.0:
            raise ValueError(
                "Training fraction must be between zero and one."
            )

        if stability_blocks < 2:
            raise ValueError(
                "At least two stability blocks are required."
            )

        self.training_fraction = float(
            training_fraction
        )
        self.stability_blocks = int(
            stability_blocks
        )

    def analyze(
        self,
        feature_data: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        prepared = self._prepare(
            feature_data=feature_data,
            market_data=market_data,
        )

        training_rows = int(
            np.floor(
                len(prepared)
                * self.training_fraction
            )
        )

        if training_rows < 100:
            raise ValueError(
                "Trend-signal analysis requires at least 100 training rows."
            )

        train = (
            prepared
            .iloc[
                :training_rows
            ]
            .reset_index(
                drop=True
            )
        )

        return {
            "alignment": self._alignment_summary(
                train
            ),
            "adx": self._quantile_summary(
                train=train,
                feature_column="adx_14",
                analysis_name="ADX",
            ),
            "compression": self._quantile_summary(
                train=train,
                feature_column="ma_compression",
                analysis_name="MA_COMPRESSION",
            ),
            "stability": self._alignment_stability(
                train
            ),
        }

    def _prepare(
        self,
        feature_data: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        required_features = {
            "trade_date",
            "ma_alignment_score",
            "ma_compression",
            "adx_14",
            "dmi_spread_14",
            "sma_10_slope_5",
            "sma_20_slope_5",
            "sma_50_slope_5",
            "realized_volatility_20",
        }

        missing_features = (
            required_features
            - set(
                feature_data.columns
            )
        )

        if missing_features:
            raise ValueError(
                "Trend-signal feature data is missing columns: "
                f"{sorted(missing_features)}"
            )

        required_market = {
            "trade_date",
            "close",
        }

        missing_market = (
            required_market
            - set(
                market_data.columns
            )
        )

        if missing_market:
            raise ValueError(
                "Trend-signal market data is missing columns: "
                f"{sorted(missing_market)}"
            )

        market = market_data[
            [
                "trade_date",
                "close",
            ]
        ].copy()

        market[
            "trade_date"
        ] = pd.to_datetime(
            market[
                "trade_date"
            ]
        )

        market = market.sort_values(
            "trade_date"
        ).reset_index(
            drop=True
        )

        market[
            "future_log_return"
        ] = np.log(
            market[
                "close"
            ].shift(-1)
            / market[
                "close"
            ]
        )

        market[
            "target_date"
        ] = market[
            "trade_date"
        ].shift(-1)

        features = feature_data.copy()
        features[
            "trade_date"
        ] = pd.to_datetime(
            features[
                "trade_date"
            ]
        )

        result = features.merge(
            market[
                [
                    "trade_date",
                    "target_date",
                    "future_log_return",
                ]
            ],
            on="trade_date",
            how="inner",
            validate="one_to_one",
        )

        result = result.dropna(
            subset=[
                "target_date",
                "future_log_return",
                "realized_volatility_20",
            ]
        ).sort_values(
            "trade_date"
        ).reset_index(
            drop=True
        )

        result[
            "future_up"
        ] = (
            result[
                "future_log_return"
            ]
            > 0.0
        ).astype(
            np.int64
        )

        result[
            "normalized_abs_future_move"
        ] = (
            result[
                "future_log_return"
            ]
            .abs()
            / result[
                "realized_volatility_20"
            ]
        )

        return result

    def _alignment_summary(
        self,
        train: pd.DataFrame,
    ) -> pd.DataFrame:
        overall_up_rate = float(
            train[
                "future_up"
            ].mean()
        )

        rows = []

        for score in (
            -1.0,
            0.0,
            1.0,
        ):
            subset = train.loc[
                train[
                    "ma_alignment_score"
                ]
                == score
            ]

            if subset.empty:
                continue

            up_rate = float(
                subset[
                    "future_up"
                ].mean()
            )

            down_rate = float(
                1.0
                - up_rate
            )

            rows.append(
                {
                    "alignment": (
                        self.ALIGNMENT_LABELS[
                            score
                        ]
                    ),
                    "rows": int(
                        len(
                            subset
                        )
                    ),
                    "share": float(
                        len(
                            subset
                        )
                        / len(
                            train
                        )
                    ),
                    "next_day_up_rate": up_rate,
                    "next_day_down_rate": down_rate,
                    "up_rate_lift_vs_overall": float(
                        up_rate
                        - overall_up_rate
                    ),
                    "down_rate_lift_vs_overall": float(
                        down_rate
                        - (
                            1.0
                            - overall_up_rate
                        )
                    ),
                    "mean_next_return_pct": float(
                        subset[
                            "future_log_return"
                        ].mean()
                        * 100.0
                    ),
                    "median_abs_next_return_pct": float(
                        subset[
                            "future_log_return"
                        ].abs().median()
                        * 100.0
                    ),
                    "mean_normalized_abs_move": float(
                        subset[
                            "normalized_abs_future_move"
                        ].mean()
                    ),
                    "mean_dmi_spread": float(
                        subset[
                            "dmi_spread_14"
                        ].mean()
                    ),
                }
            )

        return pd.DataFrame(
            rows
        )

    def _quantile_summary(
        self,
        train: pd.DataFrame,
        feature_column: str,
        analysis_name: str,
    ) -> pd.DataFrame:
        quantiles = train[
            feature_column
        ].quantile(
            [
                1.0 / 3.0,
                2.0 / 3.0,
            ]
        )

        lower = float(
            quantiles.iloc[
                0
            ]
        )
        upper = float(
            quantiles.iloc[
                1
            ]
        )

        buckets = np.where(
            train[
                feature_column
            ]
            <= lower,
            "LOW",
            np.where(
                train[
                    feature_column
                ]
                <= upper,
                "MID",
                "HIGH",
            ),
        )

        working = train.copy()
        working[
            "bucket"
        ] = buckets

        rows = []

        for bucket in (
            "LOW",
            "MID",
            "HIGH",
        ):
            subset = working.loc[
                working[
                    "bucket"
                ]
                == bucket
            ]

            rows.append(
                {
                    "analysis": analysis_name,
                    "bucket": bucket,
                    "rows": int(
                        len(
                            subset
                        )
                    ),
                    "feature_mean": float(
                        subset[
                            feature_column
                        ].mean()
                    ),
                    "next_day_up_rate": float(
                        subset[
                            "future_up"
                        ].mean()
                    ),
                    "mean_next_return_pct": float(
                        subset[
                            "future_log_return"
                        ].mean()
                        * 100.0
                    ),
                    "median_abs_next_return_pct": float(
                        subset[
                            "future_log_return"
                        ].abs().median()
                        * 100.0
                    ),
                    "mean_normalized_abs_move": float(
                        subset[
                            "normalized_abs_future_move"
                        ].mean()
                    ),
                }
            )

        return pd.DataFrame(
            rows
        )

    def _alignment_stability(
        self,
        train: pd.DataFrame,
    ) -> pd.DataFrame:
        blocks = np.array_split(
            np.arange(
                len(
                    train
                )
            ),
            self.stability_blocks,
        )

        rows = []

        for block_number, block_indices in enumerate(
            blocks,
            start=1,
        ):
            block = train.iloc[
                block_indices
            ]

            overall_up_rate = float(
                block[
                    "future_up"
                ].mean()
            )

            for score in (
                -1.0,
                1.0,
            ):
                subset = block.loc[
                    block[
                        "ma_alignment_score"
                    ]
                    == score
                ]

                if subset.empty:
                    continue

                up_rate = float(
                    subset[
                        "future_up"
                    ].mean()
                )

                rows.append(
                    {
                        "block": int(
                            block_number
                        ),
                        "alignment": (
                            self.ALIGNMENT_LABELS[
                                score
                            ]
                        ),
                        "rows": int(
                            len(
                                subset
                            )
                        ),
                        "next_day_up_rate": up_rate,
                        "directional_lift": (
                            float(
                                up_rate
                                - overall_up_rate
                            )
                            if score == 1.0
                            else float(
                                (
                                    1.0
                                    - up_rate
                                )
                                - (
                                    1.0
                                    - overall_up_rate
                                )
                            )
                        ),
                    }
                )

        return pd.DataFrame(
            rows
        )
