import numpy as np
import pandas as pd


class FlatTargetSensitivityAnalyzer:
    DEFAULT_WINDOWS = (
        10,
        20,
        40,
        60,
    )

    DEFAULT_MULTIPLIERS = tuple(
        round(value, 2)
        for value in np.arange(
            0.10,
            1.0001,
            0.05,
        )
    )

    def __init__(
        self,
        volatility_windows: tuple[int, ...] = DEFAULT_WINDOWS,
        threshold_multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS,
        training_fraction: float = 0.70,
        stability_blocks: int = 5,
    ):
        if not volatility_windows:
            raise ValueError(
                "At least one volatility window is required."
            )

        if any(
            int(window) < 2
            for window in volatility_windows
        ):
            raise ValueError(
                "Volatility windows must be at least two trading days."
            )

        if not threshold_multipliers:
            raise ValueError(
                "At least one threshold multiplier is required."
            )

        if any(
            float(multiplier) <= 0.0
            for multiplier in threshold_multipliers
        ):
            raise ValueError(
                "Threshold multipliers must be positive."
            )

        if not 0.0 < training_fraction < 1.0:
            raise ValueError(
                "Training fraction must be between zero and one."
            )

        if stability_blocks < 2:
            raise ValueError(
                "At least two stability blocks are required."
            )

        self.volatility_windows = tuple(
            sorted(
                {
                    int(window)
                    for window in volatility_windows
                }
            )
        )

        self.threshold_multipliers = tuple(
            sorted(
                {
                    float(multiplier)
                    for multiplier in threshold_multipliers
                }
            )
        )

        self.training_fraction = float(
            training_fraction
        )
        self.stability_blocks = int(
            stability_blocks
        )

    def analyze(
        self,
        data: pd.DataFrame,
        eligible_feature_dates=None,
    ) -> pd.DataFrame:
        prepared = self._prepare_data(
            data=data,
            eligible_feature_dates=(
                eligible_feature_dates
            ),
        )

        rows = []

        for window in self.volatility_windows:
            volatility = (
                prepared[
                    "log_return"
                ]
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .std()
            )

            candidate_frame = (
                prepared.assign(
                    rolling_volatility=(
                        volatility
                    )
                )
                .dropna(
                    subset=[
                        "rolling_volatility",
                        "future_log_return",
                        "target_date",
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            train = self._training_partition(
                candidate_frame
            )

            for multiplier in (
                self.threshold_multipliers
            ):
                rows.append(
                    self._summarize_candidate(
                        train=train,
                        window=window,
                        multiplier=multiplier,
                    )
                )

        result = pd.DataFrame(
            rows
        )

        if result.empty:
            raise ValueError(
                "Flat-target sensitivity analysis produced no candidates."
            )

        return result.sort_values(
            [
                "volatility_window",
                "threshold_multiplier",
            ]
        ).reset_index(
            drop=True
        )

    def _prepare_data(
        self,
        data: pd.DataFrame,
        eligible_feature_dates,
    ) -> pd.DataFrame:
        required_columns = {
            "trade_date",
            "close",
        }

        missing = (
            required_columns
            - set(
                data.columns
            )
        )

        if missing:
            raise ValueError(
                "Flat-target analysis is missing columns: "
                f"{sorted(missing)}"
            )

        result = data[
            [
                "trade_date",
                "close",
            ]
        ].copy()

        result[
            "trade_date"
        ] = pd.to_datetime(
            result[
                "trade_date"
            ]
        )

        result = result.sort_values(
            "trade_date"
        ).reset_index(
            drop=True
        )

        if result[
            "trade_date"
        ].duplicated().any():
            raise ValueError(
                "Flat-target analysis data contains duplicate trade dates."
            )

        if (
            result[
                "close"
            ]
            <= 0.0
        ).any():
            raise ValueError(
                "Flat-target analysis requires positive closing prices."
            )

        result[
            "log_return"
        ] = np.log(
            result[
                "close"
            ]
            / result[
                "close"
            ].shift(1)
        )

        result[
            "target_date"
        ] = result[
            "trade_date"
        ].shift(-1)

        result[
            "future_log_return"
        ] = np.log(
            result[
                "close"
            ].shift(-1)
            / result[
                "close"
            ]
        )

        if eligible_feature_dates is not None:
            eligible = pd.DatetimeIndex(
                pd.to_datetime(
                    eligible_feature_dates
                )
            )

            result = result.loc[
                result[
                    "trade_date"
                ].isin(
                    eligible
                )
            ].reset_index(
                drop=True
            )

        return result

    def _training_partition(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        if len(dataframe) < 100:
            raise ValueError(
                "Flat-target sensitivity analysis requires at least 100 rows."
            )

        training_rows = int(
            np.floor(
                len(dataframe)
                * self.training_fraction
            )
        )

        if training_rows <= 0:
            raise ValueError(
                "Training partition is empty."
            )

        return (
            dataframe
            .iloc[
                :training_rows
            ]
            .reset_index(
                drop=True
            )
        )

    def _summarize_candidate(
        self,
        train: pd.DataFrame,
        window: int,
        multiplier: float,
    ) -> dict:
        threshold = (
            train[
                "rolling_volatility"
            ]
            * multiplier
        )

        future_return = train[
            "future_log_return"
        ]

        direction = np.where(
            future_return
            > threshold,
            "UP",
            np.where(
                future_return
                < -threshold,
                "DOWN",
                "FLAT",
            ),
        )

        candidate = train[
            [
                "trade_date",
                "target_date",
                "future_log_return",
                "rolling_volatility",
            ]
        ].copy()

        candidate[
            "threshold"
        ] = threshold.to_numpy()
        candidate[
            "direction"
        ] = direction

        shares = (
            candidate[
                "direction"
            ]
            .value_counts(
                normalize=True
            )
            .reindex(
                [
                    "DOWN",
                    "FLAT",
                    "UP",
                ],
                fill_value=0.0,
            )
        )

        block_shares = (
            self._flat_share_by_block(
                candidate
            )
        )

        flat_rows = candidate.loc[
            candidate[
                "direction"
            ]
            == "FLAT"
        ]

        flat_abs_return = (
            flat_rows[
                "future_log_return"
            ]
            .abs()
        )

        return {
            "volatility_window": int(
                window
            ),
            "threshold_multiplier": float(
                multiplier
            ),
            "training_rows": int(
                len(
                    candidate
                )
            ),
            "training_start": (
                candidate[
                    "target_date"
                ]
                .min()
                .strftime(
                    "%Y-%m-%d"
                )
            ),
            "training_end": (
                candidate[
                    "target_date"
                ]
                .max()
                .strftime(
                    "%Y-%m-%d"
                )
            ),
            "down_share": float(
                shares[
                    "DOWN"
                ]
            ),
            "flat_share": float(
                shares[
                    "FLAT"
                ]
            ),
            "up_share": float(
                shares[
                    "UP"
                ]
            ),
            "minimum_class_share": float(
                shares.min()
            ),
            "up_down_share_gap": float(
                abs(
                    shares[
                        "UP"
                    ]
                    - shares[
                        "DOWN"
                    ]
                )
            ),
            "median_threshold_pct": float(
                threshold.median()
                * 100.0
            ),
            "threshold_p25_pct": float(
                threshold.quantile(
                    0.25
                )
                * 100.0
            ),
            "threshold_p75_pct": float(
                threshold.quantile(
                    0.75
                )
                * 100.0
            ),
            "median_flat_abs_return_pct": (
                float(
                    flat_abs_return.median()
                    * 100.0
                )
                if not flat_rows.empty
                else float(
                    "nan"
                )
            ),
            "flat_share_block_min": float(
                block_shares.min()
            ),
            "flat_share_block_max": float(
                block_shares.max()
            ),
            "flat_share_block_std": float(
                block_shares.std(
                    ddof=0
                )
            ),
            "flat_share_block_range": float(
                block_shares.max()
                - block_shares.min()
            ),
        }

    def _flat_share_by_block(
        self,
        candidate: pd.DataFrame,
    ) -> pd.Series:
        indices = np.array_split(
            np.arange(
                len(
                    candidate
                )
            ),
            self.stability_blocks,
        )

        shares = []

        for block_indices in indices:
            if len(
                block_indices
            ) == 0:
                continue

            block = candidate.iloc[
                block_indices
            ]

            shares.append(
                float(
                    np.mean(
                        block[
                            "direction"
                        ]
                        == "FLAT"
                    )
                )
            )

        if len(shares) < 2:
            raise ValueError(
                "Not enough observations for temporal stability analysis."
            )

        return pd.Series(
            shares,
            dtype=np.float64,
        )
