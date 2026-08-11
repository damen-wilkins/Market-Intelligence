from dataclasses import dataclass

import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class PriceSequenceSet:
    X: np.ndarray
    y: np.ndarray
    target_dates: np.ndarray
    prior_close: np.ndarray
    current_close: np.ndarray
    actual_close: np.ndarray
    denoised_target_close: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


class PaperWaveletPreprocessor:
    PAPER_NONCAUSAL = "paper_noncausal"
    CAUSAL = "causal"
    VALID_MODES = {
        PAPER_NONCAUSAL,
        CAUSAL,
    }

    def __init__(
        self,
        mode: str,
        sequence_length: int = 150,
        train_end_date: str = "2021-01-01",
        validation_end_date: str = "2022-07-01",
        wavelet: str = "db4",
        level: int = 1,
        pad_width: int = 100,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Mode must be one of {sorted(self.VALID_MODES)}."
            )

        if sequence_length <= 1:
            raise ValueError(
                "Sequence length must be greater than one."
            )

        if level <= 0:
            raise ValueError(
                "Wavelet decomposition level must be positive."
            )

        if pad_width <= 0:
            raise ValueError(
                "Wavelet padding width must be positive."
            )

        self.mode = mode
        self.sequence_length = sequence_length
        self.train_end_date = pd.Timestamp(
            train_end_date
        )
        self.validation_end_date = pd.Timestamp(
            validation_end_date
        )
        self.wavelet = wavelet
        self.level = level
        self.pad_width = pad_width
        self.scaler = MinMaxScaler(
            feature_range=(0, 1)
        )
        self.is_fitted = False

        if self.train_end_date >= self.validation_end_date:
            raise ValueError(
                "Training end date must be before validation end date."
            )

    def prepare(
        self,
        data: pd.DataFrame,
    ) -> dict[str, PriceSequenceSet]:
        prepared = self._validate_data(
            data
        )

        dates = prepared[
            "trade_date"
        ].to_numpy()
        closes = prepared[
            "close"
        ].to_numpy(
            dtype=np.float64
        )

        if len(closes) <= self.sequence_length:
            raise ValueError(
                "Not enough observations to build price sequences."
            )

        if self.mode == self.PAPER_NONCAUSAL:
            arrays = self._prepare_paper_noncausal(
                dates=dates,
                closes=closes,
            )
        else:
            arrays = self._prepare_causal(
                dates=dates,
                closes=closes,
            )

        splits = self._split_arrays(
            arrays
        )

        self.is_fitted = True

        return splits

    def denoise_full_series(
        self,
        values,
    ) -> np.ndarray:
        data = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

        if data.size < 2:
            raise ValueError(
                "Wavelet denoising requires at least two observations."
            )

        if not np.isfinite(data).all():
            raise ValueError(
                "Wavelet denoising input contains non-finite values."
            )

        padded_data = np.pad(
            data,
            self.pad_width,
            mode="edge",
        )

        coefficients = pywt.wavedec(
            padded_data,
            self.wavelet,
            mode="per",
            level=self.level,
        )

        detail = coefficients[
            -self.level
        ]

        sigma = (
            1.0 / 0.6745
        ) * np.median(
            np.abs(
                detail
                - np.median(
                    detail
                )
            )
        )

        threshold = sigma * np.sqrt(
            2.0
            * np.log(
                len(padded_data)
            )
        )

        coefficients[1:] = [
            pywt.threshold(
                coefficient,
                value=threshold,
                mode="soft",
            )
            for coefficient in coefficients[1:]
        ]

        coefficients[
            -self.level
        ] = np.zeros_like(
            coefficients[
                -self.level
            ]
        )

        denoised = pywt.waverec(
            coefficients,
            self.wavelet,
            mode="per",
        )

        denoised = denoised[
            self.pad_width:
            -self.pad_width
        ]

        if len(denoised) > len(data):
            denoised = denoised[
                :len(data)
            ]
        elif len(denoised) < len(data):
            denoised = np.pad(
                denoised,
                (
                    0,
                    len(data) - len(denoised),
                ),
                mode="edge",
            )

        return np.asarray(
            denoised,
            dtype=np.float64,
        )

    def inverse_transform(
        self,
        values,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(
                "Preprocessor must be fitted before inverse transformation."
            )

        array = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1, 1)

        return self.scaler.inverse_transform(
            array
        ).reshape(-1)

    def get_state(self) -> dict:
        if not self.is_fitted:
            raise ValueError(
                "Preprocessor must be fitted before serialization."
            )

        return {
            "mode": self.mode,
            "sequence_length": self.sequence_length,
            "train_end_date": self.train_end_date.strftime(
                "%Y-%m-%d"
            ),
            "validation_end_date": self.validation_end_date.strftime(
                "%Y-%m-%d"
            ),
            "wavelet": self.wavelet,
            "level": self.level,
            "pad_width": self.pad_width,
            "scaler": {
                "feature_range": list(
                    self.scaler.feature_range
                ),
                "scale_": self.scaler.scale_.tolist(),
                "min_": self.scaler.min_.tolist(),
                "data_min_": self.scaler.data_min_.tolist(),
                "data_max_": self.scaler.data_max_.tolist(),
                "data_range_": self.scaler.data_range_.tolist(),
                "n_features_in_": int(
                    self.scaler.n_features_in_
                ),
            },
        }

    def _prepare_paper_noncausal(
        self,
        dates: np.ndarray,
        closes: np.ndarray,
    ) -> dict[str, np.ndarray]:
        denoised = self.denoise_full_series(
            closes
        )

        scaled = self.scaler.fit_transform(
            denoised.reshape(-1, 1)
        ).reshape(-1)

        X = []
        y = []
        target_dates = []
        prior_close = []
        current_close = []
        actual_close = []
        denoised_target_close = []

        for target_index in range(
            self.sequence_length,
            len(closes),
        ):
            X.append(
                scaled[
                    target_index
                    - self.sequence_length:
                    target_index
                ]
            )
            y.append(
                scaled[
                    target_index
                ]
            )
            target_dates.append(
                dates[
                    target_index
                ]
            )
            prior_close.append(
                closes[
                    target_index - 2
                ]
            )
            current_close.append(
                closes[
                    target_index - 1
                ]
            )
            actual_close.append(
                closes[
                    target_index
                ]
            )
            denoised_target_close.append(
                denoised[
                    target_index
                ]
            )

        return self._build_array_bundle(
            X=X,
            y=y,
            target_dates=target_dates,
            prior_close=prior_close,
            current_close=current_close,
            actual_close=actual_close,
            denoised_target_close=denoised_target_close,
        )

    def _prepare_causal(
        self,
        dates: np.ndarray,
        closes: np.ndarray,
    ) -> dict[str, np.ndarray]:
        X_unscaled = []
        y_unscaled = []
        target_dates = []
        prior_close = []
        current_close = []
        actual_close = []
        denoised_target_close = []

        denoised_prefix = self.denoise_full_series(
            closes[
                :self.sequence_length
            ]
        )

        for target_index in range(
            self.sequence_length,
            len(closes),
        ):
            X_unscaled.append(
                denoised_prefix[
                    -self.sequence_length:
                ]
            )

            target_prefix = self.denoise_full_series(
                closes[
                    :target_index + 1
                ]
            )

            target_value = float(
                target_prefix[-1]
            )

            y_unscaled.append(
                target_value
            )
            target_dates.append(
                dates[
                    target_index
                ]
            )
            prior_close.append(
                closes[
                    target_index - 2
                ]
            )
            current_close.append(
                closes[
                    target_index - 1
                ]
            )
            actual_close.append(
                closes[
                    target_index
                ]
            )
            denoised_target_close.append(
                target_value
            )

            denoised_prefix = target_prefix

        X_unscaled = np.asarray(
            X_unscaled,
            dtype=np.float64,
        )
        y_unscaled = np.asarray(
            y_unscaled,
            dtype=np.float64,
        )
        target_dates_array = pd.to_datetime(
            target_dates
        ).to_numpy()

        train_mask = (
            target_dates_array
            < np.datetime64(
                self.train_end_date
            )
        )

        if not train_mask.any():
            raise ValueError(
                "No causal training sequences were created."
            )

        training_values = np.concatenate(
            [
                X_unscaled[
                    train_mask
                ].reshape(-1),
                y_unscaled[
                    train_mask
                ],
            ]
        ).reshape(-1, 1)

        self.scaler.fit(
            training_values
        )

        X_scaled = self.scaler.transform(
            X_unscaled.reshape(-1, 1)
        ).reshape(
            X_unscaled.shape
        )

        y_scaled = self.scaler.transform(
            y_unscaled.reshape(-1, 1)
        ).reshape(-1)

        return self._build_array_bundle(
            X=X_scaled,
            y=y_scaled,
            target_dates=target_dates,
            prior_close=prior_close,
            current_close=current_close,
            actual_close=actual_close,
            denoised_target_close=denoised_target_close,
        )

    def _build_array_bundle(
        self,
        X,
        y,
        target_dates,
        prior_close,
        current_close,
        actual_close,
        denoised_target_close,
    ) -> dict[str, np.ndarray]:
        X_array = np.asarray(
            X,
            dtype=np.float32,
        )

        if X_array.ndim != 2:
            raise ValueError(
                "Price sequences must form a two-dimensional array."
            )

        return {
            "X": X_array[
                :,
                :,
                np.newaxis,
            ],
            "y": np.asarray(
                y,
                dtype=np.float32,
            ).reshape(-1, 1),
            "target_dates": pd.to_datetime(
                target_dates
            ).to_numpy(),
            "prior_close": np.asarray(
                prior_close,
                dtype=np.float64,
            ),
            "current_close": np.asarray(
                current_close,
                dtype=np.float64,
            ),
            "actual_close": np.asarray(
                actual_close,
                dtype=np.float64,
            ),
            "denoised_target_close": np.asarray(
                denoised_target_close,
                dtype=np.float64,
            ),
        }

    def _split_arrays(
        self,
        arrays: dict[str, np.ndarray],
    ) -> dict[str, PriceSequenceSet]:
        target_dates = arrays[
            "target_dates"
        ]

        train_end = np.datetime64(
            self.train_end_date
        )
        validation_end = np.datetime64(
            self.validation_end_date
        )

        masks = {
            "train": (
                target_dates
                < train_end
            ),
            "validation": (
                (target_dates >= train_end)
                & (target_dates < validation_end)
            ),
            "test": (
                target_dates
                >= validation_end
            ),
        }

        splits = {}

        for split_name, mask in masks.items():
            if not mask.any():
                raise ValueError(
                    f"No observations were created for {split_name}."
                )

            splits[
                split_name
            ] = PriceSequenceSet(
                X=arrays[
                    "X"
                ][mask],
                y=arrays[
                    "y"
                ][mask],
                target_dates=target_dates[
                    mask
                ],
                prior_close=arrays[
                    "prior_close"
                ][mask],
                current_close=arrays[
                    "current_close"
                ][mask],
                actual_close=arrays[
                    "actual_close"
                ][mask],
                denoised_target_close=arrays[
                    "denoised_target_close"
                ][mask],
            )

        return splits

    @staticmethod
    def _validate_data(
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        required_columns = {
            "trade_date",
            "close",
        }

        missing_columns = (
            required_columns
            - set(
                data.columns
            )
        )

        if missing_columns:
            raise ValueError(
                "Paper replication data is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if data.empty:
            raise ValueError(
                "Paper replication data cannot be empty."
            )

        prepared = data[
            [
                "trade_date",
                "close",
            ]
        ].copy()

        prepared[
            "trade_date"
        ] = pd.to_datetime(
            prepared[
                "trade_date"
            ]
        )

        prepared[
            "close"
        ] = pd.to_numeric(
            prepared[
                "close"
            ],
            errors="coerce",
        )

        prepared = prepared.sort_values(
            "trade_date"
        ).reset_index(
            drop=True
        )

        if prepared[
            "trade_date"
        ].duplicated().any():
            raise ValueError(
                "Paper replication data contains duplicate trade dates."
            )

        if prepared[
            [
                "trade_date",
                "close",
            ]
        ].isna().any().any():
            raise ValueError(
                "Paper replication data contains missing values."
            )

        closes = prepared[
            "close"
        ].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(
            closes
        ).all():
            raise ValueError(
                "Paper replication close prices contain non-finite values."
            )

        if (
            closes <= 0
        ).any():
            raise ValueError(
                "Paper replication close prices must be positive."
            )

        return prepared
