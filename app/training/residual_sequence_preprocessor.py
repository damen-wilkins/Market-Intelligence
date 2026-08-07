from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.training.feature_contract import require_columns


RESIDUAL_HISTORY_FEATURE_COLUMNS = (
    "previous_residual",
    "previous_absolute_residual",
    "previous_squared_residual",
)


@dataclass(frozen=True)
class ResidualSequenceDataset:
    sequences: np.ndarray
    targets: np.ndarray
    trade_dates: np.ndarray
    sarimax_predictions: np.ndarray
    actual_residuals: np.ndarray

    def __len__(self) -> int:
        return len(self.targets)


class ResidualSequencePreprocessor:
    def __init__(
        self,
        sequence_length: int,
        scale_epsilon: float = 1e-12,
    ):
        if sequence_length < 2:
            raise ValueError(
                "Sequence length must be at least 2."
            )

        self.sequence_length = sequence_length
        self.scale_epsilon = scale_epsilon
        self.base_feature_columns: list[str] = []
        self.feature_columns: list[str] = []
        self.feature_center: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.target_center: float | None = None
        self.target_scale: float | None = None

    def fit(
        self,
        dataframe: pd.DataFrame,
        feature_columns: list[str],
    ) -> "ResidualSequencePreprocessor":
        self._validate_base_feature_columns(
            feature_columns
        )
        require_columns(
            dataframe,
            feature_columns,
            "Residual sequence training data",
        )
        frame = self._prepare_frame(dataframe)

        self.base_feature_columns = list(feature_columns)
        self.feature_columns = [
            *self.base_feature_columns,
            *RESIDUAL_HISTORY_FEATURE_COLUMNS,
        ]

        frame = self._add_residual_history(frame)
        valid_rows = frame[
            [
                *self.feature_columns,
                "sarimax_residual",
            ]
        ].notna().all(axis=1)
        fit_frame = frame.loc[valid_rows].copy()

        if len(fit_frame) < self.sequence_length:
            raise ValueError(
                "Training data does not contain enough valid rows "
                "to fit the sequence preprocessor."
            )

        feature_values = fit_frame[
            self.feature_columns
        ].to_numpy(dtype=np.float64)
        target_values = fit_frame[
            "sarimax_residual"
        ].to_numpy(dtype=np.float64)

        self.feature_center = np.median(
            feature_values,
            axis=0,
        )
        self.feature_scale = self._calculate_scale(
            feature_values,
            axis=0,
        )
        self.target_center = float(
            np.median(target_values)
        )
        self.target_scale = float(
            self._calculate_scale(
                target_values,
                axis=None,
            )
        )

        return self

    def build_training_sequences(
        self,
        dataframe: pd.DataFrame,
    ) -> ResidualSequenceDataset:
        self._validate_fitted()
        target = self._prepare_frame(dataframe)
        target["_is_target"] = True

        return self._build_sequences(
            combined=target,
            require_all_targets=False,
        )

    def build_inference_sequences(
        self,
        history: pd.DataFrame,
        dataframe: pd.DataFrame,
    ) -> ResidualSequenceDataset:
        self._validate_fitted()
        context = self._prepare_frame(history)
        target = self._prepare_frame(dataframe)

        if context.empty:
            raise ValueError(
                "Sequence inference requires historical context."
            )

        if target.empty:
            raise ValueError(
                "Sequence inference target data cannot be empty."
            )

        if (
            context["trade_date"].max()
            >= target["trade_date"].min()
        ):
            raise ValueError(
                "Sequence history must end before target data begins."
            )

        context["_is_target"] = False
        target["_is_target"] = True

        combined = pd.concat(
            [context, target],
            ignore_index=True,
        )

        return self._build_sequences(
            combined=combined,
            require_all_targets=True,
        )

    def inverse_transform_target(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        self._validate_fitted()
        array = np.asarray(
            values,
            dtype=np.float64,
        )

        return (
            array * self.target_scale
            + self.target_center
        )

    def get_state(self) -> dict[str, Any]:
        self._validate_fitted()

        return {
            "sequence_length": self.sequence_length,
            "scale_epsilon": self.scale_epsilon,
            "base_feature_columns": self.base_feature_columns,
            "feature_columns": self.feature_columns,
            "feature_center": self.feature_center.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "target_center": self.target_center,
            "target_scale": self.target_scale,
        }

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
    ) -> "ResidualSequencePreprocessor":
        preprocessor = cls(
            sequence_length=int(
                state["sequence_length"]
            ),
            scale_epsilon=float(
                state["scale_epsilon"]
            ),
        )
        preprocessor.base_feature_columns = list(
            state["base_feature_columns"]
        )
        preprocessor.feature_columns = list(
            state["feature_columns"]
        )
        preprocessor.feature_center = np.asarray(
            state["feature_center"],
            dtype=np.float64,
        )
        preprocessor.feature_scale = np.asarray(
            state["feature_scale"],
            dtype=np.float64,
        )
        preprocessor.target_center = float(
            state["target_center"]
        )
        preprocessor.target_scale = float(
            state["target_scale"]
        )

        preprocessor._validate_fitted()

        return preprocessor

    def _build_sequences(
        self,
        combined: pd.DataFrame,
        require_all_targets: bool,
    ) -> ResidualSequenceDataset:
        frame = self._add_residual_history(
            combined.sort_values(
                "trade_date"
            ).reset_index(drop=True)
        )

        feature_values = frame[
            self.feature_columns
        ].to_numpy(dtype=np.float64)
        scaled_features = (
            feature_values - self.feature_center
        ) / self.feature_scale

        target_values = frame[
            "sarimax_residual"
        ].to_numpy(dtype=np.float64)
        scaled_targets = (
            target_values - self.target_center
        ) / self.target_scale

        sequences = []
        targets = []
        trade_dates = []
        sarimax_predictions = []
        actual_residuals = []
        target_positions = frame.index[
            frame["_is_target"].astype(bool)
        ].tolist()

        for position in target_positions:
            start = position - self.sequence_length + 1

            if start < 0:
                if require_all_targets:
                    raise ValueError(
                        "Historical context is too short for the "
                        "configured sequence length."
                    )
                continue

            sequence = scaled_features[
                start:position + 1
            ]

            if not np.isfinite(sequence).all():
                if require_all_targets:
                    target_date = frame.loc[
                        position,
                        "trade_date",
                    ]
                    raise ValueError(
                        "Sequence contains missing or non-finite "
                        f"features for {target_date:%Y-%m-%d}."
                    )
                continue

            scaled_target = scaled_targets[position]

            if not np.isfinite(scaled_target):
                raise ValueError(
                    "Sequence target contains a missing or "
                    "non-finite residual."
                )

            sequences.append(sequence)
            targets.append(scaled_target)
            trade_dates.append(
                frame.loc[position, "trade_date"]
            )
            sarimax_predictions.append(
                frame.loc[
                    position,
                    "sarimax_prediction",
                ]
            )
            actual_residuals.append(
                frame.loc[
                    position,
                    "sarimax_residual",
                ]
            )

        if not sequences:
            raise ValueError(
                "No valid residual sequences were produced."
            )

        if (
            require_all_targets
            and len(sequences) != len(target_positions)
        ):
            raise ValueError(
                "Not every target observation produced a sequence."
            )

        return ResidualSequenceDataset(
            sequences=np.asarray(
                sequences,
                dtype=np.float32,
            ),
            targets=np.asarray(
                targets,
                dtype=np.float32,
            ),
            trade_dates=np.asarray(
                trade_dates,
                dtype="datetime64[ns]",
            ),
            sarimax_predictions=np.asarray(
                sarimax_predictions,
                dtype=np.float64,
            ),
            actual_residuals=np.asarray(
                actual_residuals,
                dtype=np.float64,
            ),
        )

    def _prepare_frame(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        required_columns = [
            "trade_date",
            "sarimax_prediction",
            "sarimax_residual",
        ]

        if self.base_feature_columns:
            required_columns.extend(
                self.base_feature_columns
            )

        require_columns(
            dataframe,
            required_columns,
            "Residual sequence data",
        )

        frame = dataframe.copy()
        frame["trade_date"] = pd.to_datetime(
            frame["trade_date"]
        )

        if frame["trade_date"].isna().any():
            raise ValueError(
                "Residual sequence data contains invalid dates."
            )

        if frame["trade_date"].duplicated().any():
            raise ValueError(
                "Residual sequence data contains duplicate dates."
            )

        return frame.sort_values(
            "trade_date"
        ).reset_index(drop=True)

    def _add_residual_history(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = dataframe.copy()
        previous_residual = frame[
            "sarimax_residual"
        ].shift(1)

        frame["previous_residual"] = previous_residual
        frame["previous_absolute_residual"] = (
            previous_residual.abs()
        )
        frame["previous_squared_residual"] = (
            previous_residual.pow(2)
        )

        return frame

    def _calculate_scale(
        self,
        values: np.ndarray,
        axis,
    ):
        lower = np.percentile(
            values,
            25,
            axis=axis,
        )
        upper = np.percentile(
            values,
            75,
            axis=axis,
        )
        scale = upper - lower
        standard_deviation = np.std(
            values,
            axis=axis,
        )

        return np.where(
            np.abs(scale) > self.scale_epsilon,
            scale,
            np.where(
                np.abs(standard_deviation)
                > self.scale_epsilon,
                standard_deviation,
                1.0,
            ),
        )

    def _validate_base_feature_columns(
        self,
        feature_columns: list[str],
    ) -> None:
        if not feature_columns:
            raise ValueError(
                "At least one base feature column is required."
            )

        if len(feature_columns) != len(set(feature_columns)):
            raise ValueError(
                "Base feature columns must be unique."
            )

        forbidden_columns = {
            "trade_date",
            "sarimax_residual",
            *RESIDUAL_HISTORY_FEATURE_COLUMNS,
        }
        invalid_columns = [
            column
            for column in feature_columns
            if column in forbidden_columns
        ]

        if invalid_columns:
            raise ValueError(
                "Invalid base feature columns: "
                f"{invalid_columns}"
            )

    def _validate_fitted(self) -> None:
        if not self.base_feature_columns:
            raise ValueError(
                "ResidualSequencePreprocessor must be fitted "
                "before use."
            )

        if (
            self.feature_center is None
            or self.feature_scale is None
            or self.target_center is None
            or self.target_scale is None
        ):
            raise ValueError(
                "ResidualSequencePreprocessor state is incomplete."
            )

        if len(self.feature_columns) != len(
            self.feature_center
        ):
            raise ValueError(
                "Preprocessor feature state is inconsistent."
            )

        if np.any(
            np.asarray(self.feature_scale)
            <= 0
        ):
            raise ValueError(
                "Preprocessor feature scales must be positive."
            )

        if self.target_scale <= 0:
            raise ValueError(
                "Preprocessor target scale must be positive."
            )
