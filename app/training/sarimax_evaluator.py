import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


class SarimaxEvaluator:
    def evaluate(
        self,
        actual: pd.Series,
        predicted: pd.Series,
    ) -> dict:
        actual = actual.reset_index(drop=True)
        predicted = predicted.reset_index(drop=True)

        residuals = actual - predicted

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        )

        mae = mean_absolute_error(
            actual,
            predicted,
        )

        mape = (
            np.abs(
                (actual - predicted) / actual.replace(0, np.nan)
            )
            .dropna()
            .mean()
            * 100
        )

        actual_direction = np.sign(actual)
        predicted_direction = np.sign(predicted)

        directional_accuracy = (
            (actual_direction == predicted_direction)
            .mean()
            * 100
        )

        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "directional_accuracy": directional_accuracy,
            "residual_mean": residuals.mean(),
            "residual_std": residuals.std(),
            "residual_min": residuals.min(),
            "residual_max": residuals.max(),
        }