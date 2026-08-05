from math import sqrt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class XGBoostEvaluator:
    def evaluate(
        self,
        actual,
        predicted,
    ):
        mse = mean_squared_error(
            actual,
            predicted,
        )

        return {
            "mae": mean_absolute_error(
                actual,
                predicted,
            ),
            "mse": mse,
            "rmse": sqrt(mse),
            "r2": r2_score(
                actual,
                predicted,
            ),
        }