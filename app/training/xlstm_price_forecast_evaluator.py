import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class XLSTMPriceForecastEvaluator:
    def evaluate(
        self,
        actual_close,
        predicted_close,
        current_close,
        prior_close,
    ) -> dict:
        actual = self._to_vector(
            actual_close,
            "actual close",
        )
        predicted = self._to_vector(
            predicted_close,
            "predicted close",
        )
        current = self._to_vector(
            current_close,
            "current close",
        )
        prior = self._to_vector(
            prior_close,
            "prior close",
        )

        lengths = {
            len(actual),
            len(predicted),
            len(current),
            len(prior),
        }

        if len(lengths) != 1:
            raise ValueError(
                "Forecast evaluation arrays must have equal lengths."
            )

        if len(actual) < 2:
            raise ValueError(
                "Forecast evaluation requires at least two observations."
            )

        paper_actual = (
            np.diff(
                actual
            )
            > 0.0
        ).astype(
            np.int64
        )

        paper_predicted = (
            np.diff(
                predicted
            )
            > 0.0
        ).astype(
            np.int64
        )

        production_actual = (
            actual
            > current
        ).astype(
            np.int64
        )

        production_predicted = (
            predicted
            > current
        ).astype(
            np.int64
        )

        momentum_predictions = (
            current
            > prior
        ).astype(
            np.int64
        )

        return {
            "price": self._price_metrics(
                actual,
                predicted,
            ),
            "naive_previous_close_price": (
                self._price_metrics(
                    actual,
                    current,
                )
            ),
            "paper_direction": self._binary_metrics(
                paper_actual,
                paper_predicted,
            ),
            "production_direction": self._binary_metrics(
                production_actual,
                production_predicted,
            ),
            "momentum_direction_baseline": (
                self._binary_metrics(
                    production_actual,
                    momentum_predictions,
                )
            ),
        }

    def paper_direction_labels(
        self,
        actual_close,
        predicted_close,
    ) -> tuple[np.ndarray, np.ndarray]:
        actual = self._to_vector(
            actual_close,
            "actual close",
        )
        predicted = self._to_vector(
            predicted_close,
            "predicted close",
        )

        if len(actual) != len(predicted):
            raise ValueError(
                "Actual and predicted close lengths do not match."
            )

        return (
            (
                np.diff(
                    actual
                )
                > 0.0
            ).astype(
                np.int64
            ),
            (
                np.diff(
                    predicted
                )
                > 0.0
            ).astype(
                np.int64
            ),
        )

    def production_direction_labels(
        self,
        actual_close,
        predicted_close,
        current_close,
    ) -> tuple[np.ndarray, np.ndarray]:
        actual = self._to_vector(
            actual_close,
            "actual close",
        )
        predicted = self._to_vector(
            predicted_close,
            "predicted close",
        )
        current = self._to_vector(
            current_close,
            "current close",
        )

        if not (
            len(actual)
            == len(predicted)
            == len(current)
        ):
            raise ValueError(
                "Production direction arrays must have equal lengths."
            )

        return (
            (
                actual
                > current
            ).astype(
                np.int64
            ),
            (
                predicted
                > current
            ).astype(
                np.int64
            ),
        )

    @staticmethod
    def _price_metrics(
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> dict:
        errors = (
            actual
            - predicted
        )

        mae = float(
            np.mean(
                np.abs(
                    errors
                )
            )
        )

        mse = float(
            np.mean(
                errors ** 2
            )
        )

        rmse = float(
            np.sqrt(
                mse
            )
        )

        naive_errors = (
            actual[1:]
            - actual[:-1]
        )

        naive_mse = float(
            np.mean(
                naive_errors ** 2
            )
        )

        naive_mae = float(
            np.mean(
                np.abs(
                    naive_errors
                )
            )
        )

        rmsse = (
            float(
                np.sqrt(
                    mse
                    / naive_mse
                )
            )
            if naive_mse > 0.0
            else float(
                "nan"
            )
        )

        mase = (
            float(
                mae
                / naive_mae
            )
            if naive_mae > 0.0
            else float(
                "nan"
            )
        )

        nonzero_mask = (
            actual != 0.0
        )

        mape = (
            float(
                np.mean(
                    np.abs(
                        errors[nonzero_mask]
                        / actual[nonzero_mask]
                    )
                )
                * 100.0
            )
            if nonzero_mask.any()
            else float(
                "nan"
            )
        )

        total_sum_of_squares = float(
            np.sum(
                (
                    actual
                    - np.mean(
                        actual
                    )
                ) ** 2
            )
        )

        residual_sum_of_squares = float(
            np.sum(
                errors ** 2
            )
        )

        r2 = (
            float(
                1.0
                - residual_sum_of_squares
                / total_sum_of_squares
            )
            if total_sum_of_squares > 0.0
            else float(
                "nan"
            )
        )

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "rmsse": rmsse,
            "mape": mape,
            "mase": mase,
            "r2": r2,
        }

    @staticmethod
    def _binary_metrics(
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> dict:
        if len(actual) != len(predicted):
            raise ValueError(
                "Directional label lengths do not match."
            )

        if len(actual) == 0:
            raise ValueError(
                "Directional metrics require observations."
            )

        matrix = confusion_matrix(
            actual,
            predicted,
            labels=[
                0,
                1,
            ],
        )

        return {
            "accuracy": float(
                accuracy_score(
                    actual,
                    predicted,
                )
            ),
            "recall_up": float(
                recall_score(
                    actual,
                    predicted,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "precision_up": float(
                precision_score(
                    actual,
                    predicted,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "precision_down": float(
                precision_score(
                    actual,
                    predicted,
                    pos_label=0,
                    zero_division=0,
                )
            ),
            "f1_up": float(
                f1_score(
                    actual,
                    predicted,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "actual_up_rate": float(
                np.mean(
                    actual
                    == 1
                )
            ),
            "predicted_up_rate": float(
                np.mean(
                    predicted
                    == 1
                )
            ),
            "confusion_matrix": matrix.tolist(),
        }

    @staticmethod
    def _to_vector(
        values,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

        if not np.isfinite(
            array
        ).all():
            raise ValueError(
                f"{name.capitalize()} contains non-finite values."
            )

        return array
