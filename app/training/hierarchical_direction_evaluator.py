import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from app.training.classification_evaluator import (
    ClassificationEvaluator,
)


class HierarchicalDirectionEvaluator:
    def evaluate(
        self,
        prediction_result: dict,
    ) -> dict:
        actual_labels = np.asarray(
            prediction_result[
                "actual_labels"
            ],
            dtype=np.int64,
        )

        final_predictions = np.asarray(
            prediction_result[
                "final_predictions"
            ],
            dtype=np.int64,
        )

        actual_directions = np.asarray(
            prediction_result[
                "actual_directions"
            ],
            dtype=object,
        )

        stage1_actual = (
            actual_directions
            != "FLAT"
        ).astype(
            np.int64
        )

        stage1_predicted = np.asarray(
            prediction_result[
                "stage1_predicted_move"
            ],
            dtype=np.int64,
        )

        stage2_move_mask = (
            actual_directions
            != "FLAT"
        )

        stage2_actual = (
            actual_directions[
                stage2_move_mask
            ]
            == "UP"
        ).astype(
            np.int64
        )

        stage2_predicted = np.asarray(
            prediction_result[
                "stage2_predicted_up"
            ],
            dtype=np.int64,
        )[
            stage2_move_mask
        ]

        return {
            "stage1_move_vs_flat": (
                self._evaluate_binary(
                    actual=stage1_actual,
                    predicted=stage1_predicted,
                    class_names=[
                        "FLAT",
                        "MOVE",
                    ],
                )
            ),
            "stage2_up_vs_down_oracle": (
                self._evaluate_binary(
                    actual=stage2_actual,
                    predicted=stage2_predicted,
                    class_names=[
                        "DOWN",
                        "UP",
                    ],
                )
            ),
            "end_to_end": (
                ClassificationEvaluator()
                .evaluate(
                    actual=actual_labels,
                    predicted=final_predictions,
                )
            ),
            "routing": {
                "actual_move_rate": float(
                    stage1_actual.mean()
                ),
                "predicted_move_rate": float(
                    stage1_predicted.mean()
                ),
                "actual_flat_rate": float(
                    1.0
                    - stage1_actual.mean()
                ),
                "predicted_flat_rate": float(
                    1.0
                    - stage1_predicted.mean()
                ),
                "stage2_oracle_rows": int(
                    stage2_move_mask.sum()
                ),
                "total_rows": int(
                    len(actual_labels)
                ),
            },
        }

    @staticmethod
    def _evaluate_binary(
        actual: np.ndarray,
        predicted: np.ndarray,
        class_names: list[str],
    ) -> dict:
        actual = np.asarray(
            actual,
            dtype=np.int64,
        )

        predicted = np.asarray(
            predicted,
            dtype=np.int64,
        )

        if actual.ndim != 1:
            raise ValueError(
                "Actual binary labels must be one-dimensional."
            )

        if predicted.ndim != 1:
            raise ValueError(
                "Predicted binary labels must be one-dimensional."
            )

        if len(actual) != len(predicted):
            raise ValueError(
                "Actual and predicted binary labels must contain "
                "the same number of observations."
            )

        if len(actual) == 0:
            raise ValueError(
                "Binary evaluation data cannot be empty."
            )

        if not set(
            np.unique(
                actual
            )
        ).issubset(
            {
                0,
                1,
            }
        ):
            raise ValueError(
                "Actual binary labels contain invalid values."
            )

        if not set(
            np.unique(
                predicted
            )
        ).issubset(
            {
                0,
                1,
            }
        ):
            raise ValueError(
                "Predicted binary labels contain invalid values."
            )

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                actual,
                predicted,
                labels=[
                    0,
                    1,
                ],
                zero_division=0,
            )
        )

        per_class = {}

        for index, class_name in enumerate(
            class_names
        ):
            per_class[
                class_name
            ] = {
                "precision": float(
                    precision[
                        index
                    ]
                ),
                "recall": float(
                    recall[
                        index
                    ]
                ),
                "f1": float(
                    f1[
                        index
                    ]
                ),
                "support": int(
                    support[
                        index
                    ]
                ),
            }

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
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    actual,
                    predicted,
                )
            ),
            "macro_f1": float(
                f1_score(
                    actual,
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            ),
            "per_class": per_class,
            "confusion_matrix": matrix.tolist(),
            "class_counts": {
                class_name: int(
                    np.sum(
                        actual
                        == class_index
                    )
                )
                for class_index, class_name in enumerate(
                    class_names
                )
            },
        }
