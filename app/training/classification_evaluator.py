import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


class ClassificationEvaluator:
    CLASS_NAMES = [
        "DOWN",
        "FLAT",
        "UP",
    ]

    def evaluate(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> dict:
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)

        if actual.ndim != 1:
            raise ValueError(
                "Actual labels must be one-dimensional."
            )

        if predicted.ndim != 1:
            raise ValueError(
                "Predicted labels must be one-dimensional."
            )

        if len(actual) != len(predicted):
            raise ValueError(
                "Actual and predicted labels must contain "
                "the same number of observations."
            )

        if len(actual) == 0:
            raise ValueError(
                "Evaluation data cannot be empty."
            )

        valid_classes = {0, 1, 2}

        if not set(np.unique(actual)).issubset(valid_classes):
            raise ValueError(
                "Actual labels contain invalid class values."
            )

        if not set(np.unique(predicted)).issubset(valid_classes):
            raise ValueError(
                "Predicted labels contain invalid class values."
            )

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                actual,
                predicted,
                labels=[0, 1, 2],
                zero_division=0,
            )
        )

        per_class = {}

        for index, class_name in enumerate(
            self.CLASS_NAMES
        ):
            per_class[class_name] = {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }

        matrix = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1, 2],
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
                        actual == class_index
                    )
                )
                for class_index, class_name in enumerate(
                    self.CLASS_NAMES
                )
            },
        }
