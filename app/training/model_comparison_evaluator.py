from collections.abc import Mapping, Sequence

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


class ModelComparisonEvaluator:
    def evaluate(
        self,
        actual_labels: Sequence,
        model_labels: Mapping[str, Sequence],
    ) -> dict[str, dict[str, float]]:
        actual = list(actual_labels)

        if not actual:
            raise ValueError(
                "Actual labels cannot be empty."
            )

        if not model_labels:
            raise ValueError(
                "At least one model prediction set is required."
            )

        metrics = {}

        for model_name, predicted_labels in model_labels.items():
            predicted = list(predicted_labels)

            if len(predicted) != len(actual):
                raise ValueError(
                    f"{model_name} produced {len(predicted)} labels "
                    f"for {len(actual)} observations."
                )

            metrics[model_name] = {
                "accuracy": accuracy_score(
                    actual,
                    predicted,
                ),
                "precision": precision_score(
                    actual,
                    predicted,
                    average="weighted",
                    zero_division=0,
                ),
                "recall": recall_score(
                    actual,
                    predicted,
                    average="weighted",
                    zero_division=0,
                ),
                "f1": f1_score(
                    actual,
                    predicted,
                    average="weighted",
                    zero_division=0,
                ),
            }

        return metrics
