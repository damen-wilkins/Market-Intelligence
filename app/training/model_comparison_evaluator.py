from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


class ModelComparisonEvaluator:
    def evaluate(
        self,
        actual_labels,
        sarimax_labels,
        xgboost_labels,
    ):
        return {
            "sarimax": {
                "accuracy": accuracy_score(
                    actual_labels,
                    sarimax_labels,
                ),
                "precision": precision_score(
                    actual_labels,
                    sarimax_labels,
                    average="weighted",
                    zero_division=0,
                ),
                "recall": recall_score(
                    actual_labels,
                    sarimax_labels,
                    average="weighted",
                    zero_division=0,
                ),
                "f1": f1_score(
                    actual_labels,
                    sarimax_labels,
                    average="weighted",
                    zero_division=0,
                ),
            },
            "xgboost": {
                "accuracy": accuracy_score(
                    actual_labels,
                    xgboost_labels,
                ),
                "precision": precision_score(
                    actual_labels,
                    xgboost_labels,
                    average="weighted",
                    zero_division=0,
                ),
                "recall": recall_score(
                    actual_labels,
                    xgboost_labels,
                    average="weighted",
                    zero_division=0,
                ),
                "f1": f1_score(
                    actual_labels,
                    xgboost_labels,
                    average="weighted",
                    zero_division=0,
                ),
            },
        }