import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from app.training.stage2_outer_validation_gate import moving_block_bootstrap_auc_ci


CLASS_TO_INDEX = {"DOWN": 0, "FLAT": 1, "UP": 2}


class Stage2RouteVerifierResearch:
    def __init__(
        self,
        bootstrap_resamples: int = 2000,
        bootstrap_block_length: int = 20,
        random_state: int = 42,
    ):
        self.bootstrap_resamples = int(bootstrap_resamples)
        self.bootstrap_block_length = int(bootstrap_block_length)
        self.random_state = int(random_state)

    def binary_metrics(
        self,
        actual,
        score,
        threshold: float,
    ) -> dict:
        actual_array = np.asarray(actual, dtype=np.int64)
        score_array = np.asarray(score, dtype=np.float64)
        if len(actual_array) == 0:
            raise ValueError("Binary evaluation data cannot be empty.")
        predicted = (score_array >= float(threshold)).astype(np.int64)
        auc = (
            float(roc_auc_score(actual_array, score_array))
            if np.unique(actual_array).size == 2
            else float("nan")
        )
        return {
            "rows": int(len(actual_array)),
            "positive_share": float(actual_array.mean()),
            "roc_auc": auc,
            "balanced_accuracy": float(
                balanced_accuracy_score(actual_array, predicted)
            ),
            "macro_f1": float(
                f1_score(
                    actual_array,
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            ),
        }

    def three_class_metrics(
        self,
        actual_direction,
        predicted_direction,
    ) -> dict:
        actual = np.asarray(
            [CLASS_TO_INDEX[str(value)] for value in actual_direction],
            dtype=np.int64,
        )
        predicted = np.asarray(
            [CLASS_TO_INDEX[str(value)] for value in predicted_direction],
            dtype=np.int64,
        )
        if len(actual) == 0:
            raise ValueError("Three-class evaluation data cannot be empty.")
        per_class = f1_score(
            actual,
            predicted,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )
        return {
            "rows": int(len(actual)),
            "accuracy": float(accuracy_score(actual, predicted)),
            "balanced_accuracy": float(
                balanced_accuracy_score(actual, predicted)
            ),
            "macro_f1": float(
                f1_score(
                    actual,
                    predicted,
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            ),
            "down_f1": float(per_class[0]),
            "flat_f1": float(per_class[1]),
            "up_f1": float(per_class[2]),
        }

    def paired_block_bootstrap_three_class_delta(
        self,
        actual_direction,
        baseline_prediction,
        candidate_prediction,
        seed_offset: int = 0,
    ) -> dict:
        actual = np.asarray(actual_direction, dtype=object)
        baseline = np.asarray(baseline_prediction, dtype=object)
        candidate = np.asarray(candidate_prediction, dtype=object)
        if not (len(actual) == len(baseline) == len(candidate)):
            raise ValueError("Paired bootstrap inputs must have equal length.")
        if len(actual) == 0:
            raise ValueError("Paired bootstrap data cannot be empty.")

        actual_idx = np.asarray(
            [CLASS_TO_INDEX[str(value)] for value in actual],
            dtype=np.int64,
        )
        baseline_idx = np.asarray(
            [CLASS_TO_INDEX[str(value)] for value in baseline],
            dtype=np.int64,
        )
        candidate_idx = np.asarray(
            [CLASS_TO_INDEX[str(value)] for value in candidate],
            dtype=np.int64,
        )

        rng = np.random.default_rng(self.random_state + int(seed_offset))
        n = len(actual_idx)
        block_length = min(self.bootstrap_block_length, n)
        starts = np.arange(max(1, n - block_length + 1))
        macro_deltas = []
        balanced_deltas = []

        for _ in range(self.bootstrap_resamples):
            sampled = []
            while len(sampled) < n:
                start = int(rng.choice(starts))
                sampled.extend(
                    range(
                        start,
                        min(start + block_length, n),
                    )
                )
            index = np.asarray(sampled[:n], dtype=np.int64)
            a = actual_idx[index]
            b = baseline_idx[index]
            c = candidate_idx[index]

            baseline_macro = f1_score(
                a,
                b,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
            candidate_macro = f1_score(
                a,
                c,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
            baseline_balanced = balanced_accuracy_score(a, b)
            candidate_balanced = balanced_accuracy_score(a, c)

            macro_deltas.append(
                float(candidate_macro - baseline_macro)
            )
            balanced_deltas.append(
                float(candidate_balanced - baseline_balanced)
            )

        macro = np.asarray(macro_deltas, dtype=np.float64)
        balanced = np.asarray(balanced_deltas, dtype=np.float64)
        return {
            "macro_f1_delta_lower_95": float(
                np.quantile(macro, 0.025)
            ),
            "macro_f1_delta_upper_95": float(
                np.quantile(macro, 0.975)
            ),
            "probability_macro_f1_delta_positive": float(
                np.mean(macro > 0.0)
            ),
            "balanced_accuracy_delta_lower_95": float(
                np.quantile(balanced, 0.025)
            ),
            "balanced_accuracy_delta_upper_95": float(
                np.quantile(balanced, 0.975)
            ),
            "probability_balanced_accuracy_delta_positive": float(
                np.mean(balanced > 0.0)
            ),
        }

    def auc_bootstrap(
        self,
        dataframe: pd.DataFrame,
        actual_column: str,
        score_column: str,
        seed_offset: int = 0,
    ) -> dict:
        if dataframe.empty:
            return self._empty_auc_bootstrap()
        actual = dataframe[actual_column].astype(int).to_numpy()
        if np.unique(actual).size != 2:
            return self._empty_auc_bootstrap()
        return moving_block_bootstrap_auc_ci(
            actual=actual,
            score=dataframe[score_column].astype(float).to_numpy(),
            resamples=self.bootstrap_resamples,
            block_length=min(
                self.bootstrap_block_length,
                len(dataframe),
            ),
            random_state=self.random_state + int(seed_offset),
        )

    @staticmethod
    def route_diagnostics(
        dataframe: pd.DataFrame,
        confirmed_column: str,
    ) -> dict:
        if dataframe.empty:
            raise ValueError("Route diagnostic data cannot be empty.")
        routed = dataframe["stage1_predicted_move"].astype(bool)
        actual_move = dataframe["actual_direction"].astype(str) != "FLAT"
        confirmed = dataframe[confirmed_column].astype(bool)

        routed_rows = int(routed.sum())
        routed_true = int((routed & actual_move).sum())
        confirmed_rows = int(confirmed.sum())
        confirmed_true = int((confirmed & actual_move).sum())
        confirmed_flat = int((confirmed & ~actual_move).sum())
        true_move_rows = int(actual_move.sum())

        return {
            "stage1_routed_rows": routed_rows,
            "stage1_routed_true_move_rows": routed_true,
            "stage1_route_move_purity": (
                float(routed_true / routed_rows)
                if routed_rows > 0
                else float("nan")
            ),
            "confirmed_rows": confirmed_rows,
            "confirmed_true_move_rows": confirmed_true,
            "confirmed_flat_rows": confirmed_flat,
            "confirmed_move_purity": (
                float(confirmed_true / confirmed_rows)
                if confirmed_rows > 0
                else float("nan")
            ),
            "confirmed_flat_contamination": (
                float(confirmed_flat / confirmed_rows)
                if confirmed_rows > 0
                else float("nan")
            ),
            "true_move_recall_after_verifier": (
                float(confirmed_true / true_move_rows)
                if true_move_rows > 0
                else float("nan")
            ),
            "route_purity_lift": (
                float(
                    confirmed_true / confirmed_rows
                    - routed_true / routed_rows
                )
                if routed_rows > 0 and confirmed_rows > 0
                else float("nan")
            ),
        }

    @staticmethod
    def _empty_auc_bootstrap() -> dict:
        return {
            "lower_95": float("nan"),
            "upper_95": float("nan"),
            "probability_auc_above_0_50": float("nan"),
            "valid_resamples": 0,
        }
