from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score


CLASS_TO_INDEX = {"DOWN": 0, "FLAT": 1, "UP": 2}
INDEX_TO_CLASS = {value: key for key, value in CLASS_TO_INDEX.items()}


@dataclass(frozen=True)
class MulticlassMetrics:
    rows: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    down_f1: float
    flat_f1: float
    up_f1: float


class Stage2RouteAwareMulticlassResearch:
    def __init__(
        self,
        bootstrap_resamples: int = 2000,
        bootstrap_block_length: int = 20,
        random_state: int = 42,
    ):
        self.bootstrap_resamples = int(bootstrap_resamples)
        self.bootstrap_block_length = int(bootstrap_block_length)
        self.random_state = int(random_state)

    def metrics(self, actual_direction, predicted_direction) -> dict:
        actual = np.asarray(
            [CLASS_TO_INDEX[str(value)] for value in actual_direction],
            dtype=np.int64,
        )
        predicted = np.asarray(
            [CLASS_TO_INDEX[str(value)] for value in predicted_direction],
            dtype=np.int64,
        )
        if len(actual) == 0:
            raise ValueError("Cannot evaluate an empty multiclass sample.")
        per_class_f1 = f1_score(
            actual,
            predicted,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )
        return MulticlassMetrics(
            rows=int(len(actual)),
            accuracy=float(accuracy_score(actual, predicted)),
            balanced_accuracy=float(
                recall_score(
                    actual,
                    predicted,
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            ),
            macro_f1=float(
                f1_score(
                    actual,
                    predicted,
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            ),
            down_f1=float(per_class_f1[0]),
            flat_f1=float(per_class_f1[1]),
            up_f1=float(per_class_f1[2]),
        ).__dict__

    def paired_block_bootstrap_delta(
        self,
        actual_direction,
        baseline_prediction,
        candidate_prediction,
    ) -> dict:
        actual = np.asarray(actual_direction, dtype=object)
        baseline = np.asarray(baseline_prediction, dtype=object)
        candidate = np.asarray(candidate_prediction, dtype=object)
        if not (len(actual) == len(baseline) == len(candidate)):
            raise ValueError("Paired bootstrap inputs must have equal length.")
        if len(actual) == 0:
            raise ValueError("Paired bootstrap data cannot be empty.")

        actual_idx = np.asarray([CLASS_TO_INDEX[str(value)] for value in actual], dtype=np.int64)
        baseline_idx = np.asarray([CLASS_TO_INDEX[str(value)] for value in baseline], dtype=np.int64)
        candidate_idx = np.asarray([CLASS_TO_INDEX[str(value)] for value in candidate], dtype=np.int64)

        rng = np.random.default_rng(self.random_state)
        n = len(actual_idx)
        block_length = min(self.bootstrap_block_length, n)
        starts = np.arange(max(1, n - block_length + 1))
        macro_deltas = []
        balanced_deltas = []

        for _ in range(self.bootstrap_resamples):
            sampled = []
            while len(sampled) < n:
                start = int(rng.choice(starts))
                sampled.extend(range(start, min(start + block_length, n)))
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
            baseline_balanced = recall_score(
                a,
                b,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
            candidate_balanced = recall_score(
                a,
                c,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
            macro_deltas.append(float(candidate_macro - baseline_macro))
            balanced_deltas.append(float(candidate_balanced - baseline_balanced))

        macro = np.asarray(macro_deltas, dtype=np.float64)
        balanced = np.asarray(balanced_deltas, dtype=np.float64)
        return {
            "macro_f1_delta_lower_95": float(np.quantile(macro, 0.025)),
            "macro_f1_delta_upper_95": float(np.quantile(macro, 0.975)),
            "probability_macro_f1_delta_positive": float(np.mean(macro > 0.0)),
            "balanced_accuracy_delta_lower_95": float(np.quantile(balanced, 0.025)),
            "balanced_accuracy_delta_upper_95": float(np.quantile(balanced, 0.975)),
            "probability_balanced_accuracy_delta_positive": float(np.mean(balanced > 0.0)),
        }

    @staticmethod
    def routed_diagnostics(frame: pd.DataFrame) -> dict:
        required = {
            "actual_direction",
            "baseline_prediction",
            "candidate_prediction",
            "candidate_down_probability",
            "candidate_flat_probability",
            "candidate_up_probability",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError("Routed diagnostics are missing columns: " + ", ".join(missing))
        if frame.empty:
            raise ValueError("Routed diagnostics data cannot be empty.")

        actual = frame["actual_direction"].astype(str)
        baseline = frame["baseline_prediction"].astype(str)
        candidate = frame["candidate_prediction"].astype(str)
        false_route_flat = actual == "FLAT"
        true_move = actual != "FLAT"

        flat_correction_rate = (
            float((candidate.loc[false_route_flat] == "FLAT").mean())
            if false_route_flat.any()
            else float("nan")
        )
        baseline_true_move_direction_accuracy = (
            float((baseline.loc[true_move] == actual.loc[true_move]).mean())
            if true_move.any()
            else float("nan")
        )
        candidate_true_move_direction_accuracy = (
            float((candidate.loc[true_move] == actual.loc[true_move]).mean())
            if true_move.any()
            else float("nan")
        )

        true_move_auc = float("nan")
        if true_move.any():
            move = frame.loc[true_move].copy()
            actual_up = (move["actual_direction"].astype(str) == "UP").astype(int).to_numpy()
            if np.unique(actual_up).size == 2:
                up = move["candidate_up_probability"].to_numpy(dtype=np.float64)
                down = move["candidate_down_probability"].to_numpy(dtype=np.float64)
                denominator = up + down
                score = np.divide(
                    up,
                    denominator,
                    out=np.full_like(up, 0.5),
                    where=denominator > 1e-12,
                )
                true_move_auc = float(roc_auc_score(actual_up, score))

        return {
            "routed_rows": int(len(frame)),
            "routed_actual_flat_rows": int(false_route_flat.sum()),
            "routed_true_move_rows": int(true_move.sum()),
            "candidate_false_route_flat_correction_rate": flat_correction_rate,
            "baseline_true_move_direction_accuracy": baseline_true_move_direction_accuracy,
            "candidate_true_move_direction_accuracy": candidate_true_move_direction_accuracy,
            "candidate_true_move_up_down_auc": true_move_auc,
        }
