from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from app.training.stage2_outer_validation_gate import moving_block_bootstrap_auc_ci


@dataclass(frozen=True)
class SelectiveSubsetMetrics:
    subset: str
    rows: int
    up_share: float
    roc_auc: float
    balanced_accuracy: float
    macro_f1: float
    sign_accuracy: float
    magnitude_weighted_sign_accuracy: float


class Stage2SelectivePredictionResearch:
    REQUIRED_COLUMNS = (
        "target_date",
        "future_log_return",
        "actual_up",
        "score",
        "predicted_up",
        "outer_fold",
        "regime",
    )

    def __init__(
        self,
        accepted_regime: str = "HIGH",
        bootstrap_resamples: int = 2000,
        bootstrap_block_length: int = 20,
        random_state: int = 42,
    ):
        if not accepted_regime:
            raise ValueError("accepted_regime is required.")
        if bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive.")
        if bootstrap_block_length <= 0:
            raise ValueError("bootstrap_block_length must be positive.")
        self.accepted_regime = str(accepted_regime).upper()
        self.bootstrap_resamples = int(bootstrap_resamples)
        self.bootstrap_block_length = int(bootstrap_block_length)
        self.random_state = int(random_state)

    def evaluate(
        self,
        predictions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict]:
        self._validate(predictions)
        data = predictions.sort_values("target_date").reset_index(drop=True).copy()
        data["selective_action"] = np.where(
            data["regime"].astype(str).str.upper() == self.accepted_regime,
            "PREDICT",
            "ABSTAIN",
        )

        fold_rows: list[dict] = []
        for outer_fold, fold in data.groupby("outer_fold", sort=True):
            full = self._subset_metrics(fold, "ALL")
            accepted = self._subset_metrics(
                fold.loc[fold["selective_action"] == "PREDICT"],
                "PREDICT",
            )
            abstained = self._subset_metrics(
                fold.loc[fold["selective_action"] == "ABSTAIN"],
                "ABSTAIN",
            )
            fold_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "move_rows": int(len(fold)),
                    "accepted_rows": int((fold["selective_action"] == "PREDICT").sum()),
                    "accepted_coverage": float(
                        (fold["selective_action"] == "PREDICT").mean()
                    ),
                    "full_auc": full.roc_auc,
                    "accepted_auc": accepted.roc_auc,
                    "abstained_auc": abstained.roc_auc,
                    "accepted_balanced_accuracy": accepted.balanced_accuracy,
                    "accepted_macro_f1": accepted.macro_f1,
                    "accepted_sign_accuracy": accepted.sign_accuracy,
                    "accepted_magnitude_weighted_sign_accuracy": (
                        accepted.magnitude_weighted_sign_accuracy
                    ),
                }
            )

        fold_frame = pd.DataFrame(fold_rows)
        full = self._subset_metrics(data, "ALL")
        accepted_data = data.loc[data["selective_action"] == "PREDICT"].copy()
        abstained_data = data.loc[data["selective_action"] == "ABSTAIN"].copy()
        accepted = self._subset_metrics(accepted_data, "PREDICT")
        abstained = self._subset_metrics(abstained_data, "ABSTAIN")

        accepted_bootstrap = self._bootstrap_auc(
            accepted_data,
            random_state=self.random_state + 1000,
        )
        abstained_bootstrap = self._bootstrap_auc(
            abstained_data,
            random_state=self.random_state + 2000,
        )

        valid_fold_auc = pd.to_numeric(
            fold_frame["accepted_auc"],
            errors="coerce",
        )
        valid_fold_auc = valid_fold_auc[np.isfinite(valid_fold_auc)]
        folds_above_055 = int((valid_fold_auc >= 0.55).sum())

        gates = {
            "accepted_auc_at_least_0_55": bool(
                np.isfinite(accepted.roc_auc) and accepted.roc_auc >= 0.55
            ),
            "accepted_bootstrap_lower_above_0_50": bool(
                np.isfinite(accepted_bootstrap["lower_95"])
                and accepted_bootstrap["lower_95"] > 0.50
            ),
            "at_least_two_folds_auc_at_least_0_55": bool(
                folds_above_055 >= 2
            ),
        }
        gates["overall_selective_prediction_gate"] = bool(all(gates.values()))

        summary = {
            "move_rows": int(len(data)),
            "accepted_rows": int(len(accepted_data)),
            "abstained_rows": int(len(abstained_data)),
            "accepted_coverage": float(len(accepted_data) / len(data)),
            "full_auc": full.roc_auc,
            "accepted_auc": accepted.roc_auc,
            "abstained_auc": abstained.roc_auc,
            "accepted_auc_lift_vs_full": (
                float(accepted.roc_auc - full.roc_auc)
                if np.isfinite(accepted.roc_auc) and np.isfinite(full.roc_auc)
                else float("nan")
            ),
            "accepted_auc_lower_95": accepted_bootstrap["lower_95"],
            "accepted_auc_upper_95": accepted_bootstrap["upper_95"],
            "accepted_probability_auc_above_0_50": accepted_bootstrap[
                "probability_auc_above_0_50"
            ],
            "abstained_auc_lower_95": abstained_bootstrap["lower_95"],
            "abstained_auc_upper_95": abstained_bootstrap["upper_95"],
            "accepted_balanced_accuracy": accepted.balanced_accuracy,
            "accepted_macro_f1": accepted.macro_f1,
            "accepted_sign_accuracy": accepted.sign_accuracy,
            "accepted_magnitude_weighted_sign_accuracy": (
                accepted.magnitude_weighted_sign_accuracy
            ),
            "abstained_balanced_accuracy": abstained.balanced_accuracy,
            "abstained_macro_f1": abstained.macro_f1,
            "folds_with_valid_accepted_auc": int(len(valid_fold_auc)),
            "folds_accepted_auc_at_least_0_55": folds_above_055,
            "gates": gates,
        }
        return fold_frame, summary

    def _bootstrap_auc(
        self,
        dataframe: pd.DataFrame,
        random_state: int,
    ) -> dict:
        if not self._has_auc(dataframe):
            return {
                "lower_95": float("nan"),
                "upper_95": float("nan"),
                "probability_auc_above_0_50": float("nan"),
                "valid_resamples": 0,
            }
        ordered = dataframe.sort_values("target_date")
        return moving_block_bootstrap_auc_ci(
            actual=ordered["actual_up"].astype(int).to_numpy(),
            score=ordered["score"].astype(float).to_numpy(),
            resamples=self.bootstrap_resamples,
            block_length=min(self.bootstrap_block_length, len(ordered)),
            random_state=random_state,
        )

    def _subset_metrics(
        self,
        dataframe: pd.DataFrame,
        subset: str,
    ) -> SelectiveSubsetMetrics:
        if dataframe.empty:
            return SelectiveSubsetMetrics(
                subset=subset,
                rows=0,
                up_share=float("nan"),
                roc_auc=float("nan"),
                balanced_accuracy=float("nan"),
                macro_f1=float("nan"),
                sign_accuracy=float("nan"),
                magnitude_weighted_sign_accuracy=float("nan"),
            )

        actual = dataframe["actual_up"].astype(int).to_numpy()
        score = dataframe["score"].astype(float).to_numpy()
        predicted = dataframe["predicted_up"].astype(int).to_numpy()

        roc_auc = (
            float(roc_auc_score(actual, score))
            if np.unique(actual).size == 2
            else float("nan")
        )
        balanced_accuracy = (
            float(balanced_accuracy_score(actual, predicted))
            if np.unique(actual).size == 2
            else float("nan")
        )
        macro_f1 = float(
            f1_score(
                actual,
                predicted,
                average="macro",
                labels=[0, 1],
                zero_division=0,
            )
        )
        sign_accuracy = float(np.mean(actual == predicted))

        magnitude = np.abs(
            pd.to_numeric(
                dataframe["future_log_return"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
        )
        correct = (actual == predicted).astype(np.float64)
        valid = np.isfinite(magnitude)
        magnitude_sum = float(magnitude[valid].sum())
        magnitude_weighted = (
            float(np.sum(correct[valid] * magnitude[valid]) / magnitude_sum)
            if magnitude_sum > 0.0
            else float("nan")
        )

        return SelectiveSubsetMetrics(
            subset=subset,
            rows=int(len(dataframe)),
            up_share=float(actual.mean()),
            roc_auc=roc_auc,
            balanced_accuracy=balanced_accuracy,
            macro_f1=macro_f1,
            sign_accuracy=sign_accuracy,
            magnitude_weighted_sign_accuracy=magnitude_weighted,
        )

    @staticmethod
    def _has_auc(dataframe: pd.DataFrame) -> bool:
        if dataframe.empty:
            return False
        actual = pd.to_numeric(
            dataframe["actual_up"],
            errors="coerce",
        ).dropna()
        return actual.nunique() == 2

    def _validate(self, predictions: pd.DataFrame) -> None:
        missing = sorted(set(self.REQUIRED_COLUMNS) - set(predictions.columns))
        if missing:
            raise ValueError(
                "Selective prediction data is missing columns: "
                + ", ".join(missing)
            )
        if predictions.empty:
            raise ValueError("Selective prediction data is empty.")
        if predictions["target_date"].duplicated().any():
            raise ValueError("Selective prediction data contains duplicate target dates.")
        if predictions["outer_fold"].isna().any():
            raise ValueError("outer_fold contains missing values.")
        if predictions["actual_up"].isna().any():
            raise ValueError("actual_up contains missing values.")
        if predictions["score"].isna().any():
            raise ValueError("score contains missing values.")
