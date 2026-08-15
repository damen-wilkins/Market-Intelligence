from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from app.training.classification_evaluator import ClassificationEvaluator
from app.training.stage2_outer_validation_gate import (
    classification_metrics,
    moving_block_bootstrap_auc_ci,
)


@dataclass(frozen=True)
class HierarchicalSelectiveGateConfig:
    stage1_min_auc: float = 0.60
    stage2_min_auc: float = 0.55
    minimum_selective_coverage: float = 0.60
    bootstrap_resamples: int = 2000
    bootstrap_block_length: int = 20
    stability_blocks: int = 3
    random_state: int = 42


class HierarchicalSelectiveOuterValidationEvaluator:
    DIRECTION_TO_CLASS = {
        "DOWN": 0,
        "FLAT": 1,
        "UP": 2,
    }

    def __init__(
        self,
        config: HierarchicalSelectiveGateConfig | None = None,
    ):
        self.config = config or HierarchicalSelectiveGateConfig()

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        stage1_threshold: float,
        stage2_threshold: float,
    ) -> dict:
        data = self._prepare(dataframe)

        actual_move = (data["actual_direction"] != "FLAT").astype(int).to_numpy()
        stage1_probability = data["stage1_move_probability"].to_numpy(dtype=np.float64)
        stage1_predicted_move = (
            stage1_probability >= float(stage1_threshold)
        ).astype(np.int64)

        stage1_auc = float(roc_auc_score(actual_move, stage1_probability))
        stage1_bootstrap = moving_block_bootstrap_auc_ci(
            actual=actual_move,
            score=stage1_probability,
            resamples=self.config.bootstrap_resamples,
            block_length=min(self.config.bootstrap_block_length, len(data)),
            random_state=self.config.random_state + 100,
        )
        stage1_class = self._binary_metrics(
            actual=actual_move,
            predicted=stage1_predicted_move,
            negative_name="FLAT",
            positive_name="MOVE",
        )

        stage2_predicted_up = (
            data["stage2_up_score"].to_numpy(dtype=np.float64)
            >= float(stage2_threshold)
        ).astype(np.int64)
        stage2_direction = np.where(stage2_predicted_up == 1, "UP", "DOWN")

        universal_prediction = np.where(
            stage1_predicted_move == 1,
            stage2_direction,
            "FLAT",
        ).astype(object)

        high_regime = data["high_volatility_regime"].astype(bool).to_numpy()
        accepted_direction = (stage1_predicted_move == 1) & high_regime
        selective_prediction = np.where(
            stage1_predicted_move == 0,
            "FLAT",
            np.where(
                accepted_direction,
                stage2_direction,
                "ABSTAIN",
            ),
        ).astype(object)

        universal_metrics = self._three_class_metrics(
            data["actual_direction"].to_numpy(dtype=object),
            universal_prediction,
        )

        covered_mask = selective_prediction != "ABSTAIN"
        selective_metrics = self._three_class_metrics(
            data.loc[covered_mask, "actual_direction"].to_numpy(dtype=object),
            selective_prediction[covered_mask],
        )

        accepted_frame = data.loc[accepted_direction].copy()
        accepted_frame["predicted_direction"] = stage2_direction[accepted_direction]
        accepted_actual_move = accepted_frame["actual_direction"] != "FLAT"
        accepted_true_move = accepted_frame.loc[accepted_actual_move].copy()

        if accepted_true_move.empty:
            raise ValueError(
                "No true MOVE rows reached the selective Stage-2 route."
            )
        actual_up = (
            accepted_true_move["actual_direction"] == "UP"
        ).astype(int).to_numpy()
        if np.unique(actual_up).size != 2:
            raise ValueError(
                "Selective Stage-2 true-MOVE rows must contain both UP and DOWN."
            )
        accepted_stage2_metrics = classification_metrics(
            actual=actual_up,
            score=accepted_true_move["stage2_up_score"].to_numpy(dtype=np.float64),
            threshold=float(stage2_threshold),
            weights=np.abs(
                accepted_true_move["future_log_return"].to_numpy(dtype=np.float64)
            ),
        )
        accepted_stage2_bootstrap = moving_block_bootstrap_auc_ci(
            actual=actual_up,
            score=accepted_true_move["stage2_up_score"].to_numpy(dtype=np.float64),
            resamples=self.config.bootstrap_resamples,
            block_length=min(
                self.config.bootstrap_block_length,
                len(accepted_true_move),
            ),
            random_state=self.config.random_state + 200,
        )

        routed_direction_accuracy = float(
            (
                accepted_frame["predicted_direction"].astype(str)
                == accepted_frame["actual_direction"].astype(str)
            ).mean()
        )
        routed_move_purity = float(accepted_actual_move.mean())

        selective_coverage = float(covered_mask.mean())
        directional_coverage = float(accepted_direction.mean())
        abstention_rate = float((selective_prediction == "ABSTAIN").mean())
        stage1_flat_output_rate = float((stage1_predicted_move == 0).mean())

        block_results = self._chronological_blocks(
            data=data,
            universal_prediction=universal_prediction,
            selective_prediction=selective_prediction,
            stage1_predicted_move=stage1_predicted_move,
            stage2_direction=stage2_direction,
            stage2_threshold=stage2_threshold,
        )

        gates = {
            "stage1_auc_at_least_0_60": bool(
                stage1_auc >= self.config.stage1_min_auc
            ),
            "stage1_bootstrap_lower_above_0_50": bool(
                stage1_bootstrap["lower_95"] > 0.50
            ),
            "accepted_stage2_auc_at_least_0_55": bool(
                accepted_stage2_metrics["roc_auc"] >= self.config.stage2_min_auc
            ),
            "accepted_stage2_bootstrap_lower_above_0_50": bool(
                accepted_stage2_bootstrap["lower_95"] > 0.50
            ),
            "selective_coverage_at_least_0_60": bool(
                selective_coverage >= self.config.minimum_selective_coverage
            ),
            "selective_balanced_accuracy_above_universal": bool(
                selective_metrics["balanced_accuracy"]
                > universal_metrics["balanced_accuracy"]
            ),
        }
        gates["overall_locked_hierarchy_gate"] = bool(all(gates.values()))

        return {
            "stage1": {
                "roc_auc": stage1_auc,
                "bootstrap_auc_lower_95": float(stage1_bootstrap["lower_95"]),
                "bootstrap_auc_upper_95": float(stage1_bootstrap["upper_95"]),
                "bootstrap_probability_auc_above_0_50": float(
                    stage1_bootstrap["probability_auc_above_0_50"]
                ),
                "decision_threshold": float(stage1_threshold),
                **stage1_class,
            },
            "stage2_selective_true_move": {
                **accepted_stage2_metrics,
                "bootstrap_auc_lower_95": float(
                    accepted_stage2_bootstrap["lower_95"]
                ),
                "bootstrap_auc_upper_95": float(
                    accepted_stage2_bootstrap["upper_95"]
                ),
                "bootstrap_probability_auc_above_0_50": float(
                    accepted_stage2_bootstrap[
                        "probability_auc_above_0_50"
                    ]
                ),
                "decision_threshold": float(stage2_threshold),
                "rows": int(len(accepted_true_move)),
            },
            "routing": {
                "total_rows": int(len(data)),
                "stage1_predicted_flat_rows": int(
                    (stage1_predicted_move == 0).sum()
                ),
                "stage1_predicted_move_rows": int(
                    (stage1_predicted_move == 1).sum()
                ),
                "accepted_direction_rows": int(accepted_direction.sum()),
                "abstained_rows": int(
                    (selective_prediction == "ABSTAIN").sum()
                ),
                "selective_covered_rows": int(covered_mask.sum()),
                "stage1_flat_output_rate": stage1_flat_output_rate,
                "directional_prediction_coverage": directional_coverage,
                "selective_total_coverage": selective_coverage,
                "abstention_rate": abstention_rate,
                "accepted_route_true_move_rows": int(accepted_actual_move.sum()),
                "accepted_route_actual_flat_rows": int(
                    (~accepted_actual_move).sum()
                ),
                "accepted_route_move_purity": routed_move_purity,
                "accepted_route_end_to_end_direction_accuracy": (
                    routed_direction_accuracy
                ),
            },
            "universal_hierarchy": universal_metrics,
            "selective_hierarchy": selective_metrics,
            "stability_blocks": block_results,
            "gates": gates,
            "predictions": self._prediction_frame(
                data=data,
                stage1_predicted_move=stage1_predicted_move,
                stage2_direction=stage2_direction,
                universal_prediction=universal_prediction,
                selective_prediction=selective_prediction,
            ),
        }

    def _chronological_blocks(
        self,
        data: pd.DataFrame,
        universal_prediction: np.ndarray,
        selective_prediction: np.ndarray,
        stage1_predicted_move: np.ndarray,
        stage2_direction: np.ndarray,
        stage2_threshold: float,
    ) -> list[dict]:
        rows: list[dict] = []
        index_blocks = np.array_split(
            np.arange(len(data)),
            self.config.stability_blocks,
        )
        for block_number, indices in enumerate(index_blocks, start=1):
            if len(indices) == 0:
                continue
            block = data.iloc[indices].copy()
            universal = universal_prediction[indices]
            selective = selective_prediction[indices]
            covered = selective != "ABSTAIN"

            universal_metrics = self._three_class_metrics(
                block["actual_direction"].to_numpy(dtype=object),
                universal,
            )
            selective_metrics = (
                self._three_class_metrics(
                    block.loc[covered, "actual_direction"].to_numpy(dtype=object),
                    selective[covered],
                )
                if covered.any()
                else None
            )

            accepted = (
                (stage1_predicted_move[indices] == 1)
                & block["high_volatility_regime"].astype(bool).to_numpy()
            )
            accepted_true_move = accepted & (
                block["actual_direction"].to_numpy(dtype=object) != "FLAT"
            )
            block_stage2_auc = float("nan")
            if accepted_true_move.sum() > 1:
                actual_up = (
                    block.loc[
                        accepted_true_move,
                        "actual_direction",
                    ].astype(str) == "UP"
                ).astype(int).to_numpy()
                if np.unique(actual_up).size == 2:
                    block_stage2_auc = float(
                        roc_auc_score(
                            actual_up,
                            block.loc[
                                accepted_true_move,
                                "stage2_up_score",
                            ].to_numpy(dtype=np.float64),
                        )
                    )

            rows.append(
                {
                    "block": int(block_number),
                    "start": pd.Timestamp(block["target_date"].iloc[0]),
                    "end": pd.Timestamp(block["target_date"].iloc[-1]),
                    "rows": int(len(block)),
                    "selective_coverage": float(covered.mean()),
                    "accepted_direction_rows": int(accepted.sum()),
                    "accepted_true_move_rows": int(accepted_true_move.sum()),
                    "accepted_true_move_auc": block_stage2_auc,
                    "universal_accuracy": float(
                        universal_metrics["accuracy"]
                    ),
                    "universal_balanced_accuracy": float(
                        universal_metrics["balanced_accuracy"]
                    ),
                    "universal_macro_f1": float(
                        universal_metrics["macro_f1"]
                    ),
                    "selective_accuracy": (
                        float(selective_metrics["accuracy"])
                        if selective_metrics is not None
                        else float("nan")
                    ),
                    "selective_balanced_accuracy": (
                        float(selective_metrics["balanced_accuracy"])
                        if selective_metrics is not None
                        else float("nan")
                    ),
                    "selective_macro_f1": (
                        float(selective_metrics["macro_f1"])
                        if selective_metrics is not None
                        else float("nan")
                    ),
                }
            )
        return rows

    def _prediction_frame(
        self,
        data: pd.DataFrame,
        stage1_predicted_move: np.ndarray,
        stage2_direction: np.ndarray,
        universal_prediction: np.ndarray,
        selective_prediction: np.ndarray,
    ) -> pd.DataFrame:
        output = data[
            [
                "feature_date",
                "target_date",
                "future_log_return",
                "actual_direction",
                "stage1_move_probability",
                "stage2_up_score",
                "realized_volatility_20",
                "high_volatility_regime",
            ]
        ].copy()
        output["stage1_predicted_move"] = stage1_predicted_move
        output["stage2_predicted_direction"] = stage2_direction
        output["universal_prediction"] = universal_prediction
        output["selective_prediction"] = selective_prediction
        return output

    def _three_class_metrics(
        self,
        actual_direction: np.ndarray,
        predicted_direction: np.ndarray,
    ) -> dict:
        if len(actual_direction) == 0:
            raise ValueError("Three-class evaluation data cannot be empty.")
        actual = np.asarray(
            [self.DIRECTION_TO_CLASS[str(value)] for value in actual_direction],
            dtype=np.int64,
        )
        predicted = np.asarray(
            [self.DIRECTION_TO_CLASS[str(value)] for value in predicted_direction],
            dtype=np.int64,
        )
        return ClassificationEvaluator().evaluate(
            actual=actual,
            predicted=predicted,
        )

    @staticmethod
    def _binary_metrics(
        actual: np.ndarray,
        predicted: np.ndarray,
        negative_name: str,
        positive_name: str,
    ) -> dict:
        actual = np.asarray(actual, dtype=np.int64)
        predicted = np.asarray(predicted, dtype=np.int64)
        precision, recall, f1, support = precision_recall_fscore_support(
            actual,
            predicted,
            labels=[0, 1],
            zero_division=0,
        )
        return {
            "balanced_accuracy": float(
                balanced_accuracy_score(actual, predicted)
            ),
            "macro_f1": float(
                f1_score(
                    actual,
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            ),
            "per_class": {
                negative_name: {
                    "precision": float(precision[0]),
                    "recall": float(recall[0]),
                    "f1": float(f1[0]),
                    "support": int(support[0]),
                },
                positive_name: {
                    "precision": float(precision[1]),
                    "recall": float(recall[1]),
                    "f1": float(f1[1]),
                    "support": int(support[1]),
                },
            },
        }

    @staticmethod
    def _prepare(dataframe: pd.DataFrame) -> pd.DataFrame:
        required = {
            "feature_date",
            "target_date",
            "future_log_return",
            "actual_direction",
            "stage1_move_probability",
            "stage2_up_score",
            "realized_volatility_20",
            "high_volatility_regime",
        }
        missing = sorted(required - set(dataframe.columns))
        if missing:
            raise ValueError(
                "Hierarchy evaluation data is missing columns: "
                + ", ".join(missing)
            )
        if dataframe.empty:
            raise ValueError("Hierarchy evaluation data is empty.")

        data = dataframe.copy()
        data["feature_date"] = pd.to_datetime(data["feature_date"])
        data["target_date"] = pd.to_datetime(data["target_date"])
        data = data.sort_values("target_date").reset_index(drop=True)

        if data["target_date"].duplicated().any():
            raise ValueError(
                "Hierarchy evaluation data contains duplicate target dates."
            )
        valid_directions = {"DOWN", "FLAT", "UP"}
        observed = set(data["actual_direction"].astype(str).unique())
        if not observed.issubset(valid_directions):
            raise ValueError(
                f"Invalid actual directions: {sorted(observed - valid_directions)}"
            )
        numeric_columns = [
            "future_log_return",
            "stage1_move_probability",
            "stage2_up_score",
            "realized_volatility_20",
        ]
        for column in numeric_columns:
            values = pd.to_numeric(data[column], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy()).all():
                raise ValueError(
                    f"Hierarchy evaluation column {column} contains non-finite values."
                )
            data[column] = values.astype(float)
        return data
