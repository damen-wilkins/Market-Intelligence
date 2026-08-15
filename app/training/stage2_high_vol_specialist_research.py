from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from app.training.stage2_conditioned_target_research import (
    moving_block_bootstrap_auc_delta,
)
from app.training.stage2_outer_validation_gate import (
    moving_block_bootstrap_auc_ci,
)


@dataclass(frozen=True)
class HighVolSpecialistFoldResult:
    outer_fold: int
    regime_feature: str
    regime_quantile: float
    regime_threshold: float
    training_move_rows: int
    training_high_rows: int
    test_move_rows: int
    test_high_rows: int
    test_high_coverage: float
    universal_auc: float
    specialist_auc: float
    delta_auc: float
    universal_decision_threshold: float
    specialist_decision_threshold: float
    specialist_threshold_oof_rows: int
    universal_balanced_accuracy: float
    specialist_balanced_accuracy: float
    universal_macro_f1: float
    specialist_macro_f1: float
    universal_sign_accuracy: float
    specialist_sign_accuracy: float
    universal_magnitude_weighted_sign_accuracy: float
    specialist_magnitude_weighted_sign_accuracy: float


class Stage2HighVolSpecialistResearch:
    def __init__(
        self,
        feature_columns: list[str],
        regime_feature: str,
        regime_quantile: float,
        inner_splits: int = 3,
        bootstrap_resamples: int = 2000,
        bootstrap_block_length: int = 20,
        random_state: int = 42,
    ):
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        if regime_feature not in feature_columns:
            raise ValueError(
                f"Regime feature {regime_feature!r} must already be present in the "
                "locked Stage-2 feature set."
            )
        if not 0.0 < regime_quantile < 1.0:
            raise ValueError("regime_quantile must be between zero and one.")
        if inner_splits < 2:
            raise ValueError("inner_splits must be at least two.")
        if bootstrap_resamples <= 0 or bootstrap_block_length <= 0:
            raise ValueError("Bootstrap settings must be positive.")
        self.feature_columns = list(feature_columns)
        self.regime_feature = regime_feature
        self.regime_quantile = float(regime_quantile)
        self.inner_splits = int(inner_splits)
        self.bootstrap_resamples = int(bootstrap_resamples)
        self.bootstrap_block_length = int(bootstrap_block_length)
        self.random_state = int(random_state)

    def validate_data(self, dataframe: pd.DataFrame) -> None:
        required = {
            "target_date",
            "direction",
            "future_log_return",
            self.regime_feature,
            *self.feature_columns,
        }
        missing = sorted(required - set(dataframe.columns))
        if missing:
            raise ValueError(
                "High-volatility specialist data is missing columns: "
                + ", ".join(missing)
            )

    def regime_threshold(self, training_move: pd.DataFrame) -> float:
        values = pd.to_numeric(
            training_move[self.regime_feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            raise ValueError(
                f"No finite training values are available for {self.regime_feature}."
            )
        return float(values.quantile(self.regime_quantile))

    def high_regime_mask(
        self,
        dataframe: pd.DataFrame,
        threshold: float,
    ) -> pd.Series:
        values = pd.to_numeric(
            dataframe[self.regime_feature], errors="coerce"
        )
        return values.notna() & np.isfinite(values) & (values > float(threshold))

    def fit_specialist(
        self,
        training_high: pd.DataFrame,
        parameters: dict,
        outer_fold: int,
    ) -> XGBClassifier:
        self._validate_binary_classes(training_high, "specialist training")
        model = self._model(
            parameters=parameters,
            random_state=self.random_state + 700000 + int(outer_fold),
        )
        model.fit(
            training_high[self.feature_columns],
            self._actual(training_high),
        )
        return model

    def select_specialist_threshold(
        self,
        outer_train: pd.DataFrame,
        parameters: dict,
        outer_fold: int,
    ) -> dict:
        move = self._move_rows(outer_train)
        splitter = TimeSeriesSplit(n_splits=self.inner_splits)
        actual_batches: list[np.ndarray] = []
        score_batches: list[np.ndarray] = []
        fold_rows: list[dict] = []

        for inner_fold, (train_index, validation_index) in enumerate(
            splitter.split(move), start=1
        ):
            inner_train = move.iloc[train_index].reset_index(drop=True)
            inner_validation = move.iloc[validation_index].reset_index(drop=True)
            threshold = self.regime_threshold(inner_train)
            train_high = inner_train.loc[
                self.high_regime_mask(inner_train, threshold)
            ].reset_index(drop=True)
            validation_high = inner_validation.loc[
                self.high_regime_mask(inner_validation, threshold)
            ].reset_index(drop=True)

            usable = (
                len(train_high) > 0
                and len(validation_high) > 0
                and self._has_both_classes(train_high)
                and self._has_both_classes(validation_high)
            )
            row = {
                "outer_fold": int(outer_fold),
                "inner_fold": int(inner_fold),
                "regime_threshold": float(threshold),
                "training_high_rows": int(len(train_high)),
                "validation_high_rows": int(len(validation_high)),
                "usable_for_threshold": bool(usable),
            }
            if not usable:
                fold_rows.append(row)
                continue

            model = self._model(
                parameters=parameters,
                random_state=(
                    self.random_state
                    + 800000
                    + int(outer_fold) * 100
                    + inner_fold
                ),
            )
            model.fit(
                train_high[self.feature_columns],
                self._actual(train_high),
            )
            actual = self._actual(validation_high)
            score = model.predict_proba(
                validation_high[self.feature_columns]
            )[:, 1]
            actual_batches.append(actual)
            score_batches.append(score)
            row["validation_auc"] = float(roc_auc_score(actual, score))
            fold_rows.append(row)

        if not actual_batches:
            return {
                "decision_threshold": float("nan"),
                "oof_rows": 0,
                "oof_auc": float("nan"),
                "folds": fold_rows,
            }

        actual = np.concatenate(actual_batches)
        score = np.concatenate(score_batches)
        if len(np.unique(actual)) < 2:
            return {
                "decision_threshold": float("nan"),
                "oof_rows": int(len(actual)),
                "oof_auc": float("nan"),
                "folds": fold_rows,
            }
        selected = self._select_probability_threshold(actual, score)
        return {
            "decision_threshold": float(selected["threshold"]),
            "oof_rows": int(len(actual)),
            "oof_auc": float(roc_auc_score(actual, score)),
            "folds": fold_rows,
        }

    def evaluate_fold(
        self,
        outer_fold: int,
        outer_train: pd.DataFrame,
        outer_test: pd.DataFrame,
        universal_saved: dict,
        parameters: dict,
    ) -> tuple[dict, pd.DataFrame, list[dict]]:
        self.validate_data(outer_train)
        self.validate_data(outer_test)
        train_move = self._move_rows(outer_train)
        test_move = self._move_rows(outer_test)
        self._validate_saved_predictions(test_move, universal_saved, outer_fold)

        threshold = self.regime_threshold(train_move)
        train_high = train_move.loc[
            self.high_regime_mask(train_move, threshold)
        ].reset_index(drop=True)
        test_high_mask = self.high_regime_mask(test_move, threshold)
        test_high = test_move.loc[test_high_mask].reset_index(drop=True)
        self._validate_binary_classes(train_high, "specialist training")
        self._validate_binary_classes(test_high, "specialist verification")

        saved_score = np.asarray(universal_saved["score"], dtype=np.float64)
        universal_high_score = saved_score[test_high_mask.to_numpy()]
        actual = self._actual(test_high)

        specialist_model = self.fit_specialist(
            training_high=train_high,
            parameters=parameters,
            outer_fold=outer_fold,
        )
        specialist_high_score = specialist_model.predict_proba(
            test_high[self.feature_columns]
        )[:, 1]
        threshold_selection = self.select_specialist_threshold(
            outer_train=outer_train,
            parameters=parameters,
            outer_fold=outer_fold,
        )
        specialist_threshold = float(threshold_selection["decision_threshold"])
        universal_threshold = float(universal_saved["decision_threshold"])

        universal_metrics = self._classification_metrics(
            dataframe=test_high,
            score=universal_high_score,
            threshold=universal_threshold,
        )
        specialist_metrics = self._classification_metrics(
            dataframe=test_high,
            score=specialist_high_score,
            threshold=specialist_threshold,
        )
        universal_auc = float(roc_auc_score(actual, universal_high_score))
        specialist_auc = float(roc_auc_score(actual, specialist_high_score))

        predictions = test_high[
            [
                "target_date",
                "feature_date",
                "future_log_return",
                "direction",
                self.regime_feature,
            ]
        ].copy()
        predictions["outer_fold"] = int(outer_fold)
        predictions["regime_threshold"] = float(threshold)
        predictions["actual_up"] = actual
        predictions["universal_score"] = universal_high_score
        predictions["specialist_score"] = specialist_high_score
        predictions["universal_decision_threshold"] = universal_threshold
        predictions["specialist_decision_threshold"] = specialist_threshold

        result = HighVolSpecialistFoldResult(
            outer_fold=int(outer_fold),
            regime_feature=self.regime_feature,
            regime_quantile=self.regime_quantile,
            regime_threshold=float(threshold),
            training_move_rows=int(len(train_move)),
            training_high_rows=int(len(train_high)),
            test_move_rows=int(len(test_move)),
            test_high_rows=int(len(test_high)),
            test_high_coverage=float(len(test_high) / len(test_move)),
            universal_auc=universal_auc,
            specialist_auc=specialist_auc,
            delta_auc=float(specialist_auc - universal_auc),
            universal_decision_threshold=universal_threshold,
            specialist_decision_threshold=specialist_threshold,
            specialist_threshold_oof_rows=int(threshold_selection["oof_rows"]),
            universal_balanced_accuracy=universal_metrics["balanced_accuracy"],
            specialist_balanced_accuracy=specialist_metrics["balanced_accuracy"],
            universal_macro_f1=universal_metrics["macro_f1"],
            specialist_macro_f1=specialist_metrics["macro_f1"],
            universal_sign_accuracy=universal_metrics["sign_accuracy"],
            specialist_sign_accuracy=specialist_metrics["sign_accuracy"],
            universal_magnitude_weighted_sign_accuracy=universal_metrics[
                "magnitude_weighted_sign_accuracy"
            ],
            specialist_magnitude_weighted_sign_accuracy=specialist_metrics[
                "magnitude_weighted_sign_accuracy"
            ],
        )
        return result.__dict__, predictions, threshold_selection["folds"]

    def summarize(self, predictions: pd.DataFrame, fold_results: pd.DataFrame) -> dict:
        predictions = predictions.sort_values("target_date").reset_index(drop=True)
        actual = predictions["actual_up"].astype(int).to_numpy()
        universal_score = predictions["universal_score"].astype(float).to_numpy()
        specialist_score = predictions["specialist_score"].astype(float).to_numpy()
        self._validate_arrays(actual, universal_score, specialist_score)

        universal_auc = float(roc_auc_score(actual, universal_score))
        specialist_auc = float(roc_auc_score(actual, specialist_score))
        specialist_bootstrap = moving_block_bootstrap_auc_ci(
            actual=actual,
            score=specialist_score,
            resamples=self.bootstrap_resamples,
            block_length=self.bootstrap_block_length,
            random_state=self.random_state + 900000,
        )
        delta_bootstrap = moving_block_bootstrap_auc_delta(
            actual=actual,
            candidate_probabilities=specialist_score,
            baseline_probabilities=universal_score,
            resamples=self.bootstrap_resamples,
            block_length=self.bootstrap_block_length,
            random_state=self.random_state + 910000,
        )
        fold_deltas = fold_results["delta_auc"].astype(float).to_numpy()
        folds_improved = int(np.sum(fold_deltas > 0.0))

        gates = {
            "specialist_auc_above_universal": bool(specialist_auc > universal_auc),
            "specialist_bootstrap_lower_above_0_50": bool(
                specialist_bootstrap["lower_95"] > 0.50
            ),
            "delta_bootstrap_lower_above_0": bool(
                delta_bootstrap["lower_95"] > 0.0
            ),
            "at_least_two_folds_improved": bool(folds_improved >= 2),
        }
        return {
            "high_regime_rows": int(len(predictions)),
            "high_regime_up_share": float(actual.mean()),
            "universal_high_regime_auc": universal_auc,
            "specialist_high_regime_auc": specialist_auc,
            "delta_auc": float(specialist_auc - universal_auc),
            "specialist_auc_lower_95": float(specialist_bootstrap["lower_95"]),
            "specialist_auc_upper_95": float(specialist_bootstrap["upper_95"]),
            "specialist_probability_auc_above_0_50": float(
                specialist_bootstrap["probability_auc_above_0_50"]
            ),
            "delta_auc_lower_95": float(delta_bootstrap["lower_95"]),
            "delta_auc_upper_95": float(delta_bootstrap["upper_95"]),
            "probability_delta_positive": float(
                delta_bootstrap["probability_delta_positive"]
            ),
            "fold_delta_auc_mean": float(np.mean(fold_deltas)),
            "fold_delta_auc_std": float(np.std(fold_deltas, ddof=0)),
            "folds_improved": folds_improved,
            "fold_count": int(len(fold_deltas)),
            "gates": gates,
            "architecture_gate_pass": bool(all(gates.values())),
        }

    def _model(self, parameters: dict, random_state: int) -> XGBClassifier:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=int(random_state),
            n_jobs=-1,
            **parameters,
        )


    @staticmethod
    def _select_probability_threshold(
        actual: np.ndarray,
        positive_probabilities: np.ndarray,
    ) -> dict:
        actual = np.asarray(actual, dtype=np.int64)
        probabilities = np.asarray(positive_probabilities, dtype=np.float64)
        if actual.ndim != 1 or probabilities.ndim != 1:
            raise ValueError("Threshold arrays must be one-dimensional.")
        if len(actual) != len(probabilities) or len(actual) == 0:
            raise ValueError("Threshold labels and probabilities must align and be non-empty.")
        if not set(np.unique(actual)).issubset({0, 1}):
            raise ValueError("Threshold labels must be binary.")
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0.0).any()
            or (probabilities > 1.0).any()
        ):
            raise ValueError("Probabilities must be finite values between zero and one.")
        unique_probabilities = np.unique(probabilities)
        if len(unique_probabilities) == 1:
            candidate_thresholds = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
        else:
            midpoints = (unique_probabilities[:-1] + unique_probabilities[1:]) / 2.0
            candidate_thresholds = np.unique(
                np.concatenate(
                    [np.asarray([0.0, 0.5, 1.0], dtype=np.float64), midpoints]
                )
            )
        best = None
        for threshold in candidate_thresholds:
            predicted = (probabilities >= threshold).astype(np.int64)
            candidate = {
                "threshold": float(threshold),
                "macro_f1": float(
                    f1_score(actual, predicted, average="macro", zero_division=0)
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(actual, predicted)
                ),
                "distance_from_half": abs(float(threshold) - 0.5),
            }
            if best is None:
                best = candidate
                continue
            candidate_key = (
                candidate["macro_f1"],
                candidate["balanced_accuracy"],
                -candidate["distance_from_half"],
            )
            best_key = (
                best["macro_f1"],
                best["balanced_accuracy"],
                -best["distance_from_half"],
            )
            if candidate_key > best_key:
                best = candidate
        return best

    @staticmethod
    def _move_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.loc[
            dataframe["direction"].astype(str) != "FLAT"
        ].sort_values("target_date").reset_index(drop=True)

    @staticmethod
    def _actual(dataframe: pd.DataFrame) -> np.ndarray:
        return (
            dataframe["direction"].astype(str) == "UP"
        ).astype(int).to_numpy(dtype=np.int64)

    @staticmethod
    def _has_both_classes(dataframe: pd.DataFrame) -> bool:
        if dataframe.empty:
            return False
        return len(np.unique(Stage2HighVolSpecialistResearch._actual(dataframe))) == 2

    @staticmethod
    def _validate_binary_classes(dataframe: pd.DataFrame, label: str) -> None:
        if dataframe.empty:
            raise ValueError(f"{label} data is empty.")
        if not Stage2HighVolSpecialistResearch._has_both_classes(dataframe):
            raise ValueError(f"{label} data does not contain both UP and DOWN classes.")

    def _classification_metrics(
        self,
        dataframe: pd.DataFrame,
        score: np.ndarray,
        threshold: float,
    ) -> dict:
        actual = self._actual(dataframe)
        if not np.isfinite(threshold):
            return {
                "balanced_accuracy": float("nan"),
                "macro_f1": float("nan"),
                "sign_accuracy": float("nan"),
                "magnitude_weighted_sign_accuracy": float("nan"),
            }
        predicted = (np.asarray(score, dtype=np.float64) >= threshold).astype(int)
        weights = np.abs(dataframe["future_log_return"].astype(float).to_numpy())
        weighted_accuracy = (
            float(np.average(predicted == actual, weights=weights))
            if float(weights.sum()) > 0.0
            else float(np.mean(predicted == actual))
        )
        return {
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
            "sign_accuracy": float(np.mean(predicted == actual)),
            "magnitude_weighted_sign_accuracy": weighted_accuracy,
        }

    @staticmethod
    def _validate_arrays(
        actual: np.ndarray,
        universal_score: np.ndarray,
        specialist_score: np.ndarray,
    ) -> None:
        if not (
            len(actual) == len(universal_score) == len(specialist_score)
        ):
            raise ValueError("Pooled specialist arrays do not align.")
        if len(np.unique(actual)) < 2:
            raise ValueError("Pooled specialist evaluation requires both classes.")

    @staticmethod
    def _validate_saved_predictions(
        test_move: pd.DataFrame,
        saved: dict,
        outer_fold: int,
    ) -> None:
        saved_dates = pd.DatetimeIndex(pd.to_datetime(saved["target_dates"]))
        expected_dates = pd.DatetimeIndex(pd.to_datetime(test_move["target_date"]))
        if not saved_dates.equals(expected_dates):
            raise ValueError(
                f"Fold {outer_fold} saved universal predictions no longer align with "
                "the reconstructed development sample."
            )
        saved_actual = np.asarray(saved["actual"], dtype=np.int64)
        expected_actual = Stage2HighVolSpecialistResearch._actual(test_move)
        if not np.array_equal(saved_actual, expected_actual):
            raise ValueError(
                f"Fold {outer_fold} saved universal labels do not match the locked target."
            )
