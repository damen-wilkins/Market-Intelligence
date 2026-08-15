from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app.training.stage2_outer_validation_gate import moving_block_bootstrap_auc_ci


@dataclass(frozen=True)
class RegimeHypothesis:
    family: str
    feature: str
    rationale: str


REGIME_HYPOTHESES = (
    RegimeHypothesis(
        family="volatility",
        feature="implied_realized_ratio_20",
        rationale="Strongest outer-validation state contrast and largest conditional AUC spread.",
    ),
    RegimeHypothesis(
        family="volatility",
        feature="realized_volatility_20",
        rationale="Realized-volatility state separated weak and strong Stage-2 periods.",
    ),
    RegimeHypothesis(
        family="volatility",
        feature="vvix_level",
        rationale="Volatility-of-volatility showed materially different conditional AUC states.",
    ),
    RegimeHypothesis(
        family="trend",
        feature="ma_alignment_score",
        rationale="Trend alignment was one of the largest block-level state differences.",
    ),
    RegimeHypothesis(
        family="trend",
        feature="adx_14",
        rationale="Trend strength showed a large block contrast and conditional AUC range.",
    ),
    RegimeHypothesis(
        family="breadth",
        feature="rsp_relative_return_20",
        rationale="Equal-weight relative leadership differed materially through time.",
    ),
    RegimeHypothesis(
        family="breadth",
        feature="sector_volume_breadth",
        rationale="Volume breadth produced one of the largest conditional AUC ranges.",
    ),
    RegimeHypothesis(
        family="rates_credit",
        feature="hyg_lqd_relative_return_5",
        rationale="Credit-risk state showed conditional directional predictability.",
    ),
    RegimeHypothesis(
        family="cross_asset",
        feature="crude_return_5d",
        rationale="Cross-asset crude state showed a large conditional AUC range.",
    ),
)

REGIME_ORDER = ("LOW", "MID", "HIGH")


class Stage2RegimeDevelopmentResearch:
    def __init__(
        self,
        hypotheses: tuple[RegimeHypothesis, ...] = REGIME_HYPOTHESES,
        bootstrap_resamples: int = 2000,
        bootstrap_block_length: int = 20,
        permutation_resamples: int = 2000,
        random_state: int = 42,
    ):
        if bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive.")
        if bootstrap_block_length <= 0:
            raise ValueError("bootstrap_block_length must be positive.")
        if permutation_resamples <= 0:
            raise ValueError("permutation_resamples must be positive.")
        features = [hypothesis.feature for hypothesis in hypotheses]
        if len(features) != len(set(features)):
            raise ValueError("Regime hypotheses contain duplicate features.")
        self.hypotheses = hypotheses
        self.bootstrap_resamples = int(bootstrap_resamples)
        self.bootstrap_block_length = int(bootstrap_block_length)
        self.permutation_resamples = int(permutation_resamples)
        self.random_state = int(random_state)

    @property
    def feature_columns(self) -> list[str]:
        return [hypothesis.feature for hypothesis in self.hypotheses]

    def validate_features(self, dataframe: pd.DataFrame) -> None:
        missing = sorted(set(self.feature_columns) - set(dataframe.columns))
        if missing:
            raise ValueError(
                "Development regime research data is missing features: "
                + ", ".join(missing)
            )

    @staticmethod
    def development_tertiles(
        training_move: pd.DataFrame,
        feature: str,
    ) -> dict:
        values = pd.to_numeric(training_move[feature], errors="coerce").dropna()
        if values.empty:
            return {
                "feature": feature,
                "training_rows": 0,
                "q33": float("nan"),
                "q67": float("nan"),
            }
        return {
            "feature": feature,
            "training_rows": int(len(values)),
            "q33": float(values.quantile(1.0 / 3.0)),
            "q67": float(values.quantile(2.0 / 3.0)),
        }

    @staticmethod
    def assign_tertile(
        values: pd.Series,
        q33: float,
        q67: float,
    ) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        assigned = pd.Series("UNKNOWN", index=values.index, dtype="object")
        valid = numeric.notna() & np.isfinite(numeric)
        if not np.isfinite(q33) or not np.isfinite(q67):
            return assigned
        if q33 > q67:
            raise ValueError("Regime q33 cannot be greater than q67.")
        assigned.loc[valid & (numeric < q33)] = "LOW"
        assigned.loc[valid & (numeric >= q33) & (numeric <= q67)] = "MID"
        assigned.loc[valid & (numeric > q67)] = "HIGH"
        return assigned

    def fold_conditional_auc(
        self,
        enriched_predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for hypothesis in self.hypotheses:
            regime_column = self.regime_column(hypothesis.feature)
            for fold_number, fold in enriched_predictions.groupby("outer_fold", sort=True):
                for regime in REGIME_ORDER:
                    subset = fold.loc[fold[regime_column] == regime].copy()
                    rows.append(
                        self._conditional_row(
                            hypothesis=hypothesis,
                            regime=regime,
                            subset=subset,
                            outer_fold=int(fold_number),
                        )
                    )
        return pd.DataFrame(rows)

    def pooled_conditional_auc(
        self,
        enriched_predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for hypothesis in self.hypotheses:
            regime_column = self.regime_column(hypothesis.feature)
            for regime in REGIME_ORDER:
                subset = enriched_predictions.loc[
                    enriched_predictions[regime_column] == regime
                ].sort_values("target_date")
                row = self._conditional_row(
                    hypothesis=hypothesis,
                    regime=regime,
                    subset=subset,
                    outer_fold=None,
                )
                if self._has_auc(subset):
                    bootstrap = moving_block_bootstrap_auc_ci(
                        actual=subset["actual_up"].astype(int).to_numpy(),
                        score=subset["score"].astype(float).to_numpy(),
                        resamples=self.bootstrap_resamples,
                        block_length=self.bootstrap_block_length,
                        random_state=(
                            self.random_state
                            + self.feature_columns.index(hypothesis.feature) * 100
                            + REGIME_ORDER.index(regime)
                        ),
                    )
                    row.update(
                        {
                            "auc_lower_95": float(bootstrap["lower_95"]),
                            "auc_upper_95": float(bootstrap["upper_95"]),
                            "probability_auc_above_0_50": float(
                                bootstrap["probability_auc_above_0_50"]
                            ),
                            "bootstrap_valid_resamples": int(
                                bootstrap["valid_resamples"]
                            ),
                        }
                    )
                else:
                    row.update(
                        {
                            "auc_lower_95": float("nan"),
                            "auc_upper_95": float("nan"),
                            "probability_auc_above_0_50": float("nan"),
                            "bootstrap_valid_resamples": 0,
                        }
                    )
                rows.append(row)
        return pd.DataFrame(rows)

    def summarize_hypotheses(
        self,
        enriched_predictions: pd.DataFrame,
        fold_auc: pd.DataFrame,
        pooled_auc: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for hypothesis_index, hypothesis in enumerate(self.hypotheses):
            feature_pooled = pooled_auc.loc[
                pooled_auc["feature"] == hypothesis.feature
            ].copy()
            valid_pooled = feature_pooled.dropna(subset=["roc_auc"]).copy()
            if valid_pooled.empty:
                best = None
                worst = None
                auc_range = float("nan")
            else:
                best = valid_pooled.loc[valid_pooled["roc_auc"].idxmax()]
                worst = valid_pooled.loc[valid_pooled["roc_auc"].idxmin()]
                auc_range = float(best["roc_auc"] - worst["roc_auc"])

            permutation = self.regime_auc_heterogeneity_permutation(
                enriched_predictions=enriched_predictions,
                feature=hypothesis.feature,
                random_state=self.random_state + 10000 + hypothesis_index * 1000,
            )
            coverage = float(
                (
                    enriched_predictions[self.regime_column(hypothesis.feature)]
                    != "UNKNOWN"
                ).mean()
            )

            row = {
                "family": hypothesis.family,
                "feature": hypothesis.feature,
                "rationale": hypothesis.rationale,
                "prediction_coverage": coverage,
                "pooled_auc_range": auc_range,
                "heterogeneity_permutation_p": float(permutation["p_value"]),
                "heterogeneity_valid_permutations": int(
                    permutation["valid_permutations"]
                ),
            }
            for regime in REGIME_ORDER:
                pooled_row = feature_pooled.loc[feature_pooled["regime"] == regime]
                fold_rows = fold_auc.loc[
                    (fold_auc["feature"] == hypothesis.feature)
                    & (fold_auc["regime"] == regime)
                ].dropna(subset=["roc_auc"])
                if pooled_row.empty:
                    row[f"{regime.lower()}_rows"] = 0
                    row[f"{regime.lower()}_auc"] = float("nan")
                    row[f"{regime.lower()}_auc_lower_95"] = float("nan")
                    row[f"{regime.lower()}_auc_upper_95"] = float("nan")
                    row[f"{regime.lower()}_fold_auc_mean"] = float("nan")
                    row[f"{regime.lower()}_fold_auc_std"] = float("nan")
                    row[f"{regime.lower()}_valid_folds"] = 0
                    continue
                pooled_row = pooled_row.iloc[0]
                row[f"{regime.lower()}_rows"] = int(pooled_row["rows"])
                row[f"{regime.lower()}_auc"] = float(pooled_row["roc_auc"])
                row[f"{regime.lower()}_auc_lower_95"] = float(
                    pooled_row["auc_lower_95"]
                )
                row[f"{regime.lower()}_auc_upper_95"] = float(
                    pooled_row["auc_upper_95"]
                )
                row[f"{regime.lower()}_fold_auc_mean"] = (
                    float(fold_rows["roc_auc"].mean())
                    if not fold_rows.empty
                    else float("nan")
                )
                row[f"{regime.lower()}_fold_auc_std"] = (
                    float(fold_rows["roc_auc"].std(ddof=0))
                    if not fold_rows.empty
                    else float("nan")
                )
                row[f"{regime.lower()}_valid_folds"] = int(len(fold_rows))

            if best is None:
                row.update(
                    {
                        "best_regime": None,
                        "best_regime_auc": float("nan"),
                        "best_regime_auc_lower_95": float("nan"),
                        "best_regime_auc_upper_95": float("nan"),
                        "best_regime_rows": 0,
                        "best_regime_valid_folds": 0,
                        "worst_regime": None,
                        "worst_regime_auc": float("nan"),
                        "worst_regime_auc_lower_95": float("nan"),
                        "worst_regime_auc_upper_95": float("nan"),
                        "worst_regime_rows": 0,
                    }
                )
            else:
                best_regime = str(best["regime"])
                worst_regime = str(worst["regime"])
                row.update(
                    {
                        "best_regime": best_regime,
                        "best_regime_auc": float(best["roc_auc"]),
                        "best_regime_auc_lower_95": float(best["auc_lower_95"]),
                        "best_regime_auc_upper_95": float(best["auc_upper_95"]),
                        "best_regime_rows": int(best["rows"]),
                        "best_regime_valid_folds": int(
                            row[f"{best_regime.lower()}_valid_folds"]
                        ),
                        "worst_regime": worst_regime,
                        "worst_regime_auc": float(worst["roc_auc"]),
                        "worst_regime_auc_lower_95": float(worst["auc_lower_95"]),
                        "worst_regime_auc_upper_95": float(worst["auc_upper_95"]),
                        "worst_regime_rows": int(worst["rows"]),
                    }
                )
            rows.append(row)

        summary = pd.DataFrame(rows)
        summary["heterogeneity_fdr_q"] = self.benjamini_hochberg(
            summary["heterogeneity_permutation_p"].to_numpy(dtype=float)
        )
        summary["development_regime_candidate"] = (
            (summary["prediction_coverage"] >= 0.95)
            & (summary["heterogeneity_fdr_q"] <= 0.10)
            & (summary["best_regime_auc"] >= 0.55)
            & (summary["best_regime_auc_lower_95"] > 0.50)
            & (summary["best_regime_valid_folds"] >= 2)
        )
        summary["development_abstention_candidate"] = (
            (summary["prediction_coverage"] >= 0.95)
            & (summary["worst_regime_auc_upper_95"] < 0.50)
        )
        return summary.sort_values(
            [
                "development_regime_candidate",
                "heterogeneity_fdr_q",
                "pooled_auc_range",
            ],
            ascending=[False, True, False],
        ).reset_index(drop=True)

    def regime_auc_heterogeneity_permutation(
        self,
        enriched_predictions: pd.DataFrame,
        feature: str,
        random_state: int,
    ) -> dict:
        regime_column = self.regime_column(feature)
        data = enriched_predictions.loc[
            enriched_predictions[regime_column].isin(REGIME_ORDER)
        ].copy()
        observed = self._pooled_regime_auc_range(data, regime_column)
        if not np.isfinite(observed):
            return {
                "observed_auc_range": float("nan"),
                "p_value": 1.0,
                "valid_permutations": 0,
            }

        rng = np.random.default_rng(random_state)
        null_ranges: list[float] = []
        fold_frames = [fold.copy() for _, fold in data.groupby("outer_fold", sort=True)]
        for _ in range(self.permutation_resamples):
            permuted_parts = []
            for fold in fold_frames:
                part = fold[["actual_up", "score", regime_column]].copy()
                labels = part[regime_column].to_numpy(copy=True)
                rng.shuffle(labels)
                part[regime_column] = labels
                permuted_parts.append(part)
            permuted = pd.concat(permuted_parts, ignore_index=True)
            value = self._pooled_regime_auc_range(permuted, regime_column)
            if np.isfinite(value):
                null_ranges.append(float(value))

        if not null_ranges:
            return {
                "observed_auc_range": observed,
                "p_value": 1.0,
                "valid_permutations": 0,
            }
        null = np.asarray(null_ranges, dtype=np.float64)
        p_value = float((1.0 + np.sum(null >= observed)) / (1.0 + len(null)))
        return {
            "observed_auc_range": observed,
            "p_value": p_value,
            "valid_permutations": int(len(null)),
        }

    @staticmethod
    def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
        values = np.asarray(p_values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("p_values must be one-dimensional.")
        result = np.full(values.shape, np.nan, dtype=np.float64)
        valid_index = np.flatnonzero(np.isfinite(values))
        if len(valid_index) == 0:
            return result
        valid = values[valid_index]
        order = np.argsort(valid)
        ranked = valid[order]
        m = len(ranked)
        adjusted = ranked * m / np.arange(1, m + 1, dtype=np.float64)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)
        restored = np.empty_like(adjusted)
        restored[order] = adjusted
        result[valid_index] = restored
        return result

    @staticmethod
    def regime_column(feature: str) -> str:
        return f"regime__{feature}"

    @staticmethod
    def _safe_auc(actual: np.ndarray, score: np.ndarray) -> float:
        actual = np.asarray(actual, dtype=np.int64)
        score = np.asarray(score, dtype=np.float64)
        if len(actual) < 2 or len(np.unique(actual)) < 2:
            return float("nan")
        return float(roc_auc_score(actual, score))

    @classmethod
    def _has_auc(cls, subset: pd.DataFrame) -> bool:
        if len(subset) < 2:
            return False
        return len(np.unique(subset["actual_up"].astype(int).to_numpy())) == 2

    def _conditional_row(
        self,
        hypothesis: RegimeHypothesis,
        regime: str,
        subset: pd.DataFrame,
        outer_fold: int | None,
    ) -> dict:
        actual = subset["actual_up"].astype(int).to_numpy() if len(subset) else np.array([])
        score = subset["score"].astype(float).to_numpy() if len(subset) else np.array([])
        return {
            "family": hypothesis.family,
            "feature": hypothesis.feature,
            "regime": regime,
            "outer_fold": outer_fold,
            "rows": int(len(subset)),
            "up_share": float(actual.mean()) if len(actual) else float("nan"),
            "roc_auc": self._safe_auc(actual, score),
            "mean_score": float(score.mean()) if len(score) else float("nan"),
        }

    def _pooled_regime_auc_range(
        self,
        dataframe: pd.DataFrame,
        regime_column: str,
    ) -> float:
        aucs = []
        for regime in REGIME_ORDER:
            subset = dataframe.loc[dataframe[regime_column] == regime]
            if not self._has_auc(subset):
                return float("nan")
            aucs.append(
                float(
                    roc_auc_score(
                        subset["actual_up"].astype(int).to_numpy(),
                        subset["score"].astype(float).to_numpy(),
                    )
                )
            )
        return float(max(aucs) - min(aucs))
