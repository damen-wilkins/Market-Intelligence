from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class RegimeFeatureSpec:
    family: str
    feature: str


REGIME_FEATURE_SPECS = (
    RegimeFeatureSpec("volatility", "target_rolling_volatility"),
    RegimeFeatureSpec("volatility", "realized_volatility_20"),
    RegimeFeatureSpec("volatility", "realized_volatility_40"),
    RegimeFeatureSpec("volatility", "vix_level"),
    RegimeFeatureSpec("volatility", "vvix_level"),
    RegimeFeatureSpec("volatility", "vix_vix3m_ratio"),
    RegimeFeatureSpec("volatility", "implied_realized_ratio_20"),
    RegimeFeatureSpec("trend", "close_vs_sma_50"),
    RegimeFeatureSpec("trend", "ma_alignment_score"),
    RegimeFeatureSpec("trend", "adx_14"),
    RegimeFeatureSpec("trend", "signed_trend_efficiency_20"),
    RegimeFeatureSpec("trend", "distance_from_252d_high"),
    RegimeFeatureSpec("breadth", "rsp_relative_return_20"),
    RegimeFeatureSpec("breadth", "sector_positive_participation_5d"),
    RegimeFeatureSpec("breadth", "sector_return_dispersion_5d"),
    RegimeFeatureSpec("breadth", "sector_average_correlation_20d"),
    RegimeFeatureSpec("breadth", "sector_volume_breadth"),
    RegimeFeatureSpec("breadth", "cyclical_defensive_spread_5d"),
    RegimeFeatureSpec("rates_credit", "tlt_ief_relative_return_5"),
    RegimeFeatureSpec("rates_credit", "hyg_lqd_relative_return_5"),
    RegimeFeatureSpec("rates_credit", "hyg_tlt_risk_on_5d"),
    RegimeFeatureSpec("liquidity", "relative_volume_20"),
    RegimeFeatureSpec("liquidity", "log_volume_change"),
    RegimeFeatureSpec("cross_asset", "iwm_relative_return_5"),
    RegimeFeatureSpec("cross_asset", "qqq_relative_return_5"),
    RegimeFeatureSpec("cross_asset", "dxy_return_5d"),
    RegimeFeatureSpec("cross_asset", "crude_return_5d"),
    RegimeFeatureSpec("cross_asset", "gold_dollar_relative_return_5d"),
    RegimeFeatureSpec("cross_asset", "futures_cash_confirmation"),
)


class Stage2RegimeDiagnostic:
    def __init__(
        self,
        feature_specs: tuple[RegimeFeatureSpec, ...] = REGIME_FEATURE_SPECS,
        block_count: int = 3,
        minimum_auc_rows: int = 20,
    ):
        if block_count <= 0:
            raise ValueError("block_count must be positive.")
        if minimum_auc_rows <= 1:
            raise ValueError("minimum_auc_rows must be greater than one.")
        features = [spec.feature for spec in feature_specs]
        if len(features) != len(set(features)):
            raise ValueError("Regime feature specifications contain duplicates.")
        self.feature_specs = feature_specs
        self.block_count = int(block_count)
        self.minimum_auc_rows = int(minimum_auc_rows)

    @property
    def feature_columns(self) -> list[str]:
        return [spec.feature for spec in self.feature_specs]

    def validate_feature_columns(self, dataframe: pd.DataFrame) -> None:
        missing = sorted(set(self.feature_columns) - set(dataframe.columns))
        if missing:
            raise ValueError(
                "Regime diagnostic data is missing required features: "
                + ", ".join(missing)
            )

    def assign_chronological_blocks(self, predictions: pd.DataFrame) -> pd.DataFrame:
        required = {"target_date", "actual_up", "score", "predicted_up"}
        missing = sorted(required - set(predictions.columns))
        if missing:
            raise ValueError(
                "Prediction data is missing required columns: " + ", ".join(missing)
            )
        result = predictions.copy()
        result["target_date"] = pd.to_datetime(result["target_date"])
        result = result.sort_values("target_date").reset_index(drop=True)
        if result["target_date"].duplicated().any():
            raise ValueError("Prediction data contains duplicate target dates.")
        blocks = np.array_split(np.arange(len(result)), self.block_count)
        result["validation_block"] = 0
        for block_number, block_index in enumerate(blocks, start=1):
            if len(block_index) > 0:
                result.loc[block_index, "validation_block"] = block_number
        return result

    def block_model_diagnostics(self, enriched: pd.DataFrame) -> pd.DataFrame:
        required = {
            "validation_block",
            "target_date",
            "actual_up",
            "score",
            "predicted_up",
            "actual_future_log_return",
        }
        missing = sorted(required - set(enriched.columns))
        if missing:
            raise ValueError(
                "Enriched diagnostic data is missing columns: " + ", ".join(missing)
            )
        rows = []
        for block_number, block in enriched.groupby("validation_block", sort=True):
            actual = block["actual_up"].astype(int).to_numpy()
            score = block["score"].astype(float).to_numpy()
            predicted = block["predicted_up"].astype(int).to_numpy()
            returns = block["actual_future_log_return"].astype(float).to_numpy()
            correct = predicted == actual
            auc = self._safe_auc(actual, score)
            rows.append(
                {
                    "validation_block": int(block_number),
                    "start": pd.Timestamp(block["target_date"].min()),
                    "end": pd.Timestamp(block["target_date"].max()),
                    "rows": int(len(block)),
                    "up_share": float(actual.mean()),
                    "roc_auc": auc,
                    "predicted_up_share": float(predicted.mean()),
                    "sign_accuracy": float(correct.mean()),
                    "magnitude_weighted_sign_accuracy": self._weighted_accuracy(
                        correct,
                        np.abs(returns),
                    ),
                    "mean_score": float(np.mean(score)),
                    "median_score": float(np.median(score)),
                    "mean_probability_distance_from_0_5": float(
                        np.mean(np.abs(score - 0.5))
                    ),
                    "median_absolute_future_return": float(np.median(np.abs(returns))),
                }
            )
        return pd.DataFrame(rows)

    def block_feature_profiles(
        self,
        development_move: pd.DataFrame,
        enriched_validation: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_feature_columns(development_move)
        self.validate_feature_columns(enriched_validation)
        rows = []
        family_by_feature = {spec.feature: spec.family for spec in self.feature_specs}
        for feature in self.feature_columns:
            development_values = pd.to_numeric(
                development_move[feature], errors="coerce"
            ).dropna()
            if development_values.empty:
                continue
            development_mean = float(development_values.mean())
            development_std = float(development_values.std(ddof=0))
            development_median = float(development_values.median())
            development_q33 = float(development_values.quantile(1.0 / 3.0))
            development_q67 = float(development_values.quantile(2.0 / 3.0))
            for block_number, block in enriched_validation.groupby(
                "validation_block", sort=True
            ):
                values = pd.to_numeric(block[feature], errors="coerce").dropna()
                if values.empty:
                    continue
                mean = float(values.mean())
                median = float(values.median())
                rows.append(
                    {
                        "family": family_by_feature[feature],
                        "feature": feature,
                        "validation_block": int(block_number),
                        "rows": int(len(values)),
                        "mean": mean,
                        "median": median,
                        "std": float(values.std(ddof=0)),
                        "q25": float(values.quantile(0.25)),
                        "q75": float(values.quantile(0.75)),
                        "development_mean": development_mean,
                        "development_median": development_median,
                        "development_std": development_std,
                        "development_q33": development_q33,
                        "development_q67": development_q67,
                        "mean_z_vs_development": self._standardized_difference(
                            mean,
                            development_mean,
                            development_std,
                        ),
                        "median_z_vs_development": self._standardized_difference(
                            median,
                            development_median,
                            development_std,
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def block_feature_contrasts(self, profiles: pd.DataFrame) -> pd.DataFrame:
        required = {
            "family",
            "feature",
            "validation_block",
            "mean",
            "median",
            "development_std",
            "mean_z_vs_development",
        }
        missing = sorted(required - set(profiles.columns))
        if missing:
            raise ValueError(
                "Block profiles are missing required columns: " + ", ".join(missing)
            )
        rows = []
        for (family, feature), group in profiles.groupby(
            ["family", "feature"], sort=False
        ):
            by_block = group.set_index("validation_block")
            if 1 not in by_block.index or 2 not in by_block.index:
                continue
            block1 = by_block.loc[1]
            block2 = by_block.loc[2]
            block3 = by_block.loc[3] if 3 in by_block.index else None
            development_std = float(block1["development_std"])
            block2_minus_block1 = self._standardized_difference(
                float(block2["mean"]),
                float(block1["mean"]),
                development_std,
            )
            block3_minus_block1 = (
                self._standardized_difference(
                    float(block3["mean"]),
                    float(block1["mean"]),
                    development_std,
                )
                if block3 is not None
                else float("nan")
            )
            rows.append(
                {
                    "family": family,
                    "feature": feature,
                    "block1_mean_z_vs_development": float(
                        block1["mean_z_vs_development"]
                    ),
                    "block2_mean_z_vs_development": float(
                        block2["mean_z_vs_development"]
                    ),
                    "block3_mean_z_vs_development": (
                        float(block3["mean_z_vs_development"])
                        if block3 is not None
                        else float("nan")
                    ),
                    "block2_minus_block1_mean_in_dev_std": block2_minus_block1,
                    "abs_block2_minus_block1_mean_in_dev_std": abs(
                        block2_minus_block1
                    ),
                    "block3_minus_block1_mean_in_dev_std": block3_minus_block1,
                    "block1_mean": float(block1["mean"]),
                    "block2_mean": float(block2["mean"]),
                    "block3_mean": (
                        float(block3["mean"]) if block3 is not None else float("nan")
                    ),
                    "block1_median": float(block1["median"]),
                    "block2_median": float(block2["median"]),
                    "block3_median": (
                        float(block3["median"])
                        if block3 is not None
                        else float("nan")
                    ),
                }
            )
        result = pd.DataFrame(rows)
        if result.empty:
            return result
        return result.sort_values(
            "abs_block2_minus_block1_mean_in_dev_std",
            ascending=False,
        ).reset_index(drop=True)

    def development_tertiles(self, development_move: pd.DataFrame) -> pd.DataFrame:
        self.validate_feature_columns(development_move)
        rows = []
        for spec in self.feature_specs:
            values = pd.to_numeric(
                development_move[spec.feature], errors="coerce"
            ).dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "family": spec.family,
                    "feature": spec.feature,
                    "development_rows": int(len(values)),
                    "lower_threshold_q33": float(values.quantile(1.0 / 3.0)),
                    "upper_threshold_q67": float(values.quantile(2.0 / 3.0)),
                    "development_mean": float(values.mean()),
                    "development_median": float(values.median()),
                    "development_std": float(values.std(ddof=0)),
                }
            )
        return pd.DataFrame(rows)

    def validation_auc_by_development_tertile(
        self,
        enriched_validation: pd.DataFrame,
        tertiles: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_feature_columns(enriched_validation)
        required = {
            "actual_up",
            "score",
            "predicted_up",
            "actual_future_log_return",
        }
        missing = sorted(required - set(enriched_validation.columns))
        if missing:
            raise ValueError(
                "Enriched validation data is missing columns: " + ", ".join(missing)
            )
        rows = []
        for threshold_row in tertiles.itertuples(index=False):
            feature = threshold_row.feature
            values = pd.to_numeric(enriched_validation[feature], errors="coerce")
            regime = pd.Series(index=enriched_validation.index, dtype="object")
            valid = values.notna()
            regime.loc[valid & (values <= threshold_row.lower_threshold_q33)] = "LOW"
            regime.loc[
                valid
                & (values > threshold_row.lower_threshold_q33)
                & (values < threshold_row.upper_threshold_q67)
            ] = "MID"
            regime.loc[valid & (values >= threshold_row.upper_threshold_q67)] = "HIGH"
            for regime_name in ("LOW", "MID", "HIGH"):
                subset = enriched_validation.loc[regime == regime_name]
                if subset.empty:
                    continue
                actual = subset["actual_up"].astype(int).to_numpy()
                score = subset["score"].astype(float).to_numpy()
                predicted = subset["predicted_up"].astype(int).to_numpy()
                returns = subset["actual_future_log_return"].astype(float).to_numpy()
                correct = predicted == actual
                auc = (
                    self._safe_auc(actual, score)
                    if len(subset) >= self.minimum_auc_rows
                    else float("nan")
                )
                rows.append(
                    {
                        "family": threshold_row.family,
                        "feature": feature,
                        "regime": regime_name,
                        "threshold_source": "development_MOVE_tertiles_only",
                        "lower_threshold_q33": float(
                            threshold_row.lower_threshold_q33
                        ),
                        "upper_threshold_q67": float(
                            threshold_row.upper_threshold_q67
                        ),
                        "rows": int(len(subset)),
                        "up_share": float(actual.mean()),
                        "roc_auc": auc,
                        "sign_accuracy": float(correct.mean()),
                        "magnitude_weighted_sign_accuracy": self._weighted_accuracy(
                            correct,
                            np.abs(returns),
                        ),
                        "mean_score": float(np.mean(score)),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _standardized_difference(
        value: float,
        reference: float,
        scale: float,
    ) -> float:
        if not np.isfinite(scale) or scale <= 0.0:
            return float("nan")
        return float((value - reference) / scale)

    @staticmethod
    def _safe_auc(actual: np.ndarray, score: np.ndarray) -> float:
        actual = np.asarray(actual, dtype=np.int64)
        score = np.asarray(score, dtype=np.float64)
        if len(actual) == 0 or len(np.unique(actual)) < 2:
            return float("nan")
        return float(roc_auc_score(actual, score))

    @staticmethod
    def _weighted_accuracy(correct: np.ndarray, weights: np.ndarray) -> float:
        correct = np.asarray(correct, dtype=bool)
        weights = np.asarray(weights, dtype=np.float64)
        if len(correct) != len(weights):
            raise ValueError("Correctness and weights must align.")
        if len(correct) == 0:
            return float("nan")
        if float(weights.sum()) <= 0.0:
            return float(np.mean(correct))
        return float(np.average(correct, weights=weights))
