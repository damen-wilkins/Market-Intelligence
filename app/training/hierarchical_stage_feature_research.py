from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class RefinedTargetCandidate:
    name: str
    volatility_window: int
    threshold_multiplier: float
    description: str


@dataclass(frozen=True)
class StageFeatureProfile:
    name: str
    stage1_feature_columns: tuple[str, ...]
    stage2_feature_columns: tuple[str, ...]
    description: str


def build_refined_target_candidates() -> list[RefinedTargetCandidate]:
    return [
        RefinedTargetCandidate(
            name="flat_20d_k035",
            volatility_window=20,
            threshold_multiplier=0.35,
            description="Refined 20-day candidate below the prior 0.45 optimum.",
        ),
        RefinedTargetCandidate(
            name="flat_20d_k040",
            volatility_window=20,
            threshold_multiplier=0.40,
            description="Refined 20-day candidate near the prior optimum region.",
        ),
        RefinedTargetCandidate(
            name="flat_20d_k045",
            volatility_window=20,
            threshold_multiplier=0.45,
            description="Prior strongest end-to-end 20-day candidate.",
        ),
        RefinedTargetCandidate(
            name="flat_20d_k050",
            volatility_window=20,
            threshold_multiplier=0.50,
            description="Research-reference 0.50 multiplier and upper refined bound.",
        ),
        RefinedTargetCandidate(
            name="flat_40d_k030",
            volatility_window=40,
            threshold_multiplier=0.30,
            description="Prior strongest 40-day Stage-1 region anchor.",
        ),
        RefinedTargetCandidate(
            name="flat_40d_k035",
            volatility_window=40,
            threshold_multiplier=0.35,
            description="Refined 40-day candidate above the prior 0.30 anchor.",
        ),
        RefinedTargetCandidate(
            name="flat_40d_k040",
            volatility_window=40,
            threshold_multiplier=0.40,
            description="Refined 40-day candidate in the wider neutral region.",
        ),
        RefinedTargetCandidate(
            name="flat_40d_k045",
            volatility_window=40,
            threshold_multiplier=0.45,
            description="Upper 40-day candidate for neutral-zone sensitivity.",
        ),
    ]


def classify_trend_features(
    base_columns: list[str] | tuple[str, ...],
    base_trend_columns: list[str] | tuple[str, ...],
) -> dict[str, list[str]]:
    base = list(base_columns)
    base_trend = list(base_trend_columns)

    if not base:
        raise ValueError("Base feature columns cannot be empty.")

    if len(base) != len(set(base)):
        raise ValueError("Base feature columns contain duplicates.")

    if len(base_trend) != len(set(base_trend)):
        raise ValueError("Base + trend feature columns contain duplicates.")

    missing_base = [
        column
        for column in base
        if column not in base_trend
    ]

    if missing_base:
        raise ValueError(
            "Base + trend contract is missing base features: "
            f"{missing_base}"
        )

    trend_columns = [
        column
        for column in base_trend
        if column not in base
    ]

    if not trend_columns:
        raise ValueError(
            "Base + trend feature contract does not contain trend features."
        )

    strength_columns = []
    directional_columns = []

    for column in trend_columns:
        normalized = column.lower()

        is_strength = any(
            token in normalized
            for token in (
                "adx",
                "compression",
                "separation",
                "absolute",
                "abs_",
            )
        )

        is_directional = any(
            token in normalized
            for token in (
                "alignment",
                "slope",
                "dmi",
                "plus_di",
                "minus_di",
                "di_diff",
                "direction",
                "spread",
            )
        )

        if is_strength:
            strength_columns.append(column)

        if is_directional and not is_strength:
            directional_columns.append(column)

    if not strength_columns:
        raise ValueError(
            "Could not identify any trend-strength features. "
            f"Observed trend features: {trend_columns}"
        )

    if not directional_columns:
        raise ValueError(
            "Could not identify any directional-trend features. "
            f"Observed trend features: {trend_columns}"
        )

    unassigned_columns = [
        column
        for column in trend_columns
        if column not in strength_columns
        and column not in directional_columns
    ]

    return {
        "trend": trend_columns,
        "strength": strength_columns,
        "directional": directional_columns,
        "unassigned": unassigned_columns,
    }


def build_stage_feature_profiles(
    base_columns: list[str] | tuple[str, ...],
    base_trend_columns: list[str] | tuple[str, ...],
) -> list[StageFeatureProfile]:
    base = list(base_columns)
    groups = classify_trend_features(
        base_columns=base_columns,
        base_trend_columns=base_trend_columns,
    )

    stage1_specific = [
        *base,
        *groups["strength"],
    ]

    stage2_specific = [
        *base,
        *groups["directional"],
    ]

    return [
        StageFeatureProfile(
            name="base_only",
            stage1_feature_columns=tuple(base),
            stage2_feature_columns=tuple(base),
            description=(
                "Controlled benchmark: both stages use only the original "
                "base feature contract."
            ),
        ),
        StageFeatureProfile(
            name="stage_specific_trend",
            stage1_feature_columns=tuple(stage1_specific),
            stage2_feature_columns=tuple(stage2_specific),
            description=(
                "Stage 1 receives trend-strength/state features; Stage 2 "
                "receives directional trend features."
            ),
        ),
    ]


def binary_probability_metrics(
    actual: np.ndarray,
    positive_probabilities: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.int64)
    probabilities = np.asarray(
        positive_probabilities,
        dtype=np.float64,
    )

    if actual.ndim != 1 or probabilities.ndim != 1:
        raise ValueError(
            "Binary probability metrics require one-dimensional arrays."
        )

    if len(actual) != len(probabilities):
        raise ValueError(
            "Actual labels and probabilities must contain the same rows."
        )

    if len(actual) == 0:
        raise ValueError(
            "Binary probability metrics cannot be calculated on zero rows."
        )

    if not np.isin(actual, [0, 1]).all():
        raise ValueError("Binary labels must contain only zero and one.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values.")

    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Probabilities must be between zero and one.")

    if len(np.unique(actual)) < 2:
        raise ValueError(
            "ROC AUC requires both binary classes to be present."
        )

    return {
        "roc_auc": float(
            roc_auc_score(
                actual,
                probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                actual,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                actual,
                probabilities,
            )
        ),
    }
