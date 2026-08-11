import numpy as np
import pytest

from app.training.hierarchical_stage_feature_research import (
    binary_probability_metrics,
    build_refined_target_candidates,
    build_stage_feature_profiles,
    classify_trend_features,
)


def test_refined_target_candidates_cover_both_windows_and_optimum_region():
    candidates = build_refined_target_candidates()

    observed = {
        (
            candidate.volatility_window,
            candidate.threshold_multiplier,
        )
        for candidate in candidates
    }

    assert len(candidates) == 8
    assert len(observed) == 8
    assert (20, 0.35) in observed
    assert (20, 0.45) in observed
    assert (20, 0.50) in observed
    assert (40, 0.30) in observed
    assert (40, 0.45) in observed


def test_classify_trend_features_separates_strength_and_directional_concepts():
    base = [
        "log_return",
        "vix_level",
    ]
    base_trend = [
        *base,
        "adx_14",
        "ma_compression",
        "sma_10_20_spread",
        "sma_20_slope_5",
        "ma_alignment",
        "dmi_direction",
    ]

    groups = classify_trend_features(
        base_columns=base,
        base_trend_columns=base_trend,
    )

    assert groups["strength"] == [
        "adx_14",
        "ma_compression",
    ]
    assert groups["directional"] == [
        "sma_10_20_spread",
        "sma_20_slope_5",
        "ma_alignment",
        "dmi_direction",
    ]
    assert groups["unassigned"] == []


def test_stage_specific_profile_keeps_base_features_in_both_stages():
    base = [
        "log_return",
        "vix_level",
    ]
    base_trend = [
        *base,
        "adx_14",
        "ma_compression",
        "sma_10_20_spread",
        "ma_alignment",
    ]

    profiles = build_stage_feature_profiles(
        base_columns=base,
        base_trend_columns=base_trend,
    )

    by_name = {
        profile.name: profile
        for profile in profiles
    }

    assert set(by_name) == {
        "base_only",
        "stage_specific_trend",
    }

    assert by_name[
        "base_only"
    ].stage1_feature_columns == tuple(base)

    assert by_name[
        "base_only"
    ].stage2_feature_columns == tuple(base)

    stage_specific = by_name[
        "stage_specific_trend"
    ]

    assert stage_specific.stage1_feature_columns == (
        "log_return",
        "vix_level",
        "adx_14",
        "ma_compression",
    )

    assert stage_specific.stage2_feature_columns == (
        "log_return",
        "vix_level",
        "sma_10_20_spread",
        "ma_alignment",
    )


def test_binary_probability_metrics_detect_perfect_ranking():
    result = binary_probability_metrics(
        actual=np.asarray(
            [0, 0, 1, 1],
            dtype=np.int64,
        ),
        positive_probabilities=np.asarray(
            [0.05, 0.20, 0.80, 0.95],
            dtype=np.float64,
        ),
    )

    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["average_precision"] == pytest.approx(1.0)
    assert 0.0 <= result["brier_score"] < 0.1


def test_binary_probability_metrics_reject_invalid_probabilities():
    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        binary_probability_metrics(
            actual=np.asarray([0, 1]),
            positive_probabilities=np.asarray(
                [-0.1, 1.1]
            ),
        )
