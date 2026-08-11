import numpy as np
import pandas as pd
import pytest

from app.training.hierarchical_target_feature_research import (
    align_datasets_on_common_target_dates,
    binary_metrics,
    build_target_candidates,
    compose_hierarchical_predictions,
    target_distribution,
    three_class_metrics,
)


def test_target_candidates_are_unique_and_include_current_target():
    candidates = build_target_candidates()

    names = [
        candidate.name
        for candidate in candidates
    ]

    assert len(candidates) == 6
    assert len(names) == len(set(names))
    assert any(
        candidate.volatility_window == 20
        and candidate.threshold_multiplier == 0.15
        for candidate in candidates
    )
    assert any(
        candidate.volatility_window == 20
        and candidate.threshold_multiplier == 0.50
        for candidate in candidates
    )


def test_align_datasets_uses_only_common_target_dates():
    dates = pd.date_range(
        "2024-01-01",
        periods=5,
        freq="D",
    )

    first = pd.DataFrame(
        {
            "target_date": dates,
            "direction": [
                "DOWN",
                "FLAT",
                "UP",
                "UP",
                "DOWN",
            ],
        }
    )

    second = pd.DataFrame(
        {
            "target_date": dates[1:],
            "direction": [
                "FLAT",
                "UP",
                "UP",
                "DOWN",
            ],
        }
    )

    aligned = align_datasets_on_common_target_dates(
        {
            "first": first,
            "second": second,
        }
    )

    assert len(aligned["first"]) == 4
    assert len(aligned["second"]) == 4
    assert pd.DatetimeIndex(
        aligned["first"]["target_date"]
    ).equals(
        pd.DatetimeIndex(
            aligned["second"]["target_date"]
        )
    )


def test_compose_hierarchical_predictions_respects_both_thresholds():
    predicted = compose_hierarchical_predictions(
        move_probabilities=np.asarray(
            [0.2, 0.7, 0.8]
        ),
        up_probabilities=np.asarray(
            [0.9, 0.3, 0.8]
        ),
        move_threshold=0.5,
        up_threshold=0.5,
    )

    assert predicted.tolist() == [
        "FLAT",
        "DOWN",
        "UP",
    ]


def test_target_distribution_returns_expected_shares():
    dataframe = pd.DataFrame(
        {
            "direction": [
                "DOWN",
                "FLAT",
                "UP",
                "UP",
            ]
        }
    )

    result = target_distribution(
        dataframe
    )

    assert result["rows"] == 4
    assert result["down_share"] == pytest.approx(
        0.25
    )
    assert result["flat_share"] == pytest.approx(
        0.25
    )
    assert result["up_share"] == pytest.approx(
        0.50
    )


def test_metric_helpers_return_expected_perfect_scores():
    binary = binary_metrics(
        actual=np.asarray([0, 1, 0, 1]),
        predicted=np.asarray([0, 1, 0, 1]),
        negative_name="FLAT",
        positive_name="MOVE",
    )

    multiclass = three_class_metrics(
        actual_directions=np.asarray(
            ["DOWN", "FLAT", "UP"]
        ),
        predicted_directions=np.asarray(
            ["DOWN", "FLAT", "UP"]
        ),
    )

    assert binary["macro_f1"] == pytest.approx(
        1.0
    )
    assert multiclass["macro_f1"] == pytest.approx(
        1.0
    )
