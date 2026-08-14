import numpy as np
import pandas as pd

from app.training.stage1_target_optimization import (
    add_neighborhood_statistics,
    align_candidate_datasets,
    build_target_grid,
    moving_block_bootstrap_auc,
    select_target_shortlist,
    target_stability_statistics,
)


def test_target_grid_is_dense_and_unique():
    candidates = build_target_grid()

    assert len(candidates) == 232
    assert len(
        {
            candidate.name
            for candidate in candidates
        }
    ) == 232
    assert {
        candidate.volatility_window
        for candidate in candidates
    } == {
        5,
        10,
        15,
        20,
        30,
        40,
        60,
        90,
    }

    multipliers = sorted(
        {
            candidate.threshold_multiplier
            for candidate in candidates
        }
    )
    assert np.isclose(
        multipliers[0],
        0.10,
    )
    assert np.isclose(
        multipliers[-1],
        0.80,
    )
    assert len(multipliers) == 29


def test_candidate_alignment_uses_exact_common_target_dates():
    target_dates = pd.date_range(
        "2020-01-02",
        periods=8,
        freq="B",
    )
    feature_dates = target_dates - pd.Timedelta(
        days=1
    )

    first = pd.DataFrame(
        {
            "feature_date": feature_dates,
            "target_date": target_dates,
            "direction": [
                "FLAT",
                "UP",
                "DOWN",
                "UP",
                "FLAT",
                "DOWN",
                "UP",
                "FLAT",
            ],
            "threshold": np.linspace(
                0.001,
                0.002,
                8,
            ),
        }
    )
    second = first.iloc[
        2:
    ].copy()

    aligned = align_candidate_datasets(
        {
            "first": first,
            "second": second,
        },
        training_end_date=target_dates[-2],
    )

    expected_dates = pd.DatetimeIndex(
        target_dates[2:-1]
    )

    assert pd.DatetimeIndex(
        aligned["first"]["target_date"]
    ).equals(
        expected_dates
    )
    assert pd.DatetimeIndex(
        aligned["second"]["target_date"]
    ).equals(
        expected_dates
    )


def test_target_stability_reports_prevalence_and_block_range():
    dataframe = pd.DataFrame(
        {
            "direction": [
                "FLAT",
                "UP",
                "FLAT",
                "DOWN",
                "FLAT",
                "UP",
                "DOWN",
                "FLAT",
                "UP",
                "DOWN",
            ],
            "threshold": np.full(
                10,
                0.003,
            ),
        }
    )

    result = target_stability_statistics(
        dataframe,
        chronological_blocks=5,
    )

    assert np.isclose(
        result["flat_share"],
        0.4,
    )
    assert np.isclose(
        result["down_share"],
        0.3,
    )
    assert np.isclose(
        result["up_share"],
        0.3,
    )
    assert np.isclose(
        result[
            "median_flat_boundary_percent"
        ],
        0.3,
    )
    assert 0.0 <= result[
        "flat_share_block_range"
    ] <= 1.0
    assert len(
        result["flat_share_by_block"]
    ) == 5


def test_shortlist_rewards_robust_signal_and_window_diversity():
    rows = []

    for window in (
        10,
        20,
        40,
    ):
        for offset, multiplier in enumerate(
            (
                0.30,
                0.325,
                0.35,
            )
        ):
            rows.append(
                {
                    "target_name": (
                        f"w{window}_{offset}"
                    ),
                    "volatility_window": window,
                    "threshold_multiplier": multiplier,
                    "flat_share": 0.35,
                    "flat_share_block_range": 0.03,
                    "stage1_roc_auc": (
                        0.62
                        - offset * 0.005
                        - window / 10000.0
                    ),
                    "stage1_roc_auc_fold_std": (
                        0.01
                        + offset * 0.002
                    ),
                    "stage1_balanced_accuracy": (
                        0.58
                        - offset * 0.003
                    ),
                    "stage1_flat_f1": (
                        0.52
                        - offset * 0.002
                    ),
                }
            )

    summary = pd.DataFrame(rows)
    summary = add_neighborhood_statistics(
        summary
    )
    shortlist = select_target_shortlist(
        summary,
        shortlist_size=6,
        max_per_window=2,
    )

    assert len(shortlist) == 6
    counts = shortlist[
        "volatility_window"
    ].value_counts()
    assert counts.max() <= 2
    assert set(
        shortlist[
            "volatility_window"
        ]
    ) == {
        10,
        20,
        40,
    }


def test_moving_block_bootstrap_auc_contains_point_estimate():
    random = np.random.default_rng(17)
    actual = np.asarray(
        [
            0,
            1,
        ]
        * 100,
        dtype=np.int64,
    )
    probabilities = np.clip(
        0.25
        + 0.5 * actual
        + random.normal(
            0.0,
            0.12,
            len(actual),
        ),
        0.0,
        1.0,
    )

    result = moving_block_bootstrap_auc(
        actual=actual,
        probabilities=probabilities,
        block_length=10,
        n_resamples=100,
        random_state=3,
    )

    assert 0.5 < result[
        "point_estimate"
    ] <= 1.0
    assert result[
        "lower_95"
    ] <= result[
        "point_estimate"
    ] <= result[
        "upper_95"
    ]
    assert result[
        "valid_resamples"
    ] == 100
