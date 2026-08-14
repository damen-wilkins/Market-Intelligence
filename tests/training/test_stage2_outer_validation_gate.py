import numpy as np
import pandas as pd
import pytest

from app.training.stage2_outer_validation_gate import (
    ValidationPeriods,
    chronological_auc_blocks,
    classification_metrics,
    moving_block_bootstrap_auc_ci,
    parameter_signature,
    split_development_and_outer_validation,
)


def test_split_development_and_outer_validation_preserves_locked_boundaries():
    data = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-09-15", "2020-09-25", freq="D"),
            "value": range(11),
        }
    )
    periods = ValidationPeriods(
        training_end=pd.Timestamp("2020-09-18"),
        validation_start=pd.Timestamp("2020-09-21"),
        validation_end=pd.Timestamp("2020-09-24"),
    )
    development, validation = split_development_and_outer_validation(data, periods)
    assert development["target_date"].max() == pd.Timestamp("2020-09-18")
    assert validation["target_date"].min() == pd.Timestamp("2020-09-21")
    assert validation["target_date"].max() == pd.Timestamp("2020-09-24")
    assert pd.Timestamp("2020-09-25") not in set(validation["target_date"])


def test_classification_metrics_uses_direct_up_score_orientation():
    actual = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = classification_metrics(actual, score, threshold=0.5)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["inverted_roc_auc"] == pytest.approx(0.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)


def test_bootstrap_auc_ci_reports_probability_above_chance():
    actual = np.tile(np.array([0, 1]), 80)
    score = actual.astype(float) + np.linspace(0.0, 0.01, len(actual))
    result = moving_block_bootstrap_auc_ci(
        actual,
        score,
        resamples=100,
        block_length=10,
        random_state=7,
    )
    assert result["auc"] > 0.99
    assert result["lower_95"] > 0.90
    assert result["probability_auc_above_0_50"] == pytest.approx(1.0)


def test_chronological_auc_blocks_keep_time_order():
    actual = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    score = np.array([0.1, 0.9] * 6)
    dates = pd.date_range("2022-01-01", periods=12, freq="D")
    rows = chronological_auc_blocks(actual, score, dates, block_count=3)
    assert len(rows) == 3
    assert rows[0]["start"] == pd.Timestamp("2022-01-01")
    assert rows[-1]["end"] == pd.Timestamp("2022-01-12")
    assert all(row["roc_auc"] == pytest.approx(1.0) for row in rows)


def test_parameter_signature_is_order_independent():
    left = parameter_signature({"max_depth": 4, "learning_rate": 0.05})
    right = parameter_signature({"learning_rate": 0.05, "max_depth": 4})
    assert left == right
