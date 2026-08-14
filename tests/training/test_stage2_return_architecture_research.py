import numpy as np
import pandas as pd
import pytest

from app.training.stage2_return_architecture_research import (
    Stage2ReturnSequencePreprocessor,
    auc_orientation,
    normal_up_probability,
    regression_direction_metrics,
    verify_stage2_orientation,
)


def sample_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    directions = [
        "FLAT",
        "UP",
        "DOWN",
        "UP",
        "FLAT",
        "DOWN",
        "UP",
        "DOWN",
        "FLAT",
        "UP",
        "DOWN",
        "UP",
    ]
    returns = [0.001, 0.020, -0.018, 0.015, -0.002, -0.016, 0.017, -0.014, 0.003, 0.019, -0.017, 0.016]
    threshold = [0.005] * len(dates)
    return pd.DataFrame(
        {
            "feature_date": dates,
            "target_date": dates + pd.Timedelta(days=1),
            "feature_a": np.arange(len(dates), dtype=float),
            "feature_b": np.linspace(1.0, 2.0, len(dates)),
            "future_log_return": returns,
            "future_return_vol_units": np.asarray(returns) / 0.01,
            "rolling_volatility": [0.01] * len(dates),
            "threshold": threshold,
            "direction": directions,
        }
    )


def test_verify_stage2_orientation_accepts_valid_labels():
    result = verify_stage2_orientation(sample_frame())
    assert result["violations"] == {"UP": 0, "DOWN": 0, "FLAT": 0}


def test_verify_stage2_orientation_rejects_inverted_up_label():
    frame = sample_frame()
    frame.loc[1, "future_log_return"] = -0.02
    with pytest.raises(ValueError, match="orientation violations"):
        verify_stage2_orientation(frame)


def test_auc_orientation_reports_inverse_score():
    actual = np.asarray([0, 0, 1, 1])
    score = np.asarray([0.1, 0.2, 0.8, 0.9])
    result = auc_orientation(actual, score)
    assert result["direct_auc"] == pytest.approx(1.0)
    assert result["inverted_auc"] == pytest.approx(0.0)


def test_normal_up_probability_is_monotonic_in_mean():
    probabilities = normal_up_probability(
        mean=np.asarray([-1.0, 0.0, 1.0]),
        scale=np.asarray([1.0, 1.0, 1.0]),
    )
    assert probabilities[0] < 0.5
    assert probabilities[1] == pytest.approx(0.5)
    assert probabilities[2] > 0.5


def test_return_preprocessor_keeps_flat_rows_as_sequence_context():
    frame = sample_frame()
    train = frame.iloc[:8].reset_index(drop=True)
    validation = frame.iloc[8:].reset_index(drop=True)
    preprocessor = Stage2ReturnSequencePreprocessor(
        feature_columns=["feature_a", "feature_b"],
        sequence_length=3,
    ).fit(train, "future_return_vol_units")
    training = preprocessor.build_training_sequences(
        train,
        "future_return_vol_units",
    )
    inference = preprocessor.build_inference_sequences(
        train,
        validation,
        "future_return_vol_units",
    )
    assert training["X"].shape[1:] == (3, 2)
    assert len(training["y"]) == 5
    assert len(inference["y"]) == 3
    assert list(inference["target_dates"]) == list(validation.loc[validation["direction"] != "FLAT", "target_date"])


def test_regression_direction_metrics_reward_correct_sign_and_ranking():
    actual = np.asarray([-0.02, -0.01, 0.01, 0.03])
    predicted = np.asarray([-0.015, -0.005, 0.007, 0.020])
    metrics = regression_direction_metrics(actual, predicted)
    assert metrics["direct_auc"] == pytest.approx(1.0)
    assert metrics["sign_accuracy"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
