import numpy as np
import pandas as pd

from app.training.residual_sequence_preprocessor import (
    RESIDUAL_HISTORY_FEATURE_COLUMNS,
    ResidualSequencePreprocessor,
)


def build_residual_frame(
    observations: int = 120,
) -> pd.DataFrame:
    random = np.random.default_rng(42)
    residuals = random.normal(
        0.0,
        0.01,
        observations,
    )

    return pd.DataFrame(
        {
            "trade_date": pd.date_range(
                "2020-01-01",
                periods=observations,
                freq="B",
            ),
            "feature_one": random.normal(
                size=observations
            ),
            "feature_two": random.normal(
                size=observations
            ),
            "sarimax_prediction": random.normal(
                0.0,
                0.001,
                observations,
            ),
            "sarimax_residual": residuals,
        }
    )


def test_sequence_preprocessor_builds_chronological_sequences():
    dataframe = build_residual_frame()
    train = dataframe.iloc[:90].reset_index(drop=True)
    validation = dataframe.iloc[90:].reset_index(drop=True)
    feature_columns = [
        "feature_one",
        "feature_two",
        "sarimax_prediction",
    ]

    preprocessor = ResidualSequencePreprocessor(
        sequence_length=20
    ).fit(
        dataframe=train,
        feature_columns=feature_columns,
    )

    training_sequences = (
        preprocessor.build_training_sequences(
            train
        )
    )
    validation_sequences = (
        preprocessor.build_inference_sequences(
            history=train,
            dataframe=validation,
        )
    )

    assert training_sequences.sequences.shape == (
        70,
        20,
        6,
    )
    assert validation_sequences.sequences.shape == (
        30,
        20,
        6,
    )
    assert list(preprocessor.feature_columns) == [
        *feature_columns,
        *RESIDUAL_HISTORY_FEATURE_COLUMNS,
    ]
    assert np.all(
        np.diff(
            validation_sequences.trade_dates
        ) > np.timedelta64(0, "ns")
    )


def test_current_residual_is_not_present_in_current_sequence_features():
    dataframe = build_residual_frame()
    train = dataframe.iloc[:90].reset_index(drop=True)
    validation = dataframe.iloc[90:].reset_index(drop=True)
    changed_validation = validation.copy()
    changed_validation.loc[
        0,
        "sarimax_residual",
    ] = 10.0
    feature_columns = [
        "feature_one",
        "feature_two",
        "sarimax_prediction",
    ]

    preprocessor = ResidualSequencePreprocessor(
        sequence_length=20
    ).fit(
        dataframe=train,
        feature_columns=feature_columns,
    )

    original = preprocessor.build_inference_sequences(
        history=train,
        dataframe=validation,
    )
    changed = preprocessor.build_inference_sequences(
        history=train,
        dataframe=changed_validation,
    )

    np.testing.assert_allclose(
        original.sequences[0],
        changed.sequences[0],
    )
    assert original.targets[0] != changed.targets[0]
    assert "sarimax_residual" not in (
        preprocessor.feature_columns
    )


def test_preprocessor_state_round_trip_is_exact():
    dataframe = build_residual_frame()
    train = dataframe.iloc[:90].reset_index(drop=True)
    validation = dataframe.iloc[90:].reset_index(drop=True)
    feature_columns = [
        "feature_one",
        "feature_two",
        "sarimax_prediction",
    ]

    preprocessor = ResidualSequencePreprocessor(
        sequence_length=20
    ).fit(
        dataframe=train,
        feature_columns=feature_columns,
    )
    restored = ResidualSequencePreprocessor.from_state(
        preprocessor.get_state()
    )

    original = preprocessor.build_inference_sequences(
        history=train,
        dataframe=validation,
    )
    reloaded = restored.build_inference_sequences(
        history=train,
        dataframe=validation,
    )

    np.testing.assert_allclose(
        original.sequences,
        reloaded.sequences,
    )
    np.testing.assert_allclose(
        original.targets,
        reloaded.targets,
    )
