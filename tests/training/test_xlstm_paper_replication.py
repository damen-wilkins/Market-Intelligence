import numpy as np
import pandas as pd

from app.training.paper_wavelet_preprocessor import (
    PaperWaveletPreprocessor,
)
from app.training.xlstm_price_forecast_evaluator import (
    XLSTMPriceForecastEvaluator,
)


def build_test_data() -> pd.DataFrame:
    dates = pd.bdate_range(
        "2020-01-01",
        periods=80,
    )

    close = (
        100.0
        + np.linspace(
            0.0,
            20.0,
            len(dates),
        )
        + np.sin(
            np.arange(
                len(dates)
            )
            / 3.0
        )
    )

    return pd.DataFrame(
        {
            "trade_date": dates,
            "close": close,
        }
    )


def build_preprocessor(
    mode: str,
) -> PaperWaveletPreprocessor:
    dates = build_test_data()[
        "trade_date"
    ]

    return PaperWaveletPreprocessor(
        mode=mode,
        sequence_length=10,
        train_end_date=(
            dates.iloc[40]
            .strftime(
                "%Y-%m-%d"
            )
        ),
        validation_end_date=(
            dates.iloc[60]
            .strftime(
                "%Y-%m-%d"
            )
        ),
        wavelet="db4",
        level=1,
        pad_width=20,
    )


def test_wavelet_output_preserves_length_and_is_finite():
    data = build_test_data()

    preprocessor = build_preprocessor(
        PaperWaveletPreprocessor.PAPER_NONCAUSAL
    )

    denoised = preprocessor.denoise_full_series(
        data[
            "close"
        ].to_numpy()
    )

    assert len(
        denoised
    ) == len(
        data
    )

    assert np.isfinite(
        denoised
    ).all()


def test_causal_training_data_is_invariant_to_future_test_changes():
    original = build_test_data()
    modified = original.copy()

    modified.loc[
        60:,
        "close",
    ] = (
        modified.loc[
            60:,
            "close",
        ]
        * 10.0
    )

    original_preprocessor = build_preprocessor(
        PaperWaveletPreprocessor.CAUSAL
    )
    modified_preprocessor = build_preprocessor(
        PaperWaveletPreprocessor.CAUSAL
    )

    original_splits = original_preprocessor.prepare(
        original
    )
    modified_splits = modified_preprocessor.prepare(
        modified
    )

    np.testing.assert_allclose(
        original_splits[
            "train"
        ].X,
        modified_splits[
            "train"
        ].X,
        rtol=0.0,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        original_splits[
            "train"
        ].y,
        modified_splits[
            "train"
        ].y,
        rtol=0.0,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        original_preprocessor.scaler.data_min_,
        modified_preprocessor.scaler.data_min_,
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        original_preprocessor.scaler.data_max_,
        modified_preprocessor.scaler.data_max_,
        rtol=0.0,
        atol=1e-12,
    )


def test_paper_global_scaler_changes_when_future_test_prices_change():
    original = build_test_data()
    modified = original.copy()

    modified.loc[
        60:,
        "close",
    ] = (
        modified.loc[
            60:,
            "close",
        ]
        * 10.0
    )

    original_preprocessor = build_preprocessor(
        PaperWaveletPreprocessor.PAPER_NONCAUSAL
    )
    modified_preprocessor = build_preprocessor(
        PaperWaveletPreprocessor.PAPER_NONCAUSAL
    )

    original_preprocessor.prepare(
        original
    )
    modified_preprocessor.prepare(
        modified
    )

    assert not np.allclose(
        original_preprocessor.scaler.data_max_,
        modified_preprocessor.scaler.data_max_,
    )


def test_sequence_metadata_aligns_target_with_prior_closes():
    data = build_test_data()

    preprocessor = build_preprocessor(
        PaperWaveletPreprocessor.PAPER_NONCAUSAL
    )

    splits = preprocessor.prepare(
        data
    )

    first = splits[
        "train"
    ]

    first_target_date = pd.Timestamp(
        first.target_dates[0]
    )

    target_index = int(
        data.index[
            data[
                "trade_date"
            ]
            == first_target_date
        ][0]
    )

    assert first.prior_close[0] == data.loc[
        target_index - 2,
        "close",
    ]

    assert first.current_close[0] == data.loc[
        target_index - 1,
        "close",
    ]

    assert first.actual_close[0] == data.loc[
        target_index,
        "close",
    ]


def test_paper_direction_metric_differs_from_production_direction_metric():
    evaluator = XLSTMPriceForecastEvaluator()

    actual = np.array(
        [
            101.0,
            102.0,
            103.0,
        ]
    )

    predicted = np.array(
        [
            99.0,
            100.0,
            101.0,
        ]
    )

    current = np.array(
        [
            100.0,
            101.0,
            102.0,
        ]
    )

    prior = np.array(
        [
            99.0,
            100.0,
            101.0,
        ]
    )

    metrics = evaluator.evaluate(
        actual_close=actual,
        predicted_close=predicted,
        current_close=current,
        prior_close=prior,
    )

    assert metrics[
        "paper_direction"
    ][
        "accuracy"
    ] == 1.0

    assert metrics[
        "production_direction"
    ][
        "accuracy"
    ] == 0.0
