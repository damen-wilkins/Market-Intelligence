import numpy as np
import pandas as pd

from app.training.direction_feature_builder import DirectionFeatureBuilder
from app.training.hierarchical_direction_evaluator import (
    HierarchicalDirectionEvaluator,
)
from app.training.hierarchical_sequence_preprocessor import (
    HierarchicalSequencePreprocessor,
)
from app.training.hierarchical_xlstm_parameter_selector import (
    HierarchicalXLSTMParameterSelector,
)


def build_sequence_frame() -> pd.DataFrame:
    directions = [
        "UP",
        "FLAT",
        "DOWN",
        "UP",
        "FLAT",
        "DOWN",
        "UP",
        "DOWN",
    ]

    return pd.DataFrame(
        {
            "feature_date": pd.date_range(
                "2026-01-01",
                periods=len(directions),
                freq="B",
            ),
            "target_date": pd.date_range(
                "2026-01-02",
                periods=len(directions),
                freq="B",
            ),
            "direction": directions,
            "feature_one": np.arange(
                len(directions),
                dtype=float,
            ),
            "feature_two": np.arange(
                len(directions),
                dtype=float,
            ) * 2.0,
        }
    )


def build_base_feature_frame(
    rows: int = 80,
) -> pd.DataFrame:
    index = np.arange(
        rows,
        dtype=float,
    )

    close = 100.0 + index * 0.2
    open_price = close * 0.999
    high = close * 1.01
    low = close * 0.99
    volume = 1_000_000.0 + index * 1000.0

    log_return = np.zeros(
        rows,
        dtype=float,
    )
    log_return[1:] = np.log(
        close[1:]
        / close[:-1]
    )

    return pd.DataFrame(
        {
            "trade_date": pd.date_range(
                "2025-01-01",
                periods=rows,
                freq="B",
            ),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "sma_10": close * 0.998,
            "sma_20": close * 0.997,
            "sma_50": close * 0.995,
            "ema_20": close * 0.996,
            "rsi_14": np.full(
                rows,
                55.0,
            ),
            "macd": np.full(
                rows,
                0.8,
            ),
            "macd_signal": np.full(
                rows,
                0.6,
            ),
            "macd_histogram": np.full(
                rows,
                0.2,
            ),
            "bollinger_upper": close * 1.02,
            "bollinger_middle": close,
            "bollinger_lower": close * 0.98,
            "log_return": log_return,
            "vix_close": 18.0 + np.sin(
                index / 10.0
            ),
            "vvix_close": 90.0 + np.cos(
                index / 10.0
            ),
        }
    )


def test_base_feature_scope_does_not_require_breadth_or_cross_asset_data():
    builder = DirectionFeatureBuilder(
        feature_scope="base"
    )

    result = builder.build(
        build_base_feature_frame()
    )

    assert builder.feature_columns == (
        DirectionFeatureBuilder
        .BASE_FEATURE_COLUMNS
    )

    assert result.columns.tolist() == [
        "trade_date",
        *DirectionFeatureBuilder.BASE_FEATURE_COLUMNS,
    ]

    assert not result.empty


def test_stage2_training_uses_only_move_targets_without_removing_context_days():
    dataframe = build_sequence_frame()

    preprocessor = (
        HierarchicalSequencePreprocessor(
            feature_columns=[
                "feature_one",
                "feature_two",
            ],
            sequence_length=3,
        )
        .fit(
            dataframe
        )
    )

    result = preprocessor.build_training_sequences(
        dataframe=dataframe,
        task="direction",
    )

    expected_target_dates = dataframe.loc[
        [
            2,
            3,
            5,
            6,
            7,
        ],
        "target_date",
    ].reset_index(
        drop=True
    )

    assert pd.DatetimeIndex(
        result[
            "target_dates"
        ]
    ).equals(
        pd.DatetimeIndex(
            expected_target_dates
        )
    )

    assert result[
        "X"
    ].shape == (
        5,
        3,
        2,
    )

    assert result[
        "y"
    ].tolist() == [
        0,
        1,
        0,
        1,
        0,
    ]


def test_stage2_inference_can_score_flat_rows_for_real_routing():
    dataframe = build_sequence_frame()
    train = dataframe.iloc[:5].reset_index(
        drop=True
    )
    validation = dataframe.iloc[5:].reset_index(
        drop=True
    )

    preprocessor = (
        HierarchicalSequencePreprocessor(
            feature_columns=[
                "feature_one",
                "feature_two",
            ],
            sequence_length=3,
        )
        .fit(
            train
        )
    )

    result = preprocessor.build_inference_sequences(
        history=train,
        dataframe=validation,
        task="direction",
        include_all=True,
    )

    assert len(
        result[
            "X"
        ]
    ) == len(
        validation
    )

    assert result[
        "directions"
    ].tolist() == validation[
        "direction"
    ].tolist()


def test_hierarchical_evaluator_reports_stage_and_end_to_end_metrics():
    prediction_result = {
        "actual_labels": np.asarray(
            [
                0,
                1,
                2,
                0,
                2,
            ],
            dtype=np.int64,
        ),
        "actual_directions": np.asarray(
            [
                "DOWN",
                "FLAT",
                "UP",
                "DOWN",
                "UP",
            ],
            dtype=object,
        ),
        "stage1_predicted_move": np.asarray(
            [
                1,
                0,
                1,
                1,
                1,
            ],
            dtype=np.int64,
        ),
        "stage2_predicted_up": np.asarray(
            [
                0,
                1,
                1,
                0,
                1,
            ],
            dtype=np.int64,
        ),
        "final_predictions": np.asarray(
            [
                0,
                1,
                2,
                0,
                2,
            ],
            dtype=np.int64,
        ),
    }

    metrics = HierarchicalDirectionEvaluator().evaluate(
        prediction_result
    )

    assert metrics[
        "stage1_move_vs_flat"
    ][
        "macro_f1"
    ] == 1.0

    assert metrics[
        "stage2_up_vs_down_oracle"
    ][
        "macro_f1"
    ] == 1.0

    assert metrics[
        "end_to_end"
    ][
        "macro_f1"
    ] == 1.0


def test_probability_threshold_selection_uses_only_supplied_oof_predictions():
    actual = np.asarray(
        [
            0,
            0,
            1,
            1,
        ],
        dtype=np.int64,
    )

    probabilities = np.asarray(
        [
            0.10,
            0.35,
            0.60,
            0.90,
        ],
        dtype=np.float64,
    )

    result = (
        HierarchicalXLSTMParameterSelector
        .select_probability_threshold(
            actual=actual,
            positive_probabilities=probabilities,
        )
    )

    assert result[
        "macro_f1"
    ] == 1.0

    assert 0.35 < result[
        "threshold"
    ] <= 0.60
