import numpy as np
import pandas as pd

from app.training.direction_feature_builder import DirectionFeatureBuilder
from app.training.flat_target_sensitivity_analyzer import (
    FlatTargetSensitivityAnalyzer,
)
from app.training.trend_signal_analyzer import TrendSignalAnalyzer


def build_market_frame(
    rows: int = 260,
) -> pd.DataFrame:
    index = np.arange(
        rows,
        dtype=float,
    )

    trend = 100.0 + index * 0.08
    cycle = 2.5 * np.sin(
        index / 8.0
    )
    close = trend + cycle

    open_price = close * (
        1.0
        + 0.001
        * np.sin(
            index / 5.0
        )
    )
    high = np.maximum(
        open_price,
        close,
    ) * 1.006
    low = np.minimum(
        open_price,
        close,
    ) * 0.994
    volume = (
        1_000_000.0
        + 50_000.0
        * np.cos(
            index / 7.0
        )
        + index * 500.0
    )

    dataframe = pd.DataFrame(
        {
            "trade_date": pd.date_range(
                "2020-01-01",
                periods=rows,
                freq="B",
            ),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    return add_stored_features(
        dataframe
    )


def add_stored_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()
    close = result[
        "close"
    ]

    result[
        "sma_10"
    ] = close.rolling(
        10,
        min_periods=1,
    ).mean()

    result[
        "sma_20"
    ] = close.rolling(
        20,
        min_periods=1,
    ).mean()

    result[
        "sma_50"
    ] = close.rolling(
        50,
        min_periods=1,
    ).mean()

    result[
        "ema_20"
    ] = close.ewm(
        span=20,
        adjust=False,
    ).mean()

    result[
        "rsi_14"
    ] = 50.0

    result[
        "macd"
    ] = (
        close.ewm(
            span=12,
            adjust=False,
        ).mean()
        - close.ewm(
            span=26,
            adjust=False,
        ).mean()
    )

    result[
        "macd_signal"
    ] = result[
        "macd"
    ].ewm(
        span=9,
        adjust=False,
    ).mean()

    result[
        "macd_histogram"
    ] = (
        result[
            "macd"
        ]
        - result[
            "macd_signal"
        ]
    )

    rolling_mean = close.rolling(
        20,
        min_periods=1,
    ).mean()
    rolling_std = close.rolling(
        20,
        min_periods=1,
    ).std().fillna(
        0.1
    )

    result[
        "bollinger_middle"
    ] = rolling_mean
    result[
        "bollinger_upper"
    ] = (
        rolling_mean
        + 2.0
        * rolling_std
    )
    result[
        "bollinger_lower"
    ] = (
        rolling_mean
        - 2.0
        * rolling_std
    )

    result[
        "log_return"
    ] = np.log(
        close
        / close.shift(1)
    ).fillna(
        0.0
    )

    result[
        "vix_close"
    ] = (
        18.0
        + np.sin(
            np.arange(
                len(
                    result
                )
            )
            / 10.0
        )
    )

    result[
        "vvix_close"
    ] = (
        90.0
        + np.cos(
            np.arange(
                len(
                    result
                )
            )
            / 10.0
        )
    )

    return result


def test_base_trend_scope_adds_relationship_and_trend_strength_features():
    builder = DirectionFeatureBuilder(
        feature_scope="base_trend"
    )

    result = builder.build(
        build_market_frame()
    )

    assert builder.feature_columns == [
        *DirectionFeatureBuilder.BASE_FEATURE_COLUMNS,
        *DirectionFeatureBuilder.TREND_STATE_FEATURE_COLUMNS,
    ]

    for column in (
        DirectionFeatureBuilder
        .TREND_STATE_FEATURE_COLUMNS
    ):
        assert column in result.columns

    assert set(
        result[
            "ma_alignment_score"
        ].unique()
    ).issubset(
        {
            -1.0,
            0.0,
            1.0,
        }
    )

    assert (
        result[
            "adx_14"
        ]
        >= 0.0
    ).all()


def test_default_all_scope_does_not_silently_change_existing_feature_contract():
    builder = DirectionFeatureBuilder(
        feature_scope="all"
    )

    assert builder.feature_columns == (
        DirectionFeatureBuilder
        .FEATURE_COLUMNS
    )

    assert not any(
        column
        in builder.feature_columns
        for column in (
            DirectionFeatureBuilder
            .TREND_STATE_FEATURE_COLUMNS
        )
    )


def test_trend_features_do_not_change_when_only_future_rows_are_modified():
    original = build_market_frame()
    modified = original.copy()

    cutoff = 180

    modified.loc[
        cutoff:,
        [
            "open",
            "high",
            "low",
            "close",
        ],
    ] = (
        modified.loc[
            cutoff:,
            [
                "open",
                "high",
                "low",
                "close",
            ],
        ]
        * 1.75
    )

    modified = add_stored_features(
        modified[
            [
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
    )

    builder = DirectionFeatureBuilder(
        feature_scope="base_trend"
    )

    original_features = builder.build(
        original
    )

    modified_features = builder.build(
        modified
    )

    comparison_date = original.loc[
        cutoff - 1,
        "trade_date",
    ]

    original_history = (
        original_features.loc[
            original_features[
                "trade_date"
            ]
            <= comparison_date
        ]
        .reset_index(
            drop=True
        )
    )

    modified_history = (
        modified_features.loc[
            modified_features[
                "trade_date"
            ]
            <= comparison_date
        ]
        .reset_index(
            drop=True
        )
    )

    pd.testing.assert_frame_equal(
        original_history,
        modified_history,
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )


def test_flat_share_increases_monotonically_as_volatility_multiplier_widens():
    data = build_market_frame()

    analyzer = FlatTargetSensitivityAnalyzer(
        volatility_windows=(20,),
        threshold_multipliers=(
            0.15,
            0.30,
            0.50,
        ),
        training_fraction=0.70,
        stability_blocks=3,
    )

    result = analyzer.analyze(
        data=data,
    )

    flat_shares = result[
        "flat_share"
    ].to_numpy()

    assert np.all(
        np.diff(
            flat_shares
        )
        >= 0.0
    )

    assert np.allclose(
        result[
            [
                "down_share",
                "flat_share",
                "up_share",
            ]
        ].sum(
            axis=1
        ),
        1.0,
    )


def test_trend_signal_analyzer_returns_alignment_adx_and_stability_outputs():
    market = build_market_frame()

    feature_data = (
        DirectionFeatureBuilder(
            feature_scope="base_trend"
        )
        .build(
            market
        )
    )

    result = TrendSignalAnalyzer(
        training_fraction=0.70,
        stability_blocks=3,
    ).analyze(
        feature_data=feature_data,
        market_data=market,
    )

    assert set(
        result
    ) == {
        "alignment",
        "adx",
        "compression",
        "stability",
    }

    assert set(
        result[
            "adx"
        ][
            "bucket"
        ]
    ) == {
        "LOW",
        "MID",
        "HIGH",
    }

    assert not result[
        "alignment"
    ].empty

    assert not result[
        "stability"
    ].empty
