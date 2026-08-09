import numpy as np
import pandas as pd

from app.training.volatility_direction_label_builder import (
    VolatilityDirectionLabelBuilder,
)


def build_market_data() -> pd.DataFrame:
    close_prices = [
        100.0,
        101.0,
        100.5,
        101.5,
        102.0,
        101.0,
        102.5,
        103.0,
        102.0,
        103.5,
        104.0,
        103.0,
        104.5,
        105.0,
        104.0,
        105.5,
        106.0,
        105.0,
        106.5,
        107.0,
        108.5,
        110.0,
        110.0,
        107.0,
    ]

    return pd.DataFrame(
        {
            "trade_date": pd.date_range(
                "2026-01-02",
                periods=len(close_prices),
                freq="B",
            ),
            "close": close_prices,
        }
    )


def test_builder_creates_next_trading_day_targets():
    market_data = build_market_data()

    result = VolatilityDirectionLabelBuilder(
        volatility_window=20,
        threshold_multiplier=0.15,
    ).build(market_data)

    assert len(result) == 3

    assert (
        result.iloc[0]["feature_date"]
        == market_data.iloc[20]["trade_date"]
    )

    assert (
        result.iloc[0]["target_date"]
        == market_data.iloc[21]["trade_date"]
    )

    assert (
        result.iloc[-1]["target_date"]
        == market_data.iloc[-1]["trade_date"]
    )


def test_builder_uses_only_information_available_on_feature_date():
    market_data = build_market_data()

    original = VolatilityDirectionLabelBuilder(
        volatility_window=20,
        threshold_multiplier=0.15,
    ).build(market_data)

    changed = market_data.copy()
    changed.loc[
        changed.index[-1],
        "close",
    ] = 500.0

    changed_result = VolatilityDirectionLabelBuilder(
        volatility_window=20,
        threshold_multiplier=0.15,
    ).build(changed)

    np.testing.assert_allclose(
        original.iloc[0]["rolling_volatility"],
        changed_result.iloc[0]["rolling_volatility"],
    )

    np.testing.assert_allclose(
        original.iloc[0]["threshold"],
        changed_result.iloc[0]["threshold"],
    )


def test_builder_assigns_up_flat_and_down_classes():
    market_data = build_market_data()

    result = VolatilityDirectionLabelBuilder(
        volatility_window=20,
        threshold_multiplier=0.15,
    ).build(market_data)

    assert result["direction"].tolist() == [
        "UP",
        "FLAT",
        "DOWN",
    ]


def test_builder_rejects_duplicate_dates():
    market_data = build_market_data()

    market_data.loc[
        market_data.index[-1],
        "trade_date",
    ] = market_data.loc[
        market_data.index[-2],
        "trade_date",
    ]

    builder = VolatilityDirectionLabelBuilder()

    try:
        builder.build(market_data)
        assert False, "Expected duplicate-date validation to fail."
    except ValueError as error:
        assert "duplicate trade dates" in str(error)