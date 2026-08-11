import numpy as np
import pandas as pd

from app.training.stage2_wide_feature_builder import (
    Stage2WideFeatureBuilder,
)
from app.training.stage2_wide_signal_search import (
    build_pair_candidates,
    expand_beam_candidates,
    select_beam,
    univariate_feature_auc_screen,
)


def build_raw_data(rows: int = 360) -> pd.DataFrame:
    dates = pd.bdate_range(
        "2020-01-02",
        periods=rows,
    )
    base = np.linspace(
        100.0,
        140.0,
        rows,
    )
    wave = np.sin(
        np.arange(rows) / 7.0
    )
    close = base + wave
    open_price = close * (
        1.0 + 0.001 * np.cos(np.arange(rows))
    )
    high = np.maximum(open_price, close) * 1.005
    low = np.minimum(open_price, close) * 0.995
    volume = 1_000_000.0 + 1000.0 * np.arange(rows)

    dataframe = pd.DataFrame(
        {
            "ticker": "SPY",
            "trade_date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    dataframe["sma_10"] = dataframe["close"].rolling(10).mean()
    dataframe["sma_20"] = dataframe["close"].rolling(20).mean()
    dataframe["sma_50"] = dataframe["close"].rolling(50).mean()
    dataframe["ema_20"] = dataframe["close"].ewm(span=20).mean()
    dataframe["rsi_14"] = 50.0 + 5.0 * np.sin(np.arange(rows) / 5.0)
    dataframe["macd"] = dataframe["close"] - dataframe["ema_20"]
    dataframe["macd_signal"] = dataframe["macd"].ewm(span=9).mean()
    dataframe["macd_histogram"] = (
        dataframe["macd"] - dataframe["macd_signal"]
    )
    rolling_std = dataframe["close"].rolling(20).std()
    dataframe["bollinger_middle"] = dataframe["sma_20"]
    dataframe["bollinger_upper"] = dataframe["sma_20"] + 2.0 * rolling_std
    dataframe["bollinger_lower"] = dataframe["sma_20"] - 2.0 * rolling_std
    dataframe["log_return"] = np.log(
        dataframe["close"] / dataframe["close"].shift(1)
    )
    dataframe["vix_close"] = 20.0 + 2.0 * np.sin(np.arange(rows) / 11.0)
    dataframe["vvix_close"] = 90.0 + 4.0 * np.cos(np.arange(rows) / 13.0)

    series = {
        "rsp_close": 1.001,
        "qqq_close": 1.002,
        "iwm_close": 0.999,
        "dia_close": 1.0005,
        "tlt_close": 0.998,
        "ief_close": 0.999,
        "hyg_close": 1.0002,
        "lqd_close": 1.0001,
        "gld_close": 1.0003,
        "vix9d_close": 20.5,
        "vix3m_close": 21.0,
        "skew_close": 135.0,
        "vxn_close": 24.0,
        "dxy_close": 100.0,
        "es_close": 1.0001,
        "nq_close": 1.0003,
        "rty_close": 0.9998,
        "cl_close": 1.0004,
    }

    for column, multiplier in series.items():
        if column in {
            "vix9d_close",
            "vix3m_close",
            "skew_close",
            "vxn_close",
            "dxy_close",
        }:
            dataframe[column] = (
                multiplier
                + 0.5 * np.sin(np.arange(rows) / 17.0)
            )
        else:
            dataframe[column] = close * multiplier

    for symbol in (
        "xlb",
        "xle",
        "xlf",
        "xli",
        "xlk",
        "xlp",
        "xlu",
        "xlv",
        "xly",
    ):
        dataframe[f"{symbol}_close"] = close * (
            1.0 + 0.001 * len(symbol)
        )
        dataframe[f"{symbol}_volume"] = volume * (
            0.8 + 0.01 * len(symbol)
        )

    return dataframe.dropna().reset_index(drop=True)


def test_wide_feature_builder_produces_complete_contract():
    dataframe = build_raw_data()
    builder = Stage2WideFeatureBuilder()
    result = builder.build(dataframe)

    assert not result.empty
    assert result.columns.tolist() == [
        "trade_date",
        *builder.feature_columns,
    ]
    assert not result.isna().any().any()
    assert not np.isinf(
        result[builder.feature_columns].to_numpy()
    ).any()


def test_short_history_features_do_not_truncate_core_groups():
    dataframe = build_raw_data()
    dataframe.loc[
        dataframe.index[:180],
        "rty_close",
    ] = np.nan
    dataframe.loc[
        dataframe.index[:120],
        "vix9d_close",
    ] = np.nan

    core_builder = Stage2WideFeatureBuilder(
        group_names=[
            "technical_dynamics",
            "futures_core",
            "volatility_options_core",
        ]
    )
    short_builder = Stage2WideFeatureBuilder(
        group_names=[
            "futures_smallcap",
            "volatility_options_short",
        ]
    )

    core = core_builder.build(dataframe)
    short = short_builder.build(dataframe)

    assert not core.empty
    assert len(core) > len(short)
    assert core["trade_date"].min() < short["trade_date"].min()


def test_feature_group_contracts_are_unique():
    all_columns = Stage2WideFeatureBuilder.columns_for_groups(
        list(Stage2WideFeatureBuilder.FEATURE_GROUPS)
    )

    assert len(all_columns) == len(set(all_columns))
    assert set(Stage2WideFeatureBuilder.BASE_FEATURE_COLUMNS).issubset(
        all_columns
    )
    assert Stage2WideFeatureBuilder.SHORT_HISTORY_GROUPS == {
        "volatility_options_short",
        "futures_smallcap",
    }


def test_pair_candidates_are_deduplicated():
    candidates = build_pair_candidates(
        ranked_single_groups=[
            "technical_dynamics",
            "breadth",
            "equity_rotation",
        ],
        top_group_count=3,
    )

    names = [candidate.name for candidate in candidates]
    assert len(names) == 3
    assert len(names) == len(set(names))


def test_beam_expansion_deduplicates_group_sets():
    candidates = expand_beam_candidates(
        beam_groups=[
            ("breadth", "technical_dynamics"),
            ("breadth", "equity_rotation"),
        ],
        all_group_names=[
            "technical_dynamics",
            "breadth",
            "equity_rotation",
        ],
    )

    keys = [candidate.groups for candidate in candidates]
    assert len(keys) == len(set(keys))
    assert (
        "breadth",
        "equity_rotation",
        "technical_dynamics",
    ) in keys


def test_beam_ranks_matched_sample_delta_before_raw_auc():
    results = [
        {
            "groups": ["a"],
            "stage2_roc_auc": 0.58,
            "matched_base_roc_auc": 0.57,
            "delta_roc_auc_vs_matched_base": 0.01,
            "stage2_roc_auc_fold_std": 0.01,
            "training_rows": 3000,
        },
        {
            "groups": ["b"],
            "stage2_roc_auc": 0.55,
            "matched_base_roc_auc": 0.51,
            "delta_roc_auc_vs_matched_base": 0.04,
            "stage2_roc_auc_fold_std": 0.02,
            "training_rows": 1000,
        },
    ]

    beam = select_beam(
        results=results,
        beam_width=1,
    )

    assert beam == [("b",)]


def test_univariate_auc_screen_handles_staggered_availability():
    rows = 300
    dates = pd.bdate_range(
        "2021-01-01",
        periods=rows,
    )
    signal = np.sin(
        np.arange(rows) / 3.0
    )
    directions = np.where(
        signal > 0.0,
        "UP",
        "DOWN",
    )
    short_signal = signal.copy()
    short_signal[:100] = np.nan

    dataframe = pd.DataFrame(
        {
            "target_date": dates,
            "direction": directions,
            "signal": signal,
            "short_signal": short_signal,
        }
    )

    result = univariate_feature_auc_screen(
        training_data=dataframe,
        feature_columns=[
            "signal",
            "short_signal",
        ],
        n_splits=3,
        minimum_rows=150,
    )

    signal_row = result[
        result["feature"] == "signal"
    ].iloc[0]
    short_row = result[
        result["feature"] == "short_signal"
    ].iloc[0]

    assert signal_row["fold_predictive_auc_mean"] > 0.9
    assert short_row["fold_predictive_auc_mean"] > 0.9
    assert short_row["move_rows"] < signal_row["move_rows"]
    assert short_row["start_date"] > signal_row["start_date"]
