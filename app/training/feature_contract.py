from collections.abc import Sequence

import pandas as pd


MARKET_PREDICTOR_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)

TECHNICAL_PREDICTOR_COLUMNS = (
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bollinger_upper",
    "bollinger_middle",
    "bollinger_lower",
    "daily_return",
)

SARIMAX_EXOGENOUS_COLUMNS = (
    "10-Year Treasury Constant Maturity",
    "2-Year Treasury Constant Maturity",
    "Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity",
)

RESIDUAL_MACRO_FEATURE_COLUMNS = SARIMAX_EXOGENOUS_COLUMNS

REFERENCE_DATE_MACRO_FEATURE_COLUMNS = (
    "Federal Funds Effective Rate",
    "3-Month Treasury Constant Maturity",
    "Industrial Production Index",
    "Consumer Price Index",
    "Unemployment Rate",
)

RESIDUAL_BASE_FEATURE_COLUMNS = (
    *MARKET_PREDICTOR_COLUMNS,
    *TECHNICAL_PREDICTOR_COLUMNS,
    *RESIDUAL_MACRO_FEATURE_COLUMNS,
)

RESIDUAL_MODEL_FEATURE_COLUMNS = (
    *RESIDUAL_BASE_FEATURE_COLUMNS,
    "sarimax_prediction",
)


def require_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    context: str,
) -> None:
    missing_columns = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{context} is missing required columns: "
            f"{missing_columns}"
        )
