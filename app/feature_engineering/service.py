from datetime import timedelta
import numpy as np
import pandas as pd

from app.feature_engineering.indicator_calculator import IndicatorCalculator

LOOKBACK_DAYS = 60

class FeatureEngineeringService:
    def __init__(
        self,
        market_repository,
        feature_repository,
        indicator_calculator: IndicatorCalculator,
    ):
        self.market_repository = market_repository
        self.feature_repository = feature_repository
        self.indicator_calculator = indicator_calculator

    def generate_features(self, ticker: str) -> dict:
        latest_feature_date = self.feature_repository.get_latest_trade_date(ticker)

        if latest_feature_date is None:
            rows = self.market_repository.get_market_data(ticker)
            insert_after = None
        else:
            start_date = latest_feature_date - timedelta(days=LOOKBACK_DAYS)
            rows = self.market_repository.get_market_data_since(
                ticker=ticker,
                start_date=start_date,
            )
            insert_after = latest_feature_date

        if not rows:
            return {
                "processed": 0,
                "inserted": 0,
                "skipped": 0,
            }

        dataframe = pd.DataFrame(rows)

        dataframe = self.indicator_calculator.calculate(dataframe)

        dataframe = dataframe.replace({np.nan: None})

        if insert_after is not None:
            dataframe = dataframe[dataframe["trade_date"] > insert_after]

        columns = [
            "ticker",
            "trade_date",
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
            "log_return",
        ]

        feature_rows = dataframe[columns].to_dict(orient="records")

        return self.feature_repository.insert_features(feature_rows)