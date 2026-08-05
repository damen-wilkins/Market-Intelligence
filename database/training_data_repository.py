import pandas as pd

from database.connection import get_connection_string
from database.feature_repository import FeatureRepository
from database.macro_feature_repository import MacroFeatureRepository
from database.market_data_repository import MarketDataRepository


class TrainingDataRepository:
    def __init__(self):
        connection_string = get_connection_string()
        self.market_repository = MarketDataRepository(connection_string)
        self.feature_repository = FeatureRepository(connection_string)
        self.macro_repository = MacroFeatureRepository()

    def get_training_data(self, ticker: str) -> pd.DataFrame:
        market = pd.DataFrame(self.market_repository.get_market_data(ticker))
        technical = self.feature_repository.get_features(ticker)
        macro = self.macro_repository.get_training_feature_data()

        macro = macro.rename(columns={"observation_date": "trade_date"})
        macro = macro.sort_values("trade_date").ffill()

        dataset = market.merge(
            technical,
            on=["ticker", "trade_date"],
            how="left"
        )

        dataset = dataset.merge(
            macro,
            on="trade_date",
            how="left"
        )

        macro_columns = [
            column
            for column in macro.columns
            if column != "trade_date"
        ]

        dataset = dataset.dropna(subset=macro_columns)
        dataset = dataset.sort_values("trade_date").reset_index(drop=True)

        dataset = dataset.dropna(
            subset=[
                "daily_return",
                "log_return",
            ]
        ).reset_index(drop=True)

        return dataset

    def get_train_validation_test(
        self,
        ticker: str,
        train_size: float = 0.70,
        validation_size: float = 0.15,
    ):
        dataset = self.get_training_data(ticker)

        train_end = int(len(dataset) * train_size)
        validation_end = train_end + int(len(dataset) * validation_size)

        train = dataset.iloc[:train_end].copy()
        validation = dataset.iloc[train_end:validation_end].copy()
        test = dataset.iloc[validation_end:].copy()

        return train, validation, test