from collections.abc import Sequence

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

    def get_training_data(
        self,
        ticker: str,
        macro_feature_names: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        market = pd.DataFrame(
            self.market_repository.get_market_data(ticker)
        )
        technical = self.feature_repository.get_features(ticker)
        macro = self.macro_repository.get_training_feature_data()

        if market.empty:
            raise ValueError(
                f"No market data was found for ticker {ticker}."
            )

        if technical.empty:
            raise ValueError(
                f"No technical feature data was found for ticker {ticker}."
            )

        for dataframe in (market, technical):
            dataframe["trade_date"] = pd.to_datetime(
                dataframe["trade_date"]
            )

        if macro.empty:
            raise ValueError(
                "No active macroeconomic training features were found."
            )

        macro = macro.rename(
            columns={"observation_date": "trade_date"}
        )
        macro["trade_date"] = pd.to_datetime(
            macro["trade_date"]
        )

        if macro_feature_names is not None:
            requested_columns = list(macro_feature_names)
            missing_columns = [
                column
                for column in requested_columns
                if column not in macro.columns
            ]

            if missing_columns:
                raise ValueError(
                    "Requested macroeconomic features are unavailable: "
                    f"{missing_columns}"
                )

            macro = macro[
                ["trade_date", *requested_columns]
            ]

        macro = macro.sort_values(
            "trade_date"
        ).ffill()

        dataset = market.merge(
            technical,
            on=["ticker", "trade_date"],
            how="left",
            validate="one_to_one",
        )

        dataset = dataset.merge(
            macro,
            on="trade_date",
            how="left",
            validate="many_to_one",
        )

        macro_columns = [
            column
            for column in macro.columns
            if column != "trade_date"
        ]

        if macro_columns:
            dataset = dataset.dropna(
                subset=macro_columns
            )

        dataset = dataset.dropna(
            subset=[
                "daily_return",
                "log_return",
            ]
        )

        return dataset.sort_values(
            "trade_date"
        ).reset_index(drop=True)

    def get_train_validation_test(
        self,
        ticker: str,
        train_size: float = 0.70,
        validation_size: float = 0.15,
        macro_feature_names: Sequence[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dataset = self.get_training_data(
            ticker=ticker,
            macro_feature_names=macro_feature_names,
        )

        train_end = int(len(dataset) * train_size)
        validation_end = train_end + int(
            len(dataset) * validation_size
        )

        train = dataset.iloc[:train_end].copy()
        validation = dataset.iloc[
            train_end:validation_end
        ].copy()
        test = dataset.iloc[validation_end:].copy()

        return train, validation, test
