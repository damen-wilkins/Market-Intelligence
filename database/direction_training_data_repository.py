import pandas as pd

from database.connection import get_connection_string
from database.feature_repository import FeatureRepository
from database.market_data_repository import MarketDataRepository


class DirectionTrainingDataRepository:
    SECTOR_SYMBOLS = (
        "XLB",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLU",
        "XLV",
        "XLY",
    )

    CROSS_ASSET_SYMBOLS = (
        "QQQ",
        "IWM",
        "DIA",
        "TLT",
        "IEF",
        "HYG",
        "LQD",
        "GLD",
    )

    def __init__(self):
        connection_string = get_connection_string()

        self.market_repository = MarketDataRepository(
            connection_string
        )

        self.feature_repository = FeatureRepository(
            connection_string
        )

    def get_training_data(
        self,
        ticker: str = "SPY",
    ) -> pd.DataFrame:
        market = pd.DataFrame(
            self.market_repository.get_market_data(
                ticker
            )
        )

        technical = pd.DataFrame(
            self.feature_repository.get_features(
                ticker
            )
        )

        if market.empty:
            raise ValueError(
                f"No market data was found for ticker {ticker}."
            )

        if technical.empty:
            raise ValueError(
                f"No technical feature data was found for ticker {ticker}."
            )

        market["trade_date"] = pd.to_datetime(
            market["trade_date"]
        )

        technical["trade_date"] = pd.to_datetime(
            technical["trade_date"]
        )

        dataset = market.merge(
            technical,
            on=[
                "ticker",
                "trade_date",
            ],
            how="inner",
            validate="one_to_one",
        )

        dataset = self._merge_close_series(
            dataset=dataset,
            symbol="^VIX",
            column_name="vix_close",
        )

        dataset = self._merge_close_series(
            dataset=dataset,
            symbol="^VVIX",
            column_name="vvix_close",
        )

        dataset = self._merge_close_series(
            dataset=dataset,
            symbol="RSP",
            column_name="rsp_close",
        )

        for symbol in self.SECTOR_SYMBOLS:
            dataset = self._merge_sector_data(
                dataset=dataset,
                symbol=symbol,
            )

        for symbol in self.CROSS_ASSET_SYMBOLS:
            dataset = self._merge_close_series(
                dataset=dataset,
                symbol=symbol,
                column_name=(
                    f"{symbol.lower()}_close"
                ),
            )

        dataset = dataset.sort_values(
            "trade_date"
        ).reset_index(
            drop=True
        )

        if dataset[
            "trade_date"
        ].duplicated().any():
            raise ValueError(
                "Direction training data contains duplicate trade dates."
            )

        return dataset

    def _merge_close_series(
        self,
        dataset: pd.DataFrame,
        symbol: str,
        column_name: str,
    ) -> pd.DataFrame:
        data = pd.DataFrame(
            self.market_repository.get_market_data(
                symbol
            )
        )

        if data.empty:
            raise ValueError(
                f"No market data was found for symbol {symbol}."
            )

        data["trade_date"] = pd.to_datetime(
            data["trade_date"]
        )

        data = data[
            [
                "trade_date",
                "close",
            ]
        ].rename(
            columns={
                "close": column_name,
            }
        )

        return dataset.merge(
            data,
            on="trade_date",
            how="inner",
            validate="one_to_one",
        )

    def _merge_sector_data(
        self,
        dataset: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        data = pd.DataFrame(
            self.market_repository.get_market_data(
                symbol
            )
        )

        if data.empty:
            raise ValueError(
                f"No market data was found for sector ETF {symbol}."
            )

        data["trade_date"] = pd.to_datetime(
            data["trade_date"]
        )

        prefix = symbol.lower()

        data = data[
            [
                "trade_date",
                "close",
                "volume",
            ]
        ].rename(
            columns={
                "close": f"{prefix}_close",
                "volume": f"{prefix}_volume",
            }
        )

        return dataset.merge(
            data,
            on="trade_date",
            how="inner",
            validate="one_to_one",
        )