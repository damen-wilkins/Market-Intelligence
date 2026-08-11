import pandas as pd

from database.connection import get_connection_string
from database.direction_training_data_repository import (
    DirectionTrainingDataRepository,
)
from database.market_data_repository import MarketDataRepository


class Stage2SignalDataRepository:
    CONTEXT_SYMBOLS = {
        "^VIX9D": "vix9d_close",
        "^VIX3M": "vix3m_close",
        "^SKEW": "skew_close",
        "^VXN": "vxn_close",
        "DX-Y.NYB": "dxy_close",
        "ES=F": "es_close",
        "NQ=F": "nq_close",
        "RTY=F": "rty_close",
        "CL=F": "cl_close",
    }

    def __init__(self):
        connection_string = get_connection_string()
        self.direction_repository = DirectionTrainingDataRepository()
        self.market_repository = MarketDataRepository(connection_string)

    def get_training_data(
        self,
        ticker: str = "SPY",
    ) -> pd.DataFrame:
        dataset = self.direction_repository.get_training_data(
            ticker=ticker,
            include_breadth=True,
            include_cross_asset=True,
        )

        for symbol, column_name in self.CONTEXT_SYMBOLS.items():
            dataset = self._merge_close_series(
                dataset=dataset,
                symbol=symbol,
                column_name=column_name,
            )

        dataset = dataset.sort_values(
            "trade_date"
        ).reset_index(drop=True)

        if dataset["trade_date"].duplicated().any():
            raise ValueError(
                "Stage-2 signal data contains duplicate trade dates."
            )

        return dataset

    def _merge_close_series(
        self,
        dataset: pd.DataFrame,
        symbol: str,
        column_name: str,
    ) -> pd.DataFrame:
        data = pd.DataFrame(
            self.market_repository.get_market_data(symbol)
        )

        if data.empty:
            raise ValueError(
                f"No market data was found for Stage-2 context symbol {symbol}. "
                "Run scripts.download_stage2_signal_context first."
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
            how="left",
            validate="one_to_one",
        )
