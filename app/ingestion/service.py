from datetime import date, timedelta
from app.ingestion.provider import DataProvider
from database.connection import get_env
from database.market_data_repository import MarketDataRepository


class IngestionService:
    def __init__(
        self,
        provider: DataProvider,
        repository: MarketDataRepository,
    ):
        self.provider = provider
        self.repository = repository

    def update_market_data(self, ticker: str):
        latest_date = self.repository.get_latest_trade_date(ticker)

        if latest_date is None:
            start_date = get_env("MARKET_DATA_START_DATE")
        else:
            start_date = (latest_date + timedelta(days=1)).isoformat()

        end_date = date.today().isoformat()

        dataframe = self.provider.get_historical_data(
            symbol=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        rows = dataframe.to_dict(orient="records")

        return self.repository.insert_market_data(rows)