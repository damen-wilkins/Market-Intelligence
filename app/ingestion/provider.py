from abc import ABC, abstractmethod


class MarketDataProvider(ABC):
    @abstractmethod
    def get_historical_data(self, ticker: str, start_date: str, end_date: str):
        """Return historical market data for a ticker."""
        pass