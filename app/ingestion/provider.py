from abc import ABC, abstractmethod


class DataProvider(ABC):
    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ):
        """
        Return historical data for a provider-specific symbol.
        """
        pass