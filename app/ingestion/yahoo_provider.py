import yfinance as yf
from app.ingestion.provider import DataProvider

class YahooProvider(DataProvider):
    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ):
        """
        Download historical market data from Yahoo Finance.
        """
        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
        )
        data = data.reset_index()
        data.columns = [
            "trade_date",
            "adj_close",
            "close",
            "high",
            "low",
            "open",
            "volume",
        ]
        data["ticker"] = symbol
        data = data[
            [
                "ticker",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
        return data