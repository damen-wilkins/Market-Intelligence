import pandas as pd
from fredapi import Fred
from app.ingestion.provider import DataProvider
from database.connection import get_env

class FredProvider(DataProvider):
    def __init__(self):
        self.fred = Fred(api_key=get_env("FRED_API_KEY"))

    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        data = self.fred.get_series(
            series_id=symbol,
            observation_start=start_date,
            observation_end=end_date,
        )
        data = (
            data.rename("value")
            .rename_axis("observation_date")
            .reset_index()
        )
        return data