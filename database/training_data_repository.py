import pandas as pd
from sqlalchemy import create_engine

from database.connection import get_connection_string


class TrainingDataRepository:
    def __init__(self):
        self.engine = create_engine(get_connection_string())

    def get_training_data(self) -> pd.DataFrame:
        query = """
            SELECT
                trade_date,
                open,
                high,
                low,
                close,
                volume
            FROM market_data
            ORDER BY trade_date;
        """

        return pd.read_sql(query, self.engine)