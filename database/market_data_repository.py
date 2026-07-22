from datetime import date, timedelta
from psycopg import connect
from psycopg.rows import dict_row

class MarketDataRepository:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def insert_market_data(self, rows: list[dict]) -> dict:
        query = """
            INSERT INTO market_data (
                ticker,
                trade_date,
                open,
                high,
                low,
                close,
                volume
            )
            VALUES (
                %(ticker)s,
                %(trade_date)s,
                %(open)s,
                %(high)s,
                %(low)s,
                %(close)s,
                %(volume)s
            )
            ON CONFLICT (ticker, trade_date)
            DO NOTHING;
        """

        inserted = 0

        with connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute(query, row)
                    inserted += cur.rowcount

        skipped = len(rows) - inserted

        return {
            "downloaded": len(rows),
            "inserted": inserted,
            "skipped": skipped,
        }

    def get_latest_trade_date(self, ticker: str) -> date | None:
        query = """
            SELECT MAX(trade_date)
            FROM market_data
            WHERE ticker = %s;
        """

        with connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (ticker,))
                return cur.fetchone()[0]

    def get_market_data(self, ticker: str) -> list[dict]:
        query = """
            SELECT
                ticker,
                trade_date,
                open,
                high,
                low,
                close,
                volume
            FROM market_data
            WHERE ticker = %s
            ORDER BY trade_date;
        """

        with connect(self.connection_string) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, (ticker,))
                return cur.fetchall()

    def get_market_data_since(
        self,
        ticker: str,
        start_date: date,
    ) -> list[dict]:
        query = """
            SELECT
                ticker,
                trade_date,
                open,
                high,
                low,
                close,
                volume
            FROM market_data
            WHERE ticker = %s
              AND trade_date >= %s
            ORDER BY trade_date;
        """

        with connect(self.connection_string) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, (ticker, start_date))
                return cur.fetchall()