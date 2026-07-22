from datetime import date
from psycopg import connect

class FeatureRepository:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def insert_features(self, rows: list[dict]) -> dict:
        query = """
            INSERT INTO market_features (
                ticker,
                trade_date,
                sma_10,
                sma_20,
                sma_50,
                ema_20,
                rsi_14,
                macd,
                macd_signal,
                macd_histogram,
                bollinger_upper,
                bollinger_middle,
                bollinger_lower,
                daily_return,
                log_return
            )
            VALUES (
                %(ticker)s,
                %(trade_date)s,
                %(sma_10)s,
                %(sma_20)s,
                %(sma_50)s,
                %(ema_20)s,
                %(rsi_14)s,
                %(macd)s,
                %(macd_signal)s,
                %(macd_histogram)s,
                %(bollinger_upper)s,
                %(bollinger_middle)s,
                %(bollinger_lower)s,
                %(daily_return)s,
                %(log_return)s
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

        processed = len(rows)

        return {
            "processed": processed,
            "inserted": inserted,
            "skipped": processed - inserted,
        }

    def get_latest_trade_date(self, ticker: str) -> date | None:
        query = """
            SELECT MAX(trade_date)
            FROM market_features
            WHERE ticker = %s;
        """

        with connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (ticker,))
                return cur.fetchone()[0]