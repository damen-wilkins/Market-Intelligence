from app.ingestion.service import IngestionService
from app.ingestion.yahoo_provider import YahooProvider
from database.connection import get_connection_string
from database.market_data_repository import MarketDataRepository


VOLATILITY_SYMBOLS = ("^VIX", "^VVIX")


def main():
    provider = YahooProvider()
    repository = MarketDataRepository(get_connection_string())
    service = IngestionService(provider, repository)

    for symbol in VOLATILITY_SYMBOLS:
        results = service.update_market_data(symbol)

        print(f"{symbol}")
        print(f"Downloaded: {results['downloaded']}")
        print(f"Inserted: {results['inserted']}")
        print(f"Skipped: {results['skipped']}")
        print()


if __name__ == "__main__":
    main()