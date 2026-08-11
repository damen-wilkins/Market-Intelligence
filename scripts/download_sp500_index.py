from app.ingestion.service import IngestionService
from app.ingestion.yahoo_provider import YahooProvider
from database.connection import get_connection_string
from database.market_data_repository import MarketDataRepository


SP500_INDEX_SYMBOL = "^GSPC"


def main():
    provider = YahooProvider()
    repository = MarketDataRepository(
        get_connection_string()
    )
    service = IngestionService(
        provider,
        repository,
    )

    results = service.update_market_data(
        SP500_INDEX_SYMBOL
    )

    print(
        SP500_INDEX_SYMBOL
    )
    print(
        f"Downloaded: {results['downloaded']}"
    )
    print(
        f"Inserted: {results['inserted']}"
    )
    print(
        f"Skipped: {results['skipped']}"
    )


if __name__ == "__main__":
    main()
