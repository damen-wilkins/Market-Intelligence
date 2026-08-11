from app.ingestion.service import IngestionService
from app.ingestion.yahoo_provider import YahooProvider
from database.connection import get_connection_string
from database.market_data_repository import MarketDataRepository


STAGE2_CONTEXT_SYMBOLS = (
    "^VIX9D",
    "^VIX3M",
    "^SKEW",
    "^VXN",
    "DX-Y.NYB",
    "ES=F",
    "NQ=F",
    "RTY=F",
    "CL=F",
)


def main():
    provider = YahooProvider()
    repository = MarketDataRepository(get_connection_string())
    service = IngestionService(provider, repository)

    failures = []

    for symbol in STAGE2_CONTEXT_SYMBOLS:
        try:
            results = service.update_market_data(symbol)
        except Exception as exc:
            failures.append((symbol, str(exc)))
            print(symbol)
            print(f"FAILED: {exc}")
            print()
            continue

        print(symbol)
        print(f"Downloaded: {results['downloaded']}")
        print(f"Inserted: {results['inserted']}")
        print(f"Skipped: {results['skipped']}")
        print()

    if failures:
        details = "; ".join(
            f"{symbol}: {message}"
            for symbol, message in failures
        )
        raise RuntimeError(
            "One or more Stage-2 context series failed to download: "
            f"{details}"
        )


if __name__ == "__main__":
    main()
