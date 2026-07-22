from app.feature_engineering.indicator_calculator import IndicatorCalculator
from app.feature_engineering.service import FeatureEngineeringService
from database.connection import get_connection_string
from database.feature_repository import FeatureRepository
from database.market_data_repository import MarketDataRepository

def main():
    connection_string = get_connection_string()

    market_repository = MarketDataRepository(connection_string)
    feature_repository = FeatureRepository(connection_string)
    indicator_calculator = IndicatorCalculator()

    service = FeatureEngineeringService(
        market_repository=market_repository,
        feature_repository=feature_repository,
        indicator_calculator=indicator_calculator,
    )

    results = service.generate_features("SPY")

    print(f"Processed: {results['processed']}")
    print(f"Inserted: {results['inserted']}")
    print(f"Skipped:  {results['skipped']}")

if __name__ == "__main__":
    main()