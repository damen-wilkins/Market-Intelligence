from datetime import date, timedelta
from app.ingestion.fred_provider import FredProvider
from database.macro_feature_repository import MacroFeatureRepository

DEFAULT_START_DATE = "1990-01-01"

def main():
    repository = MacroFeatureRepository()
    provider = FredProvider()

    features = repository.get_active_features()

    for feature in features:
        feature_id = feature["feature_id"]
        symbol = feature["provider_symbol"]

        latest_date = repository.get_latest_observation_date(feature_id)

        if latest_date is None:
            start_date = DEFAULT_START_DATE
            print(f"Loading full history for {symbol}...")
        else:
            start_date = (latest_date + timedelta(days=1)).isoformat()
            print(f"Updating {symbol} from {start_date}...")

        end_date = date.today().isoformat()
        data = provider.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        if data.empty:
            print(f"No new data for {symbol}.")
            continue
        repository.upsert_macro_features(
            feature_id=feature_id,
            dataframe=data,
        )
        print(f"{symbol}: {len(data)} rows processed.")
    print("Macroeconomic data load complete.")

if __name__ == "__main__":
    main()