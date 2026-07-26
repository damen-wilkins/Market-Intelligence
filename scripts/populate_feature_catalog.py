from database.macro_feature_repository import MacroFeatureRepository

FEATURES = [
    {
        "feature_name": "Federal Funds Effective Rate",
        "provider": "FRED",
        "provider_symbol": "FEDFUNDS",
        "active": True,
        "training_feature": True,
    },
    {
        "feature_name": "ICE BofA High Yield Option-Adjusted Spread",
        "provider": "FRED",
        "provider_symbol": "BAMLH0A0HYM2",
        "active": True,
        "training_feature": False,
    },
    {
        "feature_name": "10-Year Treasury Constant Maturity",
        "provider": "FRED",
        "provider_symbol": "DGS10",
        "active": True,
        "training_feature": True,
    },
    {
        "feature_name": "2-Year Treasury Constant Maturity",
        "provider": "FRED",
        "provider_symbol": "DGS2",
        "active": True,
        "training_feature": True,
    },
    {
        "feature_name": "3-Month Treasury Constant Maturity",
        "provider": "FRED",
        "provider_symbol": "GS3M",
        "active": True,
        "training_feature": True,
    },
    {
        "feature_name": "Industrial Production Index",
        "provider": "FRED",
        "provider_symbol": "INDPRO",
        "active": True,
        "training_feature": True,
    },
    {
        "feature_name": "Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity",
        "provider": "FRED",
        "provider_symbol": "BAA10Y",
        "active": True,
        "training_feature": True,
    },
    {
        "feature_name": "Consumer Price Index",
        "provider": "FRED",
        "provider_symbol": "CPIAUCSL",
        "active": True,
        "training_feature": True,
    },
    {
        "feature_name": "Unemployment Rate",
        "provider": "FRED",
        "provider_symbol": "UNRATE",
        "active": True,
        "training_feature": True,
    },
    {
        "feature_name": "10-Year Breakeven Inflation Rate",
        "provider": "FRED",
        "provider_symbol": "T10YIE",
        "active": True,
        "training_feature": True,
    },
]
def main():
    repository = MacroFeatureRepository()
    repository.insert_feature_catalog(FEATURES)

    print(f"Loaded {len(FEATURES)} features into feature_catalog.")

if __name__ == "__main__":
    main()