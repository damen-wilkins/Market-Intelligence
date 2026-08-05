import pandas as pd

from app.training.sarimax_parameter_selector import SarimaxParameterSelector
from database.training_data_repository import TrainingDataRepository


def main():
    repository = TrainingDataRepository()
    selector = SarimaxParameterSelector()

    dataset = repository.get_training_data("SPY")

    dataset = dataset.dropna(subset=["log_return"]).reset_index(drop=True)

    endog = dataset["log_return"]

    exclude = {
        "ticker",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "daily_return",
        "log_return",
        "return_1d",
        "return_1w",
        "return_1m",
        "return_1y",
        "label_1d",
        "label_1w",
        "label_1m",
        "label_1y",
    }

    exog_columns = [
        column
        for column in dataset.columns
        if column not in exclude
    ]

    exog = dataset[exog_columns]

    print("=" * 50)
    print("SELECTING SARIMAX PARAMETERS")
    print("=" * 50)
    print()

    print(f"Observations : {len(endog):,}")
    print(f"Exogenous Features : {len(exog_columns)}")
    print()

    results = selector.select(
        endog=endog,
        exog=exog,
    )

    print("=" * 50)
    print("BEST MODEL")
    print("=" * 50)
    print()

    print(f"Order           : {results['order']}")
    print(f"Seasonal Order  : {results['seasonal_order']}")
    print(f"AICc            : {results['aicc']:.4f}")


if __name__ == "__main__":
    main()