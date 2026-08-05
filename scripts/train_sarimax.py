from app.training.sarimax_trainer import SarimaxTrainer
from database.training_data_repository import TrainingDataRepository


def main():
    repository = TrainingDataRepository()
    trainer = SarimaxTrainer()

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
    print("TRAINING SARIMAX MODEL")
    print("=" * 50)
    print()

    results = trainer.train(
        endog=endog,
        exog=exog,
    )

    model = results["model"]

    print("=" * 50)
    print("SARIMAX TRAINING COMPLETE")
    print("=" * 50)
    print()

    print(f"Order               : {results['order']}")
    print(f"Seasonal Order      : {results['seasonal_order']}")
    print(f"AIC                 : {model.aic:.4f}")
    print(f"AICc                : {results['aicc']:.4f}")
    print(f"BIC                 : {model.bic:.4f}")
    print(f"Log Likelihood      : {model.llf:.4f}")
    print(f"Observations        : {model.nobs:,}")
    print(f"Parameters          : {len(model.params)}")


if __name__ == "__main__":
    main()