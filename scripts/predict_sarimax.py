from app.training.sarimax_predictor import SarimaxPredictor
from app.training.sarimax_trainer import SarimaxTrainer
from database.training_data_repository import TrainingDataRepository


def main():
    repository = TrainingDataRepository()
    trainer = SarimaxTrainer()
    predictor = SarimaxPredictor()

    train, validation, test = repository.get_train_validation_test("SPY")

    train = train.dropna(subset=["log_return"]).reset_index(drop=True)
    validation = validation.dropna(subset=["log_return"]).reset_index(drop=True)

    endog = train["log_return"]

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
        for column in train.columns
        if column not in exclude
    ]

    train_exog = train[exog_columns]
    validation_exog = validation[exog_columns]

    results = trainer.train(
        endog=endog,
        exog=train_exog,
    )

    model = results["model"]

    predictions = predictor.predict(
        model=model,
        start=len(train),
        end=len(train) + len(validation) - 1,
        exog=validation_exog,
    )

    residuals = predictor.residuals(model)

    print("=" * 50)
    print("SARIMAX PREDICTION")
    print("=" * 50)
    print()

    print(f"Training Observations   : {len(train):,}")
    print(f"Validation Observations : {len(validation):,}")
    print(f"Predictions Generated   : {len(predictions):,}")
    print(f"Residuals Generated     : {len(residuals):,}")
    print()

    print("First 10 Predictions")
    print("--------------------")
    print(predictions.head(10))


if __name__ == "__main__":
    main()