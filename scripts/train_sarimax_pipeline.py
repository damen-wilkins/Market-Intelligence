from pathlib import Path

from app.training.model_serializer import ModelSerializer
from app.training.sarimax_evaluator import SarimaxEvaluator
from app.training.sarimax_predictor import SarimaxPredictor
from app.training.sarimax_trainer import SarimaxTrainer
from database.training_data_repository import TrainingDataRepository


MODEL_NAME = "sarimax"


def get_exogenous_columns(dataset):
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

    return [
        column
        for column in dataset.columns
        if column not in exclude
    ]


def save_series(series, filename):
    output_directory = Path("models")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    series.reset_index(drop=True).to_csv(
        output_directory / filename,
        index=False,
    )


def main():
    repository = TrainingDataRepository()
    trainer = SarimaxTrainer()
    predictor = SarimaxPredictor()
    evaluator = SarimaxEvaluator()
    serializer = ModelSerializer()

    train, validation, _ = repository.get_train_validation_test("SPY")

    train = train.dropna(subset=["log_return"]).reset_index(drop=True)
    validation = validation.dropna(subset=["log_return"]).reset_index(drop=True)

    exogenous_columns = get_exogenous_columns(train)

    training_results = trainer.train(
        endog=train["log_return"],
        exog=train[exogenous_columns],
    )

    model = training_results["model"]

    predictions = predictor.predict(
        model=model,
        start=len(train),
        end=len(train) + len(validation) - 1,
        exog=validation[exogenous_columns],
    )

    residuals = predictor.residuals(model)

    metrics = evaluator.evaluate(
        actual=validation["log_return"],
        predicted=predictions,
    )

    metadata = {
        "order": training_results["order"],
        "seasonal_order": training_results["seasonal_order"],
        "aicc": training_results["aicc"],
        "metrics": metrics,
    }

    serializer.save(
        model=model,
        metadata=metadata,
        filename=MODEL_NAME,
    )

    save_series(
        predictions,
        "sarimax_predictions.csv",
    )

    save_series(
        residuals,
        "sarimax_residuals.csv",
    )

    save_series(
        validation["log_return"],
        "sarimax_actual.csv",
    )

    print("=" * 60)
    print("SARIMAX PIPELINE COMPLETE")
    print("=" * 60)
    print()

    print(f"Order                : {training_results['order']}")
    print(f"Seasonal Order       : {training_results['seasonal_order']}")
    print(f"AICc                 : {training_results['aicc']:.4f}")
    print()

    print("Artifacts")
    print("------------------------------")
    print("Model                : models/sarimax.joblib")
    print("Predictions          : models/sarimax_predictions.csv")
    print("Residuals            : models/sarimax_residuals.csv")
    print("Actual Values        : models/sarimax_actual.csv")
    print()

    print("Evaluation")
    print("------------------------------")
    for key, value in metrics.items():
        print(f"{key:<22} {value:.6f}")


if __name__ == "__main__":
    main()