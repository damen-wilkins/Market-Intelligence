import pandas as pd

from app.training.experiment_tracker import ExperimentTracker
from app.training.model_comparison_evaluator import ModelComparisonEvaluator
from app.training.model_serializer import ModelSerializer
from app.training.residual_dataset_builder import ResidualDatasetBuilder
from app.training.residual_forecast_corrector import ResidualForecastCorrector
from app.training.xgboost_evaluator import XGBoostEvaluator
from app.training.xgboost_parameter_selector import XGBoostParameterSelector
from app.training.xgboost_predictor import XGBoostPredictor
from app.training.xgboost_trainer import XGBoostTrainer
from database.training_data_repository import TrainingDataRepository


MODEL_NAME = "xgboost_residual"


def direction(values):
    labels = []

    for value in values:
        if value > 0:
            labels.append("UP")
        elif value < 0:
            labels.append("DOWN")
        else:
            labels.append("FLAT")

    return labels


def main():
    repository = TrainingDataRepository()
    serializer = ModelSerializer()

    dataset = repository.get_training_data("SPY")
    dataset = dataset.dropna(subset=["log_return"]).reset_index(drop=True)

    sarimax_predictions = pd.read_csv(
        "models/sarimax_predictions.csv"
    ).squeeze("columns")

    actual = pd.read_csv(
        "models/sarimax_actual.csv"
    ).squeeze("columns")

    dataset = dataset.tail(len(actual)).reset_index(drop=True)

    builder = ResidualDatasetBuilder()

    residual_dataset = builder.build(
        features=dataset,
        predictions=sarimax_predictions,
    )

    X = residual_dataset.drop(
        columns=["sarimax_residual"]
    )

    y = residual_dataset["sarimax_residual"]

    selector = XGBoostParameterSelector()

    parameters = selector.select_best_parameters(
        X,
        y,
    )

    trainer = XGBoostTrainer()

    model = trainer.train(
        residual_dataset,
        parameters,
    )

    serializer.save(
        model=model,
        metadata=parameters,
        filename=MODEL_NAME,
    )

    predictor = XGBoostPredictor()

    predicted_residuals = predictor.predict(
        model,
        residual_dataset,
    )

    corrected_predictions = ResidualForecastCorrector().apply(
        sarimax_predictions=sarimax_predictions,
        predicted_residuals=predicted_residuals,
    )

    regression_metrics = XGBoostEvaluator().evaluate(
        actual=actual,
        predicted=corrected_predictions,
    )

    comparison_metrics = ModelComparisonEvaluator().evaluate(
        actual_labels=direction(actual),
        sarimax_labels=direction(sarimax_predictions),
        xgboost_labels=direction(corrected_predictions),
    )

    ExperimentTracker("experiments").save(
        experiment_name="xgboost_residual",
        model_name="XGBoost",
        parameters=parameters,
        metrics={
            "regression": regression_metrics,
            "comparison": comparison_metrics,
        },
        features=list(X.columns),
    )

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print()

    print("SARIMAX")
    for metric, value in comparison_metrics["sarimax"].items():
        print(f"{metric:<12}{value:.4f}")

    print()

    print("XGBOOST")
    for metric, value in comparison_metrics["xgboost"].items():
        print(f"{metric:<12}{value:.4f}")


if __name__ == "__main__":
    main()