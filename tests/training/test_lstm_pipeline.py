import numpy as np
import pandas as pd

from app.training.lstm_predictor import LSTMPredictor
from app.training.lstm_residual_model import LSTMResidualModel
from app.training.lstm_trainer import LSTMTrainer
from app.training.residual_sequence_preprocessor import (
    ResidualSequencePreprocessor,
)
from app.training.torch_model_serializer import TorchModelSerializer


def build_residual_frame(
    observations: int = 180,
) -> pd.DataFrame:
    random = np.random.default_rng(7)
    residuals = np.zeros(observations)
    shocks = random.normal(
        0.0,
        0.01,
        observations,
    )

    for index in range(1, observations):
        residuals[index] = (
            0.35 * residuals[index - 1]
            + shocks[index]
        )

    return pd.DataFrame(
        {
            "trade_date": pd.date_range(
                "2019-01-01",
                periods=observations,
                freq="B",
            ),
            "feature_one": random.normal(
                size=observations
            ),
            "feature_two": random.normal(
                size=observations
            ),
            "sarimax_prediction": random.normal(
                0.0,
                0.001,
                observations,
            ),
            "sarimax_residual": residuals,
        }
    )


def test_lstm_training_prediction_and_serialization_round_trip(
    tmp_path,
):
    dataframe = build_residual_frame()
    train = dataframe.iloc[:140].reset_index(drop=True)
    validation = dataframe.iloc[140:].reset_index(drop=True)
    feature_columns = [
        "feature_one",
        "feature_two",
        "sarimax_prediction",
    ]

    preprocessor = ResidualSequencePreprocessor(
        sequence_length=15
    ).fit(
        dataframe=train,
        feature_columns=feature_columns,
    )
    training_sequences = (
        preprocessor.build_training_sequences(
            train
        )
    )
    validation_sequences = (
        preprocessor.build_inference_sequences(
            history=train,
            dataframe=validation,
        )
    )

    trainer = LSTMTrainer(
        random_state=42,
        device="cpu",
    )
    model = LSTMResidualModel(
        input_size=len(
            preprocessor.feature_columns
        ),
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )
    result = trainer.train(
        model=model,
        training_data=training_sequences,
        validation_data=validation_sequences,
        epochs=3,
        batch_size=32,
        learning_rate=0.001,
        weight_decay=1e-6,
        gradient_clip=1.0,
        patience=2,
    )

    predictor = LSTMPredictor(device="cpu")
    original_predictions = predictor.predict(
        model=result["model"],
        dataset=validation_sequences,
        preprocessor=preprocessor,
    )

    serializer = TorchModelSerializer(
        model_directory=str(tmp_path)
    )
    serializer.save(
        model=result["model"],
        preprocessor=preprocessor,
        metadata={"model_version": "test"},
        filename="lstm_test",
    )
    loaded = serializer.load(
        filename="lstm_test",
        map_location="cpu",
    )
    loaded_predictions = predictor.predict(
        model=loaded["model"],
        dataset=validation_sequences,
        preprocessor=loaded["preprocessor"],
    )

    np.testing.assert_allclose(
        original_predictions[
            "predicted_residual"
        ],
        loaded_predictions[
            "predicted_residual"
        ],
        rtol=0.0,
        atol=0.0,
    )
    assert loaded["metadata"] == {
        "model_version": "test"
    }
