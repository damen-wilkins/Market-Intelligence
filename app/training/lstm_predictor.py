import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.training.residual_sequence_preprocessor import (
    ResidualSequenceDataset,
    ResidualSequencePreprocessor,
)


class LSTMPredictor:
    def __init__(
        self,
        device: str | None = None,
    ):
        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

    def predict(
        self,
        model: nn.Module,
        dataset: ResidualSequenceDataset,
        preprocessor: ResidualSequencePreprocessor,
        batch_size: int = 256,
    ) -> pd.DataFrame:
        if len(dataset) == 0:
            raise ValueError(
                "LSTM prediction data cannot be empty."
            )

        model = model.to(self.device)
        model.eval()

        tensor_dataset = TensorDataset(
            torch.from_numpy(dataset.sequences)
        )
        data_loader = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(
                self.device.type == "cuda"
            ),
            drop_last=False,
        )

        scaled_predictions = []

        with torch.no_grad():
            for (sequences,) in data_loader:
                predictions = model(
                    sequences.to(
                        self.device,
                        non_blocking=True,
                    )
                )
                scaled_predictions.append(
                    predictions.detach().cpu().numpy()
                )

        predicted_residuals = (
            preprocessor.inverse_transform_target(
                np.concatenate(scaled_predictions)
            )
        )

        return pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    dataset.trade_dates
                ),
                "sarimax_prediction": (
                    dataset.sarimax_predictions
                ),
                "sarimax_residual": (
                    dataset.actual_residuals
                ),
                "predicted_residual": (
                    predicted_residuals
                ),
            }
        )
