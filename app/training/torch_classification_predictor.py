import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TorchClassificationPredictor:
    def __init__(
        self,
        batch_size: int = 256,
        device: str | None = None,
    ):
        if batch_size <= 0:
            raise ValueError(
                "Batch size must be greater than zero."
            )

        self.batch_size = batch_size

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

    def predict(
        self,
        model: nn.Module,
        X: np.ndarray,
    ) -> dict:
        if X.ndim != 3:
            raise ValueError(
                "X must have shape "
                "[samples, sequence_length, features]."
            )

        if len(X) == 0:
            raise ValueError(
                "Prediction data cannot be empty."
            )

        tensor = torch.tensor(
            X,
            dtype=torch.float32,
        )

        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=self.batch_size,
            shuffle=False,
        )

        model = model.to(self.device)
        model.eval()

        probability_batches = []

        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(
                    self.device
                )

                logits = model(
                    X_batch
                )

                probabilities = torch.softmax(
                    logits,
                    dim=1,
                )

                probability_batches.append(
                    probabilities.cpu().numpy()
                )

        probabilities = np.concatenate(
            probability_batches,
            axis=0,
        )

        predictions = probabilities.argmax(
            axis=1
        )

        return {
            "predictions": predictions.astype(
                np.int64
            ),
            "probabilities": probabilities.astype(
                np.float32
            ),
        }