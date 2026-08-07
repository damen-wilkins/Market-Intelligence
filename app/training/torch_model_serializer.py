from pathlib import Path

import torch

from app.training.lstm_residual_model import LSTMResidualModel
from app.training.residual_sequence_preprocessor import (
    ResidualSequencePreprocessor,
)


class TorchModelSerializer:
    def __init__(
        self,
        model_directory: str = "models",
    ):
        self.model_directory = Path(model_directory)
        self.model_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        model: LSTMResidualModel,
        preprocessor: ResidualSequencePreprocessor,
        metadata: dict,
        filename: str,
    ) -> Path:
        filepath = self.model_directory / f"{filename}.pt"
        state_dict = {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        }

        torch.save(
            {
                "model_state_dict": state_dict,
                "model_config": model.get_config(),
                "preprocessor_state": (
                    preprocessor.get_state()
                ),
                "metadata": metadata,
            },
            filepath,
        )

        return filepath

    def load(
        self,
        filename: str,
        map_location: str = "cpu",
    ) -> dict:
        filepath = self.model_directory / f"{filename}.pt"

        if not filepath.exists():
            raise FileNotFoundError(
                f"Torch model artifact was not found: {filepath}"
            )

        try:
            package = torch.load(
                filepath,
                map_location=map_location,
                weights_only=True,
            )
        except TypeError:
            package = torch.load(
                filepath,
                map_location=map_location,
            )

        model = LSTMResidualModel(
            **package["model_config"]
        )
        model.load_state_dict(
            package["model_state_dict"]
        )
        preprocessor = (
            ResidualSequencePreprocessor.from_state(
                package["preprocessor_state"]
            )
        )

        return {
            "model": model,
            "preprocessor": preprocessor,
            "metadata": package["metadata"],
            "filepath": filepath,
        }
