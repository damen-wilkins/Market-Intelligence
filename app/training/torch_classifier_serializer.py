from pathlib import Path

import torch
from torch import nn

from app.training.classification_sequence_preprocessor import (
    ClassificationSequencePreprocessor,
)


class TorchClassifierSerializer:
    @staticmethod
    def save(
        model: nn.Module,
        model_type: str,
        model_config: dict,
        preprocessor: ClassificationSequencePreprocessor,
        metadata: dict,
        filepath: str,
    ) -> str:
        if not model_type:
            raise ValueError(
                "Model type is required."
            )

        path = Path(filepath)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "model_type": model_type,
            "model_config": dict(model_config),
            "model_state_dict": model.state_dict(),
            "preprocessor_state": preprocessor.get_state(),
            "metadata": dict(metadata),
        }

        torch.save(
            payload,
            path,
        )

        return str(path)

    @staticmethod
    def load(
        filepath: str,
        model_class,
        device: str | None = None,
    ) -> dict:
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(
                f"Model file does not exist: {filepath}"
            )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        payload = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

        required_keys = {
            "model_type",
            "model_config",
            "model_state_dict",
            "preprocessor_state",
            "metadata",
        }

        missing_keys = (
            required_keys
            - set(payload)
        )

        if missing_keys:
            raise ValueError(
                f"Serialized model is missing keys: "
                f"{sorted(missing_keys)}"
            )

        model = model_class(
            **payload["model_config"]
        )

        model.load_state_dict(
            payload["model_state_dict"]
        )

        model = model.to(device)
        model.eval()

        preprocessor = (
            ClassificationSequencePreprocessor.from_state(
                payload["preprocessor_state"]
            )
        )

        return {
            "model": model,
            "model_type": payload["model_type"],
            "model_config": payload["model_config"],
            "preprocessor": preprocessor,
            "metadata": payload["metadata"],
            "device": device,
        }