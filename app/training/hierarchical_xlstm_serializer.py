from pathlib import Path

import torch

from app.training.hierarchical_sequence_preprocessor import (
    HierarchicalSequencePreprocessor,
)
from app.training.xlstm_classifier_model import XLSTMClassifier


class HierarchicalXLSTMSerializer:
    MODEL_TYPE = "xlstm_hierarchical_direction"

    @classmethod
    def save(
        cls,
        stage1_model: XLSTMClassifier,
        stage1_model_config: dict,
        stage1_preprocessor: HierarchicalSequencePreprocessor,
        stage1_threshold: float,
        stage2_model: XLSTMClassifier,
        stage2_model_config: dict,
        stage2_preprocessor: HierarchicalSequencePreprocessor,
        stage2_threshold: float,
        metadata: dict,
        filepath: str,
    ) -> str:
        path = Path(
            filepath
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "model_type": cls.MODEL_TYPE,
            "stage1": {
                "model_config": dict(
                    stage1_model_config
                ),
                "model_state_dict": stage1_model.state_dict(),
                "preprocessor_state": stage1_preprocessor.get_state(),
                "decision_threshold": float(
                    stage1_threshold
                ),
            },
            "stage2": {
                "model_config": dict(
                    stage2_model_config
                ),
                "model_state_dict": stage2_model.state_dict(),
                "preprocessor_state": stage2_preprocessor.get_state(),
                "decision_threshold": float(
                    stage2_threshold
                ),
            },
            "metadata": dict(
                metadata
            ),
        }

        torch.save(
            payload,
            path,
        )

        return str(
            path
        )

    @classmethod
    def load(
        cls,
        filepath: str,
        device: str | None = None,
    ) -> dict:
        path = Path(
            filepath
        )

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
            "stage1",
            "stage2",
            "metadata",
        }

        missing_keys = (
            required_keys
            - set(
                payload
            )
        )

        if missing_keys:
            raise ValueError(
                "Serialized hierarchical model is missing keys: "
                f"{sorted(missing_keys)}"
            )

        if payload[
            "model_type"
        ] != cls.MODEL_TYPE:
            raise ValueError(
                "Serialized model type does not match "
                "the hierarchical xLSTM serializer."
            )

        stage1 = cls._load_stage(
            payload[
                "stage1"
            ],
            device,
        )

        stage2 = cls._load_stage(
            payload[
                "stage2"
            ],
            device,
        )

        return {
            "model_type": payload[
                "model_type"
            ],
            "stage1": stage1,
            "stage2": stage2,
            "metadata": payload[
                "metadata"
            ],
            "device": device,
        }

    @staticmethod
    def _load_stage(
        stage_payload: dict,
        device: str,
    ) -> dict:
        required_keys = {
            "model_config",
            "model_state_dict",
            "preprocessor_state",
            "decision_threshold",
        }

        missing_keys = (
            required_keys
            - set(
                stage_payload
            )
        )

        if missing_keys:
            raise ValueError(
                "Serialized stage is missing keys: "
                f"{sorted(missing_keys)}"
            )

        model = XLSTMClassifier(
            **stage_payload[
                "model_config"
            ]
        )

        model.load_state_dict(
            stage_payload[
                "model_state_dict"
            ]
        )

        model = model.to(
            device
        )
        model.eval()

        preprocessor = (
            HierarchicalSequencePreprocessor.from_state(
                stage_payload[
                    "preprocessor_state"
                ]
            )
        )

        return {
            "model": model,
            "model_config": stage_payload[
                "model_config"
            ],
            "preprocessor": preprocessor,
            "decision_threshold": float(
                stage_payload[
                    "decision_threshold"
                ]
            ),
        }
