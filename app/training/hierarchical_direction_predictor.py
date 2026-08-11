import numpy as np
import pandas as pd
from torch import nn

from app.training.hierarchical_sequence_preprocessor import (
    HierarchicalSequencePreprocessor,
)
from app.training.torch_classification_predictor import (
    TorchClassificationPredictor,
)


class HierarchicalDirectionPredictor:
    DIRECTION_MAPPING = {
        "DOWN": 0,
        "FLAT": 1,
        "UP": 2,
    }

    REVERSE_DIRECTION_MAPPING = {
        0: "DOWN",
        1: "FLAT",
        2: "UP",
    }

    def __init__(
        self,
        stage1_model: nn.Module,
        stage1_preprocessor: HierarchicalSequencePreprocessor,
        stage1_move_threshold: float,
        stage2_model: nn.Module,
        stage2_preprocessor: HierarchicalSequencePreprocessor,
        stage2_up_threshold: float,
        batch_size: int = 256,
        device: str | None = None,
    ):
        self._validate_threshold(
            stage1_move_threshold,
            "Stage 1 MOVE threshold",
        )
        self._validate_threshold(
            stage2_up_threshold,
            "Stage 2 UP threshold",
        )

        self.stage1_model = stage1_model
        self.stage1_preprocessor = stage1_preprocessor
        self.stage1_move_threshold = float(
            stage1_move_threshold
        )
        self.stage2_model = stage2_model
        self.stage2_preprocessor = stage2_preprocessor
        self.stage2_up_threshold = float(
            stage2_up_threshold
        )
        self.batch_size = int(batch_size)
        self.device = device

    def predict(
        self,
        history: pd.DataFrame,
        dataframe: pd.DataFrame,
    ) -> dict:
        stage1_sequences = (
            self.stage1_preprocessor
            .build_inference_sequences(
                history=history,
                dataframe=dataframe,
                task="move",
                include_all=True,
            )
        )

        stage2_sequences = (
            self.stage2_preprocessor
            .build_inference_sequences(
                history=history,
                dataframe=dataframe,
                task="direction",
                include_all=True,
            )
        )

        self._validate_alignment(
            stage1_sequences,
            stage2_sequences,
        )

        predictor = TorchClassificationPredictor(
            batch_size=self.batch_size,
            device=self.device,
        )

        stage1_result = predictor.predict(
            model=self.stage1_model,
            X=stage1_sequences[
                "X"
            ],
        )

        stage2_result = predictor.predict(
            model=self.stage2_model,
            X=stage2_sequences[
                "X"
            ],
        )

        move_probability = (
            stage1_result[
                "probabilities"
            ][
                :,
                1,
            ]
        )

        up_probability = (
            stage2_result[
                "probabilities"
            ][
                :,
                1,
            ]
        )

        predicted_move = (
            move_probability
            >= self.stage1_move_threshold
        )

        predicted_up = (
            up_probability
            >= self.stage2_up_threshold
        )

        final_predictions = np.full(
            len(predicted_move),
            fill_value=1,
            dtype=np.int64,
        )

        final_predictions[
            predicted_move
            & ~predicted_up
        ] = 0

        final_predictions[
            predicted_move
            & predicted_up
        ] = 2

        actual_directions = np.asarray(
            stage1_sequences[
                "directions"
            ],
            dtype=object,
        )

        actual_labels = np.asarray(
            [
                self.DIRECTION_MAPPING[
                    str(direction)
                ]
                for direction in actual_directions
            ],
            dtype=np.int64,
        )

        return {
            "feature_dates": stage1_sequences[
                "feature_dates"
            ],
            "target_dates": stage1_sequences[
                "target_dates"
            ],
            "actual_directions": actual_directions,
            "actual_labels": actual_labels,
            "stage1_move_probability": (
                move_probability.astype(
                    np.float32
                )
            ),
            "stage1_predicted_move": (
                predicted_move.astype(
                    np.int64
                )
            ),
            "stage2_up_probability": (
                up_probability.astype(
                    np.float32
                )
            ),
            "stage2_predicted_up": (
                predicted_up.astype(
                    np.int64
                )
            ),
            "final_predictions": final_predictions,
            "final_predicted_directions": np.asarray(
                [
                    self.REVERSE_DIRECTION_MAPPING[
                        int(value)
                    ]
                    for value in final_predictions
                ],
                dtype=object,
            ),
        }

    @staticmethod
    def _validate_threshold(
        threshold: float,
        name: str,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"{name} must be between zero and one."
            )

    @staticmethod
    def _validate_alignment(
        stage1_sequences: dict,
        stage2_sequences: dict,
    ) -> None:
        stage1_dates = pd.DatetimeIndex(
            stage1_sequences[
                "target_dates"
            ]
        )

        stage2_dates = pd.DatetimeIndex(
            stage2_sequences[
                "target_dates"
            ]
        )

        if not stage1_dates.equals(
            stage2_dates
        ):
            raise ValueError(
                "Stage 1 and Stage 2 inference dates do not align."
            )

        if not np.array_equal(
            stage1_sequences[
                "directions"
            ],
            stage2_sequences[
                "directions"
            ],
        ):
            raise ValueError(
                "Stage 1 and Stage 2 actual directions do not align."
            )
