import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class HierarchicalSequencePreprocessor:
    VALID_TASKS = {
        "move",
        "direction",
    }

    STAGE1_MAPPING = {
        "FLAT": 0,
        "MOVE": 1,
    }

    STAGE2_MAPPING = {
        "DOWN": 0,
        "UP": 1,
    }

    def __init__(
        self,
        feature_columns: list[str],
        sequence_length: int,
    ):
        if not feature_columns:
            raise ValueError(
                "At least one feature column is required."
            )

        if sequence_length <= 0:
            raise ValueError(
                "Sequence length must be greater than zero."
            )

        self.feature_columns = list(feature_columns)
        self.sequence_length = int(sequence_length)
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(
        self,
        training_data: pd.DataFrame,
    ) -> "HierarchicalSequencePreprocessor":
        self._validate_dataframe(
            training_data,
            "training data",
        )

        training_data = self._sort(
            training_data
        )

        self.scaler.fit(
            training_data[
                self.feature_columns
            ]
        )

        self._is_fitted = True

        return self

    def build_training_sequences(
        self,
        dataframe: pd.DataFrame,
        task: str,
    ) -> dict:
        self._require_fitted()
        self._validate_task(task)
        self._validate_dataframe(
            dataframe,
            "training data",
        )

        dataframe = self._sort(
            dataframe
        )

        scaled_features = self.scaler.transform(
            dataframe[
                self.feature_columns
            ]
        )

        sequences = []
        labels = []
        directions = []
        feature_dates = []
        target_dates = []

        for row_index in range(
            self.sequence_length - 1,
            len(dataframe),
        ):
            direction = str(
                dataframe.loc[
                    row_index,
                    "direction",
                ]
            )

            if (
                task == "direction"
                and direction == "FLAT"
            ):
                continue

            start_index = (
                row_index
                - self.sequence_length
                + 1
            )

            sequences.append(
                scaled_features[
                    start_index : row_index + 1
                ]
            )

            labels.append(
                self._encode_label(
                    direction=direction,
                    task=task,
                )
            )

            directions.append(
                direction
            )

            feature_dates.append(
                dataframe.loc[
                    row_index,
                    "feature_date",
                ]
            )

            target_dates.append(
                dataframe.loc[
                    row_index,
                    "target_date",
                ]
            )

        return self._package(
            sequences=sequences,
            labels=labels,
            directions=directions,
            feature_dates=feature_dates,
            target_dates=target_dates,
            split_name="training",
        )

    def build_inference_sequences(
        self,
        history: pd.DataFrame,
        dataframe: pd.DataFrame,
        task: str,
        include_all: bool = True,
    ) -> dict:
        self._require_fitted()
        self._validate_task(task)
        self._validate_dataframe(
            history,
            "history data",
        )
        self._validate_dataframe(
            dataframe,
            "inference data",
        )

        history = self._sort(
            history
        )

        dataframe = self._sort(
            dataframe
        )

        if (
            history["target_date"].max()
            >= dataframe["target_date"].min()
        ):
            raise ValueError(
                "History must occur strictly before inference data."
            )

        context_rows = (
            self.sequence_length - 1
        )

        if len(history) < context_rows:
            raise ValueError(
                "Not enough historical rows to construct inference sequences."
            )

        history_tail = history.tail(
            context_rows
        )

        combined = pd.concat(
            [
                history_tail,
                dataframe,
            ],
            ignore_index=True,
        )

        scaled_features = self.scaler.transform(
            combined[
                self.feature_columns
            ]
        )

        sequences = []
        labels = []
        directions = []
        feature_dates = []
        target_dates = []

        for inference_offset in range(
            len(dataframe)
        ):
            row_index = (
                context_rows
                + inference_offset
            )

            direction = str(
                combined.loc[
                    row_index,
                    "direction",
                ]
            )

            if (
                task == "direction"
                and not include_all
                and direction == "FLAT"
            ):
                continue

            start_index = (
                row_index
                - self.sequence_length
                + 1
            )

            sequences.append(
                scaled_features[
                    start_index : row_index + 1
                ]
            )

            labels.append(
                self._encode_inference_label(
                    direction=direction,
                    task=task,
                )
            )

            directions.append(
                direction
            )

            feature_dates.append(
                combined.loc[
                    row_index,
                    "feature_date",
                ]
            )

            target_dates.append(
                combined.loc[
                    row_index,
                    "target_date",
                ]
            )

        return self._package(
            sequences=sequences,
            labels=labels,
            directions=directions,
            feature_dates=feature_dates,
            target_dates=target_dates,
            split_name="inference",
        )

    def get_state(self) -> dict:
        self._require_fitted()

        return {
            "feature_columns": list(
                self.feature_columns
            ),
            "sequence_length": int(
                self.sequence_length
            ),
            "scaler": {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
                "var": self.scaler.var_.tolist(),
                "n_features_in": int(
                    self.scaler.n_features_in_
                ),
                "n_samples_seen": int(
                    self.scaler.n_samples_seen_
                ),
            },
        }

    @classmethod
    def from_state(
        cls,
        state: dict,
    ) -> "HierarchicalSequencePreprocessor":
        preprocessor = cls(
            feature_columns=list(
                state["feature_columns"]
            ),
            sequence_length=int(
                state["sequence_length"]
            ),
        )

        scaler_state = state[
            "scaler"
        ]

        preprocessor.scaler.mean_ = np.asarray(
            scaler_state["mean"],
            dtype=np.float64,
        )

        preprocessor.scaler.scale_ = np.asarray(
            scaler_state["scale"],
            dtype=np.float64,
        )

        preprocessor.scaler.var_ = np.asarray(
            scaler_state["var"],
            dtype=np.float64,
        )

        preprocessor.scaler.n_features_in_ = int(
            scaler_state[
                "n_features_in"
            ]
        )

        preprocessor.scaler.n_samples_seen_ = int(
            scaler_state[
                "n_samples_seen"
            ]
        )

        preprocessor.scaler.feature_names_in_ = np.asarray(
            state["feature_columns"],
            dtype=object,
        )

        preprocessor._is_fitted = True

        return preprocessor

    def _validate_dataframe(
        self,
        dataframe: pd.DataFrame,
        name: str,
    ) -> None:
        required_columns = [
            "feature_date",
            "target_date",
            "direction",
            *self.feature_columns,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{name} is missing columns: {missing_columns}"
            )

        if dataframe.empty:
            raise ValueError(
                f"{name} is empty."
            )

        if dataframe[
            required_columns
        ].isna().any().any():
            raise ValueError(
                f"{name} contains missing values."
            )

        invalid_classes = (
            set(
                dataframe[
                    "direction"
                ].unique()
            )
            - {
                "DOWN",
                "FLAT",
                "UP",
            }
        )

        if invalid_classes:
            raise ValueError(
                f"{name} contains invalid classes: "
                f"{sorted(invalid_classes)}"
            )

        if dataframe[
            "target_date"
        ].duplicated().any():
            raise ValueError(
                f"{name} contains duplicate target dates."
            )

    @staticmethod
    def _sort(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        result = dataframe.copy()

        result["feature_date"] = pd.to_datetime(
            result["feature_date"]
        )

        result["target_date"] = pd.to_datetime(
            result["target_date"]
        )

        return result.sort_values(
            "target_date"
        ).reset_index(drop=True)

    def _encode_label(
        self,
        direction: str,
        task: str,
    ) -> int:
        if task == "move":
            return int(
                direction != "FLAT"
            )

        return int(
            self.STAGE2_MAPPING[
                direction
            ]
        )

    def _encode_inference_label(
        self,
        direction: str,
        task: str,
    ) -> int:
        if task == "move":
            return int(
                direction != "FLAT"
            )

        if direction == "FLAT":
            return -1

        return int(
            self.STAGE2_MAPPING[
                direction
            ]
        )

    def _package(
        self,
        sequences,
        labels,
        directions,
        feature_dates,
        target_dates,
        split_name: str,
    ) -> dict:
        if not sequences:
            raise ValueError(
                f"No {split_name} sequences could be created."
            )

        return {
            "X": np.asarray(
                sequences,
                dtype=np.float32,
            ),
            "y": np.asarray(
                labels,
                dtype=np.int64,
            ),
            "directions": np.asarray(
                directions,
                dtype=object,
            ),
            "feature_dates": pd.to_datetime(
                feature_dates
            ),
            "target_dates": pd.to_datetime(
                target_dates
            ),
        }

    def _validate_task(
        self,
        task: str,
    ) -> None:
        if task not in self.VALID_TASKS:
            raise ValueError(
                "Task must be 'move' or 'direction'."
            )

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError(
                "Preprocessor must be fitted first."
            )
