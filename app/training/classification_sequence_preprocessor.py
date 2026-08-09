import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class ClassificationSequencePreprocessor:
    CLASS_MAPPING = {
        "DOWN": 0,
        "FLAT": 1,
        "UP": 2,
    }

    REVERSE_CLASS_MAPPING = {
        0: "DOWN",
        1: "FLAT",
        2: "UP",
    }

    def __init__(
        self,
        feature_columns: list[str],
        sequence_length: int = 20,
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
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(
        self,
        training_data: pd.DataFrame,
    ):
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
    ) -> dict:
        self._require_fitted()

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
        feature_dates = []
        target_dates = []

        for row_index in range(
            self.sequence_length - 1,
            len(dataframe),
        ):
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
                self.CLASS_MAPPING[
                    dataframe.loc[
                        row_index,
                        "direction",
                    ]
                ]
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
            feature_dates=feature_dates,
            target_dates=target_dates,
            split_name="training",
        )

    def build_inference_sequences(
        self,
        history: pd.DataFrame,
        dataframe: pd.DataFrame,
    ) -> dict:
        self._require_fitted()

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
                "Not enough historical rows to construct "
                "inference sequences."
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
        feature_dates = []
        target_dates = []

        for inference_offset in range(
            len(dataframe)
        ):
            row_index = (
                context_rows
                + inference_offset
            )

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
                self.CLASS_MAPPING[
                    combined.loc[
                        row_index,
                        "direction",
                    ]
                ]
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
            feature_dates=feature_dates,
            target_dates=target_dates,
            split_name="inference",
        )

    def fit_transform(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
    ) -> dict:
        self.fit(
            train
        )

        validation_history = train

        test_history = pd.concat(
            [
                train,
                validation,
            ],
            ignore_index=True,
        )

        return {
            "train": self.build_training_sequences(
                train
            ),
            "validation": self.build_inference_sequences(
                history=validation_history,
                dataframe=validation,
            ),
            "test": self.build_inference_sequences(
                history=test_history,
                dataframe=test,
            ),
        }

    def transform_features(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        self._require_fitted()

        return self.scaler.transform(
            dataframe[
                self.feature_columns
            ]
        )

    def get_state(self) -> dict:
        self._require_fitted()

        return {
            "feature_columns": list(
                self.feature_columns
            ),
            "sequence_length": self.sequence_length,
            "class_mapping": dict(
                self.CLASS_MAPPING
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
    ):
        preprocessor = cls(
            feature_columns=state[
                "feature_columns"
            ],
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

        preprocessor.scaler.feature_names_in_ = (
            np.asarray(
                state[
                    "feature_columns"
                ],
                dtype=object,
            )
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
                f"{name} is missing columns: "
                f"{missing_columns}"
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
            - set(
                self.CLASS_MAPPING
            )
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

    def _package(
        self,
        sequences,
        labels,
        feature_dates,
        target_dates,
        split_name: str,
    ) -> dict:
        if not sequences:
            raise ValueError(
                f"No {split_name} sequences "
                "could be created."
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
            "feature_dates": pd.to_datetime(
                feature_dates
            ),
            "target_dates": pd.to_datetime(
                target_dates
            ),
        }

    def _require_fitted(self):
        if not self._is_fitted:
            raise ValueError(
                "Preprocessor must be fitted first."
            )