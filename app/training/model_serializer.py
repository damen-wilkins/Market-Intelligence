from pathlib import Path

import joblib


class ModelSerializer:
    def __init__(self, model_directory: str = "models"):
        self.model_directory = Path(model_directory)
        self.model_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        model,
        metadata: dict,
        filename: str,
    ):
        filepath = self.model_directory / f"{filename}.joblib"

        joblib.dump(
            {
                "model": model,
                "metadata": metadata,
            },
            filepath,
        )

        return filepath

    def load(
        self,
        filename: str,
    ):
        filepath = self.model_directory / f"{filename}.joblib"

        return joblib.load(filepath)