from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


class ExperimentTracker:
    def __init__(self, output_directory: str):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        experiment_name: str,
        model_name: str,
        parameters: dict,
        metrics: dict,
        features: list[str],
    ) -> Path:
        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d_%H%M%S_%f")

        experiment = {
            "experiment_name": experiment_name,
            "model_name": model_name,
            "timestamp_utc": timestamp,
            "parameters": parameters,
            "metrics": metrics,
            "features": features,
        }

        output_path = (
            self.output_directory
            / f"{experiment_name}_{timestamp}.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                experiment,
                file,
                indent=4,
                default=self._json_default,
            )

        return output_path

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, Path):
            return str(value)

        raise TypeError(
            f"Object of type {type(value).__name__} "
            "is not JSON serializable."
        )
