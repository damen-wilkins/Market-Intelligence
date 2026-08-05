from datetime import datetime
from pathlib import Path
import json


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
        timestamp = datetime.utcnow().strftime(
            "%Y%m%d_%H%M%S"
        )

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
            )

        return output_path