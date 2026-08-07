from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from app.training.feature_contract import require_columns


class SarimaxForecastLoader:
    def load(
        self,
        path: str | Path,
        allowed_splits: Sequence[str] = (
            "train",
            "validation",
        ),
    ) -> pd.DataFrame:
        forecast_path = Path(path)

        if not forecast_path.exists():
            raise FileNotFoundError(
                "SARIMAX forecast artifact was not found: "
                f"{forecast_path}"
            )

        forecasts = pd.read_csv(
            forecast_path,
            parse_dates=["trade_date"],
        )

        require_columns(
            forecasts,
            [
                "trade_date",
                "actual_log_return",
                "sarimax_prediction",
                "is_out_of_sample",
                "data_split",
            ],
            "SARIMAX forecast artifact",
        )

        if forecasts["trade_date"].isna().any():
            raise ValueError(
                "SARIMAX forecast artifact contains invalid dates."
            )

        if forecasts["trade_date"].duplicated().any():
            raise ValueError(
                "SARIMAX forecast artifact contains duplicate dates."
            )

        out_of_sample = (
            forecasts["is_out_of_sample"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
        )

        if out_of_sample.isna().any():
            raise ValueError(
                "SARIMAX is_out_of_sample contains invalid values."
            )

        if not out_of_sample.all():
            raise ValueError(
                "Residual learning requires out-of-sample SARIMAX "
                "predictions for every row."
            )

        allowed = set(allowed_splits)
        observed = set(
            forecasts["data_split"].dropna().unique()
        )
        invalid_splits = observed - allowed

        if invalid_splits:
            raise ValueError(
                "SARIMAX forecast artifact contains disallowed splits: "
                f"{sorted(invalid_splits)}"
            )

        forecasts["is_out_of_sample"] = out_of_sample

        return forecasts.sort_values(
            "trade_date"
        ).reset_index(drop=True)
