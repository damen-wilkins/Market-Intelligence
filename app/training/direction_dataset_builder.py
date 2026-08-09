import pandas as pd

from app.training.direction_feature_builder import DirectionFeatureBuilder
from app.training.volatility_direction_label_builder import (
    VolatilityDirectionLabelBuilder,
)


class DirectionDatasetBuilder:
    def __init__(
        self,
        feature_builder: DirectionFeatureBuilder | None = None,
        label_builder: VolatilityDirectionLabelBuilder | None = None,
    ):
        self.feature_builder = feature_builder or DirectionFeatureBuilder()
        self.label_builder = label_builder or VolatilityDirectionLabelBuilder()

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        features = self.feature_builder.build(data)

        labels = self.label_builder.build(
            data[
                [
                    "trade_date",
                    "close",
                ]
            ].copy()
        )

        features = features.rename(
            columns={
                "trade_date": "feature_date",
            }
        )

        dataset = features.merge(
            labels,
            on="feature_date",
            how="inner",
            validate="one_to_one",
        )

        dataset = dataset.sort_values(
            "feature_date"
        ).reset_index(drop=True)

        if dataset.empty:
            raise ValueError(
                "Direction dataset is empty after joining features and labels."
            )

        if dataset["feature_date"].duplicated().any():
            raise ValueError(
                "Direction dataset contains duplicate feature dates."
            )

        if dataset["target_date"].duplicated().any():
            raise ValueError(
                "Direction dataset contains duplicate target dates."
            )

        if not (
            dataset["target_date"]
            > dataset["feature_date"]
        ).all():
            raise ValueError(
                "Every target date must occur after its feature date."
            )

        if dataset.isna().any().any():
            raise ValueError(
                "Direction dataset contains missing values."
            )

        return dataset[
            [
                "feature_date",
                "target_date",
                *self.feature_builder.FEATURE_COLUMNS,
                "future_log_return",
                "rolling_volatility",
                "threshold",
                "direction",
            ]
        ]