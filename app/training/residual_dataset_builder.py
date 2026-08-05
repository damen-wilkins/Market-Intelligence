import pandas as pd


class ResidualDatasetBuilder:
    def build(
        self,
        features: pd.DataFrame,
        predictions: pd.Series,
    ) -> pd.DataFrame:
        dataset = features.copy().reset_index(drop=True)

        dataset["sarimax_prediction"] = (
            pd.Series(predictions)
            .reset_index(drop=True)
        )

        dataset["sarimax_residual"] = (
            dataset["log_return"]
            - dataset["sarimax_prediction"]
        )

        dataset = dataset.drop(
            columns=[
                "ticker",
                "trade_date",
                "return_1d",
                "return_1w",
                "return_1m",
                "return_1y",
                "label_1d",
                "label_1w",
                "label_1m",
                "label_1y",
            ],
            errors="ignore",
        )

        dataset = dataset.dropna().reset_index(drop=True)

        return dataset