import pandas as pd


class XGBoostPredictor:
    def predict(
        self,
        model,
        dataset: pd.DataFrame,
    ) -> pd.Series:
        features = dataset.drop(
            columns=[
                "sarimax_residual",
            ],
            errors="ignore",
        )

        predictions = model.predict(features)

        return pd.Series(
            predictions,
            index=dataset.index,
            name="predicted_residual",
        )