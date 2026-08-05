import pandas as pd


class SarimaxPredictor:
    def predict(
        self,
        model,
        start: int,
        end: int,
        exog: pd.DataFrame | None = None,
    ) -> pd.Series:
        predictions = model.predict(
            start=start,
            end=end,
            exog=exog,
        )

        return predictions

    def residuals(
        self,
        model,
    ) -> pd.Series:
        return model.resid