import pandas as pd

class ResidualForecastCorrector:
    def apply(
        self,
        sarimax_predictions: pd.Series,
        predicted_residuals: pd.Series,
    ) -> pd.Series:
        """
        Applies the predicted residuals to the SARIMAX forecasts.
        """

        corrected_predictions = (
            sarimax_predictions.reset_index(drop=True)
            + predicted_residuals.reset_index(drop=True)
        )

        return pd.Series(
            corrected_predictions,
            name="corrected_prediction",
        )