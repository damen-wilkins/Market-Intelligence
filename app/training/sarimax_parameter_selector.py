import itertools
import warnings

import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.training.stationarity_tester import StationarityTester


class SarimaxParameterSelector:
    def __init__(
        self,
        max_order: int = 3,
    ):
        self.max_order = max_order
        self.stationarity_tester = StationarityTester()

    def select(
        self,
        endog: pd.Series,
        exog: pd.DataFrame | None = None,
    ) -> dict:
        endog = endog.dropna()

        if exog is not None:
            exog = exog.loc[endog.index]

        d = self.stationarity_tester.run(endog)["recommended_d"]

        best_results = None
        best_order = None
        best_aicc = float("inf")

        warnings.filterwarnings(
            "ignore",
            category=ConvergenceWarning,
        )

        for p, q in itertools.product(
            range(self.max_order + 1),
            repeat=2,
        ):
            try:
                model = SARIMAX(
                    endog,
                    exog=exog,
                    order=(p, d, q),
                    seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                results = model.fit(disp=False)

                n = results.nobs
                k = len(results.params)

                if (n - k - 1) <= 0:
                    continue

                aicc = (
                    results.aic
                    + (2 * k * (k + 1))
                    / (n - k - 1)
                )

                if aicc < best_aicc:
                    best_aicc = aicc
                    best_order = (p, d, q)
                    best_results = results

            except Exception:
                continue

        return {
            "model": best_results,
            "order": best_order,
            "seasonal_order": (0, 0, 0, 0),
            "aicc": best_aicc,
        }