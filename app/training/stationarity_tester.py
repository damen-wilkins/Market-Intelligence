import pandas as pd
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import adfuller, kpss


class StationarityTester:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def run(self, series: pd.Series) -> dict:
        series = series.dropna()

        adf = adfuller(series)
        kpss_result = kpss(series, regression="c", nlags="auto")
        pp = PhillipsPerron(series)

        adf_p = adf[1]
        kpss_p = kpss_result[1]
        pp_p = pp.pvalue

        if adf_p < self.alpha and pp_p < self.alpha and kpss_p > self.alpha:
            differencing = 0
        else:
            differencing = 1

        return {
            "adf_p_value": adf_p,
            "kpss_p_value": kpss_p,
            "pp_p_value": pp_p,
            "recommended_d": differencing,
        }