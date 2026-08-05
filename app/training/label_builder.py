import pandas as pd

class LabelBuilder:
    HORIZONS = {
        "1d": 1,
        "1w": 5,
        "1m": 21,
        "1y": 252,
    }
    def build_labels(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.copy()

        for horizon_name, periods in self.HORIZONS.items():
            future_close = dataframe["close"].shift(-periods)

            returns = (
                future_close - dataframe["close"]
            ) / dataframe["close"]

            dataframe[f"return_{horizon_name}"] = returns

            upper = returns[returns > 0].median()
            lower = returns[returns < 0].median()
            labels = pd.Series(index=dataframe.index, dtype="object")
            labels.loc[returns > upper] = "UP"
            labels.loc[returns < lower] = "DOWN"
            labels.loc[(returns >= lower) & (returns <= upper)] = "FLAT"
            dataframe[f"label_{horizon_name}"] = labels

        return dataframe