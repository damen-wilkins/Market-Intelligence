import numpy as np
import pandas as pd
import pandas_ta as ta

class IndicatorCalculator:
    def calculate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.copy()

        dataframe["sma_10"] = ta.sma(dataframe["close"], length=10)
        dataframe["sma_20"] = ta.sma(dataframe["close"], length=20)
        dataframe["sma_50"] = ta.sma(dataframe["close"], length=50)

        dataframe["ema_20"] = ta.ema(dataframe["close"], length=20)

        dataframe["rsi_14"] = ta.rsi(dataframe["close"], length=14)

        macd = ta.macd(dataframe["close"])
        dataframe["macd"] = macd["MACD_12_26_9"]
        dataframe["macd_signal"] = macd["MACDs_12_26_9"]
        dataframe["macd_histogram"] = macd["MACDh_12_26_9"]

        bollinger = ta.bbands(dataframe["close"], length=20)

        dataframe["bollinger_upper"] = bollinger["BBU_20_2.0_2.0"]
        dataframe["bollinger_middle"] = bollinger["BBM_20_2.0_2.0"]
        dataframe["bollinger_lower"] = bollinger["BBL_20_2.0_2.0"]

        dataframe["daily_return"] = dataframe["close"].pct_change()

        dataframe["log_return"] = np.log(
            dataframe["close"] / dataframe["close"].shift(1)
        )

        return dataframe