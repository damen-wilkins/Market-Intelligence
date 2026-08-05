import pandas as pd


class ResidualDataSplitter:
    def split(
        self,
        dataframe: pd.DataFrame | pd.Series,
        train_size: float = 0.80,
        validation_size: float = 0.10,
    ):
        train_end = int(len(dataframe) * train_size)
        validation_end = train_end + int(len(dataframe) * validation_size)

        train = dataframe.iloc[:train_end].reset_index(drop=True)
        validation = dataframe.iloc[train_end:validation_end].reset_index(drop=True)
        test = dataframe.iloc[validation_end:].reset_index(drop=True)

        return train, validation, test