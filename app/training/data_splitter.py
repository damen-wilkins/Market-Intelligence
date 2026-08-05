import pandas as pd


class DataSplitter:
    def split(
        self,
        dataframe: pd.DataFrame,
        train_size: float = 0.70,
        validation_size: float = 0.15,
    ):
        label_columns = [
            "label_1d",
            "label_1w",
            "label_1m",
            "label_1y",
        ]

        dataframe = dataframe.dropna(subset=label_columns).reset_index(drop=True)

        rows = len(dataframe)

        train_end = int(rows * train_size)
        validation_end = train_end + int(rows * validation_size)

        train = dataframe.iloc[:train_end].copy()
        validation = dataframe.iloc[train_end:validation_end].copy()
        test = dataframe.iloc[validation_end:].copy()

        return train, validation, test