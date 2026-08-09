import pandas as pd


class DateAwareDataSplitter:
    def __init__(
        self,
        train_size: float = 0.70,
        validation_size: float = 0.15,
    ):
        if train_size <= 0 or validation_size <= 0:
            raise ValueError(
                "Train and validation sizes must be greater than zero."
            )

        if train_size + validation_size >= 1:
            raise ValueError(
                "Train and validation sizes must leave room for a test split."
            )

        self.train_size = train_size
        self.validation_size = validation_size

    def split(
        self,
        dataframe: pd.DataFrame,
        reference_dates: pd.Series | None = None,
        date_column: str = "trade_date",
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        if date_column not in dataframe.columns:
            raise ValueError(
                f"Dataframe must contain a {date_column} column."
            )

        data = dataframe.copy()
        data[date_column] = pd.to_datetime(
            data[date_column]
        )

        if data[date_column].isna().any():
            raise ValueError(
                f"Dataframe contains invalid {date_column} values."
            )

        if data[date_column].duplicated().any():
            raise ValueError(
                f"Dataframe contains duplicate {date_column} values."
            )

        data = data.sort_values(
            date_column
        ).reset_index(drop=True)

        if reference_dates is None:
            dates = data[date_column].copy()
        else:
            dates = pd.to_datetime(
                pd.Series(reference_dates)
            ).dropna()

            dates = (
                dates
                .drop_duplicates()
                .sort_values()
                .reset_index(drop=True)
            )

        if len(dates) < 3:
            raise ValueError(
                "At least three unique dates are required."
            )

        train_end_index = int(
            len(dates) * self.train_size
        )

        validation_end_index = int(
            len(dates)
            * (
                self.train_size
                + self.validation_size
            )
        )

        if train_end_index <= 0:
            raise ValueError(
                "Training split contains no dates."
            )

        if validation_end_index <= train_end_index:
            raise ValueError(
                "Validation split contains no dates."
            )

        if validation_end_index >= len(dates):
            raise ValueError(
                "Test split contains no dates."
            )

        validation_start_date = dates.iloc[
            train_end_index
        ]

        test_start_date = dates.iloc[
            validation_end_index
        ]

        train = data.loc[
            data[date_column] < validation_start_date
        ].copy()

        validation = data.loc[
            (
                data[date_column]
                >= validation_start_date
            )
            & (
                data[date_column]
                < test_start_date
            )
        ].copy()

        test = data.loc[
            data[date_column] >= test_start_date
        ].copy()

        train = train.reset_index(drop=True)
        validation = validation.reset_index(drop=True)
        test = test.reset_index(drop=True)

        if train.empty:
            raise ValueError(
                "Training split is empty."
            )

        if validation.empty:
            raise ValueError(
                "Validation split is empty."
            )

        if test.empty:
            raise ValueError(
                "Test split is empty."
            )

        return train, validation, test