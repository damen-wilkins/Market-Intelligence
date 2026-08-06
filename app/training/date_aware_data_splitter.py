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
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        if "trade_date" not in dataframe.columns:
            raise ValueError(
                "Dataframe must contain a trade_date column."
            )

        data = dataframe.copy()
        data["trade_date"] = pd.to_datetime(
            data["trade_date"]
        )

        if data["trade_date"].isna().any():
            raise ValueError(
                "Dataframe contains invalid trade dates."
            )

        if data["trade_date"].duplicated().any():
            raise ValueError(
                "Dataframe contains duplicate trade dates."
            )

        data = data.sort_values(
            "trade_date"
        ).reset_index(drop=True)

        if reference_dates is None:
            dates = data["trade_date"].copy()
        else:
            dates = pd.to_datetime(
                pd.Series(reference_dates)
            ).dropna()

            dates = dates.drop_duplicates().sort_values(
            ).reset_index(drop=True)

        if len(dates) < 3:
            raise ValueError(
                "At least three unique trade dates are required."
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
            data["trade_date"] < validation_start_date
        ].copy()

        validation = data.loc[
            (
                data["trade_date"]
                >= validation_start_date
            )
            & (
                data["trade_date"]
                < test_start_date
            )
        ].copy()

        test = data.loc[
            data["trade_date"] >= test_start_date
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

        return train, validation, test