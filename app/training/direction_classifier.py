import pandas as pd


class DirectionClassifier:
    def __init__(
        self,
        lower_threshold: float | None = None,
        upper_threshold: float | None = None,
    ):
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def fit(
        self,
        actual_returns: pd.Series,
    ) -> "DirectionClassifier":
        returns = pd.Series(actual_returns).dropna()

        lower_threshold = returns[
            returns < 0
        ].median()
        upper_threshold = returns[
            returns > 0
        ].median()

        if pd.isna(lower_threshold) or pd.isna(upper_threshold):
            raise ValueError(
                "Training data cannot produce valid direction thresholds."
            )

        self.lower_threshold = float(lower_threshold)
        self.upper_threshold = float(upper_threshold)

        return self

    def classify(
        self,
        values: pd.Series,
    ) -> pd.Series:
        self._validate_fitted()

        series = pd.Series(values).reset_index(drop=True)
        labels = pd.Series(
            "FLAT",
            index=series.index,
            dtype="object",
        )

        labels.loc[
            series < self.lower_threshold
        ] = "DOWN"
        labels.loc[
            series > self.upper_threshold
        ] = "UP"

        return labels

    def get_state(self) -> dict[str, float]:
        self._validate_fitted()

        return {
            "lower": self.lower_threshold,
            "upper": self.upper_threshold,
        }

    def _validate_fitted(self) -> None:
        if (
            self.lower_threshold is None
            or self.upper_threshold is None
        ):
            raise ValueError(
                "DirectionClassifier must be fitted before use."
            )
