from xgboost import XGBRegressor

class XGBoostTrainer:
    def train(
        self,
        dataset,
        parameters: dict,
    ):
        X = dataset.drop(
            columns=[
                "sarimax_residual",
            ]
        )

        y = dataset["sarimax_residual"]

        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            **parameters,
        )

        model.fit(X, y)

        return model