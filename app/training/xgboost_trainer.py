from xgboost import XGBRegressor


class XGBoostTrainer:
    def train(
        self,
        X_train,
        y_train,
        parameters: dict,
    ):
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            **parameters,
        )

        model.fit(
            X_train,
            y_train,
        )

        return model