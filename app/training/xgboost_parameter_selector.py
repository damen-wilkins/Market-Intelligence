import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor


class XGBoostParameterSelector:
    def __init__(
        self,
        n_splits: int = 5,
        n_trials: int = 100,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.random_state = random_state

    def select_best_parameters(
        self,
        X,
        y,
    ) -> dict:
        splitter = TimeSeriesSplit(
            n_splits=self.n_splits
        )

        def objective(trial):
            parameters = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    100,
                    1000,
                    step=50,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    2,
                    8,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.005,
                    0.20,
                    log=True,
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    0.60,
                    1.00,
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    0.60,
                    1.00,
                ),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight",
                    0.001,
                    10.0,
                    log=True,
                ),
                "gamma": trial.suggest_float(
                    "gamma",
                    0.0,
                    0.0001,
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    1e-10,
                    0.01,
                    log=True,
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    1e-4,
                    10.0,
                    log=True,
                ),
            }

            fold_scores = []

            for train_index, validation_index in splitter.split(X):
                X_train = X.iloc[train_index]
                X_validation = X.iloc[validation_index]
                y_train = y.iloc[train_index]
                y_validation = y.iloc[validation_index]

                model = XGBRegressor(
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=self.random_state,
                    n_jobs=-1,
                    **parameters,
                )

                model.fit(
                    X_train,
                    y_train,
                )

                predictions = model.predict(
                    X_validation
                )

                fold_scores.append(
                    mean_squared_error(
                        y_validation,
                        predictions,
                    )
                )

            return sum(fold_scores) / len(fold_scores)

        sampler = optuna.samplers.TPESampler(
            seed=self.random_state
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )

        study.optimize(
            objective,
            n_trials=self.n_trials,
        )

        return study.best_params