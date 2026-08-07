from statistics import median

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import torch

from app.training.lstm_predictor import LSTMPredictor
from app.training.lstm_residual_model import LSTMResidualModel
from app.training.lstm_trainer import LSTMTrainer
from app.training.residual_sequence_preprocessor import (
    ResidualSequencePreprocessor,
)


class LSTMParameterSelector:
    def __init__(
        self,
        n_splits: int = 3,
        n_trials: int = 30,
        max_epochs: int = 100,
        patience: int = 10,
        random_state: int = 42,
        device: str | None = None,
    ):
        if n_splits < 2:
            raise ValueError(
                "LSTM parameter selection requires at least two splits."
            )

        if n_trials <= 0:
            raise ValueError(
                "LSTM parameter selection requires at least one trial."
            )

        self.n_splits = n_splits
        self.n_trials = n_trials
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def select_best_parameters(
        self,
        training_data,
        feature_columns: list[str],
    ) -> dict:
        splitter = TimeSeriesSplit(
            n_splits=self.n_splits
        )

        def objective(trial: optuna.Trial) -> float:
            sequence_length = trial.suggest_categorical(
                "sequence_length",
                [10, 20, 40, 60, 90],
            )
            hidden_size = trial.suggest_categorical(
                "hidden_size",
                [16, 32, 64, 128],
            )
            num_layers = trial.suggest_int(
                "num_layers",
                1,
                3,
            )
            dropout = (
                trial.suggest_float(
                    "dropout",
                    0.0,
                    0.40,
                )
                if num_layers > 1
                else 0.0
            )
            batch_size = trial.suggest_categorical(
                "batch_size",
                [32, 64, 128],
            )
            learning_rate = trial.suggest_float(
                "learning_rate",
                1e-4,
                5e-3,
                log=True,
            )
            weight_decay = trial.suggest_float(
                "weight_decay",
                1e-8,
                1e-3,
                log=True,
            )
            gradient_clip = trial.suggest_float(
                "gradient_clip",
                0.5,
                5.0,
            )

            fold_scores = []
            fold_best_epochs = []

            for fold_number, (
                train_indices,
                validation_indices,
            ) in enumerate(
                splitter.split(training_data),
                start=1,
            ):
                fold_train = training_data.iloc[
                    train_indices
                ].reset_index(drop=True)
                fold_validation = training_data.iloc[
                    validation_indices
                ].reset_index(drop=True)

                preprocessor = ResidualSequencePreprocessor(
                    sequence_length=sequence_length
                ).fit(
                    dataframe=fold_train,
                    feature_columns=feature_columns,
                )
                train_sequences = (
                    preprocessor.build_training_sequences(
                        fold_train
                    )
                )
                validation_sequences = (
                    preprocessor.build_inference_sequences(
                        history=fold_train,
                        dataframe=fold_validation,
                    )
                )

                fold_seed = (
                    self.random_state
                    + trial.number * 100
                    + fold_number
                )
                trainer = LSTMTrainer(
                    random_state=fold_seed,
                    device=self.device,
                )
                model = LSTMResidualModel(
                    input_size=len(
                        preprocessor.feature_columns
                    ),
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                training_result = trainer.train(
                    model=model,
                    training_data=train_sequences,
                    validation_data=validation_sequences,
                    epochs=self.max_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    gradient_clip=gradient_clip,
                    patience=self.patience,
                )

                predictions = LSTMPredictor(
                    device=self.device
                ).predict(
                    model=training_result["model"],
                    dataset=validation_sequences,
                    preprocessor=preprocessor,
                    batch_size=batch_size,
                )
                fold_score = mean_squared_error(
                    predictions["sarimax_residual"],
                    predictions["predicted_residual"],
                )

                fold_scores.append(fold_score)
                fold_best_epochs.append(
                    training_result["best_epoch"]
                )

                running_score = float(
                    np.mean(fold_scores)
                )
                trial.report(
                    running_score,
                    step=fold_number,
                )

                if trial.should_prune():
                    raise optuna.TrialPruned()

                del model
                del training_result

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            trial.set_user_attr(
                "effective_dropout",
                dropout,
            )
            trial.set_user_attr(
                "fold_best_epochs",
                fold_best_epochs,
            )

            return float(np.mean(fold_scores))

        sampler = optuna.samplers.TPESampler(
            seed=self.random_state
        )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
        )
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
        )

        best_parameters = dict(
            study.best_trial.params
        )
        best_parameters["dropout"] = float(
            study.best_trial.user_attrs[
                "effective_dropout"
            ]
        )
        fold_best_epochs = list(
            study.best_trial.user_attrs[
                "fold_best_epochs"
            ]
        )
        best_parameters["epochs"] = max(
            1,
            int(round(median(fold_best_epochs))),
        )

        return {
            "parameters": best_parameters,
            "cv_mse": float(study.best_value),
            "best_trial_number": (
                study.best_trial.number
            ),
            "fold_best_epochs": fold_best_epochs,
            "completed_trials": len(study.trials),
        }
