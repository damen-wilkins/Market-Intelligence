from statistics import median

import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.model_selection import TimeSeriesSplit

from app.training.classification_sequence_preprocessor import (
    ClassificationSequencePreprocessor,
)
from app.training.torch_classification_trainer import (
    TorchClassificationTrainer,
)
from app.training.torch_reproducibility import (
    TorchReproducibility,
)
from app.training.xlstm_classifier_model import (
    XLSTMClassifier,
)


class XLSTMClassifierParameterSelector:
    def __init__(
        self,
        feature_columns: list[str],
        n_splits: int = 3,
        n_trials: int = 20,
        max_epochs: int = 60,
        patience: int = 8,
        random_state: int = 42,
        device: str | None = None,
    ):
        if not feature_columns:
            raise ValueError(
                "Feature columns are required."
            )

        if n_splits < 2:
            raise ValueError(
                "At least two walk-forward splits are required."
            )

        if n_trials <= 0:
            raise ValueError(
                "Number of trials must be greater than zero."
            )

        self.feature_columns = list(
            feature_columns
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
    ) -> dict:
        training_data = (
            training_data
            .sort_values(
                "target_date"
            )
            .reset_index(
                drop=True
            )
        )

        splitter = TimeSeriesSplit(
            n_splits=self.n_splits
        )

        sampler = optuna.samplers.TPESampler(
            seed=self.random_state
        )

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        def objective(
            trial: optuna.Trial,
        ) -> float:
            sequence_length = (
                trial.suggest_categorical(
                    "sequence_length",
                    [
                        10,
                        20,
                        40,
                        60,
                    ],
                )
            )

            embedding_dim = (
                trial.suggest_categorical(
                    "embedding_dim",
                    [
                        32,
                        64,
                        128,
                    ],
                )
            )

            num_blocks = trial.suggest_int(
                "num_blocks",
                1,
                3,
            )

            num_heads = (
                trial.suggest_categorical(
                    "num_heads",
                    [
                        4,
                        8,
                    ],
                )
            )

            conv1d_kernel_size = (
                trial.suggest_categorical(
                    "conv1d_kernel_size",
                    [
                        2,
                        4,
                    ],
                )
            )

            qkv_proj_blocksize = (
                trial.suggest_categorical(
                    "qkv_proj_blocksize",
                    [
                        4,
                        8,
                    ],
                )
            )

            proj_factor = (
                trial.suggest_categorical(
                    "proj_factor",
                    [
                        1.5,
                        2.0,
                    ],
                )
            )

            dropout = trial.suggest_float(
                "dropout",
                0.0,
                0.35,
            )

            learning_rate = (
                trial.suggest_float(
                    "learning_rate",
                    1e-4,
                    3e-3,
                    log=True,
                )
            )

            batch_size = (
                trial.suggest_categorical(
                    "batch_size",
                    [
                        32,
                        64,
                        128,
                    ],
                )
            )

            weight_decay = (
                trial.suggest_float(
                    "weight_decay",
                    1e-7,
                    1e-2,
                    log=True,
                )
            )

            gradient_clip = (
                trial.suggest_categorical(
                    "gradient_clip",
                    [
                        0.5,
                        1.0,
                        2.0,
                    ],
                )
            )

            loss_name = (
                trial.suggest_categorical(
                    "loss_name",
                    [
                        "focal",
                        "weighted_cross_entropy",
                    ],
                )
            )

            if loss_name == "focal":
                focal_gamma = (
                    trial.suggest_float(
                        "focal_gamma",
                        1.0,
                        3.0,
                    )
                )
            else:
                focal_gamma = 2.0

            fold_scores = []
            fold_best_epochs = []

            for fold_number, (
                train_indices,
                validation_indices,
            ) in enumerate(
                splitter.split(
                    training_data
                ),
                start=1,
            ):
                fold_train = (
                    training_data
                    .iloc[
                        train_indices
                    ]
                    .reset_index(
                        drop=True
                    )
                )

                fold_validation = (
                    training_data
                    .iloc[
                        validation_indices
                    ]
                    .reset_index(
                        drop=True
                    )
                )

                preprocessor = (
                    ClassificationSequencePreprocessor(
                        feature_columns=(
                            self.feature_columns
                        ),
                        sequence_length=(
                            sequence_length
                        ),
                    )
                )

                preprocessor.fit(
                    fold_train
                )

                training_sequences = (
                    preprocessor
                    .build_training_sequences(
                        fold_train
                    )
                )

                validation_sequences = (
                    preprocessor
                    .build_inference_sequences(
                        history=fold_train,
                        dataframe=fold_validation,
                    )
                )

                fold_seed = (
                    self.random_state
                    + (
                        trial.number
                        * 1000
                    )
                    + fold_number
                )

                TorchReproducibility.configure(
                    seed=fold_seed,
                    deterministic=True,
                )

                model = XLSTMClassifier(
                    input_size=len(
                        self.feature_columns
                    ),
                    context_length=int(
                        sequence_length
                    ),
                    embedding_dim=int(
                        embedding_dim
                    ),
                    num_blocks=int(
                        num_blocks
                    ),
                    num_heads=int(
                        num_heads
                    ),
                    conv1d_kernel_size=int(
                        conv1d_kernel_size
                    ),
                    qkv_proj_blocksize=int(
                        qkv_proj_blocksize
                    ),
                    proj_factor=float(
                        proj_factor
                    ),
                    dropout=float(
                        dropout
                    ),
                    num_classes=3,
                )

                trainer = (
                    TorchClassificationTrainer(
                        learning_rate=float(
                            learning_rate
                        ),
                        batch_size=int(
                            batch_size
                        ),
                        max_epochs=(
                            self.max_epochs
                        ),
                        patience=(
                            self.patience
                        ),
                        loss_name=loss_name,
                        focal_gamma=float(
                            focal_gamma
                        ),
                        weight_decay=float(
                            weight_decay
                        ),
                        gradient_clip=float(
                            gradient_clip
                        ),
                        seed=fold_seed,
                        deterministic=True,
                        device=self.device,
                    )
                )

                result = trainer.train(
                    model=model,
                    X_train=(
                        training_sequences[
                            "X"
                        ]
                    ),
                    y_train=(
                        training_sequences[
                            "y"
                        ]
                    ),
                    X_validation=(
                        validation_sequences[
                            "X"
                        ]
                    ),
                    y_validation=(
                        validation_sequences[
                            "y"
                        ]
                    ),
                )

                fold_scores.append(
                    float(
                        result[
                            "best_validation_macro_f1"
                        ]
                    )
                )

                fold_best_epochs.append(
                    int(
                        result[
                            "best_epoch"
                        ]
                    )
                )

                trial.report(
                    float(
                        np.mean(
                            fold_scores
                        )
                    ),
                    step=fold_number,
                )

                if trial.should_prune():
                    raise (
                        optuna.TrialPruned()
                    )

            trial.set_user_attr(
                "fold_scores",
                fold_scores,
            )

            trial.set_user_attr(
                "fold_best_epochs",
                fold_best_epochs,
            )

            return float(
                np.mean(
                    fold_scores
                )
            )

        study.optimize(
            objective,
            n_trials=self.n_trials,
            gc_after_trial=True,
        )

        best_trial = study.best_trial

        parameters = dict(
            best_trial.params
        )

        if (
            parameters[
                "loss_name"
            ]
            == "weighted_cross_entropy"
        ):
            parameters[
                "focal_gamma"
            ] = 2.0

        fold_best_epochs = [
            int(value)
            for value in (
                best_trial.user_attrs[
                    "fold_best_epochs"
                ]
            )
        ]

        parameters["epochs"] = max(
            1,
            int(
                round(
                    median(
                        fold_best_epochs
                    )
                )
            ),
        )

        fold_scores = [
            float(value)
            for value in (
                best_trial.user_attrs[
                    "fold_scores"
                ]
            )
        ]

        completed_trials = len(
            [
                trial
                for trial in study.trials
                if (
                    trial.state
                    == TrialState.COMPLETE
                )
            ]
        )

        return {
            "parameters": parameters,
            "cv_macro_f1": float(
                best_trial.value
            ),
            "cv_macro_f1_std": float(
                np.std(
                    fold_scores,
                    ddof=0,
                )
            ),
            "fold_macro_f1": fold_scores,
            "fold_best_epochs": (
                fold_best_epochs
            ),
            "best_trial": int(
                best_trial.number
            ),
            "completed_trials": int(
                completed_trials
            ),
        }