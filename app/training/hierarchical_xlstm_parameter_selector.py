from gc import collect
from statistics import median

import numpy as np
import optuna
import torch
from optuna.trial import TrialState
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from app.training.hierarchical_sequence_preprocessor import (
    HierarchicalSequencePreprocessor,
)
from app.training.torch_classification_predictor import (
    TorchClassificationPredictor,
)
from app.training.torch_classification_trainer import (
    TorchClassificationTrainer,
)
from app.training.torch_reproducibility import (
    TorchReproducibility,
)
from app.training.xlstm_classifier_model import XLSTMClassifier


class HierarchicalXLSTMParameterSelector:
    def __init__(
        self,
        feature_columns: list[str],
        task: str,
        n_splits: int = 3,
        n_trials: int = 20,
        max_epochs: int = 60,
        patience: int = 8,
        random_state: int = 42,
        device: str | None = None,
        objective_metric: str = "macro_f1",
        study_name: str | None = None,
        storage_url: str | None = None,
    ):
        if not feature_columns:
            raise ValueError(
                "At least one feature column is required."
            )

        if task not in {
            "move",
            "direction",
        }:
            raise ValueError(
                "Task must be 'move' or 'direction'."
            )

        if objective_metric not in {
            "macro_f1",
            "roc_auc",
        }:
            raise ValueError(
                "Objective metric must be 'macro_f1' or 'roc_auc'."
            )

        if n_splits < 2:
            raise ValueError(
                "At least two time-series splits are required."
            )

        if n_trials <= 0:
            raise ValueError(
                "Number of trials must be greater than zero."
            )

        if max_epochs <= 0:
            raise ValueError(
                "Maximum epochs must be greater than zero."
            )

        if patience <= 0:
            raise ValueError(
                "Patience must be greater than zero."
            )

        self.feature_columns = list(feature_columns)
        self.task = task
        self.n_splits = int(n_splits)
        self.n_trials = int(n_trials)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.random_state = int(random_state)
        self.device = device
        self.objective_metric = objective_metric
        self.study_name = study_name
        self.storage_url = storage_url

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
            study_name=self.study_name,
            storage=self.storage_url,
            load_if_exists=(
                self.study_name is not None
                and self.storage_url is not None
            ),
        )

        def objective(
            trial: optuna.Trial,
        ) -> float:
            parameters = self._suggest_parameters(
                trial
            )

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
                    HierarchicalSequencePreprocessor(
                        feature_columns=self.feature_columns,
                        sequence_length=int(
                            parameters[
                                "sequence_length"
                            ]
                        ),
                    )
                )

                preprocessor.fit(
                    fold_train
                )

                training_sequences = (
                    preprocessor
                    .build_training_sequences(
                        dataframe=fold_train,
                        task=self.task,
                    )
                )

                validation_sequences = (
                    preprocessor
                    .build_inference_sequences(
                        history=fold_train,
                        dataframe=fold_validation,
                        task=self.task,
                        include_all=False,
                    )
                )

                fold_seed = (
                    self.random_state
                    + trial.number * 1000
                    + fold_number
                )

                TorchReproducibility.configure(
                    seed=fold_seed,
                    deterministic=True,
                )

                model = self._build_model(
                    parameters
                )

                trainer = self._build_trainer(
                    parameters=parameters,
                    seed=fold_seed,
                    max_epochs=self.max_epochs,
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
                            "best_validation_score"
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

                self._cleanup_cuda()

                if trial.should_prune():
                    raise optuna.TrialPruned()

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

        existing_trials = len(
            study.trials
        )

        remaining_trials = max(
            0,
            self.n_trials
            - existing_trials,
        )

        if remaining_trials > 0:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                gc_after_trial=True,
            )

        if not study.trials:
            raise RuntimeError(
                "Optuna study did not contain any trials."
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
            for value in best_trial.user_attrs[
                "fold_best_epochs"
            ]
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
            for value in best_trial.user_attrs[
                "fold_scores"
            ]
        ]

        threshold_selection = (
            self._select_oof_threshold(
                training_data=training_data,
                parameters=parameters,
            )
        )

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

        selection_result = {
            "task": self.task,
            "objective_metric": (
                self.objective_metric
            ),
            "parameters": parameters,
            "cv_objective_score": float(
                best_trial.value
            ),
            "cv_objective_score_std": float(
                np.std(
                    fold_scores,
                    ddof=0,
                )
            ),
            "fold_objective_scores": (
                fold_scores
            ),
            "fold_best_epochs": (
                fold_best_epochs
            ),
            "decision_threshold": float(
                threshold_selection[
                    "threshold"
                ]
            ),
            "threshold_oof_macro_f1": float(
                threshold_selection[
                    "macro_f1"
                ]
            ),
            "threshold_oof_balanced_accuracy": float(
                threshold_selection[
                    "balanced_accuracy"
                ]
            ),
            "threshold_oof_roc_auc": float(
                threshold_selection[
                    "roc_auc"
                ]
            ),
            "threshold_oof_roc_auc_fold_mean": float(
                threshold_selection[
                    "roc_auc_fold_mean"
                ]
            ),
            "threshold_oof_roc_auc_fold_std": float(
                threshold_selection[
                    "roc_auc_fold_std"
                ]
            ),
            "threshold_oof_rows": int(
                threshold_selection[
                    "rows"
                ]
            ),
            "best_trial": int(
                best_trial.number
            ),
            "completed_trials": int(
                completed_trials
            ),
            "study_name": self.study_name,
            "storage_url": self.storage_url,
        }

        if self.objective_metric == "macro_f1":
            selection_result.update(
                {
                    "cv_macro_f1": float(
                        best_trial.value
                    ),
                    "cv_macro_f1_std": float(
                        np.std(
                            fold_scores,
                            ddof=0,
                        )
                    ),
                    "fold_macro_f1": (
                        fold_scores
                    ),
                }
            )
        else:
            selection_result.update(
                {
                    "cv_roc_auc": float(
                        best_trial.value
                    ),
                    "cv_roc_auc_std": float(
                        np.std(
                            fold_scores,
                            ddof=0,
                        )
                    ),
                    "fold_roc_auc": (
                        fold_scores
                    ),
                }
            )

        return selection_result

    def _select_oof_threshold(
        self,
        training_data,
        parameters: dict,
    ) -> dict:
        splitter = TimeSeriesSplit(
            n_splits=self.n_splits
        )

        actual_batches = []
        probability_batches = []
        fold_roc_auc = []

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
                HierarchicalSequencePreprocessor(
                    feature_columns=self.feature_columns,
                    sequence_length=int(
                        parameters[
                            "sequence_length"
                        ]
                    ),
                )
            )

            preprocessor.fit(
                fold_train
            )

            training_sequences = (
                preprocessor
                .build_training_sequences(
                    dataframe=fold_train,
                    task=self.task,
                )
            )

            validation_sequences = (
                preprocessor
                .build_inference_sequences(
                    history=fold_train,
                    dataframe=fold_validation,
                    task=self.task,
                    include_all=False,
                )
            )

            fold_seed = (
                self.random_state
                + 100000
                + fold_number
            )

            TorchReproducibility.configure(
                seed=fold_seed,
                deterministic=True,
            )

            model = self._build_model(
                parameters
            )

            trainer = self._build_trainer(
                parameters=parameters,
                seed=fold_seed,
                max_epochs=int(
                    parameters[
                        "epochs"
                    ]
                ),
            )

            training_result = (
                trainer.fit_fixed_epochs(
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
                    epochs=int(
                        parameters[
                            "epochs"
                        ]
                    ),
                )
            )

            prediction_result = (
                TorchClassificationPredictor(
                    batch_size=int(
                        parameters[
                            "batch_size"
                        ]
                    ),
                    device=self.device,
                )
                .predict(
                    model=(
                        training_result[
                            "model"
                        ]
                    ),
                    X=(
                        validation_sequences[
                            "X"
                        ]
                    ),
                )
            )

            actual_batches.append(
                validation_sequences[
                    "y"
                ]
            )

            fold_actual = (
                validation_sequences[
                    "y"
                ]
            )

            fold_probabilities = (
                prediction_result[
                    "probabilities"
                ][
                    :,
                    1,
                ]
            )

            probability_batches.append(
                fold_probabilities
            )

            if len(
                np.unique(
                    fold_actual
                )
            ) < 2:
                fold_roc_auc.append(
                    0.5
                )
            else:
                fold_roc_auc.append(
                    float(
                        roc_auc_score(
                            fold_actual,
                            fold_probabilities,
                        )
                    )
                )

            self._cleanup_cuda()

        actual = np.concatenate(
            actual_batches
        ).astype(
            np.int64
        )

        probabilities = np.concatenate(
            probability_batches
        ).astype(
            np.float64
        )

        threshold_result = (
            self.select_probability_threshold(
                actual=actual,
                positive_probabilities=probabilities,
            )
        )

        if len(
            np.unique(
                actual
            )
        ) < 2:
            overall_roc_auc = 0.5
        else:
            overall_roc_auc = float(
                roc_auc_score(
                    actual,
                    probabilities,
                )
            )

        return {
            **threshold_result,
            "roc_auc": (
                overall_roc_auc
            ),
            "roc_auc_fold_mean": float(
                np.mean(
                    fold_roc_auc
                )
            ),
            "roc_auc_fold_std": float(
                np.std(
                    fold_roc_auc,
                    ddof=0,
                )
            ),
        }

    @staticmethod
    def select_probability_threshold(
        actual: np.ndarray,
        positive_probabilities: np.ndarray,
    ) -> dict:
        actual = np.asarray(
            actual,
            dtype=np.int64,
        )

        positive_probabilities = np.asarray(
            positive_probabilities,
            dtype=np.float64,
        )

        if actual.ndim != 1:
            raise ValueError(
                "Actual labels must be one-dimensional."
            )

        if positive_probabilities.ndim != 1:
            raise ValueError(
                "Positive probabilities must be one-dimensional."
            )

        if len(actual) != len(
            positive_probabilities
        ):
            raise ValueError(
                "Labels and probabilities must contain the same rows."
            )

        if len(actual) == 0:
            raise ValueError(
                "Threshold selection data cannot be empty."
            )

        if not set(
            np.unique(
                actual
            )
        ).issubset(
            {
                0,
                1,
            }
        ):
            raise ValueError(
                "Threshold labels must be binary."
            )

        if (
            not np.isfinite(
                positive_probabilities
            ).all()
            or (
                positive_probabilities
                < 0.0
            ).any()
            or (
                positive_probabilities
                > 1.0
            ).any()
        ):
            raise ValueError(
                "Probabilities must be finite values between zero and one."
            )

        unique_probabilities = np.unique(
            positive_probabilities
        )

        if len(unique_probabilities) == 1:
            candidate_thresholds = np.asarray(
                [
                    0.0,
                    0.5,
                    1.0,
                ],
                dtype=np.float64,
            )
        else:
            midpoints = (
                unique_probabilities[:-1]
                + unique_probabilities[1:]
            ) / 2.0

            candidate_thresholds = np.unique(
                np.concatenate(
                    [
                        np.asarray(
                            [0.0, 0.5, 1.0],
                            dtype=np.float64,
                        ),
                        midpoints,
                    ]
                )
            )

        best = None

        for threshold in candidate_thresholds:
            predicted = (
                positive_probabilities
                >= threshold
            ).astype(
                np.int64
            )

            macro_f1 = float(
                f1_score(
                    actual,
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            )

            balanced_accuracy = float(
                balanced_accuracy_score(
                    actual,
                    predicted,
                )
            )

            candidate = {
                "threshold": float(
                    threshold
                ),
                "macro_f1": macro_f1,
                "balanced_accuracy": balanced_accuracy,
                "distance_from_half": abs(
                    float(threshold)
                    - 0.5
                ),
            }

            if best is None:
                best = candidate
                continue

            candidate_key = (
                candidate[
                    "macro_f1"
                ],
                candidate[
                    "balanced_accuracy"
                ],
                -candidate[
                    "distance_from_half"
                ],
            )

            best_key = (
                best[
                    "macro_f1"
                ],
                best[
                    "balanced_accuracy"
                ],
                -best[
                    "distance_from_half"
                ],
            )

            if candidate_key > best_key:
                best = candidate

        if best is None:
            raise RuntimeError(
                "No decision threshold could be selected."
            )

        return {
            "threshold": float(
                best[
                    "threshold"
                ]
            ),
            "macro_f1": float(
                best[
                    "macro_f1"
                ]
            ),
            "balanced_accuracy": float(
                best[
                    "balanced_accuracy"
                ]
            ),
            "rows": int(
                len(actual)
            ),
        }

    def _suggest_parameters(
        self,
        trial: optuna.Trial,
    ) -> dict:
        parameters = {
            "sequence_length": trial.suggest_categorical(
                "sequence_length",
                [
                    10,
                    20,
                    40,
                    60,
                ],
            ),
            "embedding_dim": trial.suggest_categorical(
                "embedding_dim",
                [
                    32,
                    64,
                    128,
                ],
            ),
            "num_blocks": trial.suggest_int(
                "num_blocks",
                1,
                3,
            ),
            "num_heads": trial.suggest_categorical(
                "num_heads",
                [
                    4,
                    8,
                ],
            ),
            "conv1d_kernel_size": trial.suggest_categorical(
                "conv1d_kernel_size",
                [
                    2,
                    4,
                ],
            ),
            "qkv_proj_blocksize": trial.suggest_categorical(
                "qkv_proj_blocksize",
                [
                    4,
                    8,
                ],
            ),
            "proj_factor": trial.suggest_categorical(
                "proj_factor",
                [
                    1.5,
                    2.0,
                ],
            ),
            "dropout": trial.suggest_float(
                "dropout",
                0.0,
                0.35,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                1e-4,
                3e-3,
                log=True,
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size",
                [
                    32,
                    64,
                    128,
                ],
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay",
                1e-7,
                1e-2,
                log=True,
            ),
            "gradient_clip": trial.suggest_categorical(
                "gradient_clip",
                [
                    0.5,
                    1.0,
                    2.0,
                ],
            ),
            "loss_name": trial.suggest_categorical(
                "loss_name",
                [
                    "focal",
                    "weighted_cross_entropy",
                ],
            ),
        }

        if parameters[
            "loss_name"
        ] == "focal":
            parameters[
                "focal_gamma"
            ] = trial.suggest_float(
                "focal_gamma",
                1.0,
                3.0,
            )
        else:
            parameters[
                "focal_gamma"
            ] = 2.0

        return parameters

    def _build_model(
        self,
        parameters: dict,
    ) -> XLSTMClassifier:
        return XLSTMClassifier(
            input_size=len(
                self.feature_columns
            ),
            context_length=int(
                parameters[
                    "sequence_length"
                ]
            ),
            embedding_dim=int(
                parameters[
                    "embedding_dim"
                ]
            ),
            num_blocks=int(
                parameters[
                    "num_blocks"
                ]
            ),
            num_heads=int(
                parameters[
                    "num_heads"
                ]
            ),
            conv1d_kernel_size=int(
                parameters[
                    "conv1d_kernel_size"
                ]
            ),
            qkv_proj_blocksize=int(
                parameters[
                    "qkv_proj_blocksize"
                ]
            ),
            proj_factor=float(
                parameters[
                    "proj_factor"
                ]
            ),
            dropout=float(
                parameters[
                    "dropout"
                ]
            ),
            num_classes=2,
        )

    def _build_trainer(
        self,
        parameters: dict,
        seed: int,
        max_epochs: int,
    ) -> TorchClassificationTrainer:
        return TorchClassificationTrainer(
            learning_rate=float(
                parameters[
                    "learning_rate"
                ]
            ),
            batch_size=int(
                parameters[
                    "batch_size"
                ]
            ),
            max_epochs=int(
                max_epochs
            ),
            patience=self.patience,
            loss_name=str(
                parameters[
                    "loss_name"
                ]
            ),
            focal_gamma=float(
                parameters[
                    "focal_gamma"
                ]
            ),
            weight_decay=float(
                parameters[
                    "weight_decay"
                ]
            ),
            gradient_clip=float(
                parameters[
                    "gradient_clip"
                ]
            ),
            seed=int(
                seed
            ),
            deterministic=True,
            num_classes=2,
            device=self.device,
            selection_metric=(
                self.objective_metric
            ),
        )

    def _cleanup_cuda(self) -> None:
        collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
