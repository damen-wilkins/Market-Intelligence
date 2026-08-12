import copy

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.training.focal_loss import FocalLoss
from app.training.torch_reproducibility import (
    TorchReproducibility,
)


class TorchClassificationTrainer:
    def __init__(
        self,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        loss_name: str = "focal",
        focal_gamma: float = 2.0,
        weight_decay: float = 0.0,
        gradient_clip: float = 1.0,
        seed: int = 42,
        deterministic: bool = True,
        num_classes: int = 3,
        device: str | None = None,
        selection_metric: str = "macro_f1",
    ):
        if learning_rate <= 0:
            raise ValueError(
                "Learning rate must be greater than zero."
            )

        if batch_size <= 0:
            raise ValueError(
                "Batch size must be greater than zero."
            )

        if max_epochs <= 0:
            raise ValueError(
                "Maximum epochs must be greater than zero."
            )

        if patience <= 0:
            raise ValueError(
                "Patience must be greater than zero."
            )

        if weight_decay < 0:
            raise ValueError(
                "Weight decay cannot be negative."
            )

        if gradient_clip <= 0:
            raise ValueError(
                "Gradient clip must be greater than zero."
            )

        if num_classes <= 1:
            raise ValueError(
                "Number of classes must be greater than one."
            )

        if loss_name not in {
            "focal",
            "weighted_cross_entropy",
        }:
            raise ValueError(
                "Loss name must be 'focal' or "
                "'weighted_cross_entropy'."
            )

        if selection_metric not in {
            "macro_f1",
            "roc_auc",
        }:
            raise ValueError(
                "Selection metric must be 'macro_f1' or 'roc_auc'."
            )

        if (
            selection_metric == "roc_auc"
            and num_classes != 2
        ):
            raise ValueError(
                "ROC AUC model selection requires exactly two classes."
            )

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.loss_name = loss_name
        self.focal_gamma = focal_gamma
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.seed = seed
        self.deterministic = deterministic
        self.num_classes = num_classes
        self.selection_metric = selection_metric

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(
                device
            )

    def train(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
    ) -> dict:
        TorchReproducibility.configure(
            seed=self.seed,
            deterministic=self.deterministic,
        )

        self._validate_arrays(
            X_train,
            y_train,
        )

        self._validate_arrays(
            X_validation,
            y_validation,
        )

        X_train_tensor = torch.tensor(
            X_train,
            dtype=torch.float32,
        )

        y_train_tensor = torch.tensor(
            y_train,
            dtype=torch.long,
        )

        X_validation_tensor = torch.tensor(
            X_validation,
            dtype=torch.float32,
        ).to(self.device)

        y_validation_tensor = torch.tensor(
            y_validation,
            dtype=torch.long,
        ).to(self.device)

        class_weights = self._calculate_class_weights(
            y_train_tensor
        ).to(self.device)

        criterion = self._build_loss(
            class_weights
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        loader = self._build_loader(
            X_train_tensor,
            y_train_tensor,
        )

        model = model.to(
            self.device
        )

        best_state = None
        best_selection_score = -np.inf
        best_macro_f1 = -np.inf
        best_roc_auc = None
        best_epoch = 0
        epochs_without_improvement = 0
        history = []

        for epoch in range(
            1,
            self.max_epochs + 1,
        ):
            train_loss = self._train_epoch(
                model=model,
                loader=loader,
                criterion=criterion,
                optimizer=optimizer,
            )

            model.eval()

            with torch.no_grad():
                validation_logits = model(
                    X_validation_tensor
                )

                validation_loss = criterion(
                    validation_logits,
                    y_validation_tensor,
                ).item()

                validation_probabilities = (
                    torch.softmax(
                        validation_logits,
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )

                predictions = (
                    validation_logits
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )

            validation_macro_f1 = f1_score(
                y_validation,
                predictions,
                labels=list(
                    range(
                        self.num_classes
                    )
                ),
                average="macro",
                zero_division=0,
            )

            validation_roc_auc = None

            if self.num_classes == 2:
                if len(
                    np.unique(
                        y_validation
                    )
                ) < 2:
                    validation_roc_auc = 0.5
                else:
                    validation_roc_auc = float(
                        roc_auc_score(
                            y_validation,
                            validation_probabilities[
                                :,
                                1,
                            ],
                        )
                    )

            selection_score = (
                float(
                    validation_macro_f1
                )
                if self.selection_metric
                == "macro_f1"
                else float(
                    validation_roc_auc
                )
            )

            history_row = {
                "epoch": epoch,
                "train_loss": float(
                    train_loss
                ),
                "validation_loss": float(
                    validation_loss
                ),
                "validation_macro_f1": float(
                    validation_macro_f1
                ),
                "selection_metric": (
                    self.selection_metric
                ),
                "selection_score": float(
                    selection_score
                ),
            }

            if validation_roc_auc is not None:
                history_row[
                    "validation_roc_auc"
                ] = float(
                    validation_roc_auc
                )

            history.append(
                history_row
            )

            if (
                selection_score
                > best_selection_score
            ):
                best_selection_score = (
                    selection_score
                )

                best_macro_f1 = float(
                    validation_macro_f1
                )

                best_roc_auc = (
                    None
                    if validation_roc_auc
                    is None
                    else float(
                        validation_roc_auc
                    )
                )

                best_epoch = epoch

                best_state = copy.deepcopy(
                    model.state_dict()
                )

                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= self.patience
            ):
                break

        if best_state is None:
            raise RuntimeError(
                "Training did not produce "
                "a valid model state."
            )

        model.load_state_dict(
            best_state
        )

        model.eval()

        return {
            "model": model,
            "best_epoch": int(
                best_epoch
            ),
            "best_validation_score": float(
                best_selection_score
            ),
            "best_validation_macro_f1": float(
                best_macro_f1
            ),
            "best_validation_roc_auc": (
                None
                if best_roc_auc is None
                else float(
                    best_roc_auc
                )
            ),
            "selection_metric": (
                self.selection_metric
            ),
            "history": history,
            "class_weights": (
                class_weights
                .detach()
                .cpu()
                .numpy()
            ),
            "loss_name": self.loss_name,
            "device": str(
                self.device
            ),
        }

    def fit_fixed_epochs(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
    ) -> dict:
        if epochs <= 0:
            raise ValueError(
                "Epochs must be greater than zero."
            )

        TorchReproducibility.configure(
            seed=self.seed,
            deterministic=self.deterministic,
        )

        self._validate_arrays(
            X_train,
            y_train,
        )

        X_train_tensor = torch.tensor(
            X_train,
            dtype=torch.float32,
        )

        y_train_tensor = torch.tensor(
            y_train,
            dtype=torch.long,
        )

        class_weights = self._calculate_class_weights(
            y_train_tensor
        ).to(self.device)

        criterion = self._build_loss(
            class_weights
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        loader = self._build_loader(
            X_train_tensor,
            y_train_tensor,
        )

        model = model.to(
            self.device
        )

        history = []

        for epoch in range(
            1,
            epochs + 1,
        ):
            train_loss = self._train_epoch(
                model=model,
                loader=loader,
                criterion=criterion,
                optimizer=optimizer,
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(
                        train_loss
                    ),
                }
            )

        model.eval()

        return {
            "model": model,
            "epochs": int(
                epochs
            ),
            "history": history,
            "class_weights": (
                class_weights
                .detach()
                .cpu()
                .numpy()
            ),
            "loss_name": self.loss_name,
            "device": str(
                self.device
            ),
        }

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        model.train()

        total_loss = 0.0
        total_rows = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(
                self.device
            )

            y_batch = y_batch.to(
                self.device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch,
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Training loss became non-finite."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=self.gradient_clip,
            )

            optimizer.step()

            batch_rows = int(
                X_batch.shape[0]
            )

            total_loss += (
                float(loss.item())
                * batch_rows
            )

            total_rows += batch_rows

        return (
            total_loss
            / total_rows
        )

    def _build_loader(
        self,
        X_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
    ) -> DataLoader:
        generator = torch.Generator()

        generator.manual_seed(
            self.seed
        )

        return DataLoader(
            TensorDataset(
                X_tensor,
                y_tensor,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

    def _build_loss(
        self,
        class_weights: torch.Tensor,
    ) -> nn.Module:
        if self.loss_name == "focal":
            return FocalLoss(
                alpha=class_weights,
                gamma=self.focal_gamma,
            )

        return nn.CrossEntropyLoss(
            weight=class_weights
        )

    def _calculate_class_weights(
        self,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        class_counts = torch.bincount(
            y_train,
            minlength=self.num_classes,
        ).float()

        if len(class_counts) != self.num_classes:
            raise ValueError(
                "Training labels contain a class index outside "
                "the configured class range."
            )

        if (
            class_counts == 0
        ).any():
            raise ValueError(
                "Training data must contain every configured class."
            )

        total_rows = class_counts.sum()

        return (
            total_rows
            / (
                len(class_counts)
                * class_counts
            )
        )

    def _validate_arrays(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        if X.ndim != 3:
            raise ValueError(
                "X must have shape "
                "[samples, sequence_length, features]."
            )

        if y.ndim != 1:
            raise ValueError(
                "y must be one-dimensional."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain "
                "the same number of observations."
            )

        if len(X) == 0:
            raise ValueError(
                "Training arrays cannot be empty."
            )
        if not np.issubdtype(
            y.dtype,
            np.integer,
        ):
            raise ValueError(
                "Classification labels must be integers."
            )

        if y.min() < 0 or y.max() >= self.num_classes:
            raise ValueError(
                "Classification labels contain a class index outside "
                "the configured class range."
            )
