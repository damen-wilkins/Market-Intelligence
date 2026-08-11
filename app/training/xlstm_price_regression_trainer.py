from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from app.training.torch_reproducibility import (
    TorchReproducibility,
)


class XLSTMPriceRegressionTrainer:
    AUTHORS_BACKBONE_CHECKPOINT = "authors_backbone_checkpoint"
    BEST_FULL_CHECKPOINT = "best_full_checkpoint"
    VALID_CHECKPOINT_VARIANTS = {
        AUTHORS_BACKBONE_CHECKPOINT,
        BEST_FULL_CHECKPOINT,
    }

    def __init__(
        self,
        learning_rate: float = 0.0001,
        batch_size: int = 16,
        max_epochs: int = 200,
        patience: int = 40,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        gradient_clip: float = 1.0,
        seed: int = 42,
    ):
        if learning_rate <= 0:
            raise ValueError(
                "Learning rate must be positive."
            )

        if batch_size <= 0:
            raise ValueError(
                "Batch size must be positive."
            )

        if max_epochs <= 0:
            raise ValueError(
                "Maximum epochs must be positive."
            )

        if patience <= 0:
            raise ValueError(
                "Early-stopping patience must be positive."
            )

        if scheduler_patience <= 0:
            raise ValueError(
                "Scheduler patience must be positive."
            )

        if not 0.0 < scheduler_factor < 1.0:
            raise ValueError(
                "Scheduler factor must be between zero and one."
            )

        if gradient_clip <= 0:
            raise ValueError(
                "Gradient clipping threshold must be positive."
            )

        if seed < 0:
            raise ValueError(
                "Seed must be non-negative."
            )

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.gradient_clip = gradient_clip
        self.seed = seed

        if not torch.cuda.is_available():
            raise RuntimeError(
                "The paper xLSTM architecture uses the CUDA sLSTM backend. "
                "A CUDA-capable PyTorch environment is required."
            )

        self.device = torch.device(
            "cuda"
        )

    def train(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
    ) -> dict:
        self._validate_arrays(
            X_train,
            y_train,
            "training",
        )
        self._validate_arrays(
            X_validation,
            y_validation,
            "validation",
        )

        TorchReproducibility.configure(
            seed=self.seed,
            deterministic=False,
        )

        model = model.to(
            self.device
        )

        train_loader = self._build_loader(
            X_train,
            y_train,
            shuffle=True,
        )

        validation_loader = self._build_loader(
            X_validation,
            y_validation,
            shuffle=False,
        )

        loss_function = nn.MSELoss()

        optimizer = Adam(
            model.parameters(),
            lr=self.learning_rate,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )

        best_validation_loss = float(
            "inf"
        )
        best_epoch = 0
        best_full_state = None
        best_backbone_state = None
        epochs_without_improvement = 0
        history = []

        for epoch in range(
            1,
            self.max_epochs + 1,
        ):
            training_loss = self._train_epoch(
                model=model,
                loader=train_loader,
                loss_function=loss_function,
                optimizer=optimizer,
            )

            validation_loss = self._evaluate_loss(
                model=model,
                loader=validation_loader,
                loss_function=loss_function,
            )

            scheduler.step(
                validation_loss
            )

            current_learning_rate = float(
                optimizer.param_groups[0][
                    "lr"
                ]
            )

            history.append(
                {
                    "epoch": epoch,
                    "training_loss": training_loss,
                    "validation_loss": validation_loss,
                    "learning_rate": current_learning_rate,
                }
            )

            print(
                f"Epoch [{epoch}/{self.max_epochs}] "
                f"Loss: {training_loss:.8f}, "
                f"Validation Loss: {validation_loss:.8f}, "
                f"LR: {current_learning_rate:.8f}"
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_full_state = self._cpu_state_dict(
                    model
                )
                best_backbone_state = self._cpu_state_dict(
                    model.backbone
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= self.patience
            ):
                print(
                    "Early stopping."
                )
                break

        if (
            best_full_state is None
            or best_backbone_state is None
        ):
            raise RuntimeError(
                "Training did not produce a valid checkpoint."
            )

        final_full_state = self._cpu_state_dict(
            model
        )

        model.load_state_dict(
            best_full_state
        )
        model = model.to(
            self.device
        )

        return {
            "model": model,
            "best_epoch": best_epoch,
            "best_validation_loss": float(
                best_validation_loss
            ),
            "epochs_trained": len(
                history
            ),
            "history": history,
            "best_full_state_dict": best_full_state,
            "best_backbone_state_dict": best_backbone_state,
            "final_full_state_dict": final_full_state,
        }

    def load_checkpoint_variant(
        self,
        model: nn.Module,
        training_result: dict,
        checkpoint_variant: str,
    ) -> nn.Module:
        if checkpoint_variant not in self.VALID_CHECKPOINT_VARIANTS:
            raise ValueError(
                "Checkpoint variant must be one of "
                f"{sorted(self.VALID_CHECKPOINT_VARIANTS)}."
            )

        if checkpoint_variant == self.BEST_FULL_CHECKPOINT:
            model.load_state_dict(
                training_result[
                    "best_full_state_dict"
                ]
            )
        else:
            model.load_state_dict(
                training_result[
                    "final_full_state_dict"
                ]
            )
            model.backbone.load_state_dict(
                training_result[
                    "best_backbone_state_dict"
                ]
            )

        return model.to(
            self.device
        )

    def predict(
        self,
        model: nn.Module,
        X: np.ndarray,
    ) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(
                "Prediction input must be three-dimensional."
            )

        if len(X) == 0:
            raise ValueError(
                "Prediction input cannot be empty."
            )

        model = model.to(
            self.device
        )
        model.eval()

        dataset = TensorDataset(
            torch.from_numpy(
                X.astype(
                    np.float32,
                    copy=False,
                )
            )
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        predictions = []

        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(
                    self.device,
                    non_blocking=True,
                )

                batch_predictions = model(
                    batch_X
                )

                predictions.append(
                    batch_predictions
                    .detach()
                    .cpu()
                    .numpy()
                )

        return np.concatenate(
            predictions,
            axis=0,
        ).reshape(-1)

    def get_config(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_factor": self.scheduler_factor,
            "gradient_clip": self.gradient_clip,
            "optimizer": "Adam",
            "loss": "MSE",
            "seed": self.seed,
            "device": str(
                self.device
            ),
        }

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_function: nn.Module,
        optimizer: Adam,
    ) -> float:
        model.train()
        batch_losses = []

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(
                self.device,
                non_blocking=True,
            )
            batch_y = batch_y.to(
                self.device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = model(
                batch_X
            )

            loss = loss_function(
                predictions,
                batch_y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.backbone.parameters(),
                max_norm=self.gradient_clip,
            )

            optimizer.step()

            batch_losses.append(
                float(
                    loss.item()
                )
            )

        return float(
            np.mean(
                batch_losses
            )
        )

    def _evaluate_loss(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_function: nn.Module,
    ) -> float:
        model.eval()
        batch_losses = []

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(
                    self.device,
                    non_blocking=True,
                )
                batch_y = batch_y.to(
                    self.device,
                    non_blocking=True,
                )

                predictions = model(
                    batch_X
                )

                loss = loss_function(
                    predictions,
                    batch_y,
                )

                batch_losses.append(
                    float(
                        loss.item()
                    )
                )

        return float(
            np.mean(
                batch_losses
            )
        )

    def _build_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(
            self.seed
        )

        dataset = TensorDataset(
            torch.from_numpy(
                X.astype(
                    np.float32,
                    copy=False,
                )
            ),
            torch.from_numpy(
                y.astype(
                    np.float32,
                    copy=False,
                )
            ),
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=(
                generator
                if shuffle
                else None
            ),
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

    @staticmethod
    def _cpu_state_dict(
        module: nn.Module,
    ) -> dict:
        return deepcopy(
            {
                key: value.detach().cpu()
                for key, value in module.state_dict().items()
            }
        )

    @staticmethod
    def _validate_arrays(
        X: np.ndarray,
        y: np.ndarray,
        name: str,
    ) -> None:
        if X.ndim != 3:
            raise ValueError(
                f"{name.capitalize()} X must be three-dimensional."
            )

        if y.ndim != 2:
            raise ValueError(
                f"{name.capitalize()} y must be two-dimensional."
            )

        if len(X) != len(y):
            raise ValueError(
                f"{name.capitalize()} X and y lengths do not match."
            )

        if len(X) == 0:
            raise ValueError(
                f"{name.capitalize()} data cannot be empty."
            )

        if not np.isfinite(
            X
        ).all() or not np.isfinite(
            y
        ).all():
            raise ValueError(
                f"{name.capitalize()} data contains non-finite values."
            )
