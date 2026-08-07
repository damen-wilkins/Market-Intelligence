from copy import deepcopy
import random

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from app.training.residual_sequence_preprocessor import (
    ResidualSequenceDataset,
)


class LSTMTrainer:
    def __init__(
        self,
        random_state: int = 42,
        device: str | None = None,
    ):
        self.random_state = random_state
        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )
        self.set_deterministic_seed()

    def train(
        self,
        model: nn.Module,
        training_data: ResidualSequenceDataset,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        gradient_clip: float,
        validation_data: ResidualSequenceDataset | None = None,
        patience: int = 10,
        min_delta: float = 1e-6,
    ) -> dict:
        self._validate_parameters(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            patience=patience,
        )
        self.set_deterministic_seed()

        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        loss_function = nn.MSELoss()
        training_loader = self._build_loader(
            training_data,
            batch_size=batch_size,
        )
        validation_loader = (
            self._build_loader(
                validation_data,
                batch_size=batch_size,
            )
            if validation_data is not None
            else None
        )

        history = {
            "training_loss": [],
            "validation_loss": [],
        }
        best_state = None
        best_validation_loss = float("inf")
        best_epoch = epochs
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            training_loss = self._train_epoch(
                model=model,
                data_loader=training_loader,
                optimizer=optimizer,
                loss_function=loss_function,
                gradient_clip=gradient_clip,
            )
            history["training_loss"].append(
                training_loss
            )

            if validation_loader is None:
                continue

            validation_loss = self._evaluate_loss(
                model=model,
                data_loader=validation_loader,
                loss_function=loss_function,
            )
            history["validation_loss"].append(
                validation_loss
            )

            if (
                validation_loss
                < best_validation_loss - min_delta
            ):
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = deepcopy(
                    model.state_dict()
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        if validation_loader is not None:
            if best_state is None:
                raise RuntimeError(
                    "LSTM training did not produce a valid "
                    "validation checkpoint."
                )

            model.load_state_dict(best_state)

        return {
            "model": model,
            "history": history,
            "best_epoch": best_epoch,
            "best_validation_loss": (
                best_validation_loss
                if validation_loader is not None
                else None
            ),
            "epochs_completed": len(
                history["training_loss"]
            ),
            "device": str(self.device),
        }

    def _train_epoch(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        optimizer,
        loss_function: nn.Module,
        gradient_clip: float,
    ) -> float:
        model.train()
        total_loss = 0.0
        total_observations = 0

        for sequences, targets in data_loader:
            sequences = sequences.to(
                self.device,
                non_blocking=True,
            )
            targets = targets.to(
                self.device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)
            predictions = model(sequences)
            loss = loss_function(
                predictions,
                targets,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "LSTM training produced a non-finite loss."
                )

            loss.backward()
            clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip,
            )
            optimizer.step()

            batch_size = len(targets)
            total_loss += float(loss.item()) * batch_size
            total_observations += batch_size

        return total_loss / total_observations

    def _evaluate_loss(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        loss_function: nn.Module,
    ) -> float:
        model.eval()
        total_loss = 0.0
        total_observations = 0

        with torch.no_grad():
            for sequences, targets in data_loader:
                sequences = sequences.to(
                    self.device,
                    non_blocking=True,
                )
                targets = targets.to(
                    self.device,
                    non_blocking=True,
                )
                predictions = model(sequences)
                loss = loss_function(
                    predictions,
                    targets,
                )

                batch_size = len(targets)
                total_loss += float(loss.item()) * batch_size
                total_observations += batch_size

        return total_loss / total_observations

    def _build_loader(
        self,
        dataset: ResidualSequenceDataset,
        batch_size: int,
    ) -> DataLoader:
        if len(dataset) == 0:
            raise ValueError(
                "LSTM datasets cannot be empty."
            )

        tensor_dataset = TensorDataset(
            torch.from_numpy(dataset.sequences),
            torch.from_numpy(dataset.targets),
        )

        return DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(
                self.device.type == "cuda"
            ),
            drop_last=False,
        )

    def set_deterministic_seed(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                self.random_state
            )

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )

    def _validate_parameters(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        gradient_clip: float,
        patience: int,
    ) -> None:
        if epochs <= 0:
            raise ValueError(
                "Training epochs must be greater than zero."
            )

        if batch_size <= 0:
            raise ValueError(
                "Batch size must be greater than zero."
            )

        if learning_rate <= 0:
            raise ValueError(
                "Learning rate must be greater than zero."
            )

        if weight_decay < 0:
            raise ValueError(
                "Weight decay cannot be negative."
            )

        if gradient_clip <= 0:
            raise ValueError(
                "Gradient clipping threshold must be positive."
            )

        if patience <= 0:
            raise ValueError(
                "Early-stopping patience must be positive."
            )
