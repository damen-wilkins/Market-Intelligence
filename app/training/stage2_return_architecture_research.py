from dataclasses import dataclass
from gc import collect
from math import erf, sqrt

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier, XGBRegressor

from app.training.torch_reproducibility import TorchReproducibility


@dataclass(frozen=True)
class ReturnArchitectureSpec:
    name: str
    family: str
    target_name: str
    output_mode: str


class Stage2ReturnSequencePreprocessor:
    def __init__(self, feature_columns: list[str], sequence_length: int):
        if not feature_columns:
            raise ValueError("At least one feature column is required.")
        if sequence_length <= 0:
            raise ValueError("Sequence length must be greater than zero.")
        self.feature_columns = list(feature_columns)
        self.sequence_length = int(sequence_length)
        self.feature_scaler = StandardScaler()
        self.target_mean = 0.0
        self.target_scale = 1.0
        self._fitted = False

    def fit(self, dataframe: pd.DataFrame, target_column: str):
        data = self._sort(dataframe)
        self.feature_scaler.fit(data[self.feature_columns])
        move = data.loc[data["direction"] != "FLAT", target_column].astype(float)
        if move.empty:
            raise ValueError("Training data contains no MOVE rows.")
        self.target_mean = float(move.mean())
        scale = float(move.std(ddof=0))
        self.target_scale = scale if scale > 1e-12 else 1.0
        self._fitted = True
        return self

    def build_training_sequences(self, dataframe: pd.DataFrame, target_column: str) -> dict:
        self._require_fitted()
        data = self._sort(dataframe)
        scaled = self.feature_scaler.transform(data[self.feature_columns])
        X, y, raw_y, dates = [], [], [], []
        for row_index in range(self.sequence_length - 1, len(data)):
            if str(data.loc[row_index, "direction"]) == "FLAT":
                continue
            start = row_index - self.sequence_length + 1
            value = float(data.loc[row_index, target_column])
            X.append(scaled[start : row_index + 1])
            y.append((value - self.target_mean) / self.target_scale)
            raw_y.append(value)
            dates.append(data.loc[row_index, "target_date"])
        return self._package(X, y, raw_y, dates)

    def build_inference_sequences(
        self,
        history: pd.DataFrame,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> dict:
        self._require_fitted()
        history = self._sort(history)
        data = self._sort(dataframe)
        if history["target_date"].max() >= data["target_date"].min():
            raise ValueError("History must occur strictly before inference data.")
        context_rows = self.sequence_length - 1
        if len(history) < context_rows:
            raise ValueError("Not enough history for inference sequences.")
        combined = pd.concat([history.tail(context_rows), data], ignore_index=True)
        scaled = self.feature_scaler.transform(combined[self.feature_columns])
        X, y, raw_y, dates = [], [], [], []
        for offset in range(len(data)):
            row_index = context_rows + offset
            if str(combined.loc[row_index, "direction"]) == "FLAT":
                continue
            start = row_index - self.sequence_length + 1
            value = float(combined.loc[row_index, target_column])
            X.append(scaled[start : row_index + 1])
            y.append((value - self.target_mean) / self.target_scale)
            raw_y.append(value)
            dates.append(combined.loc[row_index, "target_date"])
        return self._package(X, y, raw_y, dates)

    def inverse_target(self, standardized: np.ndarray) -> np.ndarray:
        return np.asarray(standardized, dtype=np.float64) * self.target_scale + self.target_mean

    @staticmethod
    def _sort(dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.sort_values("target_date").reset_index(drop=True)

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before sequence construction.")

    @staticmethod
    def _package(X, y, raw_y, dates) -> dict:
        if not X:
            raise ValueError("No MOVE sequences were constructed.")
        return {
            "X": np.asarray(X, dtype=np.float32),
            "y": np.asarray(y, dtype=np.float32),
            "raw_y": np.asarray(raw_y, dtype=np.float64),
            "target_dates": pd.DatetimeIndex(dates),
        }


class TorchReturnTrainer:
    def __init__(
        self,
        learning_rate: float,
        batch_size: int,
        max_epochs: int,
        patience: int,
        weight_decay: float,
        gradient_clip: float,
        loss_name: str,
        seed: int,
        device: str | None = None,
    ):
        if loss_name not in {"huber", "gaussian_nll"}:
            raise ValueError("Loss must be 'huber' or 'gaussian_nll'.")
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.weight_decay = float(weight_decay)
        self.gradient_clip = float(gradient_clip)
        self.loss_name = loss_name
        self.seed = int(seed)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def train(self, model: nn.Module, X_train, y_train, X_validation, y_validation) -> dict:
        TorchReproducibility.configure(seed=self.seed, deterministic=True)
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        train_loader = self._loader(X_train, y_train, shuffle=True)
        validation_loader = self._loader(X_validation, y_validation, shuffle=False)
        best_state = None
        best_loss = float("inf")
        best_epoch = 1
        stale = 0
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                output = model(X_batch)
                loss = self._loss(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                optimizer.step()
            validation_loss = self._validation_loss(model, validation_loader)
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("Return model training did not produce a checkpoint.")
        model.load_state_dict(best_state)
        return {"model": model, "best_epoch": best_epoch, "best_validation_loss": best_loss}

    def fit_fixed_epochs(self, model: nn.Module, X_train, y_train, epochs: int) -> nn.Module:
        TorchReproducibility.configure(seed=self.seed, deterministic=True)
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        train_loader = self._loader(X_train, y_train, shuffle=True)
        for _ in range(int(epochs)):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                output = model(X_batch)
                loss = self._loss(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                optimizer.step()
        return model

    def predict(self, model: nn.Module, X: np.ndarray) -> dict:
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        model = model.to(self.device)
        model.eval()
        means, scales = [], []
        with torch.no_grad():
            for (X_batch,) in loader:
                output = model(X_batch.to(self.device)).detach().cpu().numpy()
                if self.loss_name == "gaussian_nll":
                    means.append(output[:, 0])
                    scales.append(np.exp(output[:, 1]))
                else:
                    means.append(output[:, 0])
        result = {"mean": np.concatenate(means).astype(np.float64)}
        if scales:
            result["scale"] = np.concatenate(scales).astype(np.float64)
        return result

    def _loader(self, X, y, shuffle: bool):
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
            ),
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
        )

    def _loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "huber":
            return nn.functional.smooth_l1_loss(output[:, 0], target)
        mean = output[:, 0]
        log_scale = output[:, 1].clamp(min=-5.0, max=3.0)
        variance = torch.exp(2.0 * log_scale)
        return torch.mean(log_scale + 0.5 * ((target - mean) ** 2) / variance)

    def _validation_loss(self, model: nn.Module, loader: DataLoader) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                output = model(X_batch.to(self.device))
                losses.append(float(self._loss(output, y_batch.to(self.device)).item()))
        return float(np.mean(losses))


def verify_stage2_orientation(dataframe: pd.DataFrame) -> dict:
    required = {"direction", "future_log_return", "threshold"}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Orientation audit missing columns: {missing}")
    data = dataframe.dropna(subset=list(required)).copy()
    direction = data["direction"].astype(str)
    future_return = data["future_log_return"].astype(float)
    threshold = data["threshold"].astype(float)
    violations = {
        "UP": int(((direction == "UP") & ~(future_return > threshold)).sum()),
        "DOWN": int(((direction == "DOWN") & ~(future_return < -threshold)).sum()),
        "FLAT": int(((direction == "FLAT") & ~((future_return >= -threshold) & (future_return <= threshold))).sum()),
    }
    if any(violations.values()):
        raise ValueError(f"Direction-label orientation violations detected: {violations}")
    return {
        "rows": int(len(data)),
        "up_rows": int((direction == "UP").sum()),
        "down_rows": int((direction == "DOWN").sum()),
        "flat_rows": int((direction == "FLAT").sum()),
        "violations": violations,
    }


def auc_orientation(actual: np.ndarray, positive_score: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.int64)
    score = np.asarray(positive_score, dtype=np.float64)
    if len(np.unique(actual)) < 2:
        return {"direct_auc": 0.5, "inverted_auc": 0.5}
    direct = float(roc_auc_score(actual, score))
    inverted = float(roc_auc_score(actual, -score))
    return {"direct_auc": direct, "inverted_auc": inverted}


def normal_up_probability(mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-8)
    z = mean / scale
    erf_values = np.vectorize(erf)(z / sqrt(2.0))
    return 0.5 * (1.0 + erf_values)


def regression_direction_metrics(
    actual_return: np.ndarray,
    predicted_return: np.ndarray,
    direction_score: np.ndarray | None = None,
) -> dict:
    actual_return = np.asarray(actual_return, dtype=np.float64)
    predicted_return = np.asarray(predicted_return, dtype=np.float64)
    actual_direction = (actual_return > 0.0).astype(np.int64)
    score = predicted_return if direction_score is None else np.asarray(direction_score, dtype=np.float64)
    orientation = auc_orientation(actual_direction, score)
    predicted_direction = (predicted_return > 0.0).astype(np.int64)
    absolute_weights = np.abs(actual_return)
    weighted_accuracy = float(
        np.average(predicted_direction == actual_direction, weights=absolute_weights)
    ) if absolute_weights.sum() > 0 else float(np.mean(predicted_direction == actual_direction))
    correlation = 0.0
    if np.std(actual_return) > 0 and np.std(predicted_return) > 0:
        correlation = float(np.corrcoef(actual_return, predicted_return)[0, 1])
    return {
        **orientation,
        "balanced_accuracy": float(balanced_accuracy_score(actual_direction, predicted_direction)),
        "macro_f1": float(f1_score(actual_direction, predicted_direction, average="macro", zero_division=0)),
        "sign_accuracy": float(np.mean(predicted_direction == actual_direction)),
        "magnitude_weighted_sign_accuracy": weighted_accuracy,
        "mae": float(mean_absolute_error(actual_return, predicted_return)),
        "rmse": float(mean_squared_error(actual_return, predicted_return) ** 0.5),
        "return_correlation": correlation,
    }


def cleanup_cuda() -> None:
    collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class XLSTMReturnParameterSelector:
    def __init__(
        self,
        feature_columns: list[str],
        target_column: str,
        output_mode: str,
        n_splits: int,
        n_trials: int,
        max_epochs: int,
        patience: int,
        random_state: int,
        study_name: str,
        storage_url: str,
    ):
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.output_mode = output_mode
        self.n_splits = int(n_splits)
        self.n_trials = int(n_trials)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.random_state = int(random_state)
        self.study_name = study_name
        self.storage_url = storage_url

    def select(self, training_data: pd.DataFrame) -> dict:
        data = training_data.sort_values("target_date").reset_index(drop=True)
        splitter = TimeSeriesSplit(n_splits=self.n_splits)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
            study_name=self.study_name,
            storage=self.storage_url,
            load_if_exists=True,
        )

        def objective(trial):
            parameters = self._suggest(trial)
            fold_auc, fold_epochs = [], []
            for fold_number, (train_index, validation_index) in enumerate(splitter.split(data), start=1):
                fold_train = data.iloc[train_index].reset_index(drop=True)
                fold_validation = data.iloc[validation_index].reset_index(drop=True)
                preprocessor = Stage2ReturnSequencePreprocessor(
                    self.feature_columns,
                    int(parameters["sequence_length"]),
                ).fit(fold_train, self.target_column)
                train_sequences = preprocessor.build_training_sequences(fold_train, self.target_column)
                validation_sequences = preprocessor.build_inference_sequences(
                    fold_train,
                    fold_validation,
                    self.target_column,
                )
                seed = self.random_state + trial.number * 1000 + fold_number
                model = self._build_model(parameters)
                trainer = self._build_trainer(parameters, seed)
                result = trainer.train(
                    model,
                    train_sequences["X"],
                    train_sequences["y"],
                    validation_sequences["X"],
                    validation_sequences["y"],
                )
                prediction = trainer.predict(result["model"], validation_sequences["X"])
                predicted_mean = preprocessor.inverse_target(prediction["mean"])
                if self.output_mode == "gaussian":
                    predicted_scale = prediction["scale"] * preprocessor.target_scale
                    score = normal_up_probability(predicted_mean, predicted_scale)
                else:
                    score = predicted_mean
                actual_direction = (validation_sequences["raw_y"] > 0.0).astype(np.int64)
                fold_auc.append(float(roc_auc_score(actual_direction, score)))
                fold_epochs.append(int(result["best_epoch"]))
                trial.report(float(np.mean(fold_auc)), step=fold_number)
                cleanup_cuda()
                if trial.should_prune():
                    raise optuna.TrialPruned()
            trial.set_user_attr("fold_auc", fold_auc)
            trial.set_user_attr("fold_epochs", fold_epochs)
            return float(np.mean(fold_auc))

        completed = len(study.trials)
        remaining = max(0, self.n_trials - completed)
        if remaining:
            study.optimize(objective, n_trials=remaining, gc_after_trial=True)
        best = study.best_trial
        parameters = dict(best.params)
        epochs = best.user_attrs["fold_epochs"]
        parameters["epochs"] = max(1, int(round(float(np.median(epochs)))))
        return {
            "parameters": parameters,
            "cv_auc": float(best.value),
            "cv_auc_fold_std": float(np.std(best.user_attrs["fold_auc"], ddof=0)),
            "fold_auc": [float(value) for value in best.user_attrs["fold_auc"]],
            "epochs": parameters["epochs"],
            "best_trial": int(best.number),
        }

    @staticmethod
    def _suggest(trial) -> dict:
        return {
            "sequence_length": trial.suggest_categorical("sequence_length", [10, 20, 40, 60]),
            "embedding_dim": trial.suggest_categorical("embedding_dim", [32, 64, 128]),
            "num_blocks": trial.suggest_int("num_blocks", 1, 3),
            "num_heads": trial.suggest_categorical("num_heads", [4, 8]),
            "conv1d_kernel_size": trial.suggest_categorical("conv1d_kernel_size", [2, 4]),
            "qkv_proj_blocksize": trial.suggest_categorical("qkv_proj_blocksize", [4, 8]),
            "proj_factor": trial.suggest_categorical("proj_factor", [1.5, 2.0]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.35),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
            "gradient_clip": trial.suggest_categorical("gradient_clip", [0.5, 1.0, 2.0]),
        }

    def _build_model(self, parameters: dict):
        from app.training.xlstm_return_model import XLSTMReturnModel

        return XLSTMReturnModel(
            input_size=len(self.feature_columns),
            context_length=int(parameters["sequence_length"]),
            embedding_dim=int(parameters["embedding_dim"]),
            num_blocks=int(parameters["num_blocks"]),
            num_heads=int(parameters["num_heads"]),
            conv1d_kernel_size=int(parameters["conv1d_kernel_size"]),
            qkv_proj_blocksize=int(parameters["qkv_proj_blocksize"]),
            proj_factor=float(parameters["proj_factor"]),
            dropout=float(parameters["dropout"]),
            output_mode=self.output_mode,
        )

    def _build_trainer(self, parameters: dict, seed: int) -> TorchReturnTrainer:
        loss_name = "gaussian_nll" if self.output_mode == "gaussian" else "huber"
        return TorchReturnTrainer(
            learning_rate=float(parameters["learning_rate"]),
            batch_size=int(parameters["batch_size"]),
            max_epochs=self.max_epochs,
            patience=self.patience,
            weight_decay=float(parameters["weight_decay"]),
            gradient_clip=float(parameters["gradient_clip"]),
            loss_name=loss_name,
            seed=seed,
        )


def xgboost_parameter_search(
    training_data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    task: str,
    n_splits: int,
    n_trials: int,
    random_state: int,
    study_name: str,
    storage_url: str,
) -> dict:
    if task not in {"classification", "regression"}:
        raise ValueError("XGBoost task must be classification or regression.")
    move = training_data.loc[training_data["direction"] != "FLAT"].reset_index(drop=True)
    X = move[feature_columns]
    y = (move["direction"] == "UP").astype(int) if task == "classification" else move[target_column].astype(float)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    def objective(trial):
        parameters = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 0.001),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-10, 0.1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
        }
        aucs = []
        for train_index, validation_index in splitter.split(X):
            if task == "classification":
                model = XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="auc",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=-1,
                    **parameters,
                )
                model.fit(X.iloc[train_index], y.iloc[train_index])
                score = model.predict_proba(X.iloc[validation_index])[:, 1]
                actual = y.iloc[validation_index].to_numpy(dtype=np.int64)
            else:
                model = XGBRegressor(
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=-1,
                    **parameters,
                )
                model.fit(X.iloc[train_index], y.iloc[train_index])
                score = model.predict(X.iloc[validation_index])
                actual = (move.iloc[validation_index]["direction"] == "UP").astype(int).to_numpy()
            aucs.append(float(roc_auc_score(actual, score)))
        trial.set_user_attr("fold_auc", aucs)
        return float(np.mean(aucs))

    completed = len(study.trials)
    remaining = max(0, n_trials - completed)
    if remaining:
        study.optimize(objective, n_trials=remaining)
    best = study.best_trial
    return {
        "parameters": dict(best.params),
        "cv_auc": float(best.value),
        "cv_auc_fold_std": float(np.std(best.user_attrs["fold_auc"], ddof=0)),
        "fold_auc": [float(value) for value in best.user_attrs["fold_auc"]],
        "best_trial": int(best.number),
    }
