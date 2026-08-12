from datetime import datetime, timezone
from gc import collect
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit

from app.training.date_aware_data_splitter import DateAwareDataSplitter
from app.training.experiment_tracker import ExperimentTracker
from app.training.hierarchical_sequence_preprocessor import (
    HierarchicalSequencePreprocessor,
)
from app.training.hierarchical_stage_feature_research import (
    binary_probability_metrics,
)
from app.training.hierarchical_target_feature_research import (
    binary_metrics,
    load_latest_hierarchical_experiment,
    target_distribution,
)
from app.training.hierarchical_xlstm_parameter_selector import (
    HierarchicalXLSTMParameterSelector,
)
from app.training.stage1_wide_feature_builder import (
    Stage1WideFeatureBuilder,
)
from app.training.stage1_wide_signal_search import (
    SignalCandidate,
    build_pair_candidates,
    build_single_candidates,
    expand_beam_candidates,
    select_beam,
    univariate_feature_auc_screen,
)
from app.training.torch_classification_predictor import (
    TorchClassificationPredictor,
)
from app.training.torch_classification_trainer import (
    TorchClassificationTrainer,
)
from app.training.torch_reproducibility import TorchReproducibility
from app.training.volatility_direction_label_builder import (
    VolatilityDirectionLabelBuilder,
)
from app.training.xlstm_classifier_model import XLSTMClassifier
from database.stage2_signal_data_repository import (
    Stage2SignalDataRepository,
)


TICKER = "SPY"
TARGET_VOLATILITY_WINDOW = 40
TARGET_THRESHOLD_MULTIPLIER = 0.45
CV_SPLITS = 3
RANDOM_STATE = 42
EARLY_STOPPING_PATIENCE = 8
TOP_SINGLE_GROUPS_FOR_PAIRS = 10
BEAM_WIDTH = 10
MAX_GROUP_DEPTH = 6
MIN_TRAINING_ROWS = 2500
EXPERIMENT_DIRECTORY = Path("experiments")
EXPERIMENT_NAME = "xlstm_stage1_long_history_optimization_v2"
MODEL_NAME = "xlstm_stage1_long_history_optimization_v2"
CHECKPOINT_PATH = (
    EXPERIMENT_DIRECTORY
    / "stage1_long_history_search_v2_checkpoint.json"
)

OPTUNA_TRIALS = 50
MAX_SELECTION_EPOCHS = 80
OPTUNA_PATIENCE = 10
OPTUNA_SHORTLIST_SIZE = 3
OPTUNA_STORAGE_URL = (
    "sqlite:///experiments/optuna_stage1_long_history_optimization.db"
)
OPTIMIZATION_PROGRESS_PATH = (
    EXPERIMENT_DIRECTORY
    / "stage1_long_history_optimization_v2_progress.json"
)



def build_model_config(
    parameters: dict,
    feature_count: int,
) -> dict:
    return {
        "input_size": int(feature_count),
        "context_length": int(parameters["sequence_length"]),
        "embedding_dim": int(parameters["embedding_dim"]),
        "num_blocks": int(parameters["num_blocks"]),
        "num_heads": int(parameters["num_heads"]),
        "conv1d_kernel_size": int(parameters["conv1d_kernel_size"]),
        "qkv_proj_blocksize": int(parameters["qkv_proj_blocksize"]),
        "proj_factor": float(parameters["proj_factor"]),
        "dropout": float(parameters["dropout"]),
        "num_classes": 2,
    }


def build_trainer(
    parameters: dict,
    seed: int,
) -> TorchClassificationTrainer:
    return TorchClassificationTrainer(
        learning_rate=float(parameters["learning_rate"]),
        batch_size=int(parameters["batch_size"]),
        max_epochs=int(parameters["epochs"]),
        patience=EARLY_STOPPING_PATIENCE,
        loss_name=str(parameters["loss_name"]),
        focal_gamma=float(parameters["focal_gamma"]),
        weight_decay=float(parameters["weight_decay"]),
        gradient_clip=float(parameters["gradient_clip"]),
        seed=int(seed),
        deterministic=True,
        num_classes=2,
    )


def cleanup_cuda() -> None:
    collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_stage1_fold(
    fold_train: pd.DataFrame,
    fold_validation: pd.DataFrame,
    feature_columns: list[str],
    parameters: dict,
    seed: int,
) -> dict:
    preprocessor = HierarchicalSequencePreprocessor(
        feature_columns=feature_columns,
        sequence_length=int(parameters["sequence_length"]),
    )
    preprocessor.fit(fold_train)

    training_sequences = preprocessor.build_training_sequences(
        dataframe=fold_train,
        task="move",
    )

    validation_sequences = preprocessor.build_inference_sequences(
        history=fold_train,
        dataframe=fold_validation,
        task="move",
        include_all=True,
    )

    TorchReproducibility.configure(
        seed=seed,
        deterministic=True,
    )

    model = XLSTMClassifier(
        **build_model_config(
            parameters=parameters,
            feature_count=len(feature_columns),
        )
    )

    training_result = build_trainer(
        parameters=parameters,
        seed=seed,
    ).fit_fixed_epochs(
        model=model,
        X_train=training_sequences["X"],
        y_train=training_sequences["y"],
        epochs=int(parameters["epochs"]),
    )

    prediction_result = TorchClassificationPredictor(
        batch_size=int(parameters["batch_size"])
    ).predict(
        model=training_result["model"],
        X=validation_sequences["X"],
    )

    result = {
        "actual": validation_sequences["y"].astype(np.int64),
        "target_dates": pd.DatetimeIndex(
            validation_sequences["target_dates"]
        ),
        "move_probabilities": prediction_result[
            "probabilities"
        ][:, 1].astype(np.float64),
    }

    del model
    del training_result
    del prediction_result
    del training_sequences
    del validation_sequences
    del preprocessor
    cleanup_cuda()

    return result


def evaluate_candidate(
    training_data: pd.DataFrame,
    feature_columns: list[str],
    parameters: dict,
) -> dict:
    splitter = TimeSeriesSplit(
        n_splits=CV_SPLITS
    )

    fold_batches = []

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        splitter.split(training_data),
        start=1,
    ):
        fold_train = (
            training_data.iloc[train_indices]
            .reset_index(drop=True)
        )
        fold_validation = (
            training_data.iloc[validation_indices]
            .reset_index(drop=True)
        )

        print(
            f"      fold {fold_number}/{CV_SPLITS}"
        )

        fold_batches.append(
            train_stage1_fold(
                fold_train=fold_train,
                fold_validation=fold_validation,
                feature_columns=feature_columns,
                parameters=parameters,
                seed=RANDOM_STATE + 20000 + fold_number,
            )
        )

    actual = np.concatenate(
        [
            batch["actual"]
            for batch in fold_batches
        ]
    ).astype(np.int64)
    probabilities = np.concatenate(
        [
            batch["move_probabilities"]
            for batch in fold_batches
        ]
    ).astype(np.float64)

    probability_metrics = binary_probability_metrics(
        actual=actual,
        positive_probabilities=probabilities,
    )

    threshold_result = (
        HierarchicalXLSTMParameterSelector
        .select_probability_threshold(
            actual=actual,
            positive_probabilities=probabilities,
        )
    )
    threshold = float(
        threshold_result["threshold"]
    )
    predicted = (
        probabilities >= threshold
    ).astype(np.int64)

    metrics = binary_metrics(
        actual=actual,
        predicted=predicted,
        negative_name="FLAT",
        positive_name="MOVE",
    )

    fold_metrics = []
    offset = 0

    for batch in fold_batches:
        fold_rows = len(batch["actual"])
        fold_actual = actual[
            offset : offset + fold_rows
        ]
        fold_probabilities = probabilities[
            offset : offset + fold_rows
        ]
        fold_probability_metrics = binary_probability_metrics(
            actual=fold_actual,
            positive_probabilities=fold_probabilities,
        )
        fold_predicted = (
            fold_probabilities >= threshold
        ).astype(np.int64)
        fold_class_metrics = binary_metrics(
            actual=fold_actual,
            predicted=fold_predicted,
            negative_name="FLAT",
            positive_name="MOVE",
        )

        fold_metrics.append(
            {
                "roc_auc": float(
                    fold_probability_metrics["roc_auc"]
                ),
                "average_precision": float(
                    fold_probability_metrics[
                        "average_precision"
                    ]
                ),
                "balanced_accuracy": float(
                    fold_class_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "flat_f1": float(
                    fold_class_metrics[
                        "per_class"
                    ]["FLAT"]["f1"]
                ),
                "flat_recall": float(
                    fold_class_metrics[
                        "per_class"
                    ]["FLAT"]["recall"]
                ),
            }
        )
        offset += fold_rows

    return {
        "decision_threshold": threshold,
        "stage1_roc_auc": float(
            probability_metrics["roc_auc"]
        ),
        "stage1_average_precision": float(
            probability_metrics["average_precision"]
        ),
        "stage1_brier_score": float(
            probability_metrics["brier_score"]
        ),
        "stage1_balanced_accuracy": float(
            metrics["balanced_accuracy"]
        ),
        "stage1_macro_f1": float(
            metrics["macro_f1"]
        ),
        "stage1_flat_precision": float(
            metrics["per_class"]["FLAT"]["precision"]
        ),
        "stage1_flat_recall": float(
            metrics["per_class"]["FLAT"]["recall"]
        ),
        "stage1_flat_f1": float(
            metrics["per_class"]["FLAT"]["f1"]
        ),
        "stage1_move_f1": float(
            metrics["per_class"]["MOVE"]["f1"]
        ),
        "stage1_roc_auc_fold_mean": float(
            np.mean(
                [
                    row["roc_auc"]
                    for row in fold_metrics
                ]
            )
        ),
        "stage1_roc_auc_fold_std": float(
            np.std(
                [
                    row["roc_auc"]
                    for row in fold_metrics
                ],
                ddof=0,
            )
        ),
        "stage1_flat_f1_fold_mean": float(
            np.mean(
                [
                    row["flat_f1"]
                    for row in fold_metrics
                ]
            )
        ),
        "stage1_flat_f1_fold_std": float(
            np.std(
                [
                    row["flat_f1"]
                    for row in fold_metrics
                ],
                ddof=0,
            )
        ),
        "fold_metrics": fold_metrics,
        "oof_rows": int(len(actual)),
    }


def build_master_training_data(
    raw_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    feature_library = Stage1WideFeatureBuilder().build_library(
        raw_data
    )
    labels = VolatilityDirectionLabelBuilder(
        volatility_window=TARGET_VOLATILITY_WINDOW,
        threshold_multiplier=TARGET_THRESHOLD_MULTIPLIER,
    ).build(
        raw_data[
            [
                "trade_date",
                "close",
            ]
        ].copy()
    )

    master = feature_library.rename(
        columns={
            "trade_date": "feature_date",
        }
    ).merge(
        labels,
        on="feature_date",
        how="inner",
        validate="one_to_one",
    )

    master = master.sort_values(
        "target_date"
    ).reset_index(drop=True)

    base_columns = list(
        Stage1WideFeatureBuilder.BASE_FEATURE_COLUMNS
    )
    base_dataset = master.dropna(
        subset=base_columns
    ).reset_index(drop=True)

    base_train, _, _ = DateAwareDataSplitter().split(
        base_dataset,
        date_column="target_date",
    )
    training_end_date = pd.Timestamp(
        base_train["target_date"].max()
    )

    master = master[
        pd.to_datetime(master["target_date"])
        <= training_end_date
    ].reset_index(drop=True)

    return master, training_end_date


def candidate_training_data(
    master_training_data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    subset = master_training_data.dropna(
        subset=feature_columns
    ).copy()

    return subset[
        [
            "feature_date",
            "target_date",
            *feature_columns,
            "future_log_return",
            "rolling_volatility",
            "threshold",
            "direction",
        ]
    ].sort_values(
        "target_date"
    ).reset_index(drop=True)


def sample_signature(
    training_data: pd.DataFrame,
) -> str:
    dates = pd.to_datetime(
        training_data["target_date"]
    ).astype("int64").to_numpy()
    return hashlib.sha1(
        dates.tobytes()
    ).hexdigest()


def load_checkpoint(
    filepath: Path,
) -> dict[str, dict]:
    if not filepath.exists():
        return {}

    with filepath.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    return {
        row["candidate_name"]: row
        for row in payload.get("results", [])
    }


def save_checkpoint(
    filepath: Path,
    results: dict[str, dict],
    metadata: dict,
) -> None:
    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    payload = {
        "metadata": metadata,
        "results": list(results.values()),
    }
    with filepath.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def candidate_result(
    candidate: SignalCandidate,
    master_training_data: pd.DataFrame,
    parameters: dict,
    baseline_cache: dict[str, dict],
) -> dict:
    feature_columns = (
        Stage1WideFeatureBuilder
        .columns_for_groups(
            candidate.groups
        )
    )
    training_data = candidate_training_data(
        master_training_data=master_training_data,
        feature_columns=feature_columns,
    )

    print()
    print(
        f"    {candidate.name}"
    )
    print(
        f"      groups: {list(candidate.groups)}"
    )
    print(
        f"      features: {len(feature_columns)}"
    )

    if len(training_data) < MIN_TRAINING_ROWS:
        print(
            "      skipped: only "
            f"{len(training_data)} training rows available"
        )
        return {
            "candidate_name": candidate.name,
            "groups": list(candidate.groups),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "status": "skipped",
            "training_rows": int(len(training_data)),
        }

    start_date = pd.Timestamp(
        training_data["target_date"].min()
    )
    end_date = pd.Timestamp(
        training_data["target_date"].max()
    )
    signature = sample_signature(training_data)

    print(
        "      sample:",
        f"{len(training_data)} rows",
        f"({start_date.date()} -> {end_date.date()})",
    )

    metrics = evaluate_candidate(
        training_data=training_data,
        feature_columns=feature_columns,
        parameters=parameters,
    )

    if not candidate.groups:
        matched_baseline = metrics
        baseline_cache[signature] = metrics
    else:
        matched_baseline = baseline_cache.get(signature)
        if matched_baseline is None:
            print(
                "      matched base control on the SAME dates"
            )
            matched_baseline = evaluate_candidate(
                training_data=training_data,
                feature_columns=list(
                    Stage1WideFeatureBuilder.BASE_FEATURE_COLUMNS
                ),
                parameters=parameters,
            )
            baseline_cache[signature] = matched_baseline

    delta_auc = float(
        metrics["stage1_roc_auc"]
        - matched_baseline["stage1_roc_auc"]
    )
    delta_flat_f1 = float(
        metrics["stage1_flat_f1"]
        - matched_baseline["stage1_flat_f1"]
    )

    print(
        "      ROC AUC:",
        round(
            metrics["stage1_roc_auc"],
            4,
        ),
    )
    print(
        "      matched base AUC:",
        round(
            matched_baseline["stage1_roc_auc"],
            4,
        ),
    )
    print(
        "      delta AUC vs matched base:",
        round(
            delta_auc,
            4,
        ),
    )
    print(
        "      FLAT F1:",
        round(
            metrics["stage1_flat_f1"],
            4,
        ),
    )
    print(
        "      delta FLAT F1 vs matched base:",
        round(
            delta_flat_f1,
            4,
        ),
    )
    print(
        "      AUC fold std:",
        round(
            metrics["stage1_roc_auc_fold_std"],
            4,
        ),
    )

    return {
        "candidate_name": candidate.name,
        "groups": list(candidate.groups),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "status": "ok",
        "training_rows": int(len(training_data)),
        "training_start": start_date.strftime("%Y-%m-%d"),
        "training_end": end_date.strftime("%Y-%m-%d"),
        "sample_signature": signature,
        "matched_base_roc_auc": float(
            matched_baseline["stage1_roc_auc"]
        ),
        "matched_base_flat_f1": float(
            matched_baseline["stage1_flat_f1"]
        ),
        "delta_roc_auc_vs_matched_base": delta_auc,
        "delta_flat_f1_vs_matched_base": delta_flat_f1,
        **metrics,
    }


def evaluate_candidates(
    candidates: list[SignalCandidate],
    master_training_data: pd.DataFrame,
    parameters: dict,
    completed: dict[str, dict],
    baseline_cache: dict[str, dict],
    checkpoint_path: Path,
    checkpoint_metadata: dict,
) -> list[dict]:
    round_results = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        if candidate.name in completed:
            print(
                f"    [{index}/{len(candidates)}] "
                f"{candidate.name} -- already completed"
            )
            result = completed[candidate.name]
            if result.get("status") == "ok":
                round_results.append(result)
            continue

        print(
            f"    [{index}/{len(candidates)}] evaluating"
        )
        result = candidate_result(
            candidate=candidate,
            master_training_data=master_training_data,
            parameters=parameters,
            baseline_cache=baseline_cache,
        )
        completed[candidate.name] = result
        if result.get("status") == "ok":
            round_results.append(result)

        save_checkpoint(
            filepath=checkpoint_path,
            results=completed,
            metadata=checkpoint_metadata,
        )

    return round_results


def build_summary(
    results: dict[str, dict],
) -> pd.DataFrame:
    rows = []

    for result in results.values():
        if result.get("status") != "ok":
            continue

        rows.append(
            {
                "candidate_name": result["candidate_name"],
                "groups": "|".join(result["groups"]),
                "group_count": len(result["groups"]),
                "feature_count": result["feature_count"],
                "training_rows": result["training_rows"],
                "training_start": result["training_start"],
                "training_end": result["training_end"],
                "stage1_roc_auc": result["stage1_roc_auc"],
                "matched_base_roc_auc": result[
                    "matched_base_roc_auc"
                ],
                "delta_roc_auc_vs_matched_base": result[
                    "delta_roc_auc_vs_matched_base"
                ],
                "stage1_roc_auc_fold_std": result[
                    "stage1_roc_auc_fold_std"
                ],
                "stage1_balanced_accuracy": result[
                    "stage1_balanced_accuracy"
                ],
                "stage1_flat_precision": result[
                    "stage1_flat_precision"
                ],
                "stage1_flat_recall": result[
                    "stage1_flat_recall"
                ],
                "stage1_flat_f1": result[
                    "stage1_flat_f1"
                ],
                "matched_base_flat_f1": result[
                    "matched_base_flat_f1"
                ],
                "delta_flat_f1_vs_matched_base": result[
                    "delta_flat_f1_vs_matched_base"
                ],
                "stage1_move_f1": result[
                    "stage1_move_f1"
                ],
                "stage1_brier_score": result[
                    "stage1_brier_score"
                ],
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "delta_roc_auc_vs_matched_base",
                "stage1_roc_auc_fold_std",
                "delta_flat_f1_vs_matched_base",
                "training_rows",
            ],
            ascending=[
                False,
                True,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def print_context_coverage(
    raw_data: pd.DataFrame,
) -> None:
    print()
    print(
        "Context-series coverage inside the base SPY sample:"
    )
    for column in (
        "vix9d_close",
        "vix3m_close",
        "skew_close",
        "vxn_close",
        "dxy_close",
        "es_close",
        "nq_close",
        "rty_close",
        "cl_close",
    ):
        available = raw_data[
            ["trade_date", column]
        ].dropna()
        if available.empty:
            print(
                f"  {column}: no aligned rows"
            )
            continue
        print(
            f"  {column}: {len(available)} rows "
            f"({pd.Timestamp(available['trade_date'].min()).date()} "
            f"-> {pd.Timestamp(available['trade_date'].max()).date()})"
        )



def feature_signature(
    feature_columns: list[str],
) -> str:
    payload = "|".join(
        feature_columns
    ).encode(
        "utf-8"
    )
    return hashlib.sha1(
        payload
    ).hexdigest()[:10]


def optuna_study_name(
    prefix: str,
    training_data: pd.DataFrame,
    feature_columns: list[str],
) -> str:
    return (
        f"{prefix}_"
        f"{sample_signature(training_data)[:12]}_"
        f"{feature_signature(feature_columns)}"
    )


def optimize_feature_contract(
    name: str,
    training_data: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    print()
    print(
        f"OPTUNA: {name}"
    )
    print(
        f"Rows: {len(training_data)}"
    )
    print(
        "Period:",
        pd.Timestamp(
            training_data[
                "target_date"
            ].min()
        ).date(),
        "->",
        pd.Timestamp(
            training_data[
                "target_date"
            ].max()
        ).date(),
    )
    print(
        f"Features: {len(feature_columns)}"
    )

    selector = (
        HierarchicalXLSTMParameterSelector(
            feature_columns=(
                feature_columns
            ),
            task="move",
            n_splits=CV_SPLITS,
            n_trials=OPTUNA_TRIALS,
            max_epochs=(
                MAX_SELECTION_EPOCHS
            ),
            patience=(
                OPTUNA_PATIENCE
            ),
            random_state=(
                RANDOM_STATE
            ),
            objective_metric="roc_auc",
            study_name=(
                optuna_study_name(
                    prefix=name,
                    training_data=(
                        training_data
                    ),
                    feature_columns=(
                        feature_columns
                    ),
                )
            ),
            storage_url=(
                OPTUNA_STORAGE_URL
            ),
        )
    )

    return (
        selector
        .select_best_parameters(
            training_data=(
                training_data
            )
        )
    )


def matched_base_training_data(
    candidate_data: pd.DataFrame,
) -> pd.DataFrame:
    base_columns = list(
        Stage1WideFeatureBuilder
        .BASE_FEATURE_COLUMNS
    )

    return candidate_data[
        [
            "feature_date",
            "target_date",
            *base_columns,
            "future_log_return",
            "rolling_volatility",
            "threshold",
            "direction",
        ]
    ].copy()


def shortlist_for_optimization(
    summary: pd.DataFrame,
    core_groups: tuple[str, ...],
) -> list[SignalCandidate]:
    if summary.empty:
        raise RuntimeError(
            "Stage-1 screening did not produce results."
        )

    eligible = summary[
        (
            summary[
                "group_count"
            ]
            > 0
        )
        & (
            summary[
                "training_rows"
            ]
            >= MIN_TRAINING_ROWS
        )
        & (
            summary[
                "delta_roc_auc_vs_matched_base"
            ]
            > 0.0
        )
        & (
            summary[
                "stage1_roc_auc_fold_std"
            ]
            <= 0.03
        )
    ].copy()

    eligible = eligible.sort_values(
        [
            "delta_roc_auc_vs_matched_base",
            "delta_flat_f1_vs_matched_base",
            "stage1_roc_auc_fold_std",
            "training_rows",
        ],
        ascending=[
            False,
            False,
            True,
            False,
        ],
    )

    selected_groups = []

    for _, row in eligible.head(
        OPTUNA_SHORTLIST_SIZE
    ).iterrows():
        groups = tuple(
            group
            for group in str(
                row[
                    "groups"
                ]
            ).split("|")
            if group
        )
        if groups:
            selected_groups.append(
                groups
            )

    if len(
        selected_groups
    ) < OPTUNA_SHORTLIST_SIZE:
        fallback = summary[
            (
                summary[
                    "group_count"
                ]
                > 0
            )
            & (
                summary[
                    "training_rows"
                ]
                >= MIN_TRAINING_ROWS
            )
        ].sort_values(
            [
                "delta_roc_auc_vs_matched_base",
                "stage1_roc_auc_fold_std",
                "delta_flat_f1_vs_matched_base",
            ],
            ascending=[
                False,
                True,
                False,
            ],
        )

        for _, row in fallback.iterrows():
            groups = tuple(
                group
                for group in str(
                    row[
                        "groups"
                    ]
                ).split("|")
                if group
            )
            if (
                groups
                and groups
                not in selected_groups
            ):
                selected_groups.append(
                    groups
                )

            if (
                len(
                    selected_groups
                )
                >= OPTUNA_SHORTLIST_SIZE
            ):
                break

    if (
        core_groups
        and core_groups
        not in selected_groups
    ):
        selected_groups.append(
            core_groups
        )

    return [
        SignalCandidate(
            tuple(
                sorted(
                    groups
                )
            )
        )
        for groups in selected_groups
    ]


def save_optimization_progress(
    rows: list[dict],
) -> None:
    OPTIMIZATION_PROGRESS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OPTIMIZATION_PROGRESS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            rows,
            file,
            indent=2,
        )


def optimize_shortlist(
    candidates: list[SignalCandidate],
    master_training_data: pd.DataFrame,
) -> pd.DataFrame:
    base_columns = list(
        Stage1WideFeatureBuilder
        .BASE_FEATURE_COLUMNS
    )

    baseline_cache = {}
    results = []

    print()
    print(
        "=" * 88
    )
    print(
        "STAGE-1 AUC OPTUNA RETUNING"
    )
    print(
        "=" * 88
    )
    print(
        f"{OPTUNA_TRIALS} trials per candidate/control, "
        f"{CV_SPLITS}-fold train-only walk-forward CV."
    )
    print(
        "Objective metric: ROC AUC."
    )
    print(
        "Every candidate is compared against a separately tuned "
        "BASE model on the exact same dates."
    )

    for candidate in candidates:
        feature_columns = (
            Stage1WideFeatureBuilder
            .columns_for_groups(
                candidate.groups
            )
        )

        data = candidate_training_data(
            master_training_data=(
                master_training_data
            ),
            feature_columns=(
                feature_columns
            ),
        )

        signature = sample_signature(
            data
        )

        print()
        print(
            "-" * 88
        )
        print(
            "Candidate:",
            candidate.name,
        )
        print(
            "Groups:",
            list(
                candidate.groups
            ),
        )
        print(
            "-" * 88
        )

        candidate_selection = (
            optimize_feature_contract(
                name=(
                    "stage1_candidate_"
                    f"{candidate.name}"
                ),
                training_data=data,
                feature_columns=(
                    feature_columns
                ),
            )
        )

        if signature not in baseline_cache:
            baseline_cache[
                signature
            ] = (
                optimize_feature_contract(
                    name=(
                        "stage1_matched_base_"
                        f"{signature[:12]}"
                    ),
                    training_data=(
                        matched_base_training_data(
                            data
                        )
                    ),
                    feature_columns=(
                        base_columns
                    ),
                )
            )

        baseline_selection = (
            baseline_cache[
                signature
            ]
        )

        delta_auc = (
            float(
                candidate_selection[
                    "threshold_oof_roc_auc"
                ]
            )
            - float(
                baseline_selection[
                    "threshold_oof_roc_auc"
                ]
            )
        )

        row = {
            "candidate_name": (
                candidate.name
            ),
            "groups": list(
                candidate.groups
            ),
            "feature_count": len(
                feature_columns
            ),
            "training_rows": int(
                len(
                    data
                )
            ),
            "training_start": (
                pd.Timestamp(
                    data[
                        "target_date"
                    ].min()
                )
                .strftime(
                    "%Y-%m-%d"
                )
            ),
            "training_end": (
                pd.Timestamp(
                    data[
                        "target_date"
                    ].max()
                )
                .strftime(
                    "%Y-%m-%d"
                )
            ),
            "candidate_oof_roc_auc": float(
                candidate_selection[
                    "threshold_oof_roc_auc"
                ]
            ),
            "candidate_oof_roc_auc_fold_std": float(
                candidate_selection[
                    "threshold_oof_roc_auc_fold_std"
                ]
            ),
            "candidate_oof_balanced_accuracy": float(
                candidate_selection[
                    "threshold_oof_balanced_accuracy"
                ]
            ),
            "candidate_oof_macro_f1": float(
                candidate_selection[
                    "threshold_oof_macro_f1"
                ]
            ),
            "candidate_decision_threshold": float(
                candidate_selection[
                    "decision_threshold"
                ]
            ),
            "matched_base_oof_roc_auc": float(
                baseline_selection[
                    "threshold_oof_roc_auc"
                ]
            ),
            "delta_oof_roc_auc_vs_tuned_base": float(
                delta_auc
            ),
            "candidate_selection": (
                candidate_selection
            ),
            "matched_base_selection": (
                baseline_selection
            ),
        }

        results.append(
            row
        )

        save_optimization_progress(
            results
        )

        print()
        print(
            "TUNED RESULT"
        )
        print(
            "Candidate OOF AUC:",
            round(
                row[
                    "candidate_oof_roc_auc"
                ],
                4,
            ),
        )
        print(
            "Tuned matched-base AUC:",
            round(
                row[
                    "matched_base_oof_roc_auc"
                ],
                4,
            ),
        )
        print(
            "Delta:",
            round(
                row[
                    "delta_oof_roc_auc_vs_tuned_base"
                ],
                4,
            ),
        )
        print(
            "Fold std:",
            round(
                row[
                    "candidate_oof_roc_auc_fold_std"
                ],
                4,
            ),
        )

    summary = (
        pd.DataFrame(
            [
                {
                    key: value
                    for key, value
                    in row.items()
                    if key not in {
                        "candidate_selection",
                        "matched_base_selection",
                    }
                }
                for row in results
            ]
        )
        .sort_values(
            [
                "delta_oof_roc_auc_vs_tuned_base",
                "candidate_oof_roc_auc_fold_std",
                "candidate_oof_roc_auc",
            ],
            ascending=[
                False,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return summary



def main():
    print(
        "Loading Stage-1 long-history FLAT/MOVE signal dataset..."
    )

    raw_data = Stage2SignalDataRepository().get_training_data(
        ticker=TICKER
    )

    print(
        f"Base rows preserved: {len(raw_data)}"
    )
    print(
        "Base period:",
        pd.Timestamp(raw_data["trade_date"].min()).date(),
        "->",
        pd.Timestamp(raw_data["trade_date"].max()).date(),
    )
    print_context_coverage(raw_data)

    master_training_data, training_end_date = (
        build_master_training_data(
            raw_data=raw_data
        )
    )

    base_training_data = candidate_training_data(
        master_training_data=master_training_data,
        feature_columns=list(
            Stage1WideFeatureBuilder.BASE_FEATURE_COLUMNS
        ),
    )

    print()
    print(
        f"Fixed training cutoff: {training_end_date.date()}"
    )
    print(
        f"Base training rows: {len(base_training_data)}"
    )
    print(
        "Training target distribution:",
        target_distribution(base_training_data),
    )

    reference_path, reference_experiment = (
        load_latest_hierarchical_experiment(
            EXPERIMENT_DIRECTORY
        )
    )
    stage1_parameters = dict(
        reference_experiment[
            "parameters"
        ][
            "stage1"
        ][
            "parameters"
        ]
    )

    print(
        "Locked Stage-1 parameter source:",
        reference_path,
    )
    print(
        "Target is temporarily locked at 40-day volatility x 0.45."
    )
    print(
        "This search isolates the best FLAT/MOVE information set first."
    )
    print(
        "Afterward, the target window/multiplier can be jointly refined "
        "around the winning Stage-1 feature set."
    )
    print(
        "Short-history feature groups are EXCLUDED from this experiment."
    )
    print(
        "Screening and Optuna finalists use matched BASE controls on the exact same dates."
    )
    print(
        "Outer validation and held-out test are NOT evaluated."
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    completed = load_checkpoint(
        CHECKPOINT_PATH
    )
    baseline_cache = {}

    for result in completed.values():
        if (
            result.get("status") == "ok"
            and result.get("candidate_name") == "base_only"
        ):
            signature = result.get("sample_signature")
            if signature:
                baseline_cache[signature] = {
                    key: value
                    for key, value in result.items()
                    if key.startswith("stage1_")
                    or key in {
                        "decision_threshold",
                        "fold_metrics",
                        "oof_rows",
                    }
                }

    checkpoint_metadata = {
        "version": 1,
        "target_volatility_window": TARGET_VOLATILITY_WINDOW,
        "target_threshold_multiplier": TARGET_THRESHOLD_MULTIPLIER,
        "training_end_date": training_end_date.strftime("%Y-%m-%d"),
        "cv_splits": CV_SPLITS,
        "reference_experiment": str(reference_path),
        "beam_width": BEAM_WIDTH,
        "max_group_depth": MAX_GROUP_DEPTH,
        "feature_groups": Stage1WideFeatureBuilder.FEATURE_GROUPS,
        "short_history_groups": sorted(
            Stage1WideFeatureBuilder.SHORT_HISTORY_GROUPS
        ),
    }

    group_names = [
        group_name
        for group_name in (
            Stage1WideFeatureBuilder
            .FEATURE_GROUPS
        )
        if group_name not in (
            Stage1WideFeatureBuilder
            .SHORT_HISTORY_GROUPS
        )
    ]

    print()
    print(
        "Running univariate FLAT/MOVE feature AUC screen..."
    )
    univariate = univariate_feature_auc_screen(
        training_data=master_training_data,
        feature_columns=[
            feature
            for group_name in group_names
            for feature in (
                Stage1WideFeatureBuilder
                .FEATURE_GROUPS[
                    group_name
                ]
            )
        ],
        n_splits=CV_SPLITS,
        minimum_rows=MIN_TRAINING_ROWS,
    )
    univariate_path = (
        EXPERIMENT_DIRECTORY
        / f"stage1_long_history_univariate_auc_v2_{timestamp}.csv"
    )
    univariate.to_csv(
        univariate_path,
        index=False,
    )

    print()
    print(
        "Top 30 individual FLAT/MOVE signals:"
    )
    print(
        univariate.head(30).round(4).to_string(index=False)
    )

    baseline_candidate = SignalCandidate(())
    evaluate_candidates(
        candidates=[baseline_candidate],
        master_training_data=master_training_data,
        parameters=stage1_parameters,
        completed=completed,
        baseline_cache=baseline_cache,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_metadata=checkpoint_metadata,
    )

    print()
    print(
        "PHASE 1 - individual feature groups"
    )
    single_results = evaluate_candidates(
        candidates=build_single_candidates(
            group_names
        ),
        master_training_data=master_training_data,
        parameters=stage1_parameters,
        completed=completed,
        baseline_cache=baseline_cache,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_metadata=checkpoint_metadata,
    )

    ranked_singles = sorted(
        single_results,
        key=lambda row: (
            -float(
                row["delta_roc_auc_vs_matched_base"]
            ),
            float(row["stage1_roc_auc_fold_std"]),
            -float(row["stage1_flat_f1"]),
            -int(row["training_rows"]),
        ),
    )
    ranked_single_groups = [
        row["groups"][0]
        for row in ranked_singles
    ]

    print()
    print(
        "PHASE 2 - all pairs among the top "
        f"{TOP_SINGLE_GROUPS_FOR_PAIRS} individual groups"
    )
    pair_candidates = build_pair_candidates(
        ranked_single_groups=ranked_single_groups,
        top_group_count=TOP_SINGLE_GROUPS_FOR_PAIRS,
    )
    pair_results = evaluate_candidates(
        candidates=pair_candidates,
        master_training_data=master_training_data,
        parameters=stage1_parameters,
        completed=completed,
        baseline_cache=baseline_cache,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_metadata=checkpoint_metadata,
    )

    beam = select_beam(
        results=pair_results,
        beam_width=BEAM_WIDTH,
    )

    current_depth = 3
    while (
        current_depth <= MAX_GROUP_DEPTH
        and beam
    ):
        print()
        print(
            f"PHASE {current_depth} - beam expansion to "
            f"{current_depth} groups"
        )
        expanded_candidates = [
            candidate
            for candidate in expand_beam_candidates(
                beam_groups=beam,
                all_group_names=group_names,
            )
            if len(candidate.groups) == current_depth
        ]

        expanded_results = evaluate_candidates(
            candidates=expanded_candidates,
            master_training_data=master_training_data,
            parameters=stage1_parameters,
            completed=completed,
            baseline_cache=baseline_cache,
            checkpoint_path=CHECKPOINT_PATH,
            checkpoint_metadata=checkpoint_metadata,
        )
        beam = select_beam(
            results=expanded_results,
            beam_width=BEAM_WIDTH,
        )
        current_depth += 1

    core_groups = tuple(
        sorted(
            group_name
            for group_name in group_names
            if group_name not in (
                Stage1WideFeatureBuilder
                .SHORT_HISTORY_GROUPS
            )
        )
    )
    print()
    print(
        "FINAL CONTROL - all long-history groups"
    )
    evaluate_candidates(
        candidates=[
            SignalCandidate(core_groups),
        ],
        master_training_data=master_training_data,
        parameters=stage1_parameters,
        completed=completed,
        baseline_cache=baseline_cache,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_metadata=checkpoint_metadata,
    )

    summary = build_summary(
        results=completed,
    )
    summary_path = (
        EXPERIMENT_DIRECTORY
        / f"stage1_long_history_search_v2_{timestamp}.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    optimization_candidates = (
        shortlist_for_optimization(
            summary=summary,
            core_groups=core_groups,
        )
    )

    print()
    print(
        "Candidates promoted to Optuna retuning:"
    )
    for candidate in optimization_candidates:
        print(
            "  -",
            candidate.name,
        )

    optimization_summary = (
        optimize_shortlist(
            candidates=(
                optimization_candidates
            ),
            master_training_data=(
                master_training_data
            ),
        )
    )

    optimization_timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )

    optimization_summary_path = (
        EXPERIMENT_DIRECTORY
        / (
            "stage1_long_history_optimization_v2_"
            f"{optimization_timestamp}.csv"
        )
    )

    optimization_summary.to_csv(
        optimization_summary_path,
        index=False,
    )

    experiment_path = ExperimentTracker(
        str(EXPERIMENT_DIRECTORY)
    ).save(
        experiment_name=EXPERIMENT_NAME,
        model_name=MODEL_NAME,
        parameters={
            "target_volatility_window": TARGET_VOLATILITY_WINDOW,
            "target_threshold_multiplier": TARGET_THRESHOLD_MULTIPLIER,
            "training_end_date": training_end_date.strftime("%Y-%m-%d"),
            "cv_splits": CV_SPLITS,
            "beam_width": BEAM_WIDTH,
            "max_group_depth": MAX_GROUP_DEPTH,
            "top_single_groups_for_pairs": (
                TOP_SINGLE_GROUPS_FOR_PAIRS
            ),
            "minimum_training_rows": MIN_TRAINING_ROWS,
            "locked_stage1_parameters": stage1_parameters,
            "optuna_trials_per_promoted_candidate": OPTUNA_TRIALS,
            "optuna_objective_metric": "roc_auc",
            "optuna_promoted_candidates": [
                candidate.name
                for candidate in optimization_candidates
            ],
            "reference_experiment": str(reference_path),
            "short_history_groups": sorted(
                Stage1WideFeatureBuilder.SHORT_HISTORY_GROUPS
            ),
        },
        metrics={
            "screening_best_candidates": summary.head(30).to_dict(
                orient="records"
            ),
            "optimized_candidates": (
                optimization_summary.to_dict(
                    orient="records"
                )
            ),
        },
        features=list(
            Stage1WideFeatureBuilder.FEATURE_COLUMNS
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "STAGE-1 LONG-HISTORY FLAT/MOVE SEARCH V2 - SCREENING TOP 30"
    )
    print(
        "============================================================"
    )
    display = summary.head(30)[
        [
            "candidate_name",
            "group_count",
            "feature_count",
            "training_rows",
            "training_start",
            "stage1_roc_auc",
            "matched_base_roc_auc",
            "delta_roc_auc_vs_matched_base",
            "stage1_roc_auc_fold_std",
            "stage1_balanced_accuracy",
            "stage1_flat_precision",
            "stage1_flat_recall",
            "stage1_flat_f1",
            "delta_flat_f1_vs_matched_base",
            "stage1_move_f1",
        ]
    ].copy()
    numeric_columns = [
        column
        for column in display.columns
        if column not in {
            "candidate_name",
            "training_start",
        }
    ]
    display[numeric_columns] = display[
        numeric_columns
    ].round(4)
    print(
        display.to_string(index=False)
    )

    print()
    print(
        "Primary comparison: delta ROC AUC versus a BASE model trained "
        "and evaluated on the exact same dates."
    )
    print(
        "FLAT F1/recall and fold stability are secondary gates so a "
        "candidate cannot win only by ranking MOVE probabilities better."
    )
    print(
        "The target remains 40d x 0.45 only for this feature search; "
        "we can reopen the target definition after the best Stage-1 "
        "information set is known."
    )
    print()
    print(
        "Univariate screen:",
        univariate_path,
    )
    print(
        "Screening summary:",
        summary_path,
    )
    print(
        "Optimized finalists:",
        optimization_summary_path,
    )
    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )
    print(
        "Experiment:",
        experiment_path,
    )
    print()
    print(
        "Outer validation and held-out test were NOT evaluated."
    )


if __name__ == "__main__":
    main()
