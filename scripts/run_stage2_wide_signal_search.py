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
from app.training.stage2_wide_feature_builder import (
    Stage2WideFeatureBuilder,
)
from app.training.stage2_wide_signal_search import (
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
TOP_SINGLE_GROUPS_FOR_PAIRS = 8
BEAM_WIDTH = 8
MAX_GROUP_DEPTH = 5
MIN_TRAINING_ROWS = 600
EXPERIMENT_DIRECTORY = Path("experiments")
EXPERIMENT_NAME = "xlstm_stage2_wide_signal_search_v2"
MODEL_NAME = "xlstm_stage2_wide_signal_search_v2"
CHECKPOINT_PATH = (
    EXPERIMENT_DIRECTORY
    / "stage2_wide_signal_search_v2_checkpoint.json"
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


def train_stage2_fold(
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
        task="direction",
    )

    validation_sequences = preprocessor.build_inference_sequences(
        history=fold_train,
        dataframe=fold_validation,
        task="direction",
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
        "directions": validation_sequences["directions"].copy(),
        "target_dates": pd.DatetimeIndex(
            validation_sequences["target_dates"]
        ),
        "up_probabilities": prediction_result[
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
            train_stage2_fold(
                fold_train=fold_train,
                fold_validation=fold_validation,
                feature_columns=feature_columns,
                parameters=parameters,
                seed=RANDOM_STATE + 10000 + fold_number,
            )
        )

    directions = np.concatenate(
        [
            batch["directions"]
            for batch in fold_batches
        ]
    ).astype(object)
    probabilities = np.concatenate(
        [
            batch["up_probabilities"]
            for batch in fold_batches
        ]
    ).astype(np.float64)

    move_mask = directions != "FLAT"
    actual = np.asarray(
        [
            1 if direction == "UP" else 0
            for direction in directions[move_mask]
        ],
        dtype=np.int64,
    )
    move_probabilities = probabilities[move_mask]

    probability_metrics = binary_probability_metrics(
        actual=actual,
        positive_probabilities=move_probabilities,
    )

    threshold_result = (
        HierarchicalXLSTMParameterSelector
        .select_probability_threshold(
            actual=actual,
            positive_probabilities=move_probabilities,
        )
    )
    threshold = float(
        threshold_result["threshold"]
    )
    predicted = (
        move_probabilities >= threshold
    ).astype(np.int64)

    metrics = binary_metrics(
        actual=actual,
        predicted=predicted,
        negative_name="DOWN",
        positive_name="UP",
    )

    fold_metrics = []
    offset = 0

    for batch in fold_batches:
        fold_rows = len(batch["directions"])
        fold_directions = directions[
            offset : offset + fold_rows
        ]
        fold_probabilities = probabilities[
            offset : offset + fold_rows
        ]
        fold_move_mask = fold_directions != "FLAT"
        fold_actual = np.asarray(
            [
                1 if direction == "UP" else 0
                for direction in fold_directions[
                    fold_move_mask
                ]
            ],
            dtype=np.int64,
        )
        fold_move_probabilities = fold_probabilities[
            fold_move_mask
        ]
        fold_probability_metrics = binary_probability_metrics(
            actual=fold_actual,
            positive_probabilities=fold_move_probabilities,
        )
        fold_predicted = (
            fold_move_probabilities >= threshold
        ).astype(np.int64)
        fold_class_metrics = binary_metrics(
            actual=fold_actual,
            predicted=fold_predicted,
            negative_name="DOWN",
            positive_name="UP",
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
            }
        )
        offset += fold_rows

    return {
        "decision_threshold": threshold,
        "stage2_roc_auc": float(
            probability_metrics["roc_auc"]
        ),
        "stage2_average_precision": float(
            probability_metrics["average_precision"]
        ),
        "stage2_brier_score": float(
            probability_metrics["brier_score"]
        ),
        "stage2_balanced_accuracy": float(
            metrics["balanced_accuracy"]
        ),
        "stage2_macro_f1": float(
            metrics["macro_f1"]
        ),
        "stage2_down_f1": float(
            metrics["per_class"]["DOWN"]["f1"]
        ),
        "stage2_up_f1": float(
            metrics["per_class"]["UP"]["f1"]
        ),
        "stage2_roc_auc_fold_mean": float(
            np.mean(
                [
                    row["roc_auc"]
                    for row in fold_metrics
                ]
            )
        ),
        "stage2_roc_auc_fold_std": float(
            np.std(
                [
                    row["roc_auc"]
                    for row in fold_metrics
                ],
                ddof=0,
            )
        ),
        "fold_metrics": fold_metrics,
        "oof_move_rows": int(len(actual)),
    }


def build_master_training_data(
    raw_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    feature_library = Stage2WideFeatureBuilder().build_library(
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
        Stage2WideFeatureBuilder.BASE_FEATURE_COLUMNS
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
        Stage2WideFeatureBuilder
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
                    Stage2WideFeatureBuilder.BASE_FEATURE_COLUMNS
                ),
                parameters=parameters,
            )
            baseline_cache[signature] = matched_baseline

    delta_auc = float(
        metrics["stage2_roc_auc"]
        - matched_baseline["stage2_roc_auc"]
    )

    print(
        "      ROC AUC:",
        round(
            metrics["stage2_roc_auc"],
            4,
        ),
    )
    print(
        "      matched base AUC:",
        round(
            matched_baseline["stage2_roc_auc"],
            4,
        ),
    )
    print(
        "      delta vs matched base:",
        round(
            delta_auc,
            4,
        ),
    )
    print(
        "      fold std:",
        round(
            metrics["stage2_roc_auc_fold_std"],
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
            matched_baseline["stage2_roc_auc"]
        ),
        "delta_roc_auc_vs_matched_base": delta_auc,
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
                "stage2_roc_auc": result["stage2_roc_auc"],
                "matched_base_roc_auc": result[
                    "matched_base_roc_auc"
                ],
                "delta_roc_auc_vs_matched_base": result[
                    "delta_roc_auc_vs_matched_base"
                ],
                "stage2_roc_auc_fold_mean": result[
                    "stage2_roc_auc_fold_mean"
                ],
                "stage2_roc_auc_fold_std": result[
                    "stage2_roc_auc_fold_std"
                ],
                "stage2_average_precision": result[
                    "stage2_average_precision"
                ],
                "stage2_brier_score": result[
                    "stage2_brier_score"
                ],
                "stage2_balanced_accuracy": result[
                    "stage2_balanced_accuracy"
                ],
                "stage2_macro_f1": result[
                    "stage2_macro_f1"
                ],
                "stage2_down_f1": result[
                    "stage2_down_f1"
                ],
                "stage2_up_f1": result[
                    "stage2_up_f1"
                ],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "delta_roc_auc_vs_matched_base",
                "stage2_roc_auc_fold_std",
                "training_rows",
            ],
            ascending=[
                False,
                True,
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


def main():
    print(
        "Loading wide Stage-2 signal dataset..."
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
            Stage2WideFeatureBuilder.BASE_FEATURE_COLUMNS
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
    stage2_parameters = dict(
        reference_experiment[
            "parameters"
        ][
            "stage2"
        ][
            "parameters"
        ]
    )

    print(
        "Locked Stage-2 parameter source:",
        reference_path,
    )
    print(
        "Target is locked at 40-day volatility x 0.45."
    )
    print(
        "Short-history signals use their available TRAINING-period rows only."
    )
    print(
        "Every candidate is compared with a BASE model on the exact same dates."
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
                    if key.startswith("stage2_")
                    or key in {
                        "decision_threshold",
                        "fold_metrics",
                        "oof_move_rows",
                    }
                }

    checkpoint_metadata = {
        "version": 2,
        "target_volatility_window": TARGET_VOLATILITY_WINDOW,
        "target_threshold_multiplier": TARGET_THRESHOLD_MULTIPLIER,
        "training_end_date": training_end_date.strftime("%Y-%m-%d"),
        "cv_splits": CV_SPLITS,
        "reference_experiment": str(reference_path),
        "beam_width": BEAM_WIDTH,
        "max_group_depth": MAX_GROUP_DEPTH,
        "feature_groups": Stage2WideFeatureBuilder.FEATURE_GROUPS,
        "short_history_groups": sorted(
            Stage2WideFeatureBuilder.SHORT_HISTORY_GROUPS
        ),
    }

    print()
    print(
        "Running univariate feature AUC screen..."
    )
    univariate = univariate_feature_auc_screen(
        training_data=master_training_data,
        feature_columns=[
            feature
            for group_features in (
                Stage2WideFeatureBuilder
                .FEATURE_GROUPS.values()
            )
            for feature in group_features
        ],
        n_splits=CV_SPLITS,
        minimum_rows=MIN_TRAINING_ROWS,
    )
    univariate_path = (
        EXPERIMENT_DIRECTORY
        / f"stage2_univariate_signal_auc_v2_{timestamp}.csv"
    )
    univariate.to_csv(
        univariate_path,
        index=False,
    )

    print()
    print(
        "Top 25 individual directional signals:"
    )
    print(
        univariate.head(25).round(4).to_string(index=False)
    )

    baseline_candidate = SignalCandidate(())
    evaluate_candidates(
        candidates=[baseline_candidate],
        master_training_data=master_training_data,
        parameters=stage2_parameters,
        completed=completed,
        baseline_cache=baseline_cache,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_metadata=checkpoint_metadata,
    )

    group_names = list(
        Stage2WideFeatureBuilder.FEATURE_GROUPS
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
        parameters=stage2_parameters,
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
            float(row["stage2_roc_auc_fold_std"]),
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
        parameters=stage2_parameters,
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
            parameters=stage2_parameters,
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
                Stage2WideFeatureBuilder
                .SHORT_HISTORY_GROUPS
            )
        )
    )
    all_groups = tuple(
        sorted(group_names)
    )

    print()
    print(
        "FINAL CONTROLS - all core groups and all groups"
    )
    evaluate_candidates(
        candidates=[
            SignalCandidate(core_groups),
            SignalCandidate(all_groups),
        ],
        master_training_data=master_training_data,
        parameters=stage2_parameters,
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
        / f"stage2_wide_signal_search_v2_{timestamp}.csv"
    )
    summary.to_csv(
        summary_path,
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
            "locked_stage2_parameters": stage2_parameters,
            "reference_experiment": str(reference_path),
            "short_history_groups": sorted(
                Stage2WideFeatureBuilder.SHORT_HISTORY_GROUPS
            ),
        },
        metrics={
            "best_candidates": summary.head(25).to_dict(
                orient="records"
            ),
        },
        features=list(
            Stage2WideFeatureBuilder.FEATURE_COLUMNS
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "STAGE-2 WIDE DIRECTIONAL SIGNAL SEARCH V2 - TOP 25"
    )
    print(
        "============================================================"
    )
    display = summary.head(25)[
        [
            "candidate_name",
            "group_count",
            "feature_count",
            "training_rows",
            "training_start",
            "stage2_roc_auc",
            "matched_base_roc_auc",
            "delta_roc_auc_vs_matched_base",
            "stage2_roc_auc_fold_std",
            "stage2_balanced_accuracy",
            "stage2_down_f1",
            "stage2_up_f1",
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
        "This prevents VIX9D/RTY's shorter histories from making an "
        "apples-to-oranges comparison."
    )
    print(
        "No Optuna retuning occurred in this screening experiment."
    )
    print(
        "Outer validation and held-out test were NOT evaluated."
    )
    print()
    print(
        "Univariate feature screen:",
        univariate_path,
    )
    print(
        "Combination summary:",
        summary_path,
    )
    print(
        "Checkpoint (safe to resume):",
        CHECKPOINT_PATH,
    )
    print(
        "Experiment:",
        experiment_path,
    )


if __name__ == "__main__":
    main()
