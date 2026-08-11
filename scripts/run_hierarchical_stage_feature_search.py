from datetime import datetime, timezone
from gc import collect
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit

from app.training.date_aware_data_splitter import DateAwareDataSplitter
from app.training.direction_dataset_builder import DirectionDatasetBuilder
from app.training.direction_feature_builder import DirectionFeatureBuilder
from app.training.experiment_tracker import ExperimentTracker
from app.training.hierarchical_sequence_preprocessor import (
    HierarchicalSequencePreprocessor,
)
from app.training.hierarchical_stage_feature_research import (
    binary_probability_metrics,
    build_refined_target_candidates,
    build_stage_feature_profiles,
    classify_trend_features,
)
from app.training.hierarchical_target_feature_research import (
    align_datasets_on_common_target_dates,
    binary_metrics,
    compose_hierarchical_predictions,
    load_latest_hierarchical_experiment,
    target_distribution,
    three_class_metrics,
)
from app.training.hierarchical_xlstm_parameter_selector import (
    HierarchicalXLSTMParameterSelector,
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
from database.direction_training_data_repository import (
    DirectionTrainingDataRepository,
)


TICKER = "SPY"
EXPERIMENT_NAME = "xlstm_refined_flat_stage_feature_search"
MODEL_NAME = "xlstm_refined_flat_stage_feature_search_v1"
EXPERIMENT_DIRECTORY = Path("experiments")
CV_SPLITS = 3
RANDOM_STATE = 42
EARLY_STOPPING_PATIENCE = 8


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


def train_fold_stage(
    fold_train: pd.DataFrame,
    fold_validation: pd.DataFrame,
    feature_columns: list[str],
    task: str,
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
        task=task,
    )

    validation_sequences = preprocessor.build_inference_sequences(
        history=fold_train,
        dataframe=fold_validation,
        task=task,
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
        "actual": validation_sequences["y"].copy(),
        "directions": validation_sequences["directions"].copy(),
        "target_dates": pd.DatetimeIndex(
            validation_sequences["target_dates"]
        ),
        "positive_probabilities": prediction_result[
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
    stage1_feature_columns: list[str],
    stage2_feature_columns: list[str],
    stage1_parameters: dict,
    stage2_parameters: dict,
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
            training_data
            .iloc[train_indices]
            .reset_index(drop=True)
        )
        fold_validation = (
            training_data
            .iloc[validation_indices]
            .reset_index(drop=True)
        )

        print(
            f"    fold {fold_number}/{CV_SPLITS}: "
            "Stage 1 MOVE/FLAT"
        )

        stage1 = train_fold_stage(
            fold_train=fold_train,
            fold_validation=fold_validation,
            feature_columns=stage1_feature_columns,
            task="move",
            parameters=stage1_parameters,
            seed=RANDOM_STATE + fold_number,
        )

        print(
            f"    fold {fold_number}/{CV_SPLITS}: "
            "Stage 2 UP/DOWN"
        )

        stage2 = train_fold_stage(
            fold_train=fold_train,
            fold_validation=fold_validation,
            feature_columns=stage2_feature_columns,
            task="direction",
            parameters=stage2_parameters,
            seed=RANDOM_STATE + 10000 + fold_number,
        )

        if not stage1["target_dates"].equals(
            stage2["target_dates"]
        ):
            raise ValueError(
                "Stage 1 and Stage 2 OOF target dates do not align."
            )

        if not np.array_equal(
            stage1["directions"],
            stage2["directions"],
        ):
            raise ValueError(
                "Stage 1 and Stage 2 OOF directions do not align."
            )

        fold_batches.append(
            {
                "fold": fold_number,
                "directions": stage1["directions"],
                "stage1_actual": stage1["actual"],
                "stage1_probability": stage1[
                    "positive_probabilities"
                ],
                "stage2_probability": stage2[
                    "positive_probabilities"
                ],
            }
        )

    directions = np.concatenate(
        [batch["directions"] for batch in fold_batches]
    ).astype(object)

    stage1_actual = np.concatenate(
        [batch["stage1_actual"] for batch in fold_batches]
    ).astype(np.int64)

    stage1_probability = np.concatenate(
        [batch["stage1_probability"] for batch in fold_batches]
    ).astype(np.float64)

    stage2_probability = np.concatenate(
        [batch["stage2_probability"] for batch in fold_batches]
    ).astype(np.float64)

    actual_move_mask = directions != "FLAT"

    stage2_actual = np.asarray(
        [
            1 if direction == "UP" else 0
            for direction in directions[actual_move_mask]
        ],
        dtype=np.int64,
    )

    stage1_probability_metrics = binary_probability_metrics(
        actual=stage1_actual,
        positive_probabilities=stage1_probability,
    )

    stage2_probability_metrics = binary_probability_metrics(
        actual=stage2_actual,
        positive_probabilities=stage2_probability[actual_move_mask],
    )

    move_threshold_result = (
        HierarchicalXLSTMParameterSelector
        .select_probability_threshold(
            actual=stage1_actual,
            positive_probabilities=stage1_probability,
        )
    )

    up_threshold_result = (
        HierarchicalXLSTMParameterSelector
        .select_probability_threshold(
            actual=stage2_actual,
            positive_probabilities=stage2_probability[actual_move_mask],
        )
    )

    move_threshold = float(
        move_threshold_result["threshold"]
    )
    up_threshold = float(
        up_threshold_result["threshold"]
    )

    stage1_predicted = (
        stage1_probability >= move_threshold
    ).astype(np.int64)

    stage2_predicted = (
        stage2_probability[actual_move_mask] >= up_threshold
    ).astype(np.int64)

    final_predicted = compose_hierarchical_predictions(
        move_probabilities=stage1_probability,
        up_probabilities=stage2_probability,
        move_threshold=move_threshold,
        up_threshold=up_threshold,
    )

    stage1_metrics = binary_metrics(
        actual=stage1_actual,
        predicted=stage1_predicted,
        negative_name="FLAT",
        positive_name="MOVE",
    )

    stage2_metrics = binary_metrics(
        actual=stage2_actual,
        predicted=stage2_predicted,
        negative_name="DOWN",
        positive_name="UP",
    )

    end_to_end_metrics = three_class_metrics(
        actual_directions=directions,
        predicted_directions=final_predicted,
    )

    fold_metrics = []
    offset = 0

    for batch in fold_batches:
        fold_rows = len(batch["directions"])
        fold_slice = slice(offset, offset + fold_rows)

        fold_directions = directions[fold_slice]
        fold_stage1_actual = stage1_actual[fold_slice]
        fold_stage1_probability = stage1_probability[fold_slice]
        fold_stage2_probability = stage2_probability[fold_slice]

        fold_stage1_probability_metrics = binary_probability_metrics(
            actual=fold_stage1_actual,
            positive_probabilities=fold_stage1_probability,
        )

        fold_move_mask = fold_directions != "FLAT"
        fold_stage2_actual = np.asarray(
            [
                1 if direction == "UP" else 0
                for direction in fold_directions[fold_move_mask]
            ],
            dtype=np.int64,
        )

        fold_stage2_probability_metrics = binary_probability_metrics(
            actual=fold_stage2_actual,
            positive_probabilities=fold_stage2_probability[fold_move_mask],
        )

        fold_stage1_predicted = (
            fold_stage1_probability >= move_threshold
        ).astype(np.int64)

        fold_stage2_predicted = (
            fold_stage2_probability[fold_move_mask] >= up_threshold
        ).astype(np.int64)

        fold_final_predicted = compose_hierarchical_predictions(
            move_probabilities=fold_stage1_probability,
            up_probabilities=fold_stage2_probability,
            move_threshold=move_threshold,
            up_threshold=up_threshold,
        )

        fold_stage1_metrics = binary_metrics(
            actual=fold_stage1_actual,
            predicted=fold_stage1_predicted,
            negative_name="FLAT",
            positive_name="MOVE",
        )

        fold_stage2_metrics = binary_metrics(
            actual=fold_stage2_actual,
            predicted=fold_stage2_predicted,
            negative_name="DOWN",
            positive_name="UP",
        )

        fold_end_to_end_metrics = three_class_metrics(
            actual_directions=fold_directions,
            predicted_directions=fold_final_predicted,
        )

        fold_metrics.append(
            {
                "fold": int(batch["fold"]),
                "stage1_roc_auc": float(
                    fold_stage1_probability_metrics["roc_auc"]
                ),
                "stage1_balanced_accuracy": float(
                    fold_stage1_metrics["balanced_accuracy"]
                ),
                "stage1_flat_f1": float(
                    fold_stage1_metrics["per_class"]["FLAT"]["f1"]
                ),
                "stage2_roc_auc": float(
                    fold_stage2_probability_metrics["roc_auc"]
                ),
                "stage2_balanced_accuracy": float(
                    fold_stage2_metrics["balanced_accuracy"]
                ),
                "end_to_end_macro_f1": float(
                    fold_end_to_end_metrics["macro_f1"]
                ),
            }
        )

        offset += fold_rows

    def fold_mean(metric: str) -> float:
        return float(
            np.mean(
                [row[metric] for row in fold_metrics]
            )
        )

    def fold_std(metric: str) -> float:
        return float(
            np.std(
                [row[metric] for row in fold_metrics],
                ddof=0,
            )
        )

    return {
        "move_threshold": move_threshold,
        "up_threshold": up_threshold,
        "stage1_probability": stage1_probability_metrics,
        "stage2_probability_oracle_move_rows": stage2_probability_metrics,
        "stage1": stage1_metrics,
        "stage2_oracle_move_rows": stage2_metrics,
        "end_to_end": end_to_end_metrics,
        "fold_metrics": fold_metrics,
        "fold_summary": {
            "stage1_roc_auc_mean": fold_mean("stage1_roc_auc"),
            "stage1_roc_auc_std": fold_std("stage1_roc_auc"),
            "stage1_balanced_accuracy_mean": fold_mean(
                "stage1_balanced_accuracy"
            ),
            "stage1_flat_f1_mean": fold_mean("stage1_flat_f1"),
            "stage2_roc_auc_mean": fold_mean("stage2_roc_auc"),
            "stage2_roc_auc_std": fold_std("stage2_roc_auc"),
            "stage2_balanced_accuracy_mean": fold_mean(
                "stage2_balanced_accuracy"
            ),
            "end_to_end_macro_f1_mean": fold_mean(
                "end_to_end_macro_f1"
            ),
            "end_to_end_macro_f1_std": fold_std(
                "end_to_end_macro_f1"
            ),
        },
        "oof_rows": int(len(directions)),
    }


def build_target_datasets(
    raw_data: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    datasets = {}
    metadata = {}

    feature_builder = DirectionFeatureBuilder(
        feature_scope="base_trend"
    )

    for target in build_refined_target_candidates():
        label_builder = VolatilityDirectionLabelBuilder(
            volatility_window=target.volatility_window,
            threshold_multiplier=target.threshold_multiplier,
        )

        dataset = DirectionDatasetBuilder(
            feature_builder=feature_builder,
            label_builder=label_builder,
        ).build(raw_data)

        datasets[target.name] = dataset
        metadata[target.name] = {
            "target_name": target.name,
            "target_description": target.description,
            "volatility_window": int(target.volatility_window),
            "threshold_multiplier": float(target.threshold_multiplier),
        }

    return datasets, metadata


def build_summary(
    results: list[dict],
) -> pd.DataFrame:
    rows = []

    for result in results:
        distribution = result["training_target_distribution"]
        metrics = result["metrics"]
        stage1 = metrics["stage1"]
        stage2 = metrics["stage2_oracle_move_rows"]
        end_to_end = metrics["end_to_end"]
        fold_summary = metrics["fold_summary"]

        rows.append(
            {
                "target_name": result["target_name"],
                "volatility_window": result["volatility_window"],
                "threshold_multiplier": result["threshold_multiplier"],
                "feature_profile": result["feature_profile"],
                "stage1_feature_count": len(
                    result["stage1_feature_columns"]
                ),
                "stage2_feature_count": len(
                    result["stage2_feature_columns"]
                ),
                "train_down_share": distribution["down_share"],
                "train_flat_share": distribution["flat_share"],
                "train_up_share": distribution["up_share"],
                "stage1_roc_auc": metrics[
                    "stage1_probability"
                ]["roc_auc"],
                "stage1_roc_auc_fold_std": fold_summary[
                    "stage1_roc_auc_std"
                ],
                "stage1_balanced_accuracy": stage1[
                    "balanced_accuracy"
                ],
                "stage1_flat_f1": stage1[
                    "per_class"
                ]["FLAT"]["f1"],
                "stage1_flat_recall": stage1[
                    "per_class"
                ]["FLAT"]["recall"],
                "stage2_roc_auc": metrics[
                    "stage2_probability_oracle_move_rows"
                ]["roc_auc"],
                "stage2_roc_auc_fold_std": fold_summary[
                    "stage2_roc_auc_std"
                ],
                "stage2_balanced_accuracy": stage2[
                    "balanced_accuracy"
                ],
                "stage2_down_f1": stage2[
                    "per_class"
                ]["DOWN"]["f1"],
                "stage2_up_f1": stage2[
                    "per_class"
                ]["UP"]["f1"],
                "end_to_end_macro_f1": end_to_end["macro_f1"],
                "end_to_end_balanced_accuracy": end_to_end[
                    "balanced_accuracy"
                ],
                "end_to_end_down_f1": end_to_end[
                    "per_class"
                ]["DOWN"]["f1"],
                "end_to_end_flat_f1": end_to_end[
                    "per_class"
                ]["FLAT"]["f1"],
                "end_to_end_up_f1": end_to_end[
                    "per_class"
                ]["UP"]["f1"],
                "end_to_end_macro_f1_fold_std": fold_summary[
                    "end_to_end_macro_f1_std"
                ],
                "move_threshold": metrics["move_threshold"],
                "up_threshold": metrics["up_threshold"],
            }
        )

    summary = pd.DataFrame(rows)

    base_rows = (
        summary[
            summary["feature_profile"] == "base_only"
        ]
        .set_index("target_name")
    )

    delta_metrics = [
        "stage1_roc_auc",
        "stage1_balanced_accuracy",
        "stage1_flat_f1",
        "stage2_roc_auc",
        "stage2_balanced_accuracy",
        "end_to_end_macro_f1",
    ]

    for metric in delta_metrics:
        summary[f"stage_specific_delta_{metric}"] = summary.apply(
            lambda row: (
                row[metric]
                - base_rows.loc[
                    row["target_name"],
                    metric,
                ]
                if row["feature_profile"] == "stage_specific_trend"
                else 0.0
            ),
            axis=1,
        )

    return summary.sort_values(
        [
            "volatility_window",
            "threshold_multiplier",
            "feature_profile",
        ]
    ).reset_index(drop=True)


def print_summary(
    summary: pd.DataFrame,
) -> None:
    print()
    print(
        "============================================================"
    )
    print(
        "REFINED FLAT TARGET x STAGE-SPECIFIC FEATURE SEARCH"
    )
    print(
        "============================================================"
    )
    print(
        "Same common dates, locked xLSTM parameters, train-only "
        "walk-forward folds."
    )
    print(
        "ROC AUC is included so probability discrimination can be "
        "compared without relying only on tuned decision thresholds."
    )
    print(
        "Outer validation and held-out test were NOT used."
    )
    print()

    display = summary[
        [
            "target_name",
            "feature_profile",
            "train_flat_share",
            "stage1_roc_auc",
            "stage1_balanced_accuracy",
            "stage1_flat_f1",
            "stage2_roc_auc",
            "stage2_balanced_accuracy",
            "end_to_end_macro_f1",
            "end_to_end_macro_f1_fold_std",
            "stage_specific_delta_stage1_roc_auc",
            "stage_specific_delta_stage2_roc_auc",
            "stage_specific_delta_end_to_end_macro_f1",
        ]
    ].copy()

    display["train_flat_share"] = (
        display["train_flat_share"] * 100.0
    )

    numeric_columns = [
        column
        for column in display.columns
        if column not in {
            "target_name",
            "feature_profile",
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
        "Selection rule: look for a stable target region where Stage 1 "
        "separates FLAT/MOVE above chance, Stage 2 retains directional "
        "signal, and gains persist across folds. Do not promote a target "
        "only because one aggregate Macro F1 is largest."
    )


def main():
    print(
        "Loading base SPY/VIX/VVIX data..."
    )

    raw_data = (
        DirectionTrainingDataRepository()
        .get_training_data(
            ticker=TICKER,
            include_breadth=False,
            include_cross_asset=False,
        )
    )

    print(
        f"Raw rows: {len(raw_data)}"
    )

    base_builder = DirectionFeatureBuilder(
        feature_scope="base"
    )
    base_trend_builder = DirectionFeatureBuilder(
        feature_scope="base_trend"
    )

    base_columns = list(
        base_builder.feature_columns
    )
    base_trend_columns = list(
        base_trend_builder.feature_columns
    )

    trend_groups = classify_trend_features(
        base_columns=base_columns,
        base_trend_columns=base_trend_columns,
    )

    feature_profiles = build_stage_feature_profiles(
        base_columns=base_columns,
        base_trend_columns=base_trend_columns,
    )

    print()
    print(
        "Stage-specific trend contract:"
    )
    print(
        "  Stage 1 strength/state:",
        trend_groups["strength"],
    )
    print(
        "  Stage 2 directional:",
        trend_groups["directional"],
    )

    if trend_groups["unassigned"]:
        print(
            "  Trend features intentionally not used in stage-specific profile:",
            trend_groups["unassigned"],
        )

    print()
    print(
        "Building refined target datasets..."
    )

    datasets, target_metadata = build_target_datasets(
        raw_data
    )

    datasets = align_datasets_on_common_target_dates(
        datasets
    )

    print(
        "Common aligned rows:",
        len(next(iter(datasets.values()))),
    )

    reference_path, reference_experiment = (
        load_latest_hierarchical_experiment(
            EXPERIMENT_DIRECTORY
        )
    )

    stage1_parameters = dict(
        reference_experiment["parameters"]
        ["stage1"]["parameters"]
    )
    stage2_parameters = dict(
        reference_experiment["parameters"]
        ["stage2"]["parameters"]
    )

    print(
        "Locked parameter source:",
        reference_path,
    )
    print(
        "This experiment does NOT run Optuna and does NOT touch outer validation."
    )

    split_reference_name = next(iter(datasets))
    split_reference_train, _, _ = (
        DateAwareDataSplitter().split(
            datasets[split_reference_name],
            date_column="target_date",
        )
    )

    training_target_dates = set(
        pd.to_datetime(
            split_reference_train["target_date"]
        )
    )

    target_candidates = build_refined_target_candidates()
    total_candidates = len(target_candidates) * len(feature_profiles)
    results = []
    candidate_number = 0

    for target in target_candidates:
        dataset = datasets[target.name]
        train = (
            dataset[
                pd.to_datetime(
                    dataset["target_date"]
                ).isin(training_target_dates)
            ]
            .sort_values("target_date")
            .reset_index(drop=True)
        )

        if set(
            pd.to_datetime(train["target_date"])
        ) != training_target_dates:
            raise ValueError(
                f"Training dates are not aligned for {target.name}."
            )

        distribution = target_distribution(train)

        for profile in feature_profiles:
            candidate_number += 1

            print()
            print(
                f"[{candidate_number}/{total_candidates}] "
                f"{target.name} | {profile.name}"
            )
            print(
                f"  target: {target.volatility_window}d x "
                f"{target.threshold_multiplier:.2f}"
            )
            print(
                "  train DOWN / FLAT / UP: "
                f"{distribution['down_share'] * 100:.1f}% / "
                f"{distribution['flat_share'] * 100:.1f}% / "
                f"{distribution['up_share'] * 100:.1f}%"
            )
            print(
                "  Stage 1 features:",
                len(profile.stage1_feature_columns),
            )
            print(
                "  Stage 2 features:",
                len(profile.stage2_feature_columns),
            )

            metrics = evaluate_candidate(
                training_data=train,
                stage1_feature_columns=list(
                    profile.stage1_feature_columns
                ),
                stage2_feature_columns=list(
                    profile.stage2_feature_columns
                ),
                stage1_parameters=stage1_parameters,
                stage2_parameters=stage2_parameters,
            )

            print(
                "  Stage 1 ROC AUC:",
                round(
                    metrics["stage1_probability"]["roc_auc"],
                    4,
                ),
            )
            print(
                "  Stage 1 balanced accuracy:",
                round(
                    metrics["stage1"]["balanced_accuracy"],
                    4,
                ),
            )
            print(
                "  Stage 1 FLAT F1:",
                round(
                    metrics["stage1"]["per_class"]["FLAT"]["f1"],
                    4,
                ),
            )
            print(
                "  Stage 2 ROC AUC:",
                round(
                    metrics[
                        "stage2_probability_oracle_move_rows"
                    ]["roc_auc"],
                    4,
                ),
            )
            print(
                "  Stage 2 balanced accuracy:",
                round(
                    metrics["stage2_oracle_move_rows"][
                        "balanced_accuracy"
                    ],
                    4,
                ),
            )
            print(
                "  End-to-end Macro F1:",
                round(
                    metrics["end_to_end"]["macro_f1"],
                    4,
                ),
            )

            results.append(
                {
                    **target_metadata[target.name],
                    "feature_profile": profile.name,
                    "feature_profile_description": profile.description,
                    "stage1_feature_columns": list(
                        profile.stage1_feature_columns
                    ),
                    "stage2_feature_columns": list(
                        profile.stage2_feature_columns
                    ),
                    "training_target_distribution": distribution,
                    "metrics": metrics,
                }
            )

    summary = build_summary(results)
    print_summary(summary)

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    EXPERIMENT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        EXPERIMENT_DIRECTORY
        / f"refined_flat_stage_feature_search_{timestamp}.csv"
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
            "reference_hierarchical_experiment": str(reference_path),
            "cv_splits": CV_SPLITS,
            "stage1_locked_parameters": stage1_parameters,
            "stage2_locked_parameters": stage2_parameters,
            "target_candidates": [
                {
                    "name": target.name,
                    "volatility_window": target.volatility_window,
                    "threshold_multiplier": target.threshold_multiplier,
                    "description": target.description,
                }
                for target in target_candidates
            ],
            "feature_profiles": [
                {
                    "name": profile.name,
                    "stage1_features": list(
                        profile.stage1_feature_columns
                    ),
                    "stage2_features": list(
                        profile.stage2_feature_columns
                    ),
                    "description": profile.description,
                }
                for profile in feature_profiles
            ],
            "trend_groups": trend_groups,
            "selection_policy": (
                "Research sweep only. Candidates use common dates and locked "
                "model parameters. ROC AUC is reported as a threshold-free "
                "discrimination metric. Probability thresholds are calibrated "
                "only from training-period OOF predictions for research "
                "comparison. Any promoted target/profile must be independently "
                "retuned before outer-validation evaluation."
            ),
        },
        metrics={
            "candidates": results,
        },
        features=list(base_trend_columns),
    )

    print()
    print(
        "Saved summary:",
        summary_path,
    )
    print(
        "Saved experiment:",
        experiment_path,
    )
    print()
    print(
        "Outer validation was NOT evaluated."
    )
    print(
        "Held-out test set was NOT evaluated."
    )


if __name__ == "__main__":
    main()
