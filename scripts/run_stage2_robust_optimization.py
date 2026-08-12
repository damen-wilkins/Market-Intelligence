from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from app.training.date_aware_data_splitter import DateAwareDataSplitter
from app.training.experiment_tracker import ExperimentTracker
from app.training.hierarchical_xlstm_parameter_selector import (
    HierarchicalXLSTMParameterSelector,
)
from app.training.stage2_wide_feature_builder import Stage2WideFeatureBuilder
from app.training.volatility_direction_label_builder import (
    VolatilityDirectionLabelBuilder,
)
from database.stage2_signal_data_repository import (
    Stage2SignalDataRepository,
)


TICKER = "SPY"
TARGET_VOLATILITY_WINDOW = 40
TARGET_THRESHOLD_MULTIPLIER = 0.45

CV_SPLITS = 3
OPTUNA_TRIALS = 60
MAX_SELECTION_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 10
RANDOM_STATE = 42

EXPERIMENT_DIRECTORY = Path("experiments")
OPTUNA_STORAGE_URL = (
    "sqlite:///experiments/optuna_stage2_robust_optimization.db"
)
EXPERIMENT_NAME = "xlstm_stage2_robust_optimization_v1"
MODEL_NAME = "xlstm_stage2_robust_optimization_v1"

CANDIDATES = {
    "breadth_calendar": (
        "breadth",
        "calendar",
    ),
    "futures_core": (
        "futures_core",
    ),
    "broad_risk_directional": (
        "breadth",
        "futures_core",
        "interaction_consensus",
        "rates_credit",
        "trend_direction",
    ),
}


def build_master_training_data(
    raw_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    feature_library = (
        Stage2WideFeatureBuilder()
        .build_library(
            raw_data
        )
    )

    labels = (
        VolatilityDirectionLabelBuilder(
            volatility_window=(
                TARGET_VOLATILITY_WINDOW
            ),
            threshold_multiplier=(
                TARGET_THRESHOLD_MULTIPLIER
            ),
        )
        .build(
            raw_data[
                [
                    "trade_date",
                    "close",
                ]
            ].copy()
        )
    )

    master = (
        feature_library
        .rename(
            columns={
                "trade_date": "feature_date",
            }
        )
        .merge(
            labels,
            on="feature_date",
            how="inner",
            validate="one_to_one",
        )
        .sort_values(
            "target_date"
        )
        .reset_index(
            drop=True
        )
    )

    base_columns = list(
        Stage2WideFeatureBuilder
        .BASE_FEATURE_COLUMNS
    )

    base_dataset = (
        master
        .dropna(
            subset=base_columns
        )
        .reset_index(
            drop=True
        )
    )

    train, _, _ = (
        DateAwareDataSplitter()
        .split(
            base_dataset,
            date_column="target_date",
        )
    )

    training_end = pd.Timestamp(
        train[
            "target_date"
        ].max()
    )

    master = (
        master[
            pd.to_datetime(
                master[
                    "target_date"
                ]
            )
            <= training_end
        ]
        .reset_index(
            drop=True
        )
    )

    return master, training_end


def candidate_training_data(
    master: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    data = (
        master
        .dropna(
            subset=feature_columns
        )
        .copy()
    )

    return (
        data[
            [
                "feature_date",
                "target_date",
                *feature_columns,
                "future_log_return",
                "rolling_volatility",
                "threshold",
                "direction",
            ]
        ]
        .sort_values(
            "target_date"
        )
        .reset_index(
            drop=True
        )
    )


def matched_base_data(
    candidate_data: pd.DataFrame,
) -> pd.DataFrame:
    base_columns = list(
        Stage2WideFeatureBuilder
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


def sample_signature(
    data: pd.DataFrame,
) -> str:
    dates = (
        pd.to_datetime(
            data[
                "target_date"
            ]
        )
        .astype(
            "int64"
        )
        .to_numpy()
    )

    return hashlib.sha1(
        dates.tobytes()
    ).hexdigest()[:12]


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


def study_name(
    prefix: str,
    data: pd.DataFrame,
    feature_columns: list[str],
) -> str:
    normalized = re.sub(
        r"[^a-zA-Z0-9_]+",
        "_",
        prefix,
    )

    return (
        f"{normalized}_"
        f"{sample_signature(data)}_"
        f"{feature_signature(feature_columns)}"
    )


def optimize(
    name: str,
    data: pd.DataFrame,
    feature_columns: list[str],
    task: str,
) -> dict:
    print()
    print(
        f"OPTUNA: {name}"
    )
    print(
        f"Rows: {len(data)}"
    )
    print(
        "Period:",
        pd.Timestamp(
            data[
                "target_date"
            ].min()
        ).date(),
        "->",
        pd.Timestamp(
            data[
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
            task=task,
            n_splits=CV_SPLITS,
            n_trials=OPTUNA_TRIALS,
            max_epochs=(
                MAX_SELECTION_EPOCHS
            ),
            patience=(
                EARLY_STOPPING_PATIENCE
            ),
            random_state=(
                RANDOM_STATE
            ),
            objective_metric="roc_auc",
            study_name=study_name(
                prefix=name,
                data=data,
                feature_columns=(
                    feature_columns
                ),
            ),
            storage_url=(
                OPTUNA_STORAGE_URL
            ),
        )
    )

    return (
        selector
        .select_best_parameters(
            training_data=data
        )
    )


def save_progress(
    rows: list[dict],
) -> Path:
    EXPERIMENT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        EXPERIMENT_DIRECTORY
        / "stage2_robust_optimization_v1_progress.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            rows,
            file,
            indent=2,
        )

    return path


def main():
    print(
        "Loading Stage-2 wide signal data..."
    )

    raw_data = (
        Stage2SignalDataRepository()
        .get_training_data(
            ticker=TICKER
        )
    )

    master, training_end = (
        build_master_training_data(
            raw_data
        )
    )

    print(
        "Target:",
        f"{TARGET_VOLATILITY_WINDOW}d x "
        f"{TARGET_THRESHOLD_MULTIPLIER}",
    )
    print(
        "Fixed training cutoff:",
        training_end.date(),
    )
    print(
        "Optuna objective: train-only "
        "walk-forward ROC AUC."
    )
    print(
        "Each candidate gets a separately tuned "
        "BASE control on the exact same dates."
    )
    print(
        "Outer validation and held-out test "
        "are NOT evaluated."
    )

    base_columns = list(
        Stage2WideFeatureBuilder
        .BASE_FEATURE_COLUMNS
    )

    baseline_cache = {}
    results = []

    for (
        candidate_name,
        groups,
    ) in CANDIDATES.items():
        feature_columns = (
            Stage2WideFeatureBuilder
            .columns_for_groups(
                groups
            )
        )

        data = candidate_training_data(
            master=master,
            feature_columns=(
                feature_columns
            ),
        )

        signature = sample_signature(
            data
        )

        print()
        print(
            "=" * 72
        )
        print(
            f"CANDIDATE: {candidate_name}"
        )
        print(
            "Groups:",
            list(
                groups
            ),
        )
        print(
            "=" * 72
        )

        candidate_selection = optimize(
            name=(
                f"stage2_{candidate_name}"
            ),
            data=data,
            feature_columns=(
                feature_columns
            ),
            task="direction",
        )

        if signature not in baseline_cache:
            baseline_cache[
                signature
            ] = optimize(
                name=(
                    "stage2_matched_base_"
                    f"{signature}"
                ),
                data=(
                    matched_base_data(
                        data
                    )
                ),
                feature_columns=(
                    base_columns
                ),
                task="direction",
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
                candidate_name
            ),
            "groups": list(
                groups
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

        save_progress(
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

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )

    summary_path = (
        EXPERIMENT_DIRECTORY
        / (
            "stage2_robust_optimization_v1_"
            f"{timestamp}.csv"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    experiment_path = (
        ExperimentTracker(
            str(
                EXPERIMENT_DIRECTORY
            )
        )
        .save(
            experiment_name=(
                EXPERIMENT_NAME
            ),
            model_name=(
                MODEL_NAME
            ),
            parameters={
                "target_volatility_window": (
                    TARGET_VOLATILITY_WINDOW
                ),
                "target_threshold_multiplier": (
                    TARGET_THRESHOLD_MULTIPLIER
                ),
                "cv_splits": (
                    CV_SPLITS
                ),
                "optuna_trials_per_study": (
                    OPTUNA_TRIALS
                ),
                "max_selection_epochs": (
                    MAX_SELECTION_EPOCHS
                ),
                "objective_metric": (
                    "roc_auc"
                ),
                "candidate_groups": (
                    CANDIDATES
                ),
                "outer_validation_used": (
                    False
                ),
                "held_out_test_used": (
                    False
                ),
            },
            metrics={
                "results": (
                    results
                ),
            },
            features=list(
                Stage2WideFeatureBuilder
                .FEATURE_COLUMNS
            ),
        )
    )

    print()
    print(
        "=" * 88
    )
    print(
        "STAGE-2 ROBUST OPTIMIZATION - FINAL RANKING"
    )
    print(
        "=" * 88
    )

    display_columns = [
        "candidate_name",
        "feature_count",
        "training_rows",
        "candidate_oof_roc_auc",
        "matched_base_oof_roc_auc",
        "delta_oof_roc_auc_vs_tuned_base",
        "candidate_oof_roc_auc_fold_std",
        "candidate_oof_balanced_accuracy",
        "candidate_oof_macro_f1",
    ]

    print(
        summary[
            display_columns
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Summary:",
        summary_path,
    )
    print(
        "Experiment:",
        experiment_path,
    )
    print(
        "Optuna DB:",
        OPTUNA_STORAGE_URL,
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
