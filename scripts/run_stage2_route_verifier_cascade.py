from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from app.training.hierarchical_xlstm_parameter_selector import (
    HierarchicalXLSTMParameterSelector,
)
from app.training.stage2_route_verifier_research import (
    Stage2RouteVerifierResearch,
)
from scripts.run_stage2_route_aware_multiclass import (
    build_stage1_oof_routes,
    fit_binary_control,
    load_outer_test_route,
    route_aware_test_frame,
)
from scripts.run_stage2_route_compatibility_diagnostic import (
    BOOTSTRAP_BLOCK_LENGTH,
    BOOTSTRAP_RESAMPLES,
    EXPERIMENT_DIRECTORY,
    OUTER_SPLITS,
    RANDOM_STATE,
    REGIME_FEATURE,
    REGIME_QUANTILE,
    TARGET_MULTIPLIER,
    TARGET_NAME,
    TARGET_WINDOW,
    build_stage1_development,
    build_stage2_development,
    load_locked_stage1,
    load_stage2_saved_oof,
    save_json,
    stage1_fold_data,
)


INNER_SPLITS = 3
VERIFIER_OPTUNA_TRIALS = 50
VERIFIER_STORAGE_URL = (
    "sqlite:///experiments/optuna_stage2_route_verifier_v1.db"
)
EXPERIMENT_NAME = "stage2_route_verifier_cascade_v1"
PROGRESS_PATH = (
    EXPERIMENT_DIRECTORY
    / f"{EXPERIMENT_NAME}_progress.json"
)
ROUTE_DIAGNOSTIC_PREFIX = "stage2_route_compatibility_diagnostic_v1_"
MULTICLASS_PREFIX = "stage2_route_aware_multiclass_v1_"
MIN_VERIFIER_AUC = 0.55
MIN_SELECTIVE_COVERAGE = 0.60


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Cannot serialize {type(value).__name__}."
    )


def latest_completed_json(
    prefix: str,
) -> tuple[Path | None, dict | None]:
    candidates = []
    for path in EXPERIMENT_DIRECTORY.glob(f"{prefix}*.json"):
        if path.name.endswith("_progress.json"):
            continue
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception:
            continue
        candidates.append((path, payload))
    if not candidates:
        return None, None
    return max(
        candidates,
        key=lambda item: item[0].stat().st_mtime,
    )


def verify_prerequisites() -> tuple[Path, Path]:
    diagnostic_path, diagnostic = latest_completed_json(
        ROUTE_DIAGNOSTIC_PREFIX
    )
    if diagnostic_path is None:
        raise RuntimeError(
            "No completed Stage-2 route compatibility diagnostic was found."
        )
    if (
        diagnostic.get("summary", {}).get(
            "development_confirms_route_compatibility_problem"
        )
        is not True
    ):
        raise RuntimeError(
            "The route compatibility diagnostic does not confirm the "
            "Stage1->Stage2 population mismatch."
        )

    multiclass_path, multiclass = latest_completed_json(
        MULTICLASS_PREFIX
    )
    if multiclass_path is None:
        raise RuntimeError(
            "No completed route-aware multiclass experiment was found."
        )
    multiclass_gate = (
        multiclass.get("summary", {})
        .get("gates", {})
        .get("overall_route_aware_architecture_gate")
    )
    if multiclass_gate is not False:
        raise RuntimeError(
            "This cascade experiment is intended only after the route-aware "
            "multiclass architecture has failed development gating."
        )
    return diagnostic_path, multiclass_path


def build_verifier_training_frame(
    stage2_outer_train: pd.DataFrame,
    stage1_oof: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    stage2 = stage2_outer_train[
        [
            "target_date",
            "direction",
            *feature_columns,
        ]
    ].copy()
    merged = stage2.merge(
        stage1_oof[
            [
                "target_date",
                "direction",
                "stage1_move_probability",
                "stage1_predicted_move",
            ]
        ],
        on="target_date",
        how="inner",
        suffixes=("_stage2", "_stage1"),
        validate="one_to_one",
    )
    mismatch = (
        merged["direction_stage2"].astype(str)
        != merged["direction_stage1"].astype(str)
    )
    if mismatch.any():
        raise ValueError(
            "Stage-1 and Stage-2 labels disagree in verifier training alignment."
        )
    merged = merged.rename(
        columns={"direction_stage2": "direction"}
    ).drop(
        columns=["direction_stage1"]
    )
    routed = merged.loc[
        merged["stage1_predicted_move"].astype(bool)
    ].copy()
    if routed.empty:
        raise ValueError(
            "Stage-1 OOF routing produced no verifier training rows."
        )
    routed["actual_move"] = (
        routed["direction"].astype(str) != "FLAT"
    ).astype(int)
    if routed["actual_move"].nunique() != 2:
        raise ValueError(
            "Verifier training requires both FLAT and MOVE labels."
        )
    return routed.sort_values(
        "target_date"
    ).reset_index(
        drop=True
    )


def verifier_parameter_space(
    trial: optuna.Trial,
) -> dict:
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            1000,
            step=50,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            2,
            8,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.005,
            0.2,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0,
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            0.001,
            10.0,
            log=True,
        ),
        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            0.001,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-10,
            0.1,
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-4,
            20.0,
            log=True,
        ),
    }


def build_verifier_model(
    parameters: dict,
    seed: int,
) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=int(seed),
        n_jobs=-1,
        **parameters,
    )


def tune_verifier(
    training: pd.DataFrame,
    model_features: list[str],
    outer_fold: int,
) -> tuple[dict, float, int]:
    splitter = TimeSeriesSplit(
        n_splits=INNER_SPLITS
    )

    def objective(
        trial: optuna.Trial,
    ) -> float:
        parameters = verifier_parameter_space(
            trial
        )
        fold_scores = []
        for inner_fold, (
            train_index,
            validation_index,
        ) in enumerate(
            splitter.split(training),
            start=1,
        ):
            train = training.iloc[
                train_index
            ].reset_index(
                drop=True
            )
            validation = training.iloc[
                validation_index
            ].reset_index(
                drop=True
            )
            y_train = train[
                "actual_move"
            ].astype(
                int
            ).to_numpy()
            y_validation = validation[
                "actual_move"
            ].astype(
                int
            ).to_numpy()
            if np.unique(y_train).size != 2:
                continue
            if np.unique(y_validation).size != 2:
                continue
            weights = compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )
            model = build_verifier_model(
                parameters=parameters,
                seed=(
                    RANDOM_STATE
                    + 100000
                    + outer_fold * 100
                    + inner_fold
                ),
            )
            model.fit(
                train[model_features],
                y_train,
                sample_weight=weights,
            )
            score = model.predict_proba(
                validation[model_features]
            )[:, 1]
            fold_scores.append(
                float(
                    roc_auc_score(
                        y_validation,
                        score,
                    )
                )
            )
        if not fold_scores:
            return 0.5
        return float(
            np.mean(
                fold_scores
            )
        )

    study_name = (
        f"stage2_route_verifier_outer_{outer_fold}"
    )
    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE + outer_fold
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=VERIFIER_STORAGE_URL,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    completed = len(
        [
            trial
            for trial in study.trials
            if trial.state
            == optuna.trial.TrialState.COMPLETE
        ]
    )
    remaining = max(
        0,
        VERIFIER_OPTUNA_TRIALS - completed,
    )
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            show_progress_bar=False,
        )
    return (
        dict(study.best_trial.params),
        float(study.best_value),
        int(len(study.trials)),
    )


def verifier_oof_threshold(
    training: pd.DataFrame,
    model_features: list[str],
    parameters: dict,
    outer_fold: int,
) -> tuple[float, pd.DataFrame]:
    splitter = TimeSeriesSplit(
        n_splits=INNER_SPLITS
    )
    parts = []
    for inner_fold, (
        train_index,
        validation_index,
    ) in enumerate(
        splitter.split(training),
        start=1,
    ):
        train = training.iloc[
            train_index
        ].reset_index(
            drop=True
        )
        validation = training.iloc[
            validation_index
        ].reset_index(
            drop=True
        )
        y_train = train[
            "actual_move"
        ].astype(
            int
        ).to_numpy()
        if np.unique(y_train).size != 2:
            raise ValueError(
                f"Verifier OOF outer {outer_fold} inner {inner_fold} "
                "training sample has only one class."
            )
        weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )
        model = build_verifier_model(
            parameters=parameters,
            seed=(
                RANDOM_STATE
                + 200000
                + outer_fold * 100
                + inner_fold
            ),
        )
        model.fit(
            train[model_features],
            y_train,
            sample_weight=weights,
        )
        score = model.predict_proba(
            validation[model_features]
        )[:, 1].astype(
            np.float64
        )
        frame = validation[
            [
                "target_date",
                "actual_move",
            ]
        ].copy()
        frame[
            "verifier_move_probability"
        ] = score
        parts.append(
            frame
        )

    oof = pd.concat(
        parts,
        ignore_index=True,
    ).sort_values(
        "target_date"
    ).reset_index(
        drop=True
    )
    selection = (
        HierarchicalXLSTMParameterSelector
        .select_probability_threshold(
            actual=oof[
                "actual_move"
            ].astype(
                int
            ).to_numpy(),
            positive_probabilities=oof[
                "verifier_move_probability"
            ].astype(
                float
            ).to_numpy(),
        )
    )
    return (
        float(selection["threshold"]),
        oof,
    )


def fit_verifier(
    training: pd.DataFrame,
    test: pd.DataFrame,
    model_features: list[str],
    parameters: dict,
    outer_fold: int,
) -> np.ndarray:
    y_train = training[
        "actual_move"
    ].astype(
        int
    ).to_numpy()
    weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )
    model = build_verifier_model(
        parameters=parameters,
        seed=(
            RANDOM_STATE
            + 300000
            + outer_fold
        ),
    )
    model.fit(
        training[model_features],
        y_train,
        sample_weight=weights,
    )
    return model.predict_proba(
        test[model_features]
    )[:, 1].astype(
        np.float64
    )


def build_predictions(
    test: pd.DataFrame,
    binary_scores: np.ndarray,
    binary_threshold: float,
    verifier_scores: np.ndarray,
    verifier_threshold: float,
    high_volatility_threshold: float,
) -> pd.DataFrame:
    frame = test[
        [
            "target_date",
            "direction",
            "future_log_return",
            "stage1_move_probability",
            "stage1_predicted_move",
            REGIME_FEATURE,
        ]
    ].copy()
    frame = frame.rename(
        columns={
            "direction": "actual_direction",
        }
    )
    frame[
        "binary_up_score"
    ] = np.asarray(
        binary_scores,
        dtype=np.float64,
    )
    frame[
        "binary_direction"
    ] = np.where(
        frame[
            "binary_up_score"
        ].to_numpy()
        >= float(binary_threshold),
        "UP",
        "DOWN",
    )
    frame[
        "verifier_move_probability"
    ] = np.asarray(
        verifier_scores,
        dtype=np.float64,
    )

    routed = frame[
        "stage1_predicted_move"
    ].astype(
        bool
    ).to_numpy()
    verifier_move = (
        frame[
            "verifier_move_probability"
        ].to_numpy()
        >= float(verifier_threshold)
    )
    confirmed = (
        routed
        & verifier_move
    )
    high_vol = (
        frame[
            REGIME_FEATURE
        ].astype(
            float
        ).to_numpy()
        > float(
            high_volatility_threshold
        )
    )
    selective_direction = (
        confirmed
        & high_vol
    )

    frame[
        "verifier_confirmed_move"
    ] = confirmed
    frame[
        "high_volatility_regime"
    ] = high_vol
    frame[
        "baseline_prediction"
    ] = np.where(
        routed,
        frame[
            "binary_direction"
        ],
        "FLAT",
    )
    frame[
        "verifier_cascade_prediction"
    ] = np.where(
        confirmed,
        frame[
            "binary_direction"
        ],
        "FLAT",
    )

    selective = np.full(
        len(frame),
        "ABSTAIN",
        dtype=object,
    )
    selective[
        ~routed
    ] = "FLAT"
    selective[
        routed
        & ~verifier_move
    ] = "FLAT"
    selective[
        selective_direction
    ] = frame.loc[
        selective_direction,
        "binary_direction",
    ].astype(
        str
    ).to_numpy()
    frame[
        "selective_prediction"
    ] = selective
    return frame


def cached_fold_map() -> dict[int, dict]:
    if not PROGRESS_PATH.exists():
        return {}
    with PROGRESS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )
    return {
        int(row["outer_fold"]): row
        for row in payload.get(
            "folds",
            [],
        )
    }


def main():
    EXPERIMENT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    diagnostic_path, multiclass_path = verify_prerequisites()
    locked_stage1 = load_locked_stage1()
    saved_stage2 = load_stage2_saved_oof()
    (
        stage2_data,
        winner_groups,
        winner_features,
        cutoff,
    ) = build_stage2_development()
    stage1_data = build_stage1_development(
        cutoff
    )
    model_features = [
        *winner_features,
        "stage1_move_probability",
    ]
    research = Stage2RouteVerifierResearch(
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_block_length=BOOTSTRAP_BLOCK_LENGTH,
        random_state=RANDOM_STATE,
    )

    print("=" * 88)
    print("STAGE-2 ROUTE VERIFIER CASCADE V1")
    print("=" * 88)
    print(
        f"Target: {TARGET_WINDOW}d x {TARGET_MULTIPLIER:.3f}"
    )
    print(
        "Architecture: Stage 1 FLAT/MOVE -> route verifier FLAT/MOVE -> "
        "existing binary Stage 2 DOWN/UP"
    )
    print(
        "Verifier training population: Stage-1-predicted MOVE rows from "
        "Stage-1 inner OOF predictions only"
    )
    print(
        f"Verifier inputs: locked {len(winner_features)} Stage-2 features "
        "+ Stage-1 MOVE probability"
    )
    print(
        f"Verifier Optuna: {VERIFIER_OPTUNA_TRIALS} train-only trials per outer fold, "
        "3-fold walk-forward AUC objective"
    )
    print(
        "Verifier threshold: selected from verifier OOF predictions inside each outer fold"
    )
    print(
        "Direction model: existing binary XGBoost; exact original outer-fold "
        "parameters and thresholds"
    )
    print(
        "Secondary pre-specified policy: after verifier confirmation, allow "
        "UP/DOWN only in HIGH realized_volatility_20; otherwise ABSTAIN"
    )
    print(
        f"Development cutoff: {cutoff.date()}"
    )
    print(
        "Confirmed route diagnostic:",
        diagnostic_path,
    )
    print(
        "Failed multiclass predecessor:",
        multiclass_path,
    )
    print()
    print(
        "DEVELOPMENT ONLY: no target search, feature search, regime search, "
        "or direction-model retuning."
    )
    print(
        "The new verifier is a genuinely new binary task, so its XGBoost "
        "hyperparameters are selected independently inside each outer fold."
    )
    print(
        "Outer validation and the final held-out test are NOT loaded or evaluated."
    )

    splitter = TimeSeriesSplit(
        n_splits=OUTER_SPLITS
    )
    completed = cached_fold_map()
    fold_rows = []
    prediction_parts = []

    for outer_fold, (
        train_index,
        test_index,
    ) in enumerate(
        splitter.split(
            stage2_data
        ),
        start=1,
    ):
        print()
        print(
            f"outer fold {outer_fold}/{OUTER_SPLITS}"
        )
        if outer_fold in completed:
            print(
                "  using cached verifier-cascade fold result"
            )
            cached = completed[
                outer_fold
            ]
            fold_rows.append(
                cached[
                    "metrics"
                ]
            )
            predictions = pd.DataFrame(
                cached[
                    "predictions"
                ]
            )
            predictions[
                "target_date"
            ] = pd.to_datetime(
                predictions[
                    "target_date"
                ]
            )
            prediction_parts.append(
                predictions
            )
            continue

        stage2_outer_train = stage2_data.iloc[
            train_index
        ].reset_index(
            drop=True
        )
        stage2_outer_test = stage2_data.iloc[
            test_index
        ].reset_index(
            drop=True
        )
        stage1_train, _ = stage1_fold_data(
            stage1_data,
            stage2_outer_train,
            stage2_outer_test,
        )

        print(
            "  reconstructing Stage-1 inner OOF routes..."
        )
        stage1_oof, _ = build_stage1_oof_routes(
            training_data=stage1_train,
            locked_parameters=dict(
                locked_stage1[
                    "parameters"
                ]
            ),
            outer_fold=outer_fold,
        )
        verifier_training = build_verifier_training_frame(
            stage2_outer_train=stage2_outer_train,
            stage1_oof=stage1_oof,
            feature_columns=winner_features,
        )
        print(
            f"  verifier training rows: {len(verifier_training)} | "
            f"MOVE share {verifier_training['actual_move'].mean():.2%}"
        )

        print(
            "  tuning route verifier on outer-train only..."
        )
        (
            verifier_parameters,
            verifier_cv_auc,
            verifier_study_trials,
        ) = tune_verifier(
            training=verifier_training,
            model_features=model_features,
            outer_fold=outer_fold,
        )
        print(
            f"  verifier best train-only CV AUC: {verifier_cv_auc:.4f}"
        )

        (
            verifier_threshold,
            verifier_oof,
        ) = verifier_oof_threshold(
            training=verifier_training,
            model_features=model_features,
            parameters=verifier_parameters,
            outer_fold=outer_fold,
        )
        verifier_oof_auc = float(
            roc_auc_score(
                verifier_oof[
                    "actual_move"
                ].astype(
                    int
                ),
                verifier_oof[
                    "verifier_move_probability"
                ].astype(
                    float
                ),
            )
        )
        print(
            f"  verifier OOF threshold: {verifier_threshold:.6f} | "
            f"OOF AUC {verifier_oof_auc:.4f}"
        )

        route_test = load_outer_test_route(
            outer_fold
        )
        stage2_test = route_aware_test_frame(
            stage2_outer_test=stage2_outer_test,
            route=route_test,
            feature_columns=winner_features,
        )
        binary_parameters = (
            __import__(
                "scripts.run_stage2_route_aware_multiclass",
                fromlist=["load_fold_parameters"],
            )
            .load_fold_parameters(
                outer_fold
            )
        )
        binary_scores = fit_binary_control(
            outer_fold=outer_fold,
            outer_train=stage2_outer_train,
            outer_test=stage2_outer_test,
            feature_columns=winner_features,
            parameters=binary_parameters,
            saved=saved_stage2[
                outer_fold
            ],
        )
        verifier_scores = fit_verifier(
            training=verifier_training,
            test=stage2_test,
            model_features=model_features,
            parameters=verifier_parameters,
            outer_fold=outer_fold,
        )

        development_move = stage2_outer_train.loc[
            stage2_outer_train[
                "direction"
            ].astype(
                str
            )
            != "FLAT"
        ]
        high_volatility_threshold = float(
            development_move[
                REGIME_FEATURE
            ].astype(
                float
            ).quantile(
                REGIME_QUANTILE
            )
        )

        predictions = build_predictions(
            test=stage2_test,
            binary_scores=binary_scores,
            binary_threshold=float(
                saved_stage2[
                    outer_fold
                ][
                    "decision_threshold"
                ]
            ),
            verifier_scores=verifier_scores,
            verifier_threshold=verifier_threshold,
            high_volatility_threshold=high_volatility_threshold,
        )
        predictions[
            "outer_fold"
        ] = outer_fold

        actual = predictions[
            "actual_direction"
        ].astype(
            str
        )
        baseline_metrics = research.three_class_metrics(
            actual,
            predictions[
                "baseline_prediction"
            ],
        )
        cascade_metrics = research.three_class_metrics(
            actual,
            predictions[
                "verifier_cascade_prediction"
            ],
        )

        routed = predictions.loc[
            predictions[
                "stage1_predicted_move"
            ].astype(
                bool
            )
        ].copy()
        routed[
            "actual_move"
        ] = (
            routed[
                "actual_direction"
            ].astype(
                str
            )
            != "FLAT"
        ).astype(
            int
        )
        verifier_test_metrics = research.binary_metrics(
            actual=routed[
                "actual_move"
            ],
            score=routed[
                "verifier_move_probability"
            ],
            threshold=verifier_threshold,
        )
        route_diagnostics = research.route_diagnostics(
            predictions,
            confirmed_column="verifier_confirmed_move",
        )

        confirmed_true_move = predictions.loc[
            predictions[
                "verifier_confirmed_move"
            ].astype(
                bool
            )
            & (
                predictions[
                    "actual_direction"
                ].astype(
                    str
                )
                != "FLAT"
            )
        ].copy()
        confirmed_auc = float(
            "nan"
        )
        if (
            not confirmed_true_move.empty
            and confirmed_true_move[
                "actual_direction"
            ].astype(
                str
            ).nunique()
            == 2
        ):
            confirmed_auc = float(
                roc_auc_score(
                    (
                        confirmed_true_move[
                            "actual_direction"
                        ].astype(
                            str
                        )
                        == "UP"
                    ).astype(
                        int
                    ),
                    confirmed_true_move[
                        "binary_up_score"
                    ].astype(
                        float
                    ),
                )
            )

        selective_covered = (
            predictions[
                "selective_prediction"
            ].astype(
                str
            )
            != "ABSTAIN"
        )
        selective_metrics = research.three_class_metrics(
            predictions.loc[
                selective_covered,
                "actual_direction",
            ],
            predictions.loc[
                selective_covered,
                "selective_prediction",
            ],
        )
        baseline_same_sample = research.three_class_metrics(
            predictions.loc[
                selective_covered,
                "actual_direction",
            ],
            predictions.loc[
                selective_covered,
                "baseline_prediction",
            ],
        )
        selective_true_move = predictions.loc[
            (
                predictions[
                    "selective_prediction"
                ].astype(
                    str
                ).isin(
                    [
                        "DOWN",
                        "UP",
                    ]
                )
            )
            & (
                predictions[
                    "actual_direction"
                ].astype(
                    str
                )
                != "FLAT"
            )
        ].copy()
        selective_true_move_auc = float(
            "nan"
        )
        if (
            not selective_true_move.empty
            and selective_true_move[
                "actual_direction"
            ].astype(
                str
            ).nunique()
            == 2
        ):
            selective_true_move_auc = float(
                roc_auc_score(
                    (
                        selective_true_move[
                            "actual_direction"
                        ].astype(
                            str
                        )
                        == "UP"
                    ).astype(
                        int
                    ),
                    selective_true_move[
                        "binary_up_score"
                    ].astype(
                        float
                    ),
                )
            )

        metrics = {
            "outer_fold": int(
                outer_fold
            ),
            "verifier_training_rows": int(
                len(
                    verifier_training
                )
            ),
            "verifier_training_move_share": float(
                verifier_training[
                    "actual_move"
                ].mean()
            ),
            "verifier_optuna_trials": int(
                verifier_study_trials
            ),
            "verifier_best_cv_auc": float(
                verifier_cv_auc
            ),
            "verifier_oof_auc": float(
                verifier_oof_auc
            ),
            "verifier_threshold": float(
                verifier_threshold
            ),
            "routed_test_rows": int(
                len(
                    routed
                )
            ),
            "verifier_test_auc": float(
                verifier_test_metrics[
                    "roc_auc"
                ]
            ),
            "verifier_test_balanced_accuracy": float(
                verifier_test_metrics[
                    "balanced_accuracy"
                ]
            ),
            "stage1_route_move_purity": float(
                route_diagnostics[
                    "stage1_route_move_purity"
                ]
            ),
            "post_verifier_move_purity": float(
                route_diagnostics[
                    "confirmed_move_purity"
                ]
            ),
            "route_purity_lift": float(
                route_diagnostics[
                    "route_purity_lift"
                ]
            ),
            "true_move_recall_after_verifier": float(
                route_diagnostics[
                    "true_move_recall_after_verifier"
                ]
            ),
            "confirmed_direction_rows": int(
                route_diagnostics[
                    "confirmed_rows"
                ]
            ),
            "confirmed_true_move_up_down_auc": float(
                confirmed_auc
            ),
            "baseline_balanced_accuracy": float(
                baseline_metrics[
                    "balanced_accuracy"
                ]
            ),
            "cascade_balanced_accuracy": float(
                cascade_metrics[
                    "balanced_accuracy"
                ]
            ),
            "delta_balanced_accuracy": float(
                cascade_metrics[
                    "balanced_accuracy"
                ]
                - baseline_metrics[
                    "balanced_accuracy"
                ]
            ),
            "baseline_macro_f1": float(
                baseline_metrics[
                    "macro_f1"
                ]
            ),
            "cascade_macro_f1": float(
                cascade_metrics[
                    "macro_f1"
                ]
            ),
            "delta_macro_f1": float(
                cascade_metrics[
                    "macro_f1"
                ]
                - baseline_metrics[
                    "macro_f1"
                ]
            ),
            "selective_coverage": float(
                selective_covered.mean()
            ),
            "selective_balanced_accuracy": float(
                selective_metrics[
                    "balanced_accuracy"
                ]
            ),
            "selective_macro_f1": float(
                selective_metrics[
                    "macro_f1"
                ]
            ),
            "baseline_same_sample_balanced_accuracy": float(
                baseline_same_sample[
                    "balanced_accuracy"
                ]
            ),
            "baseline_same_sample_macro_f1": float(
                baseline_same_sample[
                    "macro_f1"
                ]
            ),
            "selective_true_move_up_down_auc": float(
                selective_true_move_auc
            ),
            "high_volatility_threshold": float(
                high_volatility_threshold
            ),
        }
        fold_rows.append(
            metrics
        )
        prediction_parts.append(
            predictions
        )

        completed[
            outer_fold
        ] = {
            "outer_fold": outer_fold,
            "metrics": metrics,
            "verifier_parameters": verifier_parameters,
            "predictions": (
                predictions.assign(
                    target_date=predictions[
                        "target_date"
                    ].astype(
                        str
                    )
                ).to_dict(
                    orient="records"
                )
            ),
        }
        save_json(
            PROGRESS_PATH,
            {
                "folds": [
                    completed[key]
                    for key in sorted(
                        completed
                    )
                ]
            },
        )

        print(
            f"  route purity {metrics['stage1_route_move_purity']:.2%} -> "
            f"{metrics['post_verifier_move_purity']:.2%}"
        )
        print(
            f"  end-to-end macro F1 {metrics['baseline_macro_f1']:.4f} -> "
            f"{metrics['cascade_macro_f1']:.4f} "
            f"({metrics['delta_macro_f1']:+.4f})"
        )
        print(
            f"  balanced accuracy {metrics['baseline_balanced_accuracy']:.4f} -> "
            f"{metrics['cascade_balanced_accuracy']:.4f} "
            f"({metrics['delta_balanced_accuracy']:+.4f})"
        )
        print(
            f"  selective coverage {metrics['selective_coverage']:.2%} | "
            f"same-date macro F1 "
            f"{metrics['baseline_same_sample_macro_f1']:.4f} -> "
            f"{metrics['selective_macro_f1']:.4f}"
        )

    fold_frame = pd.DataFrame(
        fold_rows
    ).sort_values(
        "outer_fold"
    ).reset_index(
        drop=True
    )
    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    ).sort_values(
        "target_date"
    ).reset_index(
        drop=True
    )

    pooled_baseline = research.three_class_metrics(
        predictions[
            "actual_direction"
        ],
        predictions[
            "baseline_prediction"
        ],
    )
    pooled_cascade = research.three_class_metrics(
        predictions[
            "actual_direction"
        ],
        predictions[
            "verifier_cascade_prediction"
        ],
    )
    cascade_bootstrap = (
        research
        .paired_block_bootstrap_three_class_delta(
            actual_direction=predictions[
                "actual_direction"
            ],
            baseline_prediction=predictions[
                "baseline_prediction"
            ],
            candidate_prediction=predictions[
                "verifier_cascade_prediction"
            ],
            seed_offset=1000,
        )
    )

    routed = predictions.loc[
        predictions[
            "stage1_predicted_move"
        ].astype(
            bool
        )
    ].copy()
    routed[
        "actual_move"
    ] = (
        routed[
            "actual_direction"
        ].astype(
            str
        )
        != "FLAT"
    ).astype(
        int
    )
    pooled_verifier = research.binary_metrics(
        actual=routed[
            "actual_move"
        ],
        score=routed[
            "verifier_move_probability"
        ],
        threshold=0.5,
    )
    pooled_verifier_auc = float(
        roc_auc_score(
            routed[
                "actual_move"
            ].astype(
                int
            ),
            routed[
                "verifier_move_probability"
            ].astype(
                float
            ),
        )
    )
    pooled_verifier[
        "roc_auc"
    ] = pooled_verifier_auc
    verifier_bootstrap = research.auc_bootstrap(
        dataframe=routed.sort_values(
            "target_date"
        ),
        actual_column="actual_move",
        score_column="verifier_move_probability",
        seed_offset=2000,
    )

    pooled_route = research.route_diagnostics(
        predictions,
        confirmed_column="verifier_confirmed_move",
    )
    confirmed_true_move = predictions.loc[
        predictions[
            "verifier_confirmed_move"
        ].astype(
            bool
        )
        & (
            predictions[
                "actual_direction"
            ].astype(
                str
            )
            != "FLAT"
        )
    ].copy()
    confirmed_true_move[
        "actual_up"
    ] = (
        confirmed_true_move[
            "actual_direction"
        ].astype(
            str
        )
        == "UP"
    ).astype(
        int
    )
    confirmed_stage2_auc = float(
        roc_auc_score(
            confirmed_true_move[
                "actual_up"
            ],
            confirmed_true_move[
                "binary_up_score"
            ],
        )
    )
    confirmed_stage2_bootstrap = research.auc_bootstrap(
        dataframe=confirmed_true_move.sort_values(
            "target_date"
        ),
        actual_column="actual_up",
        score_column="binary_up_score",
        seed_offset=3000,
    )

    selective_covered = (
        predictions[
            "selective_prediction"
        ].astype(
            str
        )
        != "ABSTAIN"
    )
    selective = predictions.loc[
        selective_covered
    ].copy()
    pooled_selective = research.three_class_metrics(
        selective[
            "actual_direction"
        ],
        selective[
            "selective_prediction"
        ],
    )
    pooled_baseline_same_sample = research.three_class_metrics(
        selective[
            "actual_direction"
        ],
        selective[
            "baseline_prediction"
        ],
    )
    selective_bootstrap = (
        research
        .paired_block_bootstrap_three_class_delta(
            actual_direction=selective[
                "actual_direction"
            ],
            baseline_prediction=selective[
                "baseline_prediction"
            ],
            candidate_prediction=selective[
                "selective_prediction"
            ],
            seed_offset=4000,
        )
    )
    selective_true_move = selective.loc[
        selective[
            "selective_prediction"
        ].astype(
            str
        ).isin(
            [
                "DOWN",
                "UP",
            ]
        )
        & (
            selective[
                "actual_direction"
            ].astype(
                str
            )
            != "FLAT"
        )
    ].copy()
    selective_true_move[
        "actual_up"
    ] = (
        selective_true_move[
            "actual_direction"
        ].astype(
            str
        )
        == "UP"
    ).astype(
        int
    )
    selective_stage2_auc = float(
        roc_auc_score(
            selective_true_move[
                "actual_up"
            ],
            selective_true_move[
                "binary_up_score"
            ],
        )
    )
    selective_stage2_bootstrap = research.auc_bootstrap(
        dataframe=selective_true_move.sort_values(
            "target_date"
        ),
        actual_column="actual_up",
        score_column="binary_up_score",
        seed_offset=5000,
    )

    fold_improvements = int(
        (
            fold_frame[
                "delta_macro_f1"
            ]
            > 0.0
        ).sum()
    )
    verifier_gates = {
        "verifier_auc_at_least_0_55": bool(
            pooled_verifier_auc
            >= MIN_VERIFIER_AUC
        ),
        "verifier_auc_bootstrap_lower_above_0_50": bool(
            verifier_bootstrap[
                "lower_95"
            ]
            > 0.50
        ),
        "post_verifier_route_purity_above_stage1_route_purity": bool(
            pooled_route[
                "confirmed_move_purity"
            ]
            > pooled_route[
                "stage1_route_move_purity"
            ]
        ),
        "cascade_macro_f1_above_binary_control": bool(
            pooled_cascade[
                "macro_f1"
            ]
            > pooled_baseline[
                "macro_f1"
            ]
        ),
        "cascade_balanced_accuracy_above_binary_control": bool(
            pooled_cascade[
                "balanced_accuracy"
            ]
            > pooled_baseline[
                "balanced_accuracy"
            ]
        ),
        "paired_macro_f1_delta_bootstrap_lower_above_0": bool(
            cascade_bootstrap[
                "macro_f1_delta_lower_95"
            ]
            > 0.0
        ),
        "at_least_two_outer_folds_macro_f1_improved": bool(
            fold_improvements
            >= 2
        ),
    }
    verifier_gates[
        "overall_verifier_cascade_gate"
    ] = bool(
        all(
            verifier_gates.values()
        )
    )

    selective_coverage = float(
        selective_covered.mean()
    )
    selective_gates = {
        "selective_coverage_at_least_0_60": bool(
            selective_coverage
            >= MIN_SELECTIVE_COVERAGE
        ),
        "selective_macro_f1_above_baseline_on_same_dates": bool(
            pooled_selective[
                "macro_f1"
            ]
            > pooled_baseline_same_sample[
                "macro_f1"
            ]
        ),
        "selective_balanced_accuracy_above_baseline_on_same_dates": bool(
            pooled_selective[
                "balanced_accuracy"
            ]
            > pooled_baseline_same_sample[
                "balanced_accuracy"
            ]
        ),
        "selective_macro_f1_delta_bootstrap_lower_above_0": bool(
            selective_bootstrap[
                "macro_f1_delta_lower_95"
            ]
            > 0.0
        ),
        "selective_stage2_auc_at_least_0_55": bool(
            selective_stage2_auc
            >= 0.55
        ),
        "selective_stage2_auc_bootstrap_lower_above_0_50": bool(
            selective_stage2_bootstrap[
                "lower_95"
            ]
            > 0.50
        ),
    }
    selective_gates[
        "overall_selective_cascade_gate"
    ] = bool(
        all(
            selective_gates.values()
        )
    )

    print()
    print(
        "NESTED DEVELOPMENT ROUTE-VERIFIER RESULTS"
    )
    print(
        fold_frame[
            [
                "outer_fold",
                "verifier_training_rows",
                "verifier_best_cv_auc",
                "verifier_oof_auc",
                "verifier_test_auc",
                "stage1_route_move_purity",
                "post_verifier_move_purity",
                "route_purity_lift",
                "true_move_recall_after_verifier",
                "baseline_balanced_accuracy",
                "cascade_balanced_accuracy",
                "delta_balanced_accuracy",
                "baseline_macro_f1",
                "cascade_macro_f1",
                "delta_macro_f1",
                "confirmed_true_move_up_down_auc",
                "selective_coverage",
                "baseline_same_sample_macro_f1",
                "selective_macro_f1",
                "selective_true_move_up_down_auc",
            ]
        ].round(
            4
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "POOLED DEVELOPMENT VERIFIER CASCADE"
    )
    print(
        f"Rows: {len(predictions)}"
    )
    print(
        f"Stage-1 routed rows: {pooled_route['stage1_routed_rows']}"
    )
    print(
        "Stage-1 route MOVE purity:",
        f"{pooled_route['stage1_route_move_purity']:.2%}",
    )
    print(
        "Verifier-confirmed rows:",
        pooled_route[
            "confirmed_rows"
        ],
    )
    print(
        "Post-verifier MOVE purity:",
        f"{pooled_route['confirmed_move_purity']:.2%}",
    )
    print(
        "Route purity lift:",
        f"{pooled_route['route_purity_lift']:+.2%}",
    )
    print(
        "True-MOVE recall after verifier:",
        f"{pooled_route['true_move_recall_after_verifier']:.2%}",
    )
    print(
        "Verifier pooled OOF test AUC:",
        f"{pooled_verifier_auc:.4f}",
    )
    print(
        "Verifier moving-block bootstrap 95% AUC CI:",
        f"[{verifier_bootstrap['lower_95']:.4f}, "
        f"{verifier_bootstrap['upper_95']:.4f}]",
    )
    print(
        "Binary-control balanced accuracy / macro F1:",
        f"{pooled_baseline['balanced_accuracy']:.4f} / "
        f"{pooled_baseline['macro_f1']:.4f}",
    )
    print(
        "Verifier-cascade balanced accuracy / macro F1:",
        f"{pooled_cascade['balanced_accuracy']:.4f} / "
        f"{pooled_cascade['macro_f1']:.4f}",
    )
    print(
        "Delta balanced accuracy / macro F1:",
        f"{pooled_cascade['balanced_accuracy'] - pooled_baseline['balanced_accuracy']:+.4f} / "
        f"{pooled_cascade['macro_f1'] - pooled_baseline['macro_f1']:+.4f}",
    )
    print(
        "Macro-F1 delta moving-block bootstrap 95% CI:",
        f"[{cascade_bootstrap['macro_f1_delta_lower_95']:+.4f}, "
        f"{cascade_bootstrap['macro_f1_delta_upper_95']:+.4f}]",
    )
    print(
        "Probability macro-F1 delta > 0:",
        f"{cascade_bootstrap['probability_macro_f1_delta_positive']:.2%}",
    )
    print(
        "Existing Stage-2 AUC after verifier on true MOVE:",
        f"{confirmed_stage2_auc:.4f}",
    )
    print(
        "Stage-2 after-verifier bootstrap 95% AUC CI:",
        f"[{confirmed_stage2_bootstrap['lower_95']:.4f}, "
        f"{confirmed_stage2_bootstrap['upper_95']:.4f}]",
    )

    print()
    print(
        "PRE-SPECIFIED VERIFIER + HIGH-VOL SELECTIVE POLICY"
    )
    print(
        "Total hierarchy coverage:",
        f"{selective_coverage:.2%}",
    )
    print(
        "Baseline vs selective macro F1 on identical covered dates:",
        f"{pooled_baseline_same_sample['macro_f1']:.4f} -> "
        f"{pooled_selective['macro_f1']:.4f}",
    )
    print(
        "Baseline vs selective balanced accuracy on identical covered dates:",
        f"{pooled_baseline_same_sample['balanced_accuracy']:.4f} -> "
        f"{pooled_selective['balanced_accuracy']:.4f}",
    )
    print(
        "Selective macro-F1 delta bootstrap 95% CI:",
        f"[{selective_bootstrap['macro_f1_delta_lower_95']:+.4f}, "
        f"{selective_bootstrap['macro_f1_delta_upper_95']:+.4f}]",
    )
    print(
        "Selective true-MOVE UP/DOWN AUC:",
        f"{selective_stage2_auc:.4f}",
    )
    print(
        "Selective Stage-2 bootstrap 95% AUC CI:",
        f"[{selective_stage2_bootstrap['lower_95']:.4f}, "
        f"{selective_stage2_bootstrap['upper_95']:.4f}]",
    )

    print()
    print(
        "PREDEFINED VERIFIER-CASCADE GATES"
    )
    for name, passed in verifier_gates.items():
        if name == "overall_verifier_cascade_gate":
            continue
        print(
            f"- {name}: {'PASS' if passed else 'FAIL'}"
        )
    print(
        "  OVERALL VERIFIER-CASCADE GATE:",
        (
            "PASS"
            if verifier_gates[
                "overall_verifier_cascade_gate"
            ]
            else "FAIL"
        ),
    )

    print()
    print(
        "PREDEFINED SELECTIVE-CASCADE GATES"
    )
    for name, passed in selective_gates.items():
        if name == "overall_selective_cascade_gate":
            continue
        print(
            f"- {name}: {'PASS' if passed else 'FAIL'}"
        )
    print(
        "  OVERALL SELECTIVE-CASCADE GATE:",
        (
            "PASS"
            if selective_gates[
                "overall_selective_cascade_gate"
            ]
            else "FAIL"
        ),
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    fold_path = (
        EXPERIMENT_DIRECTORY
        / f"{EXPERIMENT_NAME}_fold_results_{timestamp}.csv"
    )
    prediction_path = (
        EXPERIMENT_DIRECTORY
        / f"{EXPERIMENT_NAME}_predictions_{timestamp}.csv"
    )
    experiment_path = (
        EXPERIMENT_DIRECTORY
        / f"{EXPERIMENT_NAME}_{timestamp}.json"
    )
    fold_frame.to_csv(
        fold_path,
        index=False,
    )
    predictions.to_csv(
        prediction_path,
        index=False,
    )

    payload = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment_name": EXPERIMENT_NAME,
        "target": {
            "name": TARGET_NAME,
            "volatility_window": TARGET_WINDOW,
            "threshold_multiplier": TARGET_MULTIPLIER,
        },
        "development_cutoff": cutoff,
        "prerequisites": {
            "confirmed_route_diagnostic": diagnostic_path,
            "failed_multiclass_experiment": multiclass_path,
        },
        "winner_groups": list(
            winner_groups
        ),
        "winner_feature_count": len(
            winner_features
        ),
        "architecture": {
            "stage1": (
                "locked xLSTM FLAT vs MOVE candidate detector"
            ),
            "route_verifier": (
                "XGBoost FLAT vs MOVE trained only on Stage-1 OOF-routed rows"
            ),
            "route_verifier_features": (
                "locked Stage-2 winner features plus Stage-1 MOVE probability"
            ),
            "route_verifier_hyperparameters": (
                "independent train-only Optuna search inside each development outer fold"
            ),
            "route_verifier_threshold": (
                "selected from train-only verifier OOF predictions inside each outer fold"
            ),
            "stage2": (
                "existing binary XGBoost DOWN vs UP; no retuning"
            ),
            "secondary_selective_policy": (
                "after verifier confirmation, direction only in HIGH "
                "realized_volatility_20; otherwise ABSTAIN"
            ),
        },
        "statistics": {
            "outer_splits": OUTER_SPLITS,
            "inner_splits": INNER_SPLITS,
            "verifier_optuna_trials": VERIFIER_OPTUNA_TRIALS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_block_length": BOOTSTRAP_BLOCK_LENGTH,
        },
        "summary": {
            "binary_control": pooled_baseline,
            "verifier_cascade": pooled_cascade,
            "verifier": pooled_verifier,
            "verifier_bootstrap": verifier_bootstrap,
            "route_diagnostics": pooled_route,
            "cascade_paired_bootstrap": cascade_bootstrap,
            "confirmed_stage2_auc": confirmed_stage2_auc,
            "confirmed_stage2_bootstrap": confirmed_stage2_bootstrap,
            "selective_coverage": selective_coverage,
            "selective_baseline_same_sample": pooled_baseline_same_sample,
            "selective_policy": pooled_selective,
            "selective_paired_bootstrap": selective_bootstrap,
            "selective_stage2_auc": selective_stage2_auc,
            "selective_stage2_bootstrap": selective_stage2_bootstrap,
            "folds_macro_f1_improved": fold_improvements,
            "verifier_gates": verifier_gates,
            "selective_gates": selective_gates,
        },
        "methodology": {
            "development_only": True,
            "new_target_search": False,
            "new_feature_search": False,
            "new_regime_search": False,
            "new_direction_model_hyperparameter_search": False,
            "new_verifier_hyperparameter_search": True,
            "outer_validation_loaded": False,
            "held_out_test_loaded": False,
        },
        "outputs": {
            "fold_results": fold_path,
            "predictions": prediction_path,
            "experiment": experiment_path,
            "progress": PROGRESS_PATH,
            "optuna_storage": VERIFIER_STORAGE_URL,
        },
    }
    with experiment_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            default=json_default,
        )

    print()
    print(
        "Fold results:",
        fold_path,
    )
    print(
        "Predictions:",
        prediction_path,
    )
    print(
        "Experiment:",
        experiment_path,
    )
    print(
        "Progress checkpoint:",
        PROGRESS_PATH,
    )
    print(
        "Verifier Optuna DB:",
        VERIFIER_STORAGE_URL,
    )
    print(
        "Outer validation was NOT evaluated."
    )
    print(
        "Held-out test set was NOT evaluated."
    )
    print()

    verifier_pass = verifier_gates[
        "overall_verifier_cascade_gate"
    ]
    selective_pass = selective_gates[
        "overall_selective_cascade_gate"
    ]
    if verifier_pass and selective_pass:
        print(
            "NEXT DECISION RULE: BOTH PASS. Lock the verifier as the route-purification "
            "stage. Prefer the HIGH-vol selective variant only if its development-only "
            "same-date macro F1 is higher than the non-selective verifier cascade; "
            "otherwise lock the verifier cascade without the volatility abstention."
        )
    elif verifier_pass:
        print(
            "NEXT DECISION RULE: VERIFIER CASCADE PASS. Lock Stage1 -> route verifier "
            "-> binary Stage2. Do not add the HIGH-vol abstention unless its separate "
            "gate also passes."
        )
    elif selective_pass:
        print(
            "NEXT DECISION RULE: SELECTIVE CASCADE PASS ONLY. The verifier is useful "
            "only together with the pre-specified HIGH-vol eligibility rule. Lock that "
            "combined development-only architecture before any further evaluation."
        )
    else:
        print(
            "NEXT DECISION RULE: FAIL. The route verifier did not create stable "
            "development-only incremental value. Do not touch outer validation or the "
            "held-out test. The current hierarchical formulation should then be treated "
            "as exhausted rather than repeatedly patched."
        )


if __name__ == "__main__":
    main()
