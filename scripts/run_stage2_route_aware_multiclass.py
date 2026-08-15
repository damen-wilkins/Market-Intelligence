from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from app.training.hierarchical_xlstm_parameter_selector import (
    HierarchicalXLSTMParameterSelector,
)
from app.training.stage2_route_aware_multiclass_research import (
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    Stage2RouteAwareMulticlassResearch,
)
from scripts.run_stage1_long_history_optimization import train_stage1_fold
from scripts.run_stage1_target_optimization import STAGE1_FEATURE_COLUMNS
from scripts.run_stage2_conditioned_megasearch import columns_for_groups
from scripts.run_stage2_route_compatibility_diagnostic import (
    BOOTSTRAP_BLOCK_LENGTH,
    BOOTSTRAP_RESAMPLES,
    EXPERIMENT_DIRECTORY,
    OUTER_SPLITS,
    RANDOM_STATE,
    REGIME_FEATURE,
    TARGET_MULTIPLIER,
    TARGET_NAME,
    TARGET_WINDOW,
    TREE_PROGRESS,
    build_stage1_development,
    build_stage2_development,
    completed_fold_map,
    load_locked_stage1,
    load_stage2_saved_oof,
    save_json,
    stage1_fold_data,
)


INNER_SPLITS = 3
TREE_STORAGE_URL = "sqlite:///experiments/optuna_stage2_return_architecture_tree_v1.db"
DIAGNOSTIC_PREFIX = "stage2_route_compatibility_diagnostic_v1_"
EXPERIMENT_NAME = "stage2_route_aware_multiclass_v1"
PROGRESS_PATH = EXPERIMENT_DIRECTORY / f"{EXPERIMENT_NAME}_progress.json"
SCORE_REPRODUCTION_ATOL = 1e-7
SCORE_REPRODUCTION_RTOL = 1e-6


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
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def latest_confirmed_diagnostic() -> Path | None:
    candidates = []
    for path in EXPERIMENT_DIRECTORY.glob(f"{DIAGNOSTIC_PREFIX}*.json"):
        if path.name.endswith("_progress.json"):
            continue
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception:
            continue
        summary = payload.get("summary", {})
        if summary.get("development_confirms_route_compatibility_problem") is True:
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_fold_parameters(outer_fold: int) -> dict:
    study_name = f"xgboost_binary_winner_outer_{outer_fold}"
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=TREE_STORAGE_URL,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the existing development-only Optuna study {study_name!r}."
        ) from exc
    return dict(study.best_trial.params)


def build_stage1_oof_routes(
    training_data: pd.DataFrame,
    locked_parameters: dict,
    outer_fold: int,
) -> tuple[pd.DataFrame, float]:
    splitter = TimeSeriesSplit(n_splits=INNER_SPLITS)
    batches = []
    for inner_fold, (train_index, validation_index) in enumerate(
        splitter.split(training_data),
        start=1,
    ):
        print(f"      Stage-1 routing OOF {inner_fold}/{INNER_SPLITS}")
        fold_train = training_data.iloc[train_index].reset_index(drop=True)
        fold_validation = training_data.iloc[validation_index].reset_index(drop=True)
        result = train_stage1_fold(
            fold_train=fold_train,
            fold_validation=fold_validation,
            feature_columns=STAGE1_FEATURE_COLUMNS,
            parameters=dict(locked_parameters),
            seed=RANDOM_STATE + 20000 + inner_fold,
        )
        result_dates = pd.DatetimeIndex(pd.to_datetime(result["target_dates"]))
        expected_dates = pd.DatetimeIndex(pd.to_datetime(fold_validation["target_date"]))
        if not result_dates.equals(expected_dates):
            raise ValueError(
                f"Outer fold {outer_fold} inner fold {inner_fold} Stage-1 OOF dates do not align."
            )
        frame = fold_validation[["target_date", "direction"]].copy()
        frame["stage1_move_probability"] = np.asarray(
            result["move_probabilities"],
            dtype=np.float64,
        )
        frame["actual_move"] = (frame["direction"].astype(str) != "FLAT").astype(int)
        batches.append(frame)

    oof = pd.concat(batches, ignore_index=True).sort_values("target_date").reset_index(drop=True)
    threshold = float(
        HierarchicalXLSTMParameterSelector.select_probability_threshold(
            actual=oof["actual_move"].to_numpy(dtype=np.int64),
            positive_probabilities=oof["stage1_move_probability"].to_numpy(dtype=np.float64),
        )["threshold"]
    )
    oof["stage1_predicted_move"] = oof["stage1_move_probability"] >= threshold
    return oof, threshold


def load_outer_test_route(outer_fold: int) -> pd.DataFrame:
    completed = completed_fold_map()
    if outer_fold not in completed:
        raise RuntimeError(
            f"Route compatibility progress does not contain outer fold {outer_fold}. "
            "Run the route compatibility diagnostic first."
        )
    route = pd.DataFrame(completed[outer_fold]["route_predictions"])
    route["target_date"] = pd.to_datetime(route["target_date"])
    route["feature_date"] = pd.to_datetime(route["feature_date"])
    route["stage1_move_probability"] = pd.to_numeric(
        route["stage1_move_probability"],
        errors="raise",
    )
    route["stage1_predicted_move"] = route["stage1_predicted_move"].astype(bool)
    return route.sort_values("target_date").reset_index(drop=True)


def route_aware_training_frame(
    stage2_outer_train: pd.DataFrame,
    stage1_oof: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    stage2 = stage2_outer_train[
        ["target_date", "direction", *feature_columns]
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
        raise ValueError("Stage-1 and Stage-2 development labels disagree after alignment.")
    merged = merged.rename(columns={"direction_stage2": "direction"}).drop(
        columns=["direction_stage1"]
    )
    routed = merged.loc[merged["stage1_predicted_move"].astype(bool)].copy()
    if routed.empty:
        raise ValueError("Stage-1 OOF routing produced no Stage-2 training rows.")
    observed = set(routed["direction"].astype(str).unique())
    required = {"DOWN", "FLAT", "UP"}
    if not required.issubset(observed):
        raise ValueError(
            "Route-aware Stage-2 training requires DOWN, FLAT, and UP classes; "
            f"observed {sorted(observed)}."
        )
    return routed.sort_values("target_date").reset_index(drop=True)


def route_aware_test_frame(
    stage2_outer_test: pd.DataFrame,
    route: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    merged = stage2_outer_test[
        ["target_date", "direction", "future_log_return", *feature_columns]
    ].merge(
        route[
            [
                "target_date",
                "stage1_move_probability",
                "stage1_predicted_move",
            ]
        ],
        on="target_date",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(stage2_outer_test):
        raise ValueError("Stage-1 route predictions do not cover the full Stage-2 outer test.")
    return merged.sort_values("target_date").reset_index(drop=True)


def fit_binary_control(
    outer_fold: int,
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    feature_columns: list[str],
    parameters: dict,
    saved: dict,
) -> np.ndarray:
    train_move = outer_train.loc[
        outer_train["direction"].astype(str) != "FLAT"
    ].reset_index(drop=True)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE + 700000 + outer_fold,
        n_jobs=-1,
        **parameters,
    )
    model.fit(
        train_move[feature_columns],
        (train_move["direction"].astype(str) == "UP").astype(int),
    )
    all_scores = model.predict_proba(outer_test[feature_columns])[:, 1].astype(np.float64)

    test_move_mask = outer_test["direction"].astype(str) != "FLAT"
    test_move_dates = pd.DatetimeIndex(
        pd.to_datetime(outer_test.loc[test_move_mask, "target_date"])
    )
    saved_dates = pd.DatetimeIndex(pd.to_datetime(saved["target_dates"]))
    if not saved_dates.equals(test_move_dates):
        raise ValueError(f"Outer fold {outer_fold} binary reproduction dates do not align.")
    reproduced = all_scores[test_move_mask.to_numpy()]
    saved_scores = np.asarray(saved["score"], dtype=np.float64)
    if not np.allclose(
        reproduced,
        saved_scores,
        atol=SCORE_REPRODUCTION_ATOL,
        rtol=SCORE_REPRODUCTION_RTOL,
    ):
        maximum = float(np.max(np.abs(reproduced - saved_scores)))
        raise ValueError(
            f"Outer fold {outer_fold} binary control failed saved-score reproduction; "
            f"max abs difference={maximum:.12g}."
        )
    return all_scores


def fit_route_aware_multiclass(
    outer_fold: int,
    training: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    parameters: dict,
) -> tuple[np.ndarray, np.ndarray]:
    model_features = [*feature_columns, "stage1_move_probability"]
    y_train = training["direction"].astype(str).map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE + 800000 + outer_fold,
        n_jobs=-1,
        **parameters,
    )
    model.fit(
        training[model_features],
        y_train,
        sample_weight=sample_weight,
    )
    probability = model.predict_proba(test[model_features]).astype(np.float64)
    if probability.shape[1] != 3:
        raise ValueError("Route-aware multiclass model did not produce three class probabilities.")
    predicted_index = np.argmax(probability, axis=1).astype(np.int64)
    predicted = np.asarray([INDEX_TO_CLASS[int(value)] for value in predicted_index], dtype=object)
    return predicted, probability


def end_to_end_predictions(
    test: pd.DataFrame,
    binary_scores: np.ndarray,
    route_aware_prediction: np.ndarray,
    binary_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    routed = test["stage1_predicted_move"].astype(bool).to_numpy()
    binary_direction = np.where(binary_scores >= float(binary_threshold), "UP", "DOWN")
    baseline = np.where(routed, binary_direction, "FLAT").astype(object)
    candidate = np.where(routed, route_aware_prediction, "FLAT").astype(object)
    return baseline, candidate


def cached_result_map() -> dict[int, dict]:
    if not PROGRESS_PATH.exists():
        return {}
    with PROGRESS_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return {int(row["outer_fold"]): row for row in payload.get("folds", [])}


def main():
    diagnostic_path = latest_confirmed_diagnostic()
    if diagnostic_path is None:
        print()
        print("ROUTE-AWARE MULTICLASS RESEARCH SKIPPED")
        print(
            "No completed development diagnostic confirms the Stage1->Stage2 label-space mismatch."
        )
        print("Run python -m scripts.run_stage2_route_compatibility_diagnostic first.")
        return

    locked_stage1 = load_locked_stage1()
    saved_stage2 = load_stage2_saved_oof()
    stage2_data, winner_groups, winner_features, cutoff = build_stage2_development()
    stage1_data = build_stage1_development(cutoff)
    research = Stage2RouteAwareMulticlassResearch(
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_block_length=BOOTSTRAP_BLOCK_LENGTH,
        random_state=RANDOM_STATE,
    )

    print()
    print("=" * 88)
    print("STAGE-2 ROUTE-AWARE MULTICLASS ARCHITECTURE V1")
    print("=" * 88)
    print(f"Target: {TARGET_WINDOW}d x {TARGET_MULTIPLIER:.3f}")
    print("Development diagnostic:", diagnostic_path)
    print("Stage-2 label space: DOWN / FLAT / UP")
    print("Training population: Stage-1-predicted MOVE rows from Stage-1 inner OOF only")
    print("Stage-2 inputs: locked 58 winner features + Stage-1 MOVE probability")
    print("XGBoost parameters: reuse each fold's existing binary Optuna winner; NO retuning")
    print("Class treatment: train-fold balanced sample weights")
    print("Control: original binary Stage 2 refit on oracle true-MOVE rows")
    print(f"Development cutoff: {cutoff.date()}")
    print()
    print("DEVELOPMENT ONLY: no feature search, target search, regime search, or hyperparameter search.")
    print("Outer validation and the held-out test are NOT loaded or evaluated.")

    splitter = TimeSeriesSplit(n_splits=OUTER_SPLITS)
    completed = cached_result_map()
    fold_rows = []
    prediction_parts = []

    for outer_fold, (train_index, test_index) in enumerate(splitter.split(stage2_data), start=1):
        print()
        print(f"outer fold {outer_fold}/{OUTER_SPLITS}")
        if outer_fold in completed:
            print("  using cached route-aware fold result")
            row = completed[outer_fold]
            fold_rows.append(row["metrics"])
            cached_predictions = pd.DataFrame(row["predictions"])
            cached_predictions["target_date"] = pd.to_datetime(cached_predictions["target_date"])
            prediction_parts.append(cached_predictions)
            continue

        stage2_outer_train = stage2_data.iloc[train_index].reset_index(drop=True)
        stage2_outer_test = stage2_data.iloc[test_index].reset_index(drop=True)
        stage1_train, _ = stage1_fold_data(stage1_data, stage2_outer_train, stage2_outer_test)

        print("  reconstructing Stage-1 inner OOF routes for Stage-2 training...")
        stage1_oof, stage1_threshold = build_stage1_oof_routes(
            training_data=stage1_train,
            locked_parameters=dict(locked_stage1["parameters"]),
            outer_fold=outer_fold,
        )
        route_training = route_aware_training_frame(
            stage2_outer_train=stage2_outer_train,
            stage1_oof=stage1_oof,
            feature_columns=winner_features,
        )

        route_test = load_outer_test_route(outer_fold)
        stage2_test = route_aware_test_frame(
            stage2_outer_test=stage2_outer_test,
            route=route_test,
            feature_columns=winner_features,
        )

        cached_threshold = float(completed_fold_map()[outer_fold]["stage1_threshold"])
        if abs(stage1_threshold - cached_threshold) > 1e-9:
            raise ValueError(
                f"Outer fold {outer_fold} Stage-1 threshold mismatch between reconstructed "
                f"OOF routing ({stage1_threshold:.12g}) and compatibility diagnostic "
                f"({cached_threshold:.12g})."
            )

        parameters = load_fold_parameters(outer_fold)
        print(f"  route-aware Stage-2 training rows: {len(route_training)}")
        class_counts = route_training["direction"].astype(str).value_counts().to_dict()
        print("  route-aware training classes:", class_counts)

        binary_scores = fit_binary_control(
            outer_fold=outer_fold,
            outer_train=stage2_outer_train,
            outer_test=stage2_outer_test,
            feature_columns=winner_features,
            parameters=parameters,
            saved=saved_stage2[outer_fold],
        )
        candidate_prediction, candidate_probability = fit_route_aware_multiclass(
            outer_fold=outer_fold,
            training=route_training,
            test=stage2_test,
            feature_columns=winner_features,
            parameters=parameters,
        )
        baseline_prediction, end_to_end_candidate = end_to_end_predictions(
            test=stage2_test,
            binary_scores=binary_scores,
            route_aware_prediction=candidate_prediction,
            binary_threshold=float(saved_stage2[outer_fold]["decision_threshold"]),
        )

        actual = stage2_test["direction"].astype(str).to_numpy(dtype=object)
        baseline_metrics = research.metrics(actual, baseline_prediction)
        candidate_metrics = research.metrics(actual, end_to_end_candidate)

        routed_mask = stage2_test["stage1_predicted_move"].astype(bool).to_numpy()
        routed = pd.DataFrame({
            "target_date": pd.to_datetime(stage2_test.loc[routed_mask, "target_date"]),
            "actual_direction": stage2_test.loc[routed_mask, "direction"].astype(str).to_numpy(),
            "baseline_prediction": baseline_prediction[routed_mask],
            "candidate_prediction": end_to_end_candidate[routed_mask],
            "candidate_down_probability": candidate_probability[routed_mask, 0],
            "candidate_flat_probability": candidate_probability[routed_mask, 1],
            "candidate_up_probability": candidate_probability[routed_mask, 2],
        })
        routed_diagnostics = research.routed_diagnostics(routed)

        metrics = {
            "outer_fold": int(outer_fold),
            "training_oof_rows": int(len(stage1_oof)),
            "route_aware_training_rows": int(len(route_training)),
            "route_aware_training_down_rows": int(class_counts.get("DOWN", 0)),
            "route_aware_training_flat_rows": int(class_counts.get("FLAT", 0)),
            "route_aware_training_up_rows": int(class_counts.get("UP", 0)),
            "test_rows": int(len(stage2_test)),
            "routed_test_rows": int(routed_mask.sum()),
            "baseline_balanced_accuracy": float(baseline_metrics["balanced_accuracy"]),
            "candidate_balanced_accuracy": float(candidate_metrics["balanced_accuracy"]),
            "delta_balanced_accuracy": float(
                candidate_metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
            ),
            "baseline_macro_f1": float(baseline_metrics["macro_f1"]),
            "candidate_macro_f1": float(candidate_metrics["macro_f1"]),
            "delta_macro_f1": float(candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"]),
            "baseline_down_f1": float(baseline_metrics["down_f1"]),
            "baseline_flat_f1": float(baseline_metrics["flat_f1"]),
            "baseline_up_f1": float(baseline_metrics["up_f1"]),
            "candidate_down_f1": float(candidate_metrics["down_f1"]),
            "candidate_flat_f1": float(candidate_metrics["flat_f1"]),
            "candidate_up_f1": float(candidate_metrics["up_f1"]),
            **routed_diagnostics,
        }
        fold_rows.append(metrics)

        predictions = pd.DataFrame({
            "target_date": pd.to_datetime(stage2_test["target_date"]),
            "outer_fold": outer_fold,
            "actual_direction": actual,
            "stage1_move_probability": stage2_test["stage1_move_probability"].to_numpy(dtype=np.float64),
            "stage1_predicted_move": routed_mask,
            "baseline_binary_up_score": binary_scores,
            "baseline_prediction": baseline_prediction,
            "candidate_down_probability": candidate_probability[:, 0],
            "candidate_flat_probability": candidate_probability[:, 1],
            "candidate_up_probability": candidate_probability[:, 2],
            "candidate_prediction": end_to_end_candidate,
        })
        prediction_parts.append(predictions)

        completed[outer_fold] = {
            "outer_fold": outer_fold,
            "metrics": metrics,
            "predictions": predictions.assign(
                target_date=predictions["target_date"].astype(str)
            ).to_dict(orient="records"),
        }
        save_json(
            PROGRESS_PATH,
            {"folds": [completed[key] for key in sorted(completed)]},
        )

        print(
            f"  end-to-end macro F1 {baseline_metrics['macro_f1']:.4f} -> "
            f"{candidate_metrics['macro_f1']:.4f} "
            f"({metrics['delta_macro_f1']:+.4f})"
        )
        print(
            f"  balanced accuracy {baseline_metrics['balanced_accuracy']:.4f} -> "
            f"{candidate_metrics['balanced_accuracy']:.4f} "
            f"({metrics['delta_balanced_accuracy']:+.4f})"
        )
        print(
            "  false-route FLAT correction: "
            f"{routed_diagnostics['candidate_false_route_flat_correction_rate']:.2%}"
        )

    fold_frame = pd.DataFrame(fold_rows).sort_values("outer_fold").reset_index(drop=True)
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values("target_date").reset_index(drop=True)

    pooled_baseline = research.metrics(
        predictions["actual_direction"],
        predictions["baseline_prediction"],
    )
    pooled_candidate = research.metrics(
        predictions["actual_direction"],
        predictions["candidate_prediction"],
    )
    bootstrap = research.paired_block_bootstrap_delta(
        actual_direction=predictions["actual_direction"],
        baseline_prediction=predictions["baseline_prediction"],
        candidate_prediction=predictions["candidate_prediction"],
    )
    routed = predictions.loc[predictions["stage1_predicted_move"].astype(bool)].copy()
    pooled_routed = research.routed_diagnostics(routed)

    fold_improvements = int((fold_frame["delta_macro_f1"] > 0.0).sum())
    gates = {
        "pooled_macro_f1_above_binary_control": bool(
            pooled_candidate["macro_f1"] > pooled_baseline["macro_f1"]
        ),
        "pooled_balanced_accuracy_above_binary_control": bool(
            pooled_candidate["balanced_accuracy"] > pooled_baseline["balanced_accuracy"]
        ),
        "paired_macro_f1_delta_bootstrap_lower_above_0": bool(
            bootstrap["macro_f1_delta_lower_95"] > 0.0
        ),
        "at_least_two_outer_folds_macro_f1_improved": bool(fold_improvements >= 2),
    }
    gates["overall_route_aware_architecture_gate"] = bool(all(gates.values()))

    print()
    print("NESTED DEVELOPMENT ROUTE-AWARE RESULTS")
    print(
        fold_frame[
            [
                "outer_fold",
                "route_aware_training_rows",
                "routed_test_rows",
                "baseline_balanced_accuracy",
                "candidate_balanced_accuracy",
                "delta_balanced_accuracy",
                "baseline_macro_f1",
                "candidate_macro_f1",
                "delta_macro_f1",
                "candidate_false_route_flat_correction_rate",
                "baseline_true_move_direction_accuracy",
                "candidate_true_move_direction_accuracy",
                "candidate_true_move_up_down_auc",
            ]
        ].round(4).to_string(index=False)
    )

    print()
    print("POOLED DEVELOPMENT END-TO-END")
    print(f"Rows: {len(predictions)}")
    print(
        "Binary-control balanced accuracy / macro F1: "
        f"{pooled_baseline['balanced_accuracy']:.4f} / {pooled_baseline['macro_f1']:.4f}"
    )
    print(
        "Route-aware balanced accuracy / macro F1: "
        f"{pooled_candidate['balanced_accuracy']:.4f} / {pooled_candidate['macro_f1']:.4f}"
    )
    print(
        "Delta balanced accuracy / macro F1: "
        f"{pooled_candidate['balanced_accuracy'] - pooled_baseline['balanced_accuracy']:+.4f} / "
        f"{pooled_candidate['macro_f1'] - pooled_baseline['macro_f1']:+.4f}"
    )
    print(
        "Macro-F1 delta moving-block bootstrap 95% CI: "
        f"[{bootstrap['macro_f1_delta_lower_95']:+.4f}, "
        f"{bootstrap['macro_f1_delta_upper_95']:+.4f}]"
    )
    print(
        "Probability macro-F1 delta > 0: "
        f"{bootstrap['probability_macro_f1_delta_positive']:.2%}"
    )
    print(
        "False-routed FLAT correction rate: "
        f"{pooled_routed['candidate_false_route_flat_correction_rate']:.2%}"
    )
    print(
        "True-MOVE direction accuracy, binary -> route-aware: "
        f"{pooled_routed['baseline_true_move_direction_accuracy']:.4f} -> "
        f"{pooled_routed['candidate_true_move_direction_accuracy']:.4f}"
    )
    print(
        "Route-aware true-MOVE UP/DOWN AUC: "
        f"{pooled_routed['candidate_true_move_up_down_auc']:.4f}"
    )

    print()
    print("PREDEFINED ROUTE-AWARE ARCHITECTURE GATES")
    for name, passed in gates.items():
        if name == "overall_route_aware_architecture_gate":
            continue
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(
        "  OVERALL ROUTE-AWARE ARCHITECTURE GATE: "
        f"{'PASS' if gates['overall_route_aware_architecture_gate'] else 'FAIL'}"
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    fold_path = EXPERIMENT_DIRECTORY / f"{EXPERIMENT_NAME}_fold_results_{timestamp}.csv"
    prediction_path = EXPERIMENT_DIRECTORY / f"{EXPERIMENT_NAME}_predictions_{timestamp}.csv"
    experiment_path = EXPERIMENT_DIRECTORY / f"{EXPERIMENT_NAME}_{timestamp}.json"
    fold_frame.to_csv(fold_path, index=False)
    predictions.to_csv(prediction_path, index=False)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": EXPERIMENT_NAME,
        "target": {
            "name": TARGET_NAME,
            "volatility_window": TARGET_WINDOW,
            "threshold_multiplier": TARGET_MULTIPLIER,
        },
        "development_cutoff": cutoff,
        "confirmed_route_diagnostic": diagnostic_path,
        "winner_groups": list(winner_groups),
        "winner_feature_count": len(winner_features),
        "architecture": {
            "training_population": "Stage-1 predicted MOVE rows from Stage-1 inner OOF predictions",
            "labels": ["DOWN", "FLAT", "UP"],
            "features": "locked Stage-2 winner features plus Stage-1 MOVE probability",
            "class_weighting": "balanced sample weights computed inside each outer-fold training sample",
            "xgboost_parameters": "reuse existing xgboost_binary_winner outer-fold Optuna parameters; no retuning",
            "control": "original binary XGBoost trained on true MOVE rows and forced to DOWN/UP after Stage-1 routing",
        },
        "summary": {
            "binary_control": pooled_baseline,
            "route_aware_multiclass": pooled_candidate,
            "routed_diagnostics": pooled_routed,
            "paired_bootstrap": bootstrap,
            "folds_macro_f1_improved": fold_improvements,
            "gates": gates,
        },
        "methodology": {
            "development_only": True,
            "new_feature_search": False,
            "new_target_search": False,
            "new_regime_search": False,
            "new_hyperparameter_search": False,
            "outer_validation_loaded": False,
            "held_out_test_loaded": False,
        },
        "outputs": {
            "fold_results": fold_path,
            "predictions": prediction_path,
            "experiment": experiment_path,
            "progress": PROGRESS_PATH,
        },
    }
    with experiment_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=json_default)

    print()
    print("Fold results:", fold_path)
    print("Predictions:", prediction_path)
    print("Experiment:", experiment_path)
    print("Progress checkpoint:", PROGRESS_PATH)
    print("Outer validation was NOT evaluated.")
    print("Held-out test set was NOT evaluated.")
    print()
    if gates["overall_route_aware_architecture_gate"]:
        print(
            "NEXT DECISION RULE: PASS. Route-aware three-class Stage 2 demonstrated stable "
            "development-only incremental value. Next optimize this architecture inside development "
            "only before any new outer-validation characterization."
        )
    else:
        print(
            "NEXT DECISION RULE: FAIL. The label-space mismatch is real, but this first route-aware "
            "three-class XGBoost architecture did not demonstrate stable incremental value. Do not "
            "touch outer validation or the held-out test; use the failure to choose the next materially "
            "different development-only architecture."
        )


if __name__ == "__main__":
    main()
