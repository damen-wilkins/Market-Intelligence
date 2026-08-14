from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from app.training.experiment_tracker import ExperimentTracker
from app.training.hierarchical_stage_feature_research import (
    binary_probability_metrics,
)
from app.training.hierarchical_target_feature_research import binary_metrics
from app.training.hierarchical_xlstm_parameter_selector import (
    HierarchicalXLSTMParameterSelector,
)
from app.training.stage2_conditioned_target_research import (
    Stage2FeatureCandidate,
    Stage2TargetSpec,
    expand_beam,
    moving_block_bootstrap_auc_delta,
    pair_candidates,
    select_beam,
    select_finalists,
    single_candidates,
    target_distribution,
    target_specs,
)
from app.training.stage2_wide_feature_builder import Stage2WideFeatureBuilder
from app.training.stage2_wide_signal_search import univariate_feature_auc_screen
from app.training.volatility_direction_label_builder import (
    VolatilityDirectionLabelBuilder,
)
from database.stage2_signal_data_repository import Stage2SignalDataRepository
from scripts.run_stage2_wide_signal_search import (
    evaluate_candidate,
    train_stage2_fold,
)


TICKER = "SPY"
CV_SPLITS = 3
RANDOM_STATE = 42
REFERENCE_MODEL_PATH = Path("models/xlstm_hierarchical_direction.pt")
EXPERIMENT_DIRECTORY = Path("experiments")
EXPERIMENT_NAME = "xlstm_stage2_conditioned_megasearch_v1"
MODEL_NAME = "xlstm_stage2_conditioned_megasearch_v1"
OPTUNA_STORAGE_URL = (
    "sqlite:///experiments/optuna_stage2_conditioned_megasearch_v1.db"
)
SCREENING_CHECKPOINT = (
    EXPERIMENT_DIRECTORY / "stage2_conditioned_megasearch_v1_screening.json"
)
DEEP_PROGRESS = (
    EXPERIMENT_DIRECTORY / "stage2_conditioned_megasearch_v1_deep.json"
)
TARGET_STATE_FEATURE = "target_rolling_volatility"
VERIFICATION_FRACTION = 0.20
BASE_OPTUNA_TRIALS = 100
FINALIST_OPTUNA_TRIALS = 75
ROBUSTNESS_OPTUNA_TRIALS = 50
SHORT_HISTORY_OPTUNA_TRIALS = 50
MAX_SELECTION_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
SCREENING_MIN_ROWS = 1200
FINALIST_MIN_ROWS = 1800
FINALIST_MAX_FOLD_STD = 0.08
TOP_SINGLE_GROUPS_FOR_PAIRS = 10
BEAM_WIDTH = 10
MAX_GROUP_DEPTH = 6
FINALIST_COUNT = 8
ROBUSTNESS_CHAMPION_COUNT = 3
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_BLOCK_LENGTH = 20


def json_load(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def json_save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=json_default)
    temp.replace(path)


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


def load_training_cutoff() -> pd.Timestamp:
    if not REFERENCE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "models/xlstm_hierarchical_direction.pt is required to preserve "
            "the original training boundary."
        )
    try:
        package = torch.load(
            REFERENCE_MODEL_PATH,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        package = torch.load(REFERENCE_MODEL_PATH, map_location="cpu")
    end = package.get("metadata", {}).get("training_period", {}).get("end")
    if end is None:
        raise ValueError("Reference model has no training-period end date.")
    return pd.Timestamp(end)


def base_columns() -> list[str]:
    return [*Stage2WideFeatureBuilder.BASE_FEATURE_COLUMNS, TARGET_STATE_FEATURE]


def columns_for_groups(groups: tuple[str, ...]) -> list[str]:
    columns = Stage2WideFeatureBuilder.columns_for_groups(groups)
    columns.append(TARGET_STATE_FEATURE)
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate Stage-2 feature columns detected.")
    return columns


def build_master(
    raw_data: pd.DataFrame,
    target: Stage2TargetSpec,
    training_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    features = Stage2WideFeatureBuilder().build_library(raw_data)
    labels = VolatilityDirectionLabelBuilder(
        volatility_window=target.volatility_window,
        threshold_multiplier=target.threshold_multiplier,
    ).build(raw_data[["trade_date", "close"]].copy())
    master = (
        features.rename(columns={"trade_date": "feature_date"})
        .merge(labels, on="feature_date", how="inner", validate="one_to_one")
        .sort_values("target_date")
        .reset_index(drop=True)
    )
    master[TARGET_STATE_FEATURE] = master["rolling_volatility"].astype(float)
    return master.loc[
        pd.to_datetime(master["target_date"]) <= training_cutoff
    ].reset_index(drop=True)


def dataset(master: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    data = master.dropna(subset=feature_columns).copy()
    return data[
        [
            "feature_date",
            "target_date",
            *feature_columns,
            "future_log_return",
            "rolling_volatility",
            "threshold",
            "direction",
        ]
    ].sort_values("target_date").reset_index(drop=True)


def verification_boundary(master: pd.DataFrame) -> pd.Timestamp:
    base = dataset(master, base_columns())
    split_index = int(np.floor(len(base) * (1.0 - VERIFICATION_FRACTION)))
    split_index = max(1, min(split_index, len(base) - 1))
    return pd.Timestamp(base.iloc[split_index - 1]["target_date"])


def split_data(
    data: pd.DataFrame,
    boundary: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = data.loc[
        pd.to_datetime(data["target_date"]) <= boundary
    ].reset_index(drop=True)
    verification = data.loc[
        pd.to_datetime(data["target_date"]) > boundary
    ].reset_index(drop=True)
    return development, verification


def sample_signature(data: pd.DataFrame) -> str:
    values = pd.to_datetime(data["target_date"]).astype("int64").to_numpy()
    return hashlib.sha1(values.tobytes()).hexdigest()[:12]


def feature_signature(feature_columns: list[str]) -> str:
    return hashlib.sha1("|".join(feature_columns).encode("utf-8")).hexdigest()[:10]


def study_name(prefix: str, data: pd.DataFrame, feature_columns: list[str]) -> str:
    prefix = re.sub(r"[^a-zA-Z0-9_]+", "_", prefix)
    return f"{prefix}_{sample_signature(data)}_{feature_signature(feature_columns)}"


def optimize(
    name: str,
    data: pd.DataFrame,
    feature_columns: list[str],
    trials: int,
) -> dict:
    print()
    print(f"OPTUNA: {name}")
    print(
        f"Rows {len(data)} | "
        f"{pd.Timestamp(data['target_date'].min()).date()} -> "
        f"{pd.Timestamp(data['target_date'].max()).date()} | "
        f"features {len(feature_columns)}"
    )
    selector = HierarchicalXLSTMParameterSelector(
        feature_columns=feature_columns,
        task="direction",
        n_splits=CV_SPLITS,
        n_trials=trials,
        max_epochs=MAX_SELECTION_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        random_state=RANDOM_STATE,
        objective_metric="roc_auc",
        study_name=study_name(name, data, feature_columns),
        storage_url=OPTUNA_STORAGE_URL,
    )
    return selector.select_best_parameters(training_data=data)


def screening_result(
    candidate: Stage2FeatureCandidate,
    master: pd.DataFrame,
    boundary: pd.Timestamp,
    parameters: dict,
    baseline_cache: dict[str, dict],
) -> dict:
    feature_columns = columns_for_groups(candidate.groups)
    development, _ = split_data(dataset(master, feature_columns), boundary)
    print()
    print(candidate.name)
    print("  groups:", list(candidate.groups))
    print("  development rows:", len(development))
    if len(development) < SCREENING_MIN_ROWS:
        print("  skipped: insufficient long-history development rows")
        return {
            "candidate_name": candidate.name,
            "groups": list(candidate.groups),
            "status": "skipped",
            "training_rows": int(len(development)),
        }

    metrics = evaluate_candidate(development, feature_columns, parameters)
    signature = sample_signature(development)
    if not candidate.groups:
        baseline = metrics
        baseline_cache[signature] = metrics
    else:
        baseline = baseline_cache.get(signature)
        if baseline is None:
            print("  matched BASE on exact same dates")
            baseline = evaluate_candidate(development, base_columns(), parameters)
            baseline_cache[signature] = baseline
    delta = float(metrics["stage2_roc_auc"] - baseline["stage2_roc_auc"])
    print(
        f"  AUC {metrics['stage2_roc_auc']:.4f} | "
        f"base {baseline['stage2_roc_auc']:.4f} | delta {delta:+.4f} | "
        f"std {metrics['stage2_roc_auc_fold_std']:.4f}"
    )
    return {
        "candidate_name": candidate.name,
        "groups": list(candidate.groups),
        "status": "ok",
        "feature_count": len(feature_columns),
        "training_rows": len(development),
        "training_start": pd.Timestamp(development["target_date"].min()).strftime(
            "%Y-%m-%d"
        ),
        "training_end": pd.Timestamp(development["target_date"].max()).strftime(
            "%Y-%m-%d"
        ),
        "matched_base_roc_auc": float(baseline["stage2_roc_auc"]),
        "delta_roc_auc_vs_matched_base": delta,
        **{
            key: value
            for key, value in metrics.items()
            if key != "fold_metrics"
        },
    }


def run_screening_round(
    candidates: list[Stage2FeatureCandidate],
    master: pd.DataFrame,
    boundary: pd.Timestamp,
    parameters: dict,
    completed: dict[str, dict],
    baseline_cache: dict[str, dict],
    metadata: dict,
) -> list[dict]:
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}]", end=" ")
        if candidate.name in completed:
            print(candidate.name, "-- completed")
            row = completed[candidate.name]
        else:
            print("evaluating")
            row = screening_result(
                candidate,
                master,
                boundary,
                parameters,
                baseline_cache,
            )
            completed[candidate.name] = row
            json_save(
                SCREENING_CHECKPOINT,
                {"metadata": metadata, "results": list(completed.values())},
            )
        if row.get("status") == "ok":
            rows.append(row)
    return rows


def holdout_metrics(
    development: pd.DataFrame,
    verification: pd.DataFrame,
    feature_columns: list[str],
    selection: dict,
    seed: int,
) -> dict:
    batch = train_stage2_fold(
        fold_train=development,
        fold_validation=verification,
        feature_columns=feature_columns,
        parameters=selection["parameters"],
        seed=seed,
    )
    directions = np.asarray(batch["directions"], dtype=object)
    move_mask = directions != "FLAT"
    actual = np.asarray(
        [1 if value == "UP" else 0 for value in directions[move_mask]],
        dtype=np.int64,
    )
    probabilities = batch["up_probabilities"][move_mask]
    dates = pd.DatetimeIndex(batch["target_dates"])[move_mask]
    threshold = float(selection["decision_threshold"])
    probability_metrics = binary_probability_metrics(
        actual=actual,
        positive_probabilities=probabilities,
    )
    predicted = (probabilities >= threshold).astype(np.int64)
    class_metrics = binary_metrics(
        actual=actual,
        predicted=predicted,
        negative_name="DOWN",
        positive_name="UP",
    )
    return {
        "actual": actual,
        "probabilities": probabilities,
        "target_dates": dates,
        "roc_auc": float(probability_metrics["roc_auc"]),
        "balanced_accuracy": float(class_metrics["balanced_accuracy"]),
        "macro_f1": float(class_metrics["macro_f1"]),
        "down_f1": float(class_metrics["per_class"]["DOWN"]["f1"]),
        "up_f1": float(class_metrics["per_class"]["UP"]["f1"]),
        "move_rows": len(actual),
    }


def deep_tune(
    phase: str,
    target: Stage2TargetSpec,
    candidate: Stage2FeatureCandidate,
    master: pd.DataFrame,
    boundary: pd.Timestamp,
    trials: int,
    base_cache: dict[str, dict],
) -> dict:
    feature_columns = columns_for_groups(candidate.groups)
    development, verification = split_data(dataset(master, feature_columns), boundary)
    selection = optimize(
        f"{phase}_{target.name}_{candidate.name}",
        development,
        feature_columns,
        trials,
    )
    signature = sample_signature(development)
    cache_key = f"{target.name}_{signature}_{trials}"
    if cache_key not in base_cache:
        base_cache[cache_key] = optimize(
            f"{phase}_{target.name}_base_{signature}",
            development[
                [
                    "feature_date",
                    "target_date",
                    *base_columns(),
                    "future_log_return",
                    "rolling_volatility",
                    "threshold",
                    "direction",
                ]
            ].copy(),
            base_columns(),
            trials,
        )
    base_selection = base_cache[cache_key]

    candidate_holdout = holdout_metrics(
        development,
        verification,
        feature_columns,
        selection,
        RANDOM_STATE + 700001,
    )
    base_holdout = holdout_metrics(
        development,
        verification,
        base_columns(),
        base_selection,
        RANDOM_STATE + 700001,
    )
    if not candidate_holdout["target_dates"].equals(base_holdout["target_dates"]):
        raise ValueError("Candidate/base verification dates are not aligned.")
    if not np.array_equal(candidate_holdout["actual"], base_holdout["actual"]):
        raise ValueError("Candidate/base verification labels are not aligned.")

    bootstrap = moving_block_bootstrap_auc_delta(
        actual=candidate_holdout["actual"],
        candidate_probabilities=candidate_holdout["probabilities"],
        baseline_probabilities=base_holdout["probabilities"],
        resamples=BOOTSTRAP_RESAMPLES,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        random_state=RANDOM_STATE,
    )
    return {
        "phase": phase,
        "target_name": target.name,
        "target_role": target.role,
        "volatility_window": target.volatility_window,
        "threshold_multiplier": target.threshold_multiplier,
        "candidate_name": candidate.name,
        "groups": list(candidate.groups),
        "feature_count": len(feature_columns),
        "development_rows": len(development),
        "verification_rows": len(verification),
        "verification_move_rows": candidate_holdout["move_rows"],
        "development_oof_auc": float(selection["threshold_oof_roc_auc"]),
        "development_oof_auc_fold_std": float(
            selection["threshold_oof_roc_auc_fold_std"]
        ),
        "development_base_oof_auc": float(base_selection["threshold_oof_roc_auc"]),
        "development_delta_auc": float(
            selection["threshold_oof_roc_auc"]
            - base_selection["threshold_oof_roc_auc"]
        ),
        "verification_auc": candidate_holdout["roc_auc"],
        "verification_base_auc": base_holdout["roc_auc"],
        "verification_delta_auc": float(
            candidate_holdout["roc_auc"] - base_holdout["roc_auc"]
        ),
        "verification_delta_lower_95": bootstrap["lower_95"],
        "verification_delta_upper_95": bootstrap["upper_95"],
        "verification_probability_delta_positive": bootstrap[
            "probability_delta_positive"
        ],
        "verification_balanced_accuracy": candidate_holdout["balanced_accuracy"],
        "verification_macro_f1": candidate_holdout["macro_f1"],
        "verification_down_f1": candidate_holdout["down_f1"],
        "verification_up_f1": candidate_holdout["up_f1"],
        "status": "ok",
    }


def deep_key(phase: str, target_name: str, candidate_name: str) -> str:
    return f"{phase}::{target_name}::{candidate_name}"


def load_deep_progress() -> dict[str, dict]:
    payload = json_load(DEEP_PROGRESS, {"rows": []})
    return {
        deep_key(row["phase"], row["target_name"], row["candidate_name"]): row
        for row in payload.get("rows", [])
    }


def save_deep_progress(rows: dict[str, dict]) -> None:
    json_save(DEEP_PROGRESS, {"rows": list(rows.values())})


def rank_verified(rows: list[dict]) -> list[dict]:
    return sorted(
        [row for row in rows if row.get("status") == "ok"],
        key=lambda row: (
            -float(row["verification_delta_auc"]),
            -float(row["verification_auc"]),
            -float(row["verification_probability_delta_positive"]),
            float(row["development_oof_auc_fold_std"]),
        ),
    )


def main():
    EXPERIMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    cutoff = load_training_cutoff()
    raw_data = Stage2SignalDataRepository().get_training_data(ticker=TICKER)
    primary = next(spec for spec in target_specs() if spec.role == "primary")
    primary_master = build_master(raw_data, primary, cutoff)
    boundary = verification_boundary(primary_master)

    print("=" * 78)
    print("STAGE-2 CONDITIONED MEGASEARCH V1")
    print("=" * 78)
    print("PRIMARY: 90d x 0.700 -- the Stage-1 winner")
    print("Training cutoff:", cutoff.date())
    print("Internal development ends:", boundary.date())
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Outer validation and held-out test are NOT evaluated.")
    print("Primary distribution:", target_distribution(dataset(primary_master, base_columns())))

    primary_base, _ = split_data(dataset(primary_master, base_columns()), boundary)
    base_selection = optimize(
        "primary_90d_k700_screening_base",
        primary_base,
        base_columns(),
        BASE_OPTUNA_TRIALS,
    )
    screening_parameters = dict(base_selection["parameters"])

    core_groups = [
        name
        for name in Stage2WideFeatureBuilder.FEATURE_GROUPS
        if name not in Stage2WideFeatureBuilder.SHORT_HISTORY_GROUPS
    ]
    metadata = {
        "version": 1,
        "primary_target": primary.name,
        "training_cutoff": cutoff.strftime("%Y-%m-%d"),
        "verification_boundary": boundary.strftime("%Y-%m-%d"),
        "core_groups": core_groups,
        "screening_parameters": screening_parameters,
    }
    checkpoint = json_load(SCREENING_CHECKPOINT, {"results": []})
    completed = {row["candidate_name"]: row for row in checkpoint.get("results", [])}
    baseline_cache: dict[str, dict] = {}

    print()
    print("UNIVARIATE PRIMARY-TARGET SCREEN")
    development_master = primary_master.loc[
        pd.to_datetime(primary_master["target_date"]) <= boundary
    ].reset_index(drop=True)
    univariate = univariate_feature_auc_screen(
        training_data=development_master,
        feature_columns=[
            feature
            for group in core_groups
            for feature in Stage2WideFeatureBuilder.FEATURE_GROUPS[group]
        ],
        n_splits=CV_SPLITS,
        minimum_rows=SCREENING_MIN_ROWS,
    )
    univariate_path = (
        EXPERIMENT_DIRECTORY / f"stage2_90d_k700_univariate_v1_{timestamp}.csv"
    )
    univariate.to_csv(univariate_path, index=False)
    print(univariate.head(30).round(4).to_string(index=False))

    print()
    print("PHASE 1 - SINGLES")
    run_screening_round(
        [Stage2FeatureCandidate(())],
        primary_master,
        boundary,
        screening_parameters,
        completed,
        baseline_cache,
        metadata,
    )
    single_rows = run_screening_round(
        single_candidates(core_groups),
        primary_master,
        boundary,
        screening_parameters,
        completed,
        baseline_cache,
        metadata,
    )
    ranked_groups = [
        row["groups"][0]
        for row in sorted(
            single_rows,
            key=lambda row: (
                -row["delta_roc_auc_vs_matched_base"],
                row["stage2_roc_auc_fold_std"],
            ),
        )
    ]

    print()
    print("PHASE 2 - ALL PAIRS AMONG TOP 10 GROUPS")
    pair_rows = run_screening_round(
        pair_candidates(ranked_groups, TOP_SINGLE_GROUPS_FOR_PAIRS),
        primary_master,
        boundary,
        screening_parameters,
        completed,
        baseline_cache,
        metadata,
    )
    beam = select_beam(pair_rows, BEAM_WIDTH)
    for depth in range(3, MAX_GROUP_DEPTH + 1):
        if not beam:
            break
        print()
        print(f"PHASE {depth} - BEAM TO {depth} GROUPS")
        rows = run_screening_round(
            expand_beam(beam, core_groups, depth),
            primary_master,
            boundary,
            screening_parameters,
            completed,
            baseline_cache,
            metadata,
        )
        beam = select_beam(rows, BEAM_WIDTH)

    print()
    print("FINAL CONTROL - ALL LONG-HISTORY GROUPS")
    run_screening_round(
        [Stage2FeatureCandidate(tuple(sorted(core_groups)))],
        primary_master,
        boundary,
        screening_parameters,
        completed,
        baseline_cache,
        metadata,
    )

    screening_rows = [row for row in completed.values() if row.get("status") == "ok"]
    screening_summary = pd.DataFrame(screening_rows).sort_values(
        ["delta_roc_auc_vs_matched_base", "stage2_roc_auc"],
        ascending=[False, False],
    )
    screening_path = (
        EXPERIMENT_DIRECTORY / f"stage2_90d_k700_screening_v1_{timestamp}.csv"
    )
    screening_summary.to_csv(screening_path, index=False)
    finalists = select_finalists(
        screening_rows,
        FINALIST_COUNT,
        FINALIST_MIN_ROWS,
        FINALIST_MAX_FOLD_STD,
    )

    print()
    print("DEEP-TUNING 8 PRIMARY FINALISTS")
    for index, row in enumerate(finalists, start=1):
        print(
            f"{index}. {row['candidate_name']} | "
            f"AUC {row['stage2_roc_auc']:.4f} | "
            f"delta {row['delta_roc_auc_vs_matched_base']:+.4f}"
        )

    deep_progress = load_deep_progress()
    base_cache: dict[str, dict] = {}
    primary_deep = []
    for index, finalist in enumerate(finalists, start=1):
        candidate = Stage2FeatureCandidate(tuple(finalist["groups"]))
        key = deep_key("primary", primary.name, candidate.name)
        print()
        print("=" * 78)
        print(f"PRIMARY FINALIST [{index}/{len(finalists)}] {candidate.name}")
        print("=" * 78)
        if key not in deep_progress:
            deep_progress[key] = deep_tune(
                "primary",
                primary,
                candidate,
                primary_master,
                boundary,
                FINALIST_OPTUNA_TRIALS,
                base_cache,
            )
            save_deep_progress(deep_progress)
        row = deep_progress[key]
        primary_deep.append(row)
        print(
            f"Verification AUC {row['verification_auc']:.4f} | "
            f"base {row['verification_base_auc']:.4f} | "
            f"delta {row['verification_delta_auc']:+.4f} | "
            f"CI [{row['verification_delta_lower_95']:.4f}, "
            f"{row['verification_delta_upper_95']:.4f}]"
        )

    primary_ranked = rank_verified(primary_deep)
    primary_path = (
        EXPERIMENT_DIRECTORY / f"stage2_90d_k700_verified_v1_{timestamp}.csv"
    )
    pd.DataFrame(primary_ranked).to_csv(primary_path, index=False)

    print()
    print("=" * 78)
    print("PRIMARY 90d x .700 VERIFIED RANKING")
    print("=" * 78)
    print(
        pd.DataFrame(primary_ranked)[
            [
                "candidate_name",
                "development_oof_auc",
                "development_delta_auc",
                "verification_auc",
                "verification_base_auc",
                "verification_delta_auc",
                "verification_delta_lower_95",
                "verification_delta_upper_95",
                "verification_probability_delta_positive",
                "verification_balanced_accuracy",
                "verification_macro_f1",
            ]
        ].round(4).to_string(index=False)
    )

    top_candidates = [
        Stage2FeatureCandidate(tuple(row["groups"]))
        for row in primary_ranked[:ROBUSTNESS_CHAMPION_COUNT]
    ]
    robustness_rows = []
    print()
    print("=" * 78)
    print("ROBUSTNESS TARGETS")
    print("=" * 78)
    for target in [spec for spec in target_specs() if spec.role != "primary"]:
        master = build_master(raw_data, target, cutoff)
        print()
        print(
            f"TARGET {target.name}: {target.volatility_window}d x "
            f"{target.threshold_multiplier:.3f}"
        )
        print("Distribution:", target_distribution(dataset(master, base_columns())))
        target_base_cache: dict[str, dict] = {}
        for candidate in top_candidates:
            key = deep_key("robustness", target.name, candidate.name)
            if key not in deep_progress:
                deep_progress[key] = deep_tune(
                    "robustness",
                    target,
                    candidate,
                    master,
                    boundary,
                    ROBUSTNESS_OPTUNA_TRIALS,
                    target_base_cache,
                )
                save_deep_progress(deep_progress)
            row = deep_progress[key]
            robustness_rows.append(row)
            print(
                f"  {candidate.name}: verification AUC "
                f"{row['verification_auc']:.4f}, delta "
                f"{row['verification_delta_auc']:+.4f}"
            )

    robustness_path = (
        EXPERIMENT_DIRECTORY / f"stage2_target_robustness_v1_{timestamp}.csv"
    )
    pd.DataFrame(robustness_rows).to_csv(robustness_path, index=False)

    print()
    print("=" * 78)
    print("SHORT-HISTORY CHALLENGE - EXPLORATORY")
    print("=" * 78)
    champion_groups = tuple(primary_ranked[0]["groups"])
    short_candidates = [
        Stage2FeatureCandidate(
            tuple(sorted(set([*champion_groups, "futures_smallcap"])))
        ),
        Stage2FeatureCandidate(
            tuple(sorted(set([*champion_groups, "volatility_options_short"])))
        ),
        Stage2FeatureCandidate(
            tuple(
                sorted(
                    set(
                        [
                            *champion_groups,
                            "futures_smallcap",
                            "volatility_options_short",
                        ]
                    )
                )
            )
        ),
    ]
    short_rows = []
    champion_control_columns = columns_for_groups(champion_groups)
    champion_control_cache: dict[str, dict] = {}
    print(
        "Each augmented candidate is compared with the PRIMARY CHAMPION "
        "on the exact same modern-history dates."
    )
    for candidate in short_candidates:
        feature_columns = columns_for_groups(candidate.groups)
        data = dataset(primary_master, feature_columns)
        print(
            f"{candidate.name}: {len(data)} rows, "
            f"{pd.Timestamp(data['target_date'].min()).date()} -> "
            f"{pd.Timestamp(data['target_date'].max()).date()}"
        )
        key = deep_key("short_history", primary.name, candidate.name)
        if key not in deep_progress:
            selection = optimize(
                f"short_{candidate.name}",
                data,
                feature_columns,
                SHORT_HISTORY_OPTUNA_TRIALS,
            )
            signature = sample_signature(data)
            if signature not in champion_control_cache:
                champion_control_cache[signature] = optimize(
                    f"short_champion_control_{signature}",
                    data[
                        [
                            "feature_date",
                            "target_date",
                            *champion_control_columns,
                            "future_log_return",
                            "rolling_volatility",
                            "threshold",
                            "direction",
                        ]
                    ].copy(),
                    champion_control_columns,
                    SHORT_HISTORY_OPTUNA_TRIALS,
                )
            control = champion_control_cache[signature]
            deep_progress[key] = {
                "phase": "short_history",
                "target_name": primary.name,
                "candidate_name": candidate.name,
                "groups": list(candidate.groups),
                "training_rows": len(data),
                "oof_auc": float(selection["threshold_oof_roc_auc"]),
                "champion_control_oof_auc": float(
                    control["threshold_oof_roc_auc"]
                ),
                "delta_oof_auc_vs_champion": float(
                    selection["threshold_oof_roc_auc"]
                    - control["threshold_oof_roc_auc"]
                ),
                "oof_fold_std": float(
                    selection["threshold_oof_roc_auc_fold_std"]
                ),
                "status": "ok",
            }
            save_deep_progress(deep_progress)
        row = deep_progress[key]
        short_rows.append(row)
        print(
            f"  OOF AUC {row['oof_auc']:.4f} | champion control "
            f"{row['champion_control_oof_auc']:.4f} | delta "
            f"{row['delta_oof_auc_vs_champion']:+.4f}"
        )

    short_path = (
        EXPERIMENT_DIRECTORY
        / f"stage2_short_history_challenge_v1_{timestamp}.csv"
    )
    pd.DataFrame(short_rows).to_csv(short_path, index=False)

    experiment_path = ExperimentTracker(str(EXPERIMENT_DIRECTORY)).save(
        experiment_name=EXPERIMENT_NAME,
        model_name=MODEL_NAME,
        parameters={
            "primary_target": "90d x 0.700",
            "training_cutoff": cutoff.strftime("%Y-%m-%d"),
            "internal_verification_boundary": boundary.strftime("%Y-%m-%d"),
            "base_optuna_trials": BASE_OPTUNA_TRIALS,
            "finalist_optuna_trials": FINALIST_OPTUNA_TRIALS,
            "robustness_optuna_trials": ROBUSTNESS_OPTUNA_TRIALS,
            "short_history_optuna_trials": SHORT_HISTORY_OPTUNA_TRIALS,
            "beam_width": BEAM_WIDTH,
            "max_group_depth": MAX_GROUP_DEPTH,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        metrics={
            "primary_verified": primary_ranked,
            "robustness": robustness_rows,
            "short_history": short_rows,
        },
        features=list(Stage2WideFeatureBuilder.FEATURE_COLUMNS),
    )

    print()
    print("=" * 78)
    print("MEGASEARCH COMPLETE")
    print("=" * 78)
    print("Verified primary winner:", primary_ranked[0]["candidate_name"])
    print("Verification AUC:", round(primary_ranked[0]["verification_auc"], 4))
    print("Verification delta:", round(primary_ranked[0]["verification_delta_auc"], 4))
    print("Univariate:", univariate_path)
    print("Screening:", screening_path)
    print("Primary verified:", primary_path)
    print("Robustness:", robustness_path)
    print("Short-history challenge:", short_path)
    print("Experiment:", experiment_path)
    print("Screening checkpoint:", SCREENING_CHECKPOINT)
    print("Deep progress:", DEEP_PROGRESS)
    print("Optuna DB:", OPTUNA_STORAGE_URL)
    print("Outer validation was NOT evaluated.")
    print("Held-out test set was NOT evaluated.")


if __name__ == "__main__":
    main()
