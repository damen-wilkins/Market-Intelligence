from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from app.training.stage2_conditioned_target_research import target_specs
from app.training.stage2_regime_development_research import (
    REGIME_HYPOTHESES,
    REGIME_ORDER,
    Stage2RegimeDevelopmentResearch,
)
from database.stage2_signal_data_repository import Stage2SignalDataRepository
from scripts.run_stage2_conditioned_megasearch import (
    build_master,
    columns_for_groups,
    dataset,
    load_training_cutoff,
)


TICKER = "SPY"
TARGET_WINDOW = 90
TARGET_MULTIPLIER = 0.700
OUTER_SPLITS = 3
RANDOM_STATE = 42
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_BLOCK_LENGTH = 20
PERMUTATION_RESAMPLES = 2000
FALLBACK_WINNER_GROUPS = ("breadth", "calendar", "interaction_consensus")
EXPERIMENT_DIRECTORY = Path("experiments")
TREE_PROGRESS_PATH = EXPERIMENT_DIRECTORY / "stage2_return_architecture_tree_v1_progress.json"
EXPERIMENT_NAME = "stage2_regime_development_screen_v1"


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


def latest_verified_winner_groups() -> tuple[str, ...]:
    paths = sorted(
        EXPERIMENT_DIRECTORY.glob("stage2_90d_k700_verified_v1_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        return FALLBACK_WINNER_GROUPS
    frame = pd.read_csv(paths[0])
    if frame.empty or "candidate_name" not in frame.columns:
        return FALLBACK_WINNER_GROUPS
    name = str(frame.iloc[0]["candidate_name"])
    if not name or name == "base_only":
        return FALLBACK_WINNER_GROUPS
    return tuple(sorted(part for part in name.split("+") if part))


def load_nested_binary_predictions() -> dict[int, dict]:
    if not TREE_PROGRESS_PATH.exists():
        raise FileNotFoundError(
            f"{TREE_PROGRESS_PATH} is required. Run the completed Stage-2 return "
            "architecture tree research before this screen."
        )
    with TREE_PROGRESS_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    rows = [
        row
        for row in payload.get("rows", [])
        if row.get("architecture") == "xgboost_binary_winner"
    ]
    by_fold = {int(row["outer_fold"]): row for row in rows}
    missing = sorted(set(range(1, OUTER_SPLITS + 1)) - set(by_fold))
    if missing:
        raise RuntimeError(
            "Nested XGBoost binary development predictions are incomplete for folds: "
            + ", ".join(map(str, missing))
        )
    return by_fold


def primary_target_spec():
    matches = [
        spec
        for spec in target_specs()
        if spec.volatility_window == TARGET_WINDOW
        and abs(spec.threshold_multiplier - TARGET_MULTIPLIER) < 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeError("Could not resolve the locked 90d x 0.700 target spec.")
    return matches[0]


def build_research_data(
    research: Stage2RegimeDevelopmentResearch,
) -> tuple[pd.DataFrame, tuple[str, ...], list[str], pd.Timestamp]:
    cutoff = load_training_cutoff()
    raw = Stage2SignalDataRepository().get_training_data(ticker=TICKER)
    master = build_master(raw, primary_target_spec(), cutoff)
    winner_groups = latest_verified_winner_groups()
    winner_features = columns_for_groups(winner_groups)

    base = dataset(master, winner_features)
    missing_regime_features = [
        feature
        for feature in research.feature_columns
        if feature not in base.columns
    ]
    if missing_regime_features:
        regime_frame = master[["target_date", *missing_regime_features]].copy()
        if regime_frame["target_date"].duplicated().any():
            raise ValueError("Master data contains duplicate target dates.")
        base = base.merge(
            regime_frame,
            on="target_date",
            how="left",
            validate="one_to_one",
        )
    research.validate_features(base)
    return base.sort_values("target_date").reset_index(drop=True), winner_groups, winner_features, cutoff


def reconstruct_nested_oof(
    research_data: pd.DataFrame,
    saved_by_fold: dict[int, dict],
    research: Stage2RegimeDevelopmentResearch,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = TimeSeriesSplit(n_splits=OUTER_SPLITS)
    prediction_parts: list[pd.DataFrame] = []
    threshold_rows: list[dict] = []

    for fold_number, (train_index, test_index) in enumerate(
        splitter.split(research_data), start=1
    ):
        outer_train = research_data.iloc[train_index].reset_index(drop=True)
        outer_test = research_data.iloc[test_index].reset_index(drop=True)
        train_move = outer_train.loc[outer_train["direction"] != "FLAT"].reset_index(drop=True)
        test_move = outer_test.loc[outer_test["direction"] != "FLAT"].reset_index(drop=True)
        saved = saved_by_fold[fold_number]

        saved_dates = pd.DatetimeIndex(pd.to_datetime(saved["target_dates"]))
        expected_dates = pd.DatetimeIndex(pd.to_datetime(test_move["target_date"]))
        if not saved_dates.equals(expected_dates):
            raise ValueError(
                f"Fold {fold_number} saved nested predictions no longer align with "
                "the reconstructed development split. Do not continue with a changed sample."
            )

        actual = np.asarray(saved["actual"], dtype=np.int64)
        expected_actual = (test_move["direction"].astype(str) == "UP").astype(int).to_numpy()
        if not np.array_equal(actual, expected_actual):
            raise ValueError(
                f"Fold {fold_number} saved actual labels do not match the locked target."
            )

        predictions = test_move[
            [
                "feature_date",
                "target_date",
                "future_log_return",
                "direction",
                *research.feature_columns,
            ]
        ].copy()
        predictions["outer_fold"] = fold_number
        predictions["actual_up"] = actual
        predictions["score"] = np.asarray(saved["score"], dtype=np.float64)
        predictions["saved_decision_threshold"] = float(saved["decision_threshold"])
        predictions["predicted_up"] = (
            predictions["score"] >= predictions["saved_decision_threshold"]
        ).astype(int)

        for hypothesis in research.hypotheses:
            thresholds = research.development_tertiles(
                training_move=train_move,
                feature=hypothesis.feature,
            )
            regime_column = research.regime_column(hypothesis.feature)
            predictions[regime_column] = research.assign_tertile(
                predictions[hypothesis.feature],
                q33=float(thresholds["q33"]),
                q67=float(thresholds["q67"]),
            )
            assigned = predictions[regime_column].isin(REGIME_ORDER)
            threshold_rows.append(
                {
                    "outer_fold": fold_number,
                    "family": hypothesis.family,
                    "feature": hypothesis.feature,
                    "training_move_rows": int(len(train_move)),
                    "threshold_training_rows": int(thresholds["training_rows"]),
                    "q33": float(thresholds["q33"]),
                    "q67": float(thresholds["q67"]),
                    "test_move_rows": int(len(predictions)),
                    "assigned_test_rows": int(assigned.sum()),
                    "assignment_coverage": float(assigned.mean()),
                    "train_start": pd.Timestamp(train_move["target_date"].min()),
                    "train_end": pd.Timestamp(train_move["target_date"].max()),
                    "test_start": pd.Timestamp(predictions["target_date"].min()),
                    "test_end": pd.Timestamp(predictions["target_date"].max()),
                }
            )
        prediction_parts.append(predictions)

    enriched = pd.concat(prediction_parts, ignore_index=True).sort_values(
        "target_date"
    ).reset_index(drop=True)
    return enriched, pd.DataFrame(threshold_rows)


def print_summary(summary: pd.DataFrame) -> None:
    print()
    print("DEVELOPMENT REGIME HETEROGENEITY SCREEN")
    display = summary[
        [
            "family",
            "feature",
            "prediction_coverage",
            "low_auc",
            "mid_auc",
            "high_auc",
            "pooled_auc_range",
            "heterogeneity_permutation_p",
            "heterogeneity_fdr_q",
            "best_regime",
            "best_regime_auc",
            "best_regime_auc_lower_95",
            "development_regime_candidate",
        ]
    ].copy()
    print(display.round(4).to_string(index=False))

    candidates = summary.loc[summary["development_regime_candidate"]].copy()
    print()
    print("DEVELOPMENT-SUPPORTED REGIME CANDIDATES")
    if candidates.empty:
        print("NONE")
        print(
            "The outer-validation regime pattern did not reproduce strongly enough "
            "inside nested development OOF to justify a regime-specialist model yet."
        )
    else:
        print(
            candidates[
                [
                    "family",
                    "feature",
                    "best_regime",
                    "best_regime_auc",
                    "best_regime_auc_lower_95",
                    "worst_regime",
                    "worst_regime_auc",
                    "heterogeneity_fdr_q",
                ]
            ].round(4).to_string(index=False)
        )

    abstention = summary.loc[summary["development_abstention_candidate"]].copy()
    print()
    print("STRICT DEVELOPMENT ABSTENTION CANDIDATES")
    if abstention.empty:
        print("NONE")
    else:
        print(
            abstention[
                [
                    "family",
                    "feature",
                    "worst_regime",
                    "worst_regime_auc",
                    "worst_regime_auc_upper_95",
                ]
            ].round(4).to_string(index=False)
        )


def main():
    EXPERIMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    research = Stage2RegimeDevelopmentResearch(
        hypotheses=REGIME_HYPOTHESES,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_block_length=BOOTSTRAP_BLOCK_LENGTH,
        permutation_resamples=PERMUTATION_RESAMPLES,
        random_state=RANDOM_STATE,
    )
    saved_by_fold = load_nested_binary_predictions()
    research_data, winner_groups, winner_features, training_cutoff = build_research_data(
        research
    )
    enriched, thresholds = reconstruct_nested_oof(
        research_data=research_data,
        saved_by_fold=saved_by_fold,
        research=research,
    )

    actual = enriched["actual_up"].astype(int).to_numpy()
    score = enriched["score"].astype(float).to_numpy()
    nested_oof_auc = float(roc_auc_score(actual, score))

    print("=" * 78)
    print("STAGE-2 REGIME DEVELOPMENT SCREEN V1")
    print("=" * 78)
    print(f"Target: {TARGET_WINDOW}d x {TARGET_MULTIPLIER:.3f}")
    print("Source predictions:", TREE_PROGRESS_PATH)
    print("Winner feature groups:", list(winner_groups))
    print("Winner model features:", len(winner_features))
    print("Development cutoff:", training_cutoff.date())
    print("Nested OOF MOVE rows:", len(enriched))
    print("Nested OOF XGBoost AUC:", f"{nested_oof_auc:.4f}")
    print("Regime hypotheses:", len(research.hypotheses))
    print()
    print(
        "DEVELOPMENT ONLY: this experiment reconstructs the already-generated nested "
        "walk-forward XGBoost OOF predictions. It does not fit, tune, or select a model."
    )
    print(
        "For every outer fold, LOW/MID/HIGH thresholds are calculated from that fold's "
        "earlier training MOVE rows only. The fold's OOF predictions never set thresholds."
    )
    print(
        "The consumed outer-validation period and the final held-out test are NOT loaded "
        "or evaluated."
    )

    fold_auc = research.fold_conditional_auc(enriched)
    pooled_auc = research.pooled_conditional_auc(enriched)
    summary = research.summarize_hypotheses(
        enriched_predictions=enriched,
        fold_auc=fold_auc,
        pooled_auc=pooled_auc,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    paths = {
        "enriched_oof": EXPERIMENT_DIRECTORY / f"{EXPERIMENT_NAME}_enriched_oof_{timestamp}.csv",
        "fold_thresholds": EXPERIMENT_DIRECTORY / f"{EXPERIMENT_NAME}_fold_thresholds_{timestamp}.csv",
        "fold_auc": EXPERIMENT_DIRECTORY / f"{EXPERIMENT_NAME}_fold_auc_{timestamp}.csv",
        "pooled_auc": EXPERIMENT_DIRECTORY / f"{EXPERIMENT_NAME}_pooled_auc_{timestamp}.csv",
        "summary": EXPERIMENT_DIRECTORY / f"{EXPERIMENT_NAME}_{timestamp}.csv",
        "experiment": EXPERIMENT_DIRECTORY / f"{EXPERIMENT_NAME}_{timestamp}.json",
    }
    enriched.to_csv(paths["enriched_oof"], index=False)
    thresholds.to_csv(paths["fold_thresholds"], index=False)
    fold_auc.to_csv(paths["fold_auc"], index=False)
    pooled_auc.to_csv(paths["pooled_auc"], index=False)
    summary.to_csv(paths["summary"], index=False)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": EXPERIMENT_NAME,
        "target": {
            "volatility_window": TARGET_WINDOW,
            "threshold_multiplier": TARGET_MULTIPLIER,
        },
        "source_predictions": TREE_PROGRESS_PATH,
        "development_cutoff": training_cutoff,
        "winner_groups": list(winner_groups),
        "winner_feature_count": len(winner_features),
        "nested_oof_rows": int(len(enriched)),
        "nested_oof_auc": nested_oof_auc,
        "outer_folds": OUTER_SPLITS,
        "regime_hypotheses": [
            {
                "family": hypothesis.family,
                "feature": hypothesis.feature,
                "rationale": hypothesis.rationale,
            }
            for hypothesis in research.hypotheses
        ],
        "statistics": {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_block_length": BOOTSTRAP_BLOCK_LENGTH,
            "heterogeneity_permutation_resamples": PERMUTATION_RESAMPLES,
            "multiple_testing_correction": "Benjamini-Hochberg FDR",
            "development_candidate_rule": (
                "coverage >= 0.95; heterogeneity FDR q <= 0.10; best pooled regime "
                "AUC >= 0.55; best-regime moving-block bootstrap lower 95% AUC > 0.50; "
                "best regime has valid AUC in at least 2 outer folds"
            ),
            "strict_abstention_rule": (
                "coverage >= 0.95 and worst-regime moving-block bootstrap upper 95% "
                "AUC < 0.50"
            ),
        },
        "methodology": {
            "model_refit": False,
            "hyperparameter_search": False,
            "new_feature_search": False,
            "threshold_source": (
                "Each development outer fold's training MOVE rows only"
            ),
            "outer_validation_evaluated": False,
            "held_out_test_evaluated": False,
            "purpose": (
                "Require the outer-validation regime hypotheses to reproduce inside "
                "independent nested development OOF before designing a regime-aware "
                "Stage-2 architecture."
            ),
        },
        "outputs": paths,
    }
    with paths["experiment"].open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=json_default)

    print_summary(summary)
    print()
    print("=" * 78)
    print("DEVELOPMENT REGIME SCREEN COMPLETE")
    print("=" * 78)
    print("Summary:", paths["summary"])
    print("Fold thresholds:", paths["fold_thresholds"])
    print("Fold conditional AUC:", paths["fold_auc"])
    print("Pooled conditional AUC:", paths["pooled_auc"])
    print("Experiment:", paths["experiment"])
    print("Outer validation was NOT evaluated.")
    print("Held-out test set was NOT evaluated.")
    print()
    print(
        "NEXT DECISION RULE: only features marked development_regime_candidate=True "
        "are eligible to define the first regime-aware Stage-2 architecture. If none "
        "qualify, do not build regime specialists from the consumed validation pattern."
    )


if __name__ == "__main__":
    main()
