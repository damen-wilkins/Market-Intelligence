from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class Stage2TargetSpec:
    name: str
    volatility_window: int
    threshold_multiplier: float
    role: str


@dataclass(frozen=True)
class Stage2FeatureCandidate:
    groups: tuple[str, ...]

    @property
    def name(self) -> str:
        return "base_only" if not self.groups else "+".join(self.groups)


def target_specs() -> list[Stage2TargetSpec]:
    return [
        Stage2TargetSpec("primary_90d_k700", 90, 0.700, "primary"),
        Stage2TargetSpec("neighbor_90d_k675", 90, 0.675, "local_robustness"),
        Stage2TargetSpec("neighbor_90d_k725", 90, 0.725, "local_robustness"),
        Stage2TargetSpec("runnerup_30d_k725", 30, 0.725, "runner_up"),
        Stage2TargetSpec("runnerup_40d_k725", 40, 0.725, "runner_up"),
    ]


def candidate_key(groups: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(groups)))


def single_candidates(group_names: list[str]) -> list[Stage2FeatureCandidate]:
    return [Stage2FeatureCandidate((name,)) for name in group_names]


def pair_candidates(
    ranked_groups: list[str],
    top_group_count: int,
) -> list[Stage2FeatureCandidate]:
    selected = ranked_groups[:top_group_count]
    return [
        Stage2FeatureCandidate(candidate_key(pair))
        for pair in combinations(selected, 2)
    ]


def expand_beam(
    beam: list[tuple[str, ...]],
    all_group_names: list[str],
    depth: int,
) -> list[Stage2FeatureCandidate]:
    candidates: dict[tuple[str, ...], Stage2FeatureCandidate] = {}
    for groups in beam:
        for group_name in all_group_names:
            if group_name in groups:
                continue
            expanded = candidate_key([*groups, group_name])
            if len(expanded) == depth:
                candidates[expanded] = Stage2FeatureCandidate(expanded)
    return list(candidates.values())


def select_beam(rows: list[dict], width: int) -> list[tuple[str, ...]]:
    eligible = [row for row in rows if row.get("status") == "ok"]
    ranked = sorted(
        eligible,
        key=lambda row: (
            -float(row["delta_roc_auc_vs_matched_base"]),
            float(row["stage2_roc_auc_fold_std"]),
            -int(row["training_rows"]),
            len(row["groups"]),
        ),
    )
    return [tuple(row["groups"]) for row in ranked[:width]]


def select_finalists(
    rows: list[dict],
    count: int,
    minimum_rows: int,
    maximum_fold_std: float,
) -> list[dict]:
    eligible = [
        row
        for row in rows
        if (
            row.get("status") == "ok"
            and row.get("candidate_name") != "base_only"
            and int(row.get("training_rows", 0)) >= minimum_rows
            and float(row.get("stage2_roc_auc_fold_std", 1.0)) <= maximum_fold_std
        )
    ]
    return sorted(
        eligible,
        key=lambda row: (
            -float(row["delta_roc_auc_vs_matched_base"]),
            -float(row["stage2_roc_auc"]),
            float(row["stage2_roc_auc_fold_std"]),
            -int(row["training_rows"]),
            len(row["groups"]),
        ),
    )[:count]


def moving_block_bootstrap_auc_delta(
    actual: np.ndarray,
    candidate_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    resamples: int = 1000,
    block_length: int = 20,
    random_state: int = 42,
) -> dict:
    actual = np.asarray(actual, dtype=np.int64)
    candidate_probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
    baseline_probabilities = np.asarray(baseline_probabilities, dtype=np.float64)
    if not (
        len(actual) == len(candidate_probabilities) == len(baseline_probabilities)
    ):
        raise ValueError("Bootstrap arrays must have the same length.")
    if len(np.unique(actual)) < 2:
        return {
            "delta_auc": 0.0,
            "lower_95": 0.0,
            "upper_95": 0.0,
            "probability_delta_positive": 0.5,
            "valid_resamples": 0,
        }

    point_delta = float(
        roc_auc_score(actual, candidate_probabilities)
        - roc_auc_score(actual, baseline_probabilities)
    )
    rng = np.random.default_rng(random_state)
    n = len(actual)
    starts = np.arange(max(1, n - block_length + 1))
    values: list[float] = []
    for _ in range(resamples):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.choice(starts))
            indices.extend(range(start, min(start + block_length, n)))
        indices = indices[:n]
        sample_actual = actual[indices]
        if len(np.unique(sample_actual)) < 2:
            continue
        candidate_auc = roc_auc_score(
            sample_actual,
            candidate_probabilities[indices],
        )
        baseline_auc = roc_auc_score(
            sample_actual,
            baseline_probabilities[indices],
        )
        values.append(float(candidate_auc - baseline_auc))

    if not values:
        return {
            "delta_auc": point_delta,
            "lower_95": point_delta,
            "upper_95": point_delta,
            "probability_delta_positive": float(point_delta > 0.0),
            "valid_resamples": 0,
        }

    values_array = np.asarray(values, dtype=np.float64)
    return {
        "delta_auc": point_delta,
        "lower_95": float(np.quantile(values_array, 0.025)),
        "upper_95": float(np.quantile(values_array, 0.975)),
        "probability_delta_positive": float(np.mean(values_array > 0.0)),
        "valid_resamples": int(len(values_array)),
    }


def target_distribution(dataframe: pd.DataFrame) -> dict:
    directions = dataframe["direction"].astype(str)
    return {
        "down_share": float((directions == "DOWN").mean()),
        "flat_share": float((directions == "FLAT").mean()),
        "up_share": float((directions == "UP").mean()),
        "move_rows": int((directions != "FLAT").sum()),
        "rows": int(len(directions)),
    }
