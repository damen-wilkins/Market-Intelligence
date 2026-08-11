from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class SignalCandidate:
    groups: tuple[str, ...]

    @property
    def name(self) -> str:
        if not self.groups:
            return "base_only"
        return "+".join(self.groups)


def candidate_key(
    groups: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(sorted(groups))


def build_single_candidates(
    group_names: list[str],
) -> list[SignalCandidate]:
    return [
        SignalCandidate((group_name,))
        for group_name in group_names
    ]


def build_pair_candidates(
    ranked_single_groups: list[str],
    top_group_count: int,
) -> list[SignalCandidate]:
    selected = ranked_single_groups[
        : min(
            top_group_count,
            len(ranked_single_groups),
        )
    ]

    return [
        SignalCandidate(
            candidate_key(pair)
        )
        for pair in combinations(
            selected,
            2,
        )
    ]


def expand_beam_candidates(
    beam_groups: list[tuple[str, ...]],
    all_group_names: list[str],
) -> list[SignalCandidate]:
    candidates = {}

    for groups in beam_groups:
        for group_name in all_group_names:
            if group_name in groups:
                continue

            expanded = candidate_key(
                [
                    *groups,
                    group_name,
                ]
            )
            candidates[expanded] = SignalCandidate(
                expanded
            )

    return list(candidates.values())


def select_beam(
    results: list[dict],
    beam_width: int,
) -> list[tuple[str, ...]]:
    ranked = sorted(
        results,
        key=lambda row: (
            -float(
                row.get(
                    "delta_roc_auc_vs_matched_base",
                    row["stage1_roc_auc"],
                )
            ),
            float(row["stage1_roc_auc_fold_std"]),
            -float(row["stage1_flat_f1"]),
            -int(row.get("training_rows", 0)),
            len(row["groups"]),
        ),
    )

    return [
        tuple(row["groups"])
        for row in ranked[:beam_width]
    ]


def univariate_feature_auc_screen(
    training_data: pd.DataFrame,
    feature_columns: list[str],
    n_splits: int,
    minimum_rows: int = 200,
) -> pd.DataFrame:
    rows = []

    for feature in feature_columns:
        feature_rows = training_data[
            [
                "target_date",
                "direction",
                feature,
            ]
        ].dropna(
            subset=[feature]
        ).sort_values(
            "target_date"
        ).reset_index(drop=True)

        if len(feature_rows) < minimum_rows:
            continue

        labels = (
            feature_rows["direction"] != "FLAT"
        ).astype(np.int64).to_numpy()

        if len(np.unique(labels)) < 2:
            continue

        splitter = TimeSeriesSplit(
            n_splits=n_splits
        )
        fold_aucs = []

        for _, validation_indices in splitter.split(feature_rows):
            actual = labels[validation_indices]
            values = feature_rows.iloc[
                validation_indices
            ][feature].to_numpy(
                dtype=np.float64
            )

            if len(np.unique(actual)) < 2:
                continue

            if np.nanstd(values) == 0.0:
                fold_aucs.append(0.5)
                continue

            auc = float(
                roc_auc_score(
                    actual,
                    values,
                )
            )
            fold_aucs.append(
                max(
                    auc,
                    1.0 - auc,
                )
            )

        if not fold_aucs:
            continue

        full_auc = float(
            roc_auc_score(
                labels,
                feature_rows[feature].to_numpy(
                    dtype=np.float64
                ),
            )
        )

        rows.append(
            {
                "feature": feature,
                "direction": (
                    "higher_move"
                    if full_auc >= 0.5
                    else "higher_flat"
                ),
                "predictive_auc": max(
                    full_auc,
                    1.0 - full_auc,
                ),
                "fold_predictive_auc_mean": float(
                    np.mean(fold_aucs)
                ),
                "fold_predictive_auc_std": float(
                    np.std(
                        fold_aucs,
                        ddof=0,
                    )
                ),
                "rows": int(len(feature_rows)),
                "flat_share": float(
                    np.mean(labels == 0)
                ),
                "start_date": pd.Timestamp(
                    feature_rows["target_date"].min()
                ).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp(
                    feature_rows["target_date"].max()
                ).strftime("%Y-%m-%d"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "direction",
                "predictive_auc",
                "fold_predictive_auc_mean",
                "fold_predictive_auc_std",
                "rows",
                "flat_share",
                "start_date",
                "end_date",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "fold_predictive_auc_mean",
                "fold_predictive_auc_std",
                "rows",
            ],
            ascending=[
                False,
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )
