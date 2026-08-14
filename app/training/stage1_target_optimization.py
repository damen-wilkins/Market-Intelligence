from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class Stage1TargetCandidate:
    name: str
    volatility_window: int
    threshold_multiplier: float


def build_target_grid() -> list[Stage1TargetCandidate]:
    windows = (
        5,
        10,
        15,
        20,
        30,
        40,
        60,
        90,
    )
    multiplier_basis_points = range(
        100,
        801,
        25,
    )

    return [
        Stage1TargetCandidate(
            name=(
                f"flat_{window}d_"
                f"k{multiplier_basis_point:03d}"
            ),
            volatility_window=int(window),
            threshold_multiplier=float(
                multiplier_basis_point / 1000.0
            ),
        )
        for window in windows
        for multiplier_basis_point in multiplier_basis_points
    ]


def align_candidate_datasets(
    datasets: dict[str, pd.DataFrame],
    training_end_date: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    if not datasets:
        raise ValueError(
            "At least one target dataset is required."
        )

    training_end_date = pd.Timestamp(
        training_end_date
    )
    common_dates: pd.DatetimeIndex | None = None

    prepared: dict[str, pd.DataFrame] = {}

    for name, dataframe in datasets.items():
        if dataframe.empty:
            raise ValueError(
                f"Target dataset {name} is empty."
            )

        required_columns = {
            "feature_date",
            "target_date",
            "direction",
        }
        missing = required_columns - set(
            dataframe.columns
        )
        if missing:
            raise ValueError(
                f"Target dataset {name} is missing columns: "
                f"{sorted(missing)}"
            )

        candidate = dataframe.copy()
        candidate["feature_date"] = pd.to_datetime(
            candidate["feature_date"]
        )
        candidate["target_date"] = pd.to_datetime(
            candidate["target_date"]
        )

        candidate = candidate.loc[
            candidate["target_date"]
            <= training_end_date
        ].sort_values(
            "target_date"
        ).reset_index(
            drop=True
        )

        if candidate.empty:
            raise ValueError(
                f"Target dataset {name} has no training rows."
            )

        if candidate["target_date"].duplicated().any():
            raise ValueError(
                f"Target dataset {name} contains duplicate target dates."
            )

        dates = pd.DatetimeIndex(
            candidate["target_date"]
        )
        common_dates = (
            dates
            if common_dates is None
            else common_dates.intersection(
                dates
            )
        )
        prepared[name] = candidate

    if common_dates is None or common_dates.empty:
        raise ValueError(
            "Target datasets do not share common training dates."
        )

    common_dates = common_dates.sort_values()
    aligned: dict[str, pd.DataFrame] = {}

    for name, dataframe in prepared.items():
        candidate = dataframe.loc[
            dataframe["target_date"].isin(
                common_dates
            )
        ].sort_values(
            "target_date"
        ).reset_index(
            drop=True
        )

        if len(candidate) != len(common_dates):
            raise ValueError(
                f"Target dataset {name} did not align to every common date."
            )

        observed_dates = pd.DatetimeIndex(
            candidate["target_date"]
        )
        if not observed_dates.equals(
            common_dates
        ):
            raise ValueError(
                f"Target dataset {name} has misaligned target dates."
            )

        aligned[name] = candidate

    return aligned


def target_stability_statistics(
    dataframe: pd.DataFrame,
    chronological_blocks: int = 5,
) -> dict:
    if chronological_blocks < 2:
        raise ValueError(
            "At least two chronological blocks are required."
        )

    required_columns = {
        "direction",
        "threshold",
    }
    missing = required_columns - set(
        dataframe.columns
    )
    if missing:
        raise ValueError(
            "Target stability data is missing columns: "
            f"{sorted(missing)}"
        )

    if dataframe.empty:
        raise ValueError(
            "Target stability data cannot be empty."
        )

    directions = dataframe[
        "direction"
    ].astype(
        str
    )
    valid_directions = {
        "DOWN",
        "FLAT",
        "UP",
    }
    if not set(
        directions.unique()
    ).issubset(
        valid_directions
    ):
        raise ValueError(
            "Target stability data contains invalid directions."
        )

    shares = {
        label: float(
            (directions == label).mean()
        )
        for label in (
            "DOWN",
            "FLAT",
            "UP",
        )
    }

    index_blocks = np.array_split(
        np.arange(
            len(dataframe)
        ),
        chronological_blocks,
    )
    flat_shares = [
        float(
            (
                directions.iloc[
                    indices
                ]
                == "FLAT"
            ).mean()
        )
        for indices in index_blocks
        if len(indices) > 0
    ]

    threshold = pd.to_numeric(
        dataframe["threshold"],
        errors="raise",
    )

    return {
        "down_share": shares["DOWN"],
        "flat_share": shares["FLAT"],
        "up_share": shares["UP"],
        "median_flat_boundary": float(
            threshold.median()
        ),
        "median_flat_boundary_percent": float(
            threshold.median()
            * 100.0
        ),
        "flat_share_block_min": float(
            min(flat_shares)
        ),
        "flat_share_block_max": float(
            max(flat_shares)
        ),
        "flat_share_block_range": float(
            max(flat_shares)
            - min(flat_shares)
        ),
        "flat_share_by_block": flat_shares,
    }


def add_neighborhood_statistics(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "volatility_window",
        "threshold_multiplier",
        "stage1_roc_auc",
    }
    missing = required_columns - set(
        summary.columns
    )
    if missing:
        raise ValueError(
            "Screening summary is missing columns: "
            f"{sorted(missing)}"
        )

    output = summary.copy()
    output[
        "neighbor_roc_auc_mean"
    ] = np.nan
    output[
        "neighbor_roc_auc_min"
    ] = np.nan

    for index, row in output.iterrows():
        window_rows = output.loc[
            output["volatility_window"]
            == row["volatility_window"]
        ].copy()

        distance = np.abs(
            window_rows[
                "threshold_multiplier"
            ].astype(float)
            - float(
                row[
                    "threshold_multiplier"
                ]
            )
        )

        neighbors = window_rows.loc[
            distance <= 0.0250001,
            "stage1_roc_auc",
        ]

        output.loc[
            index,
            "neighbor_roc_auc_mean",
        ] = float(
            neighbors.mean()
        )
        output.loc[
            index,
            "neighbor_roc_auc_min",
        ] = float(
            neighbors.min()
        )

    return output


def select_target_shortlist(
    summary: pd.DataFrame,
    shortlist_size: int = 10,
    max_per_window: int = 2,
) -> pd.DataFrame:
    if shortlist_size <= 0:
        raise ValueError(
            "Shortlist size must be greater than zero."
        )
    if max_per_window <= 0:
        raise ValueError(
            "Maximum candidates per window must be greater than zero."
        )
    if summary.empty:
        raise ValueError(
            "Screening summary cannot be empty."
        )

    required_columns = {
        "target_name",
        "volatility_window",
        "threshold_multiplier",
        "flat_share",
        "flat_share_block_range",
        "stage1_roc_auc",
        "stage1_roc_auc_fold_std",
        "stage1_balanced_accuracy",
        "stage1_flat_f1",
    }
    missing = required_columns - set(
        summary.columns
    )
    if missing:
        raise ValueError(
            "Screening summary is missing columns: "
            f"{sorted(missing)}"
        )

    ranked = add_neighborhood_statistics(
        summary
    )

    sensible = ranked.loc[
        ranked["flat_share"].between(
            0.20,
            0.60,
            inclusive="both",
        )
    ].copy()

    if len(sensible) < shortlist_size:
        sensible = ranked.copy()

    sensible["rank_auc"] = sensible[
        "stage1_roc_auc"
    ].rank(
        ascending=False,
        method="min",
    )
    sensible["rank_neighbor_auc"] = sensible[
        "neighbor_roc_auc_mean"
    ].rank(
        ascending=False,
        method="min",
    )
    sensible["rank_balanced_accuracy"] = sensible[
        "stage1_balanced_accuracy"
    ].rank(
        ascending=False,
        method="min",
    )
    sensible["rank_flat_f1"] = sensible[
        "stage1_flat_f1"
    ].rank(
        ascending=False,
        method="min",
    )
    sensible["rank_auc_stability"] = sensible[
        "stage1_roc_auc_fold_std"
    ].rank(
        ascending=True,
        method="min",
    )
    sensible["rank_target_stability"] = sensible[
        "flat_share_block_range"
    ].rank(
        ascending=True,
        method="min",
    )

    sensible["robust_rank_score"] = (
        2.0 * sensible["rank_auc"]
        + sensible["rank_neighbor_auc"]
        + sensible[
            "rank_balanced_accuracy"
        ]
        + sensible["rank_flat_f1"]
        + sensible["rank_auc_stability"]
        + sensible[
            "rank_target_stability"
        ]
    )

    sensible = sensible.sort_values(
        [
            "robust_rank_score",
            "stage1_roc_auc",
            "stage1_roc_auc_fold_std",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    selected_indices: list[int] = []
    per_window_counts: dict[int, int] = {}

    for index, row in sensible.iterrows():
        window = int(
            row["volatility_window"]
        )
        count = per_window_counts.get(
            window,
            0,
        )
        if count >= max_per_window:
            continue

        selected_indices.append(
            int(index)
        )
        per_window_counts[
            window
        ] = count + 1

        if len(selected_indices) >= shortlist_size:
            break

    if len(selected_indices) < shortlist_size:
        for index in sensible.index:
            integer_index = int(index)
            if integer_index in selected_indices:
                continue
            selected_indices.append(
                integer_index
            )
            if len(selected_indices) >= shortlist_size:
                break

    return sensible.loc[
        selected_indices
    ].reset_index(
        drop=True
    )


def moving_block_bootstrap_auc(
    actual: np.ndarray,
    probabilities: np.ndarray,
    block_length: int = 20,
    n_resamples: int = 500,
    random_state: int = 42,
) -> dict:
    actual = np.asarray(
        actual,
        dtype=np.int64,
    ).reshape(-1)
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    ).reshape(-1)

    if len(actual) != len(probabilities):
        raise ValueError(
            "Actual labels and probabilities must have the same length."
        )
    if len(actual) < 2:
        raise ValueError(
            "Bootstrap AUC requires at least two observations."
        )
    if len(np.unique(actual)) < 2:
        raise ValueError(
            "Bootstrap AUC requires both binary classes."
        )
    if block_length <= 0:
        raise ValueError(
            "Block length must be greater than zero."
        )
    if n_resamples <= 0:
        raise ValueError(
            "Number of bootstrap resamples must be greater than zero."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Probabilities contain non-finite values."
        )

    n_rows = len(actual)
    block_length = min(
        int(block_length),
        n_rows,
    )
    random = np.random.default_rng(
        random_state
    )
    auc_values: list[float] = []

    max_start = n_rows - block_length
    blocks_needed = int(
        np.ceil(
            n_rows / block_length
        )
    )

    for _ in range(n_resamples):
        starts = random.integers(
            0,
            max_start + 1,
            size=blocks_needed,
        )
        indices = np.concatenate(
            [
                np.arange(
                    start,
                    start + block_length,
                )
                for start in starts
            ]
        )[:n_rows]

        sample_actual = actual[
            indices
        ]
        if len(np.unique(sample_actual)) < 2:
            continue

        auc_values.append(
            float(
                roc_auc_score(
                    sample_actual,
                    probabilities[
                        indices
                    ],
                )
            )
        )

    if not auc_values:
        raise RuntimeError(
            "Bootstrap did not produce any valid AUC samples."
        )

    values = np.asarray(
        auc_values,
        dtype=np.float64,
    )

    return {
        "point_estimate": float(
            roc_auc_score(
                actual,
                probabilities,
            )
        ),
        "lower_95": float(
            np.quantile(
                values,
                0.025,
            )
        ),
        "upper_95": float(
            np.quantile(
                values,
                0.975,
            )
        ),
        "bootstrap_mean": float(
            values.mean()
        ),
        "bootstrap_std": float(
            values.std(
                ddof=0
            )
        ),
        "valid_resamples": int(
            len(values)
        ),
    }
