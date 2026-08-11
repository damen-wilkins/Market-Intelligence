from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.training.direction_feature_builder import DirectionFeatureBuilder
from app.training.flat_target_sensitivity_analyzer import (
    FlatTargetSensitivityAnalyzer,
)
from app.training.trend_signal_analyzer import TrendSignalAnalyzer
from database.direction_training_data_repository import (
    DirectionTrainingDataRepository,
)


TICKER = "SPY"
EXPERIMENT_DIRECTORY = Path("experiments")


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

    base_builder = DirectionFeatureBuilder(
        feature_scope="base"
    )

    base_features = base_builder.build(
        raw_data
    )

    trend_builder = DirectionFeatureBuilder(
        feature_scope="base_trend"
    )

    trend_features = trend_builder.build(
        raw_data
    )

    print(
        f"Raw rows: {len(raw_data)}"
    )
    print(
        f"Base feature rows: {len(base_features)}"
    )
    print(
        f"Base + trend feature rows: {len(trend_features)}"
    )

    flat_results = (
        FlatTargetSensitivityAnalyzer()
        .analyze(
            data=raw_data,
            eligible_feature_dates=(
                base_features[
                    "trade_date"
                ]
            ),
        )
    )

    trend_results = (
        TrendSignalAnalyzer()
        .analyze(
            feature_data=trend_features,
            market_data=raw_data,
        )
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    EXPERIMENT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    flat_path = (
        EXPERIMENT_DIRECTORY
        / f"flat_target_sensitivity_{timestamp}.csv"
    )

    alignment_path = (
        EXPERIMENT_DIRECTORY
        / f"trend_alignment_analysis_{timestamp}.csv"
    )

    adx_path = (
        EXPERIMENT_DIRECTORY
        / f"trend_adx_analysis_{timestamp}.csv"
    )

    compression_path = (
        EXPERIMENT_DIRECTORY
        / f"trend_compression_analysis_{timestamp}.csv"
    )

    stability_path = (
        EXPERIMENT_DIRECTORY
        / f"trend_alignment_stability_{timestamp}.csv"
    )

    flat_results.to_csv(
        flat_path,
        index=False,
    )

    trend_results[
        "alignment"
    ].to_csv(
        alignment_path,
        index=False,
    )

    trend_results[
        "adx"
    ].to_csv(
        adx_path,
        index=False,
    )

    trend_results[
        "compression"
    ].to_csv(
        compression_path,
        index=False,
    )

    trend_results[
        "stability"
    ].to_csv(
        stability_path,
        index=False,
    )

    print_flat_summary(
        flat_results
    )

    print_trend_summary(
        trend_results
    )

    print()
    print(
        "Saved:"
    )
    print(
        flat_path
    )
    print(
        alignment_path
    )
    print(
        adx_path
    )
    print(
        compression_path
    )
    print(
        stability_path
    )

    print()
    print(
        "No validation or held-out test labels were used "
        "to select a model in this analysis."
    )


def print_flat_summary(
    flat_results: pd.DataFrame,
) -> None:
    print()
    print(
        "============================================================"
    )
    print(
        "FLAT TARGET SENSITIVITY - TRAINING PERIOD ONLY"
    )
    print(
        "============================================================"
    )

    current = flat_results.loc[
        (
            flat_results[
                "volatility_window"
            ]
            == 20
        )
        & (
            flat_results[
                "threshold_multiplier"
            ]
            == 0.15
        )
    ]

    research_reference = flat_results.loc[
        (
            flat_results[
                "volatility_window"
            ]
            == 20
        )
        & (
            flat_results[
                "threshold_multiplier"
            ]
            == 0.50
        )
    ]

    if not current.empty:
        print_candidate(
            "Current target (20d x 0.15)",
            current.iloc[0],
        )

    if not research_reference.empty:
        print_candidate(
            "Research reference (20d x 0.50)",
            research_reference.iloc[0],
        )

    print()
    print(
        "Candidates closest to useful FLAT prevalence levels:"
    )

    target_flat_shares = (
        0.20,
        0.30,
        0.40,
    )

    displayed = set()

    for target_share in target_flat_shares:
        working = flat_results.copy()
        working[
            "distance"
        ] = (
            working[
                "flat_share"
            ]
            - target_share
        ).abs()

        row = (
            working
            .sort_values(
                [
                    "distance",
                    "flat_share_block_std",
                    "up_down_share_gap",
                ]
            )
            .iloc[0]
        )

        key = (
            int(
                row[
                    "volatility_window"
                ]
            ),
            float(
                row[
                    "threshold_multiplier"
                ]
            ),
        )

        if key in displayed:
            continue

        displayed.add(
            key
        )

        print_candidate(
            f"Closest to {target_share:.0%} FLAT",
            row,
        )


def print_candidate(
    label: str,
    row: pd.Series,
) -> None:
    print()
    print(
        label
    )
    print(
        "  window / k:",
        f"{int(row['volatility_window'])} / "
        f"{float(row['threshold_multiplier']):.2f}",
    )
    print(
        "  DOWN / FLAT / UP:",
        f"{row['down_share']:.1%} / "
        f"{row['flat_share']:.1%} / "
        f"{row['up_share']:.1%}",
    )
    print(
        "  median flat boundary:",
        f"+/-{row['median_threshold_pct']:.3f}%",
    )
    print(
        "  FLAT share across time blocks:",
        f"{row['flat_share_block_min']:.1%} -> "
        f"{row['flat_share_block_max']:.1%}",
    )


def print_trend_summary(
    trend_results: dict[str, pd.DataFrame],
) -> None:
    print()
    print(
        "============================================================"
    )
    print(
        "TREND-STATE SIGNAL CHECK - TRAINING PERIOD ONLY"
    )
    print(
        "============================================================"
    )

    alignment = trend_results[
        "alignment"
    ]

    for _, row in alignment.iterrows():
        print(
            f"{row['alignment']}: "
            f"share={row['share']:.1%}, "
            f"next-day UP={row['next_day_up_rate']:.1%}, "
            f"UP lift={row['up_rate_lift_vs_overall']:+.1%}, "
            f"mean next return={row['mean_next_return_pct']:+.4f}%"
        )

    print()
    print(
        "ADX terciles:"
    )

    for _, row in trend_results[
        "adx"
    ].iterrows():
        print(
            f"  {row['bucket']}: "
            f"mean ADX={row['feature_mean']:.2f}, "
            f"normalized next-day move="
            f"{row['mean_normalized_abs_move']:.3f}"
        )

    stability = trend_results[
        "stability"
    ]

    print()
    print(
        "MA alignment directional lift by chronological block:"
    )

    pivot = stability.pivot(
        index="block",
        columns="alignment",
        values="directional_lift",
    )

    print(
        pivot.round(
            4
        ).to_string()
    )


if __name__ == "__main__":
    main()
