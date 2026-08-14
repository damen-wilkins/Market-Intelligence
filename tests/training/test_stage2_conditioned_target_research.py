import numpy as np

from app.training.stage2_conditioned_target_research import (
    Stage2FeatureCandidate,
    expand_beam,
    moving_block_bootstrap_auc_delta,
    select_finalists,
    target_specs,
)


def test_primary_target_is_90d_k700():
    primary = [spec for spec in target_specs() if spec.role == "primary"]
    assert len(primary) == 1
    assert primary[0].volatility_window == 90
    assert primary[0].threshold_multiplier == 0.700


def test_robustness_targets_are_present():
    names = {spec.name for spec in target_specs()}
    assert {
        "neighbor_90d_k675",
        "neighbor_90d_k725",
        "runnerup_30d_k725",
        "runnerup_40d_k725",
    }.issubset(names)


def test_expand_beam_deduplicates_candidates():
    candidates = expand_beam(
        beam=[("breadth", "calendar"), ("calendar", "breadth")],
        all_group_names=["breadth", "calendar", "rates_credit"],
        depth=3,
    )
    assert candidates == [
        Stage2FeatureCandidate(("breadth", "calendar", "rates_credit"))
    ]


def test_select_finalists_filters_unstable_rows():
    rows = [
        {
            "status": "ok",
            "candidate_name": "a",
            "groups": ["a"],
            "delta_roc_auc_vs_matched_base": 0.03,
            "stage2_roc_auc": 0.55,
            "stage2_roc_auc_fold_std": 0.02,
            "training_rows": 2500,
        },
        {
            "status": "ok",
            "candidate_name": "b",
            "groups": ["b"],
            "delta_roc_auc_vs_matched_base": 0.04,
            "stage2_roc_auc": 0.54,
            "stage2_roc_auc_fold_std": 0.01,
            "training_rows": 2500,
        },
        {
            "status": "ok",
            "candidate_name": "unstable",
            "groups": ["c"],
            "delta_roc_auc_vs_matched_base": 0.20,
            "stage2_roc_auc": 0.70,
            "stage2_roc_auc_fold_std": 0.20,
            "training_rows": 2500,
        },
    ]
    finalists = select_finalists(rows, 2, 2000, 0.08)
    assert [row["candidate_name"] for row in finalists] == ["b", "a"]


def test_bootstrap_delta_detects_clear_improvement():
    actual = np.array([0, 1] * 80, dtype=np.int64)
    candidate = np.where(actual == 1, 0.8, 0.2).astype(np.float64)
    baseline = np.linspace(0.2, 0.8, len(actual))
    result = moving_block_bootstrap_auc_delta(
        actual,
        candidate,
        baseline,
        resamples=100,
        block_length=12,
        random_state=8,
    )
    assert result["delta_auc"] > 0.2
    assert result["probability_delta_positive"] > 0.95
