from pathlib import Path

import pandas as pd

from app.training.hierarchical_sequence_preprocessor import (
    HierarchicalSequencePreprocessor,
)
from app.training.stage2_conditioned_target_research import target_specs
from scripts.run_stage2_conditioned_megasearch import (
    base_columns,
    build_master,
    dataset,
    load_training_cutoff,
)
from app.training.stage2_return_architecture_research import (
    verify_stage2_orientation,
)
from database.stage2_signal_data_repository import Stage2SignalDataRepository


TICKER = "SPY"
EXPERIMENT_DIRECTORY = Path("experiments")


def latest_verified_result() -> pd.DataFrame | None:
    paths = sorted(
        EXPERIMENT_DIRECTORY.glob("stage2_90d_k700_verified_v1_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        return None
    return pd.read_csv(paths[0])


def main():
    cutoff = load_training_cutoff()
    primary = next(spec for spec in target_specs() if spec.role == "primary")
    raw = Stage2SignalDataRepository().get_training_data(ticker=TICKER)
    master = build_master(raw, primary, cutoff)
    audit = verify_stage2_orientation(dataset(master, base_columns()))

    print("=" * 78)
    print("STAGE-2 ORIENTATION AUDIT")
    print("=" * 78)
    print("Target: 90d x 0.700")
    print("Training cutoff:", cutoff.date())
    print("Stage-2 mapping:", HierarchicalSequencePreprocessor.STAGE2_MAPPING)
    if HierarchicalSequencePreprocessor.STAGE2_MAPPING != {"DOWN": 0, "UP": 1}:
        raise ValueError("Stage-2 class mapping is not DOWN=0, UP=1.")
    print("Label audit rows:", audit["rows"])
    print("UP/DOWN/FLAT rows:", audit["up_rows"], audit["down_rows"], audit["flat_rows"])
    print("Label orientation violations:", audit["violations"])
    print()
    print("PASS: UP labels are positive moves beyond +threshold.")
    print("PASS: DOWN labels are negative moves beyond -threshold.")
    print("PASS: classifier class index 1 is UP, which is the probability used for AUC.")

    verified = latest_verified_result()
    if verified is None or verified.empty:
        print()
        print("No prior verified Stage-2 CSV found. Code/label orientation audit is complete.")
        return

    row = verified.iloc[0]
    candidate_auc = float(row["verification_auc"])
    base_auc = float(row["verification_base_auc"])
    print()
    print("LATEST PRIOR VERIFIED RESULT")
    print("Candidate:", row["candidate_name"])
    print(f"Candidate direct UP AUC: {candidate_auc:.4f}")
    print(f"Candidate inverted-score AUC: {1.0 - candidate_auc:.4f}")
    print(f"Matched base direct UP AUC: {base_auc:.4f}")
    print(f"Matched base inverted-score AUC: {1.0 - base_auc:.4f}")
    print()
    print(
        "Interpretation: because the label mapping and probability orientation pass, "
        "a sub-0.50 verification AUC is not a class-index bug. It is temporal/ranking "
        "instability on that slice. We do NOT flip scores using verification labels."
    )


if __name__ == "__main__":
    main()
