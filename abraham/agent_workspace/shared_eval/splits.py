from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class FoldSpec:
    name: str
    origin_ym: int
    target_start_ym: int
    target_end_ym: int
    regime: str


BACKTEST_FOLDS: List[FoldSpec] = [
    FoldSpec(
        name="fold_1_recovery",
        origin_ym=202107,
        target_start_ym=202108,
        target_end_ym=202307,
        regime="covid_recovery",
    ),
    FoldSpec(
        name="fold_2_transition",
        origin_ym=202201,
        target_start_ym=202202,
        target_end_ym=202401,
        regime="transition",
    ),
    FoldSpec(
        name="fold_3_steady",
        origin_ym=202207,
        target_start_ym=202208,
        target_end_ym=202407,
        regime="steady_state",
    ),
]

FINAL_TEST_FOLD = FoldSpec(
    name="final_test_holdout",
    origin_ym=202407,
    target_start_ym=202408,
    target_end_ym=202607,
    regime="untouched_holdout",
)

# Clean sub-windows inside FINAL_TEST_FOLD before May-June 2026 anomalies
TEST_SUBWINDOWS: Dict[str, tuple[int, int]] = {
    "full_24m": (202408, 202607),
    "clean_pre_anomaly_21m": (202408, 202604),
    "clean_through_2025_17m": (202408, 202512),
}


def assert_origin_cutoff(max_feature_ym: int, origin_ym: int) -> None:
    """Assert that no feature or training record crosses origin_ym."""
    if max_feature_ym > origin_ym:
        raise AssertionError(
            f"DATA LEAKAGE DETECTED: max_feature_ym={max_feature_ym} > origin_ym={origin_ym}"
        )
