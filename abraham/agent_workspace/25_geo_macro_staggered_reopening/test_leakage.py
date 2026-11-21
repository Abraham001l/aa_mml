from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.data_loader import build_balanced_panel, load_airport_info, load_raw_pax
from shared_eval.splits import BACKTEST_FOLDS

MODEL_DIR = Path(__file__).resolve().parent


def main() -> None:
    spec = importlib.util.spec_from_file_location("run_25", MODEL_DIR / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    raw_df = load_raw_pax()
    ai = load_airport_info()

    print("Verifying adversarial future-poisoning leakage for Model 25 across all folds...")

    # 1. Verify compute_geo_macro_persistent_recovery invariance across ALL 3 folds
    for fold in BACKTEST_FOLDS:
        hist_c, tgt_c = build_balanced_panel(raw_df, fold.origin_ym, fold.target_end_ym, include_targets=True)
        skel_c = tgt_c.drop(columns=["y_true", "IS_OBSERVED"]).copy()
        pred_c = mod.compute_geo_macro_persistent_recovery(fold, hist_c, skel_c, ai)

        raw_p = raw_df.copy()
        raw_p.loc[raw_p["YM"] > fold.origin_ym, "OD_PAX"] = 9.0e9
        raw_p.loc[raw_p["YM"] > fold.origin_ym, "OD_REV_USD"] = 9.0e11
        hist_p, tgt_p = build_balanced_panel(raw_p, fold.origin_ym, fold.target_end_ym, include_targets=True)
        skel_p = tgt_p.drop(columns=["y_true", "IS_OBSERVED"]).copy()
        pred_p = mod.compute_geo_macro_persistent_recovery(fold, hist_p, skel_p, ai)

        diff = float(np.max(np.abs(pred_c["y_pred"].to_numpy() - pred_p["y_pred"].to_numpy())))
        assert diff == 0.0, f"Future poisoning leak in {fold.name}: {diff}"
        print(f"PASS: {fold.name} compute_geo_macro_persistent_recovery invariance verified (diff={diff:.2e}).")

    # 2. Verify origin-cutoff enforcement
    f3 = BACKTEST_FOLDS[-1]
    hist_leak = raw_df.loc[raw_df["YM"] <= f3.origin_ym + 1].copy()
    try:
        mod.assert_origin_cutoff(int(hist_leak["YM"].max()), f3.origin_ym)
        raise AssertionError("Failed to detect origin-cutoff boundary violation!")
    except AssertionError:
        print("PASS: Strict origin-cutoff assert guard verified.")

    print("ALL LEAKAGE CHECKS PASSED FOR MODEL 25.")


if __name__ == "__main__":
    main()
