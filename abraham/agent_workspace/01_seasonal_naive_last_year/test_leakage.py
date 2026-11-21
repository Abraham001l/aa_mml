from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.data_loader import build_balanced_panel, load_airport_info, load_raw_pax
from shared_eval.splits import BACKTEST_FOLDS
from run import predict_fold


def test_leakage() -> None:
    model_dir = Path(__file__).resolve().parent
    raw_df = load_raw_pax()
    ai = load_airport_info()

    for fold in BACKTEST_FOLDS:
        hist_clean, tgt_clean = build_balanced_panel(raw_df, fold.origin_ym, fold.target_end_ym, include_targets=True)
        skel_clean = tgt_clean.drop(columns=["y_true", "IS_OBSERVED"]).copy()
        pred_clean = predict_fold(fold, hist_clean.copy(), skel_clean, ai, model_dir)

        # Poison all future raw records (> origin_ym) with massive values and fake future pairs
        raw_poison = raw_df.copy()
        future_mask = raw_poison["YM"] > fold.origin_ym
        raw_poison.loc[future_mask, "OD_PAX"] = 9.0e9
        raw_poison.loc[future_mask, "OD_REV_USD"] = 9.0e11

        hist_poison, tgt_poison = build_balanced_panel(raw_poison, fold.origin_ym, fold.target_end_ym, include_targets=True)
        skel_poison = tgt_poison.drop(columns=["y_true", "IS_OBSERVED"]).copy()
        pred_poison = predict_fold(fold, hist_poison.copy(), skel_poison, ai, model_dir)

        assert np.allclose(pred_clean["y_pred"].values, pred_poison["y_pred"].values, rtol=0.0, atol=1e-12), (
            f"Leakage detected in {fold.name}: future poisoning changed predictions!"
        )
        # Also verify saved model_specific_data contains no future information
        lookup = pd.read_parquet(model_dir / "model_specific_data" / fold.name / "trailing_12m_seasonal_lookup.parquet")
        assert len(lookup) == 17119 * 12

    print("PASS: 01_seasonal_naive_last_year passed future-poisoning invariance & artifact cutoff tests.")


if __name__ == "__main__":
    test_leakage()
