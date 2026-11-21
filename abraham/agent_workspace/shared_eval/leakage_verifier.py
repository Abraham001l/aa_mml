from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.data_loader import build_balanced_panel, load_airport_info, load_raw_pax
from shared_eval.feature_engine import FoldFeatureStore, assemble_training_dataset
from shared_eval.splits import BACKTEST_FOLDS


def verify_model_leakage(model_dir: Path, atol: float = 1e-10) -> None:
    """Memory-lean, ironclad future-poisoning invariance verifier for any model folder."""
    spec = importlib.util.spec_from_file_location(f"mod_{model_dir.name}", model_dir / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    predict_fold = mod.predict_fold

    raw_df = load_raw_pax()
    ai = load_airport_info()

    # 1. Check exact feature & target matrix invariance across ALL folds
    for fold in BACKTEST_FOLDS:
        hist_c, _ = build_balanced_panel(raw_df, fold.origin_ym, fold.target_end_ym, include_targets=True)
        sc = FoldFeatureStore(hist_c, ai)
        Xc, yc, wc, _ = assemble_training_dataset(sc, s_min=6, stride_s=8)
        xc_arr = Xc.fillna(-999.0).to_numpy(dtype=np.float32)
        del hist_c, sc, Xc
        gc.collect()

        raw_p = raw_df.copy()
        raw_p.loc[raw_p["YM"] > fold.origin_ym, "OD_PAX"] = 9.0e9
        raw_p.loc[raw_p["YM"] > fold.origin_ym, "OD_REV_USD"] = 9.0e11
        hist_p, _ = build_balanced_panel(raw_p, fold.origin_ym, fold.target_end_ym, include_targets=True)
        sp = FoldFeatureStore(hist_p, ai)
        Xp, yp, wp, _ = assemble_training_dataset(sp, s_min=6, stride_s=8)
        xp_arr = Xp.fillna(-999.0).to_numpy(dtype=np.float32)
        del raw_p, hist_p, sp, Xp
        gc.collect()

        assert np.allclose(yc, yp, rtol=0, atol=1e-12), f"Target leak in {fold.name}"
        assert np.allclose(wc, wp, rtol=0, atol=1e-12), f"Weight leak in {fold.name}"
        assert np.allclose(xc_arr, xp_arr, rtol=0, atol=1e-12), f"Feature leak in {fold.name}"

    # 2. Check full end-to-end model fit + predict invariance on Fold 3 (sequentially to save RAM)
    f3 = BACKTEST_FOLDS[-1]
    hist_c, tgt_c = build_balanced_panel(raw_df, f3.origin_ym, f3.target_end_ym, include_targets=True)
    skel_c = tgt_c.drop(columns=["y_true", "IS_OBSERVED"]).copy()
    pred_clean = predict_fold(f3, hist_c, skel_c, ai, model_dir)["y_pred"].to_numpy(dtype=np.float64)
    del hist_c, tgt_c, skel_c
    gc.collect()

    raw_p = raw_df.copy()
    raw_p.loc[raw_p["YM"] > f3.origin_ym, "OD_PAX"] = 9.0e9
    raw_p.loc[raw_p["YM"] > f3.origin_ym, "OD_REV_USD"] = 9.0e11
    hist_p, tgt_p = build_balanced_panel(raw_p, f3.origin_ym, f3.target_end_ym, include_targets=True)
    skel_p = tgt_p.drop(columns=["y_true", "IS_OBSERVED"]).copy()
    pred_poison = predict_fold(f3, hist_p, skel_p, ai, model_dir)["y_pred"].to_numpy(dtype=np.float64)
    del raw_p, hist_p, tgt_p, skel_p
    gc.collect()

    max_diff = float(np.max(np.abs(pred_clean - pred_poison)))
    assert np.allclose(pred_clean, pred_poison, rtol=0.0, atol=atol), (
        f"End-to-end leakage in {model_dir.name}: max_diff={max_diff}"
    )
    print(f"PASS: {model_dir.name} passed all-fold feature & Fold-3 end-to-end invariance (max_diff={max_diff:.2e}).")


def verify_external_model_leakage(model_dir: Path, include_macro: bool = False, atol: float = 1e-10) -> None:
    """Verify invariance to poisoning BOTH future pax_forecasting AND future external data rows (AVAILABLE_YM > origin_ym)."""
    import pandas as pd
    from shared_eval.external_feature_store import EXT_DIR, ExternalFoldFeatureStore

    spec = importlib.util.spec_from_file_location(f"mod_{model_dir.name}", model_dir / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    predict_fold = mod.predict_fold

    raw_df = load_raw_pax()
    ai = load_airport_info()

    # Check external point-in-time cutoff on ALL folds via ExternalFoldFeatureStore
    ap_ann_orig = pd.read_parquet(EXT_DIR / "airport_external_annual.parquet")
    macro_orig = pd.read_parquet(EXT_DIR / "macro_monthly_clean.parquet")
    try:
        for fold in BACKTEST_FOLDS:
            hist_c, _ = build_balanced_panel(raw_df, fold.origin_ym, fold.target_end_ym, include_targets=True)
            sc = ExternalFoldFeatureStore(hist_c, ai, include_macro=include_macro)
            Xc, yc, wc, _ = assemble_training_dataset(sc, s_min=6, stride_s=8)
            xc_arr = Xc.fillna(-999.0).to_numpy(dtype=np.float32)

            # Poison future external tables (> fold.origin_ym)
            ap_p = ap_ann_orig.copy()
            ap_p.loc[ap_p["AVAILABLE_YM"] > fold.origin_ym, ["CBSA_POP", "CBSA_POP_YOY", "CBSA_DOM_MIG_RATE", "COUNTY_POP_DENSITY", "STATE_DOM_MIG_RATE"]] = 9.0e9
            ap_p.loc[ap_p["FAA_AVAILABLE_YM"] > fold.origin_ym, "FAA_ENPLANEMENTS"] = 9.0e9
            ap_p.to_parquet(EXT_DIR / "airport_external_annual.parquet", index=False)

            mac_p = macro_orig.copy()
            mac_p.loc[mac_p["AVAILABLE_YM"] > fold.origin_ym, ["UNRATE", "CPIAUCSL_YOY", "DSPIC96_YOY", "DCOILWTICO", "UMCSENT"]] = 9.0e5
            mac_p.to_parquet(EXT_DIR / "macro_monthly_clean.parquet", index=False)

            raw_p = raw_df.copy()
            raw_p.loc[raw_p["YM"] > fold.origin_ym, "OD_PAX"] = 9.0e9
            raw_p.loc[raw_p["YM"] > fold.origin_ym, "OD_REV_USD"] = 9.0e11
            hist_p, _ = build_balanced_panel(raw_p, fold.origin_ym, fold.target_end_ym, include_targets=True)
            sp = ExternalFoldFeatureStore(hist_p, ai, include_macro=include_macro)
            Xp, yp, wp, _ = assemble_training_dataset(sp, s_min=6, stride_s=8)
            xp_arr = Xp.fillna(-999.0).to_numpy(dtype=np.float32)

            # Restore clean external tables immediately
            ap_ann_orig.to_parquet(EXT_DIR / "airport_external_annual.parquet", index=False)
            macro_orig.to_parquet(EXT_DIR / "macro_monthly_clean.parquet", index=False)

            assert np.allclose(yc, yp, rtol=0, atol=1e-12)
            assert np.allclose(wc, wp, rtol=0, atol=1e-12)
            assert np.allclose(xc_arr, xp_arr, rtol=0, atol=1e-12), f"External feature leak in {fold.name}"
    finally:
        ap_ann_orig.to_parquet(EXT_DIR / "airport_external_annual.parquet", index=False)
        macro_orig.to_parquet(EXT_DIR / "macro_monthly_clean.parquet", index=False)

    # Full end-to-end Fold 3 verification under BOTH raw pax + external poisoning
    f3 = BACKTEST_FOLDS[-1]
    hist_c, tgt_c = build_balanced_panel(raw_df, f3.origin_ym, f3.target_end_ym, include_targets=True)
    skel_c = tgt_c.drop(columns=["y_true", "IS_OBSERVED"]).copy()
    pred_clean = predict_fold(f3, hist_c, skel_c, ai, model_dir)["y_pred"].to_numpy(dtype=np.float64)

    try:
        ap_p = ap_ann_orig.copy()
        ap_p.loc[ap_p["AVAILABLE_YM"] > f3.origin_ym, ["CBSA_POP", "CBSA_POP_YOY", "CBSA_DOM_MIG_RATE"]] = 9.0e9
        ap_p.loc[ap_p["FAA_AVAILABLE_YM"] > f3.origin_ym, "FAA_ENPLANEMENTS"] = 9.0e9
        ap_p.to_parquet(EXT_DIR / "airport_external_annual.parquet", index=False)
        mac_p = macro_orig.copy()
        mac_p.loc[mac_p["AVAILABLE_YM"] > f3.origin_ym, ["UNRATE", "DCOILWTICO", "UMCSENT"]] = 9.0e5
        mac_p.to_parquet(EXT_DIR / "macro_monthly_clean.parquet", index=False)

        raw_p = raw_df.copy()
        raw_p.loc[raw_p["YM"] > f3.origin_ym, "OD_PAX"] = 9.0e9
        raw_p.loc[raw_p["YM"] > f3.origin_ym, "OD_REV_USD"] = 9.0e11
        hist_p, tgt_p = build_balanced_panel(raw_p, f3.origin_ym, f3.target_end_ym, include_targets=True)
        skel_p = tgt_p.drop(columns=["y_true", "IS_OBSERVED"]).copy()
        pred_poison = predict_fold(f3, hist_p, skel_p, ai, model_dir)["y_pred"].to_numpy(dtype=np.float64)
    finally:
        ap_ann_orig.to_parquet(EXT_DIR / "airport_external_annual.parquet", index=False)
        macro_orig.to_parquet(EXT_DIR / "macro_monthly_clean.parquet", index=False)

    max_diff = float(np.max(np.abs(pred_clean - pred_poison)))
    assert np.allclose(pred_clean, pred_poison, rtol=0.0, atol=atol)
    print(f"PASS: {model_dir.name} passed external+internal future-poisoning invariance (max_diff={max_diff:.2e}).")
