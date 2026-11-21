from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from shared_eval.data_loader import build_balanced_panel, load_airport_info, load_raw_pax
from shared_eval.metrics import evaluate_predictions, summarize_fold_results
from shared_eval.splits import BACKTEST_FOLDS, FINAL_TEST_FOLD, TEST_SUBWINDOWS, FoldSpec


def run_backtest_suite(
    model_dir: Path,
    predict_fold_fn: Callable[[FoldSpec, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path], pd.DataFrame],
    folds: List[FoldSpec] = BACKTEST_FOLDS,
) -> Dict[str, dict]:
    """
    Execute rolling-origin backtests strictly on YM <= 202407 and save standardized outputs.

    predict_fold_fn signature:
        (fold, hist_panel, tgt_skeleton_no_target, airport_df, model_dir) -> pred_df[ORG_AIRPRT, DES_AIRPRT, YM, HORIZON, y_pred]
    """
    raw_df = load_raw_pax()
    # Hard pre-test guard so backtesting NEVER reads test-window rows
    raw_pretest = raw_df.loc[raw_df["YM"] <= 202407].copy()
    airport_df = load_airport_info()

    fold_metrics: Dict[str, Dict[str, dict]] = {}
    all_oof_preds = []

    outputs_dir = model_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        hist_panel, tgt_panel = build_balanced_panel(
            raw_pretest,
            origin_ym=fold.origin_ym,
            target_end_ym=fold.target_end_ym,
            include_targets=True,
        )
        # Strip ground truth before handing skeleton to model!
        tgt_skeleton = tgt_panel.drop(columns=["y_true", "IS_OBSERVED"]).copy()

        # Pass defensive copy of hist_panel so model cannot mutate shared frames
        pred_only = predict_fold_fn(fold, hist_panel.copy(), tgt_skeleton, airport_df.copy(), model_dir)

        merged = tgt_panel.merge(
            pred_only[["ORG_AIRPRT", "DES_AIRPRT", "YM", "y_pred"]],
            on=["ORG_AIRPRT", "DES_AIRPRT", "YM"],
            how="left",
            validate="one_to_one",
        )
        if merged["y_pred"].isna().any():
            raise ValueError(f"Missing predictions in {fold.name}: {merged['y_pred'].isna().sum()} NaNs")

        fold_res = evaluate_predictions(merged, hist_panel, airport_df, fold.origin_ym)
        fold_metrics[fold.name] = fold_res

        merged_save = merged[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_true", "y_pred"]].copy()
        merged_save["fold"] = fold.name
        all_oof_preds.append(merged_save)

        print(
            f"[{model_dir.name} | {fold.name}] "
            f"Market WAPE={fold_res['market_dir_overall']['wape']:.4f} "
            f"(Y1={fold_res['market_dir_year1']['wape']:.4f}, Y2={fold_res['market_dir_year2']['wape']:.4f}) | "
            f"RMSE={fold_res['market_dir_overall']['rmse']:.1f} | "
            f"Bias={fold_res['market_dir_overall']['bias']:+.4f} | "
            f"Metro WAPE={fold_res['market_metro_dir_overall']['wape']:.4f} | "
            f"State WAPE={fold_res['state_us_interstate_overall']['wape']:.4f} | "
            f"Entity WAPE={fold_res['entity_total_airport_overall']['wape']:.4f}"
        )

    summary = summarize_fold_results(fold_metrics)
    oof_df = pd.concat(all_oof_preds, ignore_index=True)
    oof_df.to_parquet(outputs_dir / "backtest_predictions.parquet", index=False)

    payload = {
        "model": model_dir.name,
        "folds": fold_metrics,
        "summary": summary,
    }
    with open(outputs_dir / "backtest_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        f"==> [{model_dir.name} SUMMARY 3-Fold Mean] "
        f"Market WAPE={summary['market_dir_overall']['wape_mean']:.4f} +/- {summary['market_dir_overall']['wape_std']:.4f} | "
        f"RMSE={summary['market_dir_overall']['rmse_mean']:.1f} | "
        f"sMAPE={summary['market_dir_overall']['smape_mean']:.4f} | "
        f"RMSLE={summary['market_dir_overall']['rmsle_mean']:.4f} | "
        f"Bias={summary['market_dir_overall']['bias_mean']:+.4f} | "
        f"Steady(F3) WAPE={fold_metrics['fold_3_steady']['market_dir_overall']['wape']:.4f}"
    )
    return payload


def run_final_holdout_once(
    model_dir: Path,
    predict_fold_fn: Callable[[FoldSpec, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path], pd.DataFrame],
) -> Dict[str, dict]:
    """
    Evaluate the single winning model ONCE on the untouched 2024-08..2026-07 holdout,
    including clean pre-anomaly sub-windows.
    """
    raw_df = load_raw_pax()
    airport_df = load_airport_info()
    fold = FINAL_TEST_FOLD

    hist_panel, tgt_panel = build_balanced_panel(
        raw_df,
        origin_ym=fold.origin_ym,
        target_end_ym=fold.target_end_ym,
        include_targets=True,
    )
    tgt_skeleton = tgt_panel.drop(columns=["y_true", "IS_OBSERVED"]).copy()
    pred_only = predict_fold_fn(fold, hist_panel.copy(), tgt_skeleton, airport_df.copy(), model_dir)

    merged = tgt_panel.merge(
        pred_only[["ORG_AIRPRT", "DES_AIRPRT", "YM", "y_pred"]],
        on=["ORG_AIRPRT", "DES_AIRPRT", "YM"],
        how="left",
        validate="one_to_one",
    )
    if merged["y_pred"].isna().any():
        raise ValueError(f"Missing holdout predictions: {merged['y_pred'].isna().sum()} NaNs")

    sub_results = {}
    for sub_name, (ym_start, ym_end) in TEST_SUBWINDOWS.items():
        sub_df = merged.loc[(merged["YM"] >= ym_start) & (merged["YM"] <= ym_end)].copy()
        sub_results[sub_name] = evaluate_predictions(sub_df, hist_panel, airport_df, fold.origin_ym)

    outputs_dir = model_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(outputs_dir / "final_holdout_predictions.parquet", index=False)
    with open(outputs_dir / "final_holdout_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"model": model_dir.name, "subwindows": sub_results}, f, indent=2)

    return sub_results
