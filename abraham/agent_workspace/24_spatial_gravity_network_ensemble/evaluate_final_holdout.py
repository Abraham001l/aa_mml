from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.runner import run_final_holdout_once

MODEL_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run_24", MODEL_DIR / "run.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main() -> None:
    print("Running final holdout evaluation for Model 24 on strictly untouched 2024-08..2026-07 holdout...")
    sub_results = run_final_holdout_once(MODEL_DIR, mod.predict_fold)

    for sub_name, res in sub_results.items():
        m = res["market_dir_overall"]
        y1 = res["market_dir_year1"]
        y2 = res["market_dir_year2"]
        met = res["market_metro_dir_overall"]
        st = res["state_us_interstate_overall"]
        ent = res["entity_total_airport_overall"]
        print(
            f"\n=== HOLDOUT [{sub_name}] ==="
            f"\n  Market Dir Overall : WAPE={m['wape']:.4f} | MAE={m['mae']:.1f} | RMSE={m['rmse']:.1f} | sMAPE={m['smape']:.4f} | RMSLE={m['rmsle']:.4f} | Bias={m['bias']:+.4f}"
            f"\n  Year 1 vs Year 2   : Y1 WAPE={y1['wape']:.4f} (Bias={y1['bias']:+.4f}) | Y2 WAPE={y2['wape']:.4f} (Bias={y2['bias']:+.4f})"
            f"\n  Segment Dom/Intl   : Dom WAPE={res['seg_domestic']['wape']:.4f} | Intl/PR WAPE={res['seg_intl_pr']['wape']:.4f}"
            f"\n  Undirected Market  : WAPE={res['market_undir_overall']['wape']:.4f}"
            f"\n  Metro Dir Market   : WAPE={met['wape']:.4f} | RMSE={met['rmse']:.1f} | sMAPE={met['smape']:.4f}"
            f"\n  US Interstate State: WAPE={st['wape']:.4f} | RMSE={st['rmse']:.1f} | sMAPE={st['smape']:.4f}"
            f"\n  Entity Total Airpt : WAPE={ent['wape']:.4f} | RMSE={ent['rmse']:.1f} | sMAPE={ent['smape']:.4f}"
        )
        print("  Market Size Tiers:")
        for k in sorted(res.keys()):
            if k.startswith("tier_"):
                t = res[k]
                print(f"    {k:24s}: WAPE={t['wape']:.4f} | MAE={t['mae']:.1f} | sMAPE={t['smape']:.4f} | Bias={t['bias']:+.4f}")

    pred_df = pd.read_parquet(MODEL_DIR / "outputs" / "final_holdout_predictions.parquet")
    monthly = pred_df.groupby("YM", as_index=False)[["y_true", "y_pred"]].sum()
    monthly["abs_err_sum"] = pred_df.assign(ae=np.abs(pred_df["y_true"] - pred_df["y_pred"])).groupby("YM")["ae"].sum().values
    monthly["wape"] = monthly["abs_err_sum"] / np.maximum(monthly["y_true"], 1.0)
    monthly["bias"] = (monthly["y_pred"] - monthly["y_true"]) / np.maximum(monthly["y_true"], 1.0)
    monthly.to_csv(MODEL_DIR / "outputs" / "final_holdout_monthly_summary.csv", index=False)
    print("\n=== Monthly Holdout Trajectory ===")
    print(monthly.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
