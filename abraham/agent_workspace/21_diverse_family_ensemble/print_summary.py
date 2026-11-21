from __future__ import annotations

import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
d = json.load(open(MODEL_DIR / "outputs" / "backtest_metrics.json"))
s, f3 = d["summary"], d["folds"]["fold_3_steady"]
print("=== 21_diverse_family_ensemble Summary ===")
print(
    f"Mean WAPE={s['market_dir_overall']['wape_mean']:.4f} +/- {s['market_dir_overall']['wape_std']:.4f} | "
    f"F3 WAPE={f3['market_dir_overall']['wape']:.4f} | "
    f"RMSE={s['market_dir_overall']['rmse_mean']:.1f} | "
    f"sMAPE={s['market_dir_overall']['smape_mean']:.4f} | "
    f"RMSLE={s['market_dir_overall']['rmsle_mean']:.4f} | "
    f"Bias={s['market_dir_overall']['bias_mean']:+.4f}"
)
print(
    f"Metro Mean/F3={s['market_metro_dir_overall']['wape_mean']:.4f}/{f3['market_metro_dir_overall']['wape']:.4f} | "
    f"State Mean/F3={s['state_us_interstate_overall']['wape_mean']:.4f}/{f3['state_us_interstate_overall']['wape']:.4f} | "
    f"Entity Mean/F3={s['entity_total_airport_overall']['wape_mean']:.4f}/{f3['entity_total_airport_overall']['wape']:.4f}"
)
print(
    f"Undir Mean/F3={s['market_undir_overall']['wape_mean']:.4f}/{f3['market_undir_overall']['wape']:.4f} | "
    f"Y1/Y2 Mean={s['market_dir_year1']['wape_mean']:.4f}/{s['market_dir_year2']['wape_mean']:.4f} | "
    f"Y1/Y2 F3={f3['market_dir_year1']['wape']:.4f}/{f3['market_dir_year2']['wape']:.4f}"
)
print(
    f"Dom/Intl Mean={s['seg_domestic']['wape_mean']:.4f}/{s['seg_intl_pr']['wape_mean']:.4f} | "
    f"Dom/Intl F3={f3['seg_domestic']['wape']:.4f}/{f3['seg_intl_pr']['wape']:.4f}"
)
for k in sorted(s.keys()):
    if k.startswith("tier_"):
        print(f"  {k:24s}: Mean={s[k]['wape_mean']:.4f}, F3={f3[k]['wape']:.4f}, sMAPE_F3={f3[k]['smape']:.4f}, Bias_F3={f3[k]['bias']:+.4f}")
