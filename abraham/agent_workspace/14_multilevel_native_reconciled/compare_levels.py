from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_eval.data_loader import load_airport_info
from shared_eval.metrics import compute_metrics


def main() -> None:
    model_dir = Path(__file__).resolve().parent
    oof = pd.read_parquet(model_dir / "outputs" / "backtest_predictions.parquet")
    ai = load_airport_info().set_index("AIRPRT_CD")
    oof["ORG_REG"] = oof["ORG_AIRPRT"].map(ai["REGION_CD"])
    oof["DES_REG"] = oof["DES_AIRPRT"].map(ai["REGION_CD"])

    rows = []
    for fold_name, sub in oof.groupby("fold"):
        mdata = model_dir / "model_specific_data" / fold_name
        nat_ap = pd.read_parquet(mdata / "native_airport_preds.parquet")
        nat_reg = pd.read_parquet(mdata / "native_region_preds.parquet")

        # Origin airport bottom-up vs native
        org_bu = sub.groupby(["ORG_AIRPRT", "YM"], as_index=False)[["y_true", "y_pred"]].sum().rename(
            columns={"ORG_AIRPRT": "AIRPRT_CD", "y_pred": "bu_pred"}
        )
        org_cmp = org_bu.merge(nat_ap[["AIRPRT_CD", "YM", "pred_org_native"]], on=["AIRPRT_CD", "YM"])
        m_org_bu = compute_metrics(org_cmp["y_true"], org_cmp["bu_pred"])
        m_org_nat = compute_metrics(org_cmp["y_true"], org_cmp["pred_org_native"])

        # Region-pair bottom-up vs native
        reg_bu = sub.groupby(["ORG_REG", "DES_REG", "YM"], as_index=False)[["y_true", "y_pred"]].sum().rename(
            columns={"y_pred": "bu_pred"}
        )
        reg_cmp = reg_bu.merge(nat_reg, on=["ORG_REG", "DES_REG", "YM"])
        m_reg_bu = compute_metrics(reg_cmp["y_true"], reg_cmp["bu_pred"])
        m_reg_nat = compute_metrics(reg_cmp["y_true"], reg_cmp["pred_reg_native"])

        rows.append({
            "fold": fold_name,
            "org_reconciled_wape": m_org_bu["wape"],
            "org_native_wape": m_org_nat["wape"],
            "reg_reconciled_wape": m_reg_bu["wape"],
            "reg_native_wape": m_reg_nat["wape"],
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(model_dir / "outputs" / "bottom_up_vs_native_levels.csv", index=False)
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
