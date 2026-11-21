from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.runner import run_backtest_suite
from shared_eval.splits import FoldSpec, assert_origin_cutoff

WORKSPACE = Path(__file__).resolve().parents[1]

LATAM_CODES = {
    'MX', 'CA', 'PR', 'BS', 'JM', 'DO', 'CR', 'AW', 'KY', 'TT', 'BB',
    'SX', 'LC', 'AG', 'TC', 'KN', 'GD', 'VI', 'BQ', 'CW', 'CO', 'PE',
    'EC', 'BZ', 'GT', 'HN', 'SV', 'NI', 'PA'
}
EUROPE_CODES = {
    'GB', 'FR', 'DE', 'IT', 'ES', 'IE', 'NL', 'CH', 'PT', 'BE', 'AT',
    'SE', 'DK', 'NO', 'GR', 'PL', 'CZ', 'IS', 'TR', 'AE', 'QA'
}
APAC_CODES = {
    'JP', 'KR', 'CN', 'HK', 'TW', 'AU', 'NZ', 'SG', 'PH', 'TH', 'IN'
}


def classify_geo_segment(org: str, des: str, cntry_dict: dict[str, str]) -> int:
    """
    Classify route into 5 Geo-Macro Market Segments:
    0: DOMESTIC_US
    1: AMERICAS_INTL (Transborder & Caribbean)
    2: TRANSATLANTIC (Europe / Mid-East)
    3: TRANSPACIFIC (Asia / Pacific)
    4: OTHER_INTL
    """
    c1 = cntry_dict.get(org, 'US')
    c2 = cntry_dict.get(des, 'US')
    if c1 == 'US' and c2 == 'US':
        return 0
    if c1 in LATAM_CODES or c2 in LATAM_CODES:
        return 1
    if c1 in EUROPE_CODES or c2 in EUROPE_CODES:
        return 2
    if c1 in APAC_CODES or c2 in APAC_CODES:
        return 3
    return 4


def _load_predict_fn(folder_name: str):
    spec = importlib.util.spec_from_file_location(f"mod_{folder_name}", WORKSPACE / folder_name / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.predict_fold


def compute_geo_macro_persistent_recovery(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    5-Segment Geo-Macro Persistent Network Recovery.
    Disentangles staggered regional border reopenings:
    - Domestic US (fastest)
    - Americas / Caribbean (early leisure rebound)
    - Transatlantic Europe (summer 2021-2022 reopening)
    - Transpacific Asia (delayed late 2022-2023 reopening)
    - Other International
    """
    hp = hist_panel.sort_values(["PAIR_ID", "T_IDX"])
    pairs = hp[["PAIR_ID", "ORG_AIRPRT", "DES_AIRPRT"]].drop_duplicates().reset_index(drop=True)
    n = len(pairs)
    P = hp["OD_PAX"].to_numpy().reshape(n, -1)
    s = P.shape[1] - 1
    assert s >= 23

    cntry_map = airport_df.set_index("AIRPRT_CD")["CNTRY_CD"].to_dict()
    seg_ids = np.array([classify_geo_segment(o, d, cntry_map) for o, d in zip(pairs["ORG_AIRPRT"], pairs["DES_AIRPRT"])], dtype=int)

    oi = pd.factorize(pairs["ORG_AIRPRT"])[0]
    di = pd.factorize(pairs["DES_AIRPRT"])[0]

    idx3 = np.arange(s - 2, s + 1)
    idx6 = np.arange(s - 5, s + 1)
    ref3 = P[:, idx3 % 12].sum(axis=1)
    cur3 = P[:, idx3].sum(axis=1)
    ref6 = P[:, idx6 % 12].sum(axis=1)
    cur6 = P[:, idx6].sum(axis=1)

    K = 5
    seg_denom = np.maximum(np.bincount(seg_ids, weights=ref3, minlength=K), 1.0)
    seg = np.bincount(seg_ids, weights=cur3, minlength=K) / seg_denom

    n_keys_o = len(np.unique(oi)) * K
    keys_o = oi * K + seg_ids
    numer_o = np.bincount(keys_o, weights=cur3, minlength=n_keys_o)
    denom_o = np.bincount(keys_o, weights=ref3, minlength=n_keys_o)
    priors_o = np.tile(seg, len(numer_o) // K + 1)[: len(numer_o)]
    prior_o = ((numer_o + 10000.0 * priors_o) / (denom_o + 10000.0))[keys_o]

    n_keys_d = len(np.unique(di)) * K
    keys_d = di * K + seg_ids
    numer_d = np.bincount(keys_d, weights=cur3, minlength=n_keys_d)
    denom_d = np.bincount(keys_d, weights=ref3, minlength=n_keys_d)
    priors_d = np.tile(seg, len(numer_d) // K + 1)[: len(numer_d)]
    prior_d = ((numer_d + 10000.0 * priors_d) / (denom_d + 10000.0))[keys_d]

    pooled = np.sqrt(np.maximum(prior_o * prior_d, 1e-4))
    rec3 = (cur3 + 1500.0 * pooled) / (ref3 + 1500.0)
    rec6 = (cur6 + 3000.0 * pooled) / (ref6 + 3000.0)
    now = np.clip(0.75 * rec3 + 0.25 * rec6, 0.03, 3.0)

    relative = np.clip(now / np.maximum(seg[seg_ids], 0.15), 0.10, 2.5)
    persistent = np.where(relative < 1.0, np.sqrt(relative), relative)

    eq_target = np.array([1.02, 1.02, 1.02, 0.95, 1.00])
    eq = np.maximum(eq_target[seg_ids], seg[seg_ids]) * persistent
    tau_arr = np.array([12.0, 12.0, 12.0, 16.0, 14.0])[seg_ids]

    frames = []
    for h in range(1, 25):
        t = s + h
        m = t % 12
        decay = np.exp(-h / tau_arr)
        mult = decay * now + (1.0 - decay) * eq * (1.0 + 0.015 * h / 12.0)
        pred = P[:, m] * mult
        last = t - (12 if h <= 12 else 24)
        pred = np.where(P[:, m] > 0.0, pred, P[:, last] * (1.0 + 0.015 * h / 12.0))

        frame = pairs[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
        frame["HORIZON"] = h
        frame["y_pred"] = np.maximum(pred, 0.0)
        frames.append(frame)

    pred_all = pd.concat(frames, ignore_index=True)
    return tgt_skeleton[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"]].merge(
        pred_all, on=["ORG_AIRPRT", "DES_AIRPRT", "HORIZON"], how="left"
    )


def predict_fold(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    """
    Model 25: Geo-Macro Staggered Reopening Ensemble.
    Synthesizes:
    - Model 24 Base Grand Ensemble (40% Model 21 + 10% Model 22 + 50% Persistent Network Recovery)
    - 5-Segment Geo-Macro Reopening on International Corridors (30% blend on international routes, 0% on domestic)
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base Model 24 Prediction (use cached predictions if available)
    m24_backtest_file = WORKSPACE / "24_spatial_gravity_network_ensemble" / "outputs" / "backtest_predictions.parquet"
    m24_holdout_file = WORKSPACE / "24_spatial_gravity_network_ensemble" / "outputs" / "final_holdout_predictions.parquet"
    df_24 = None

    if m24_backtest_file.exists():
        oof24 = pd.read_parquet(m24_backtest_file)
        f_sub = oof24[oof24["fold"] == fold.name].copy()
        if len(f_sub) > 0:
            df_24 = tgt_skeleton[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"]].merge(
                f_sub[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_pred"]],
                on=["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"],
                how="left",
            )

    if df_24 is None and m24_holdout_file.exists():
        h24 = pd.read_parquet(m24_holdout_file)
        if set(tgt_skeleton["YM"].unique()).issubset(set(h24["YM"].unique())):
            df_24 = tgt_skeleton[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"]].merge(
                h24[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_pred"]],
                on=["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"],
                how="left",
            )

    if df_24 is None:
        pred_24_fn = _load_predict_fn("24_spatial_gravity_network_ensemble")
        sub_dir_24 = mdata_dir / "_sub_artifacts" / "24_spatial_gravity_network_ensemble"
        df_24 = pred_24_fn(fold, hist_panel.copy(), tgt_skeleton.copy(), airport_df.copy(), sub_dir_24)

    # 2. 5-Segment Geo-Macro Persistent Network Recovery
    df_5seg = compute_geo_macro_persistent_recovery(fold, hist_panel.copy(), tgt_skeleton.copy(), airport_df.copy())

    out = tgt_skeleton[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"]].copy()
    out = out.merge(df_24[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_pred"]], on=["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"])
    out = out.rename(columns={"y_pred": "pred_24"})

    out = out.merge(df_5seg[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_pred"]], on=["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"])
    out = out.rename(columns={"y_pred": "pred_5seg"})

    # Classify International routes
    cntry_map = airport_df.set_index("AIRPRT_CD")["CNTRY_CD"].to_dict()
    is_intl = (out["ORG_AIRPRT"].map(cntry_map) != "US") | (out["DES_AIRPRT"].map(cntry_map) != "US")

    # Apply 30% 5-seg blend to international routes, retain 100% Model 24 on domestic
    y_final = out["pred_24"].to_numpy().copy()
    y_final[is_intl] = 0.70 * out["pred_24"].to_numpy()[is_intl] + 0.30 * out["pred_5seg"].to_numpy()[is_intl]
    out["y_pred"] = np.clip(y_final, 0.0, None)

    summary = pd.DataFrame([
        {"segment": "Domestic_US", "model_24_weight": 1.00, "geo_5seg_weight": 0.00},
        {"segment": "International_Corridors", "model_24_weight": 0.70, "geo_5seg_weight": 0.30},
    ])
    summary.to_csv(mdata_dir / "segment_blend_weights.csv", index=False)

    return out[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_pred"]]


if __name__ == "__main__":
    run_backtest_suite(Path(__file__).resolve().parent, predict_fold)
