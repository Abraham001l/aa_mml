from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from shared_eval.data_loader import idx_to_ym
from shared_eval.feature_engine import FoldFeatureStore

EXT_DIR = Path(__file__).resolve().parents[1] / "data_external"


class ExternalFoldFeatureStore(FoldFeatureStore):
    """
    Extends FoldFeatureStore with strictly point-in-time external Census, FAA,
    County Boundaries, County-to-County Migration, and FRED Macro datasets.
    Every external record joined at pseudo-origin s strictly satisfies AVAILABLE_YM <= ym(s).
    """

    def __init__(
        self,
        hist_panel: pd.DataFrame,
        airport_df: pd.DataFrame,
        include_macro: bool = False,
    ):
        super().__init__(hist_panel, airport_df)
        self.include_macro = include_macro
        self._load_and_preindex_external()

    def _load_and_preindex_external(self) -> None:
        n_t = self.t_cut + 1
        ap_ann = pd.read_parquet(EXT_DIR / "airport_external_annual.parquet")
        mig = pd.read_parquet(EXT_DIR / "pair_migration_2015.parquet")
        macro = pd.read_parquet(EXT_DIR / "macro_monthly_clean.parquet")

        # 1. Pair-level Census 2011-2015 Migration (AVAILABLE_YM == 201801)
        p_mig = self.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].merge(
            mig, on=["ORG_AIRPRT", "DES_AIRPRT"], how="left"
        )
        self.mig_avail_ym = p_mig["AVAILABLE_YM"].fillna(999999).to_numpy(dtype=np.int32)
        self.mig_total = np.log1p(p_mig["MIG_TOTAL"].fillna(0.0).to_numpy(dtype=np.float32))
        self.mig_fwd = np.log1p(p_mig["MIG_ORG_TO_DES"].fillna(0.0).to_numpy(dtype=np.float32))
        tot = p_mig["MIG_TOTAL"].fillna(0.0).to_numpy(dtype=np.float32)
        fwd = p_mig["MIG_ORG_TO_DES"].fillna(0.0).to_numpy(dtype=np.float32)
        rev = p_mig["MIG_DES_TO_ORG"].fillna(0.0).to_numpy(dtype=np.float32)
        self.mig_asym = ((fwd - rev) / (tot + 10.0)).astype(np.float32)

        # 2. Airport-level point-in-time annual matrices (n_aps, n_t)
        fold_aps = sorted(set(self.pairs_df["ORG_AIRPRT"]).union(set(self.pairs_df["DES_AIRPRT"])))
        ap_to_idx = {a: i for i, a in enumerate(fold_aps)}

        def make_ap_mat():
            return np.full((self.n_aps, n_t), np.nan, dtype=np.float32)

        self.ap_cbsa_pop = make_ap_mat()
        self.ap_cbsa_yoy = make_ap_mat()
        self.ap_cbsa_mig = make_ap_mat()
        self.ap_cty_dens = make_ap_mat()
        self.ap_st_mig = make_ap_mat()
        self.ap_faa_enp = make_ap_mat()
        self.ap_faa_yoy = make_ap_mat()

        # Precompute FAA YoY strictly between consecutive years when BOTH are available <= ym_s
        ap_ann = ap_ann[ap_ann["AIRPRT_CD"].isin(fold_aps)].sort_values(["AIRPRT_CD", "YEAR"]).copy()

        for t in range(n_t):
            ym_s = idx_to_ym(t)
            # Census CBSA / State / County rows released <= ym_s
            c_sub = ap_ann[ap_ann["AVAILABLE_YM"] <= ym_s]
            if len(c_sub):
                latest_c = c_sub.groupby("AIRPRT_CD", as_index=False).last()
                for _, r in latest_c.iterrows():
                    idx = ap_to_idx[r["AIRPRT_CD"]]
                    if pd.notna(r["CBSA_POP"]):
                        self.ap_cbsa_pop[idx, t] = np.log1p(r["CBSA_POP"])
                    if pd.notna(r["CBSA_POP_YOY"]):
                        self.ap_cbsa_yoy[idx, t] = r["CBSA_POP_YOY"]
                    if pd.notna(r["CBSA_DOM_MIG_RATE"]):
                        self.ap_cbsa_mig[idx, t] = r["CBSA_DOM_MIG_RATE"]
                    if pd.notna(r["COUNTY_POP_DENSITY"]):
                        self.ap_cty_dens[idx, t] = np.log1p(r["COUNTY_POP_DENSITY"])
                    if pd.notna(r["STATE_DOM_MIG_RATE"]):
                        self.ap_st_mig[idx, t] = r["STATE_DOM_MIG_RATE"]

            # FAA enplanements rows strictly released <= ym_s
            f_sub = ap_ann[(ap_ann["FAA_AVAILABLE_YM"].notna()) & (ap_ann["FAA_AVAILABLE_YM"] <= ym_s)]
            if len(f_sub):
                for ap_cd, grp in f_sub.groupby("AIRPRT_CD"):
                    idx = ap_to_idx[ap_cd]
                    grp_s = grp.sort_values("YEAR")
                    last_f = grp_s.iloc[-1]
                    if pd.notna(last_f["FAA_ENPLANEMENTS"]):
                        self.ap_faa_enp[idx, t] = np.log1p(last_f["FAA_ENPLANEMENTS"])
                    if len(grp_s) >= 2:
                        prev_f = grp_s.iloc[-2]
                        if last_f["YEAR"] == prev_f["YEAR"] + 1 and prev_f["FAA_ENPLANEMENTS"] > 0:
                            self.ap_faa_yoy[idx, t] = np.clip(
                                last_f["FAA_ENPLANEMENTS"] / prev_f["FAA_ENPLANEMENTS"] - 1.0, -0.9, 3.0
                            )

        # 3. Macro monthly series strictly released <= ym_s
        self.macro_cols = ["UNRATE", "UNRATE_DIFF12", "CPIAUCSL_YOY", "DSPIC96_YOY", "DCOILWTICO", "DCOILWTICO_YOY", "UMCSENT"]
        self.macro_mat = np.full((n_t, len(self.macro_cols)), np.nan, dtype=np.float32)
        if self.include_macro:
            macro_s = macro.sort_values("YM")
            for t in range(n_t):
                ym_s = idx_to_ym(t)
                avail = macro_s[macro_s["AVAILABLE_YM"] <= ym_s]
                if len(avail):
                    last_m = avail.iloc[-1]
                    for j, col in enumerate(self.macro_cols):
                        if pd.notna(last_m[col]):
                            self.macro_mat[t, j] = float(last_m[col])

    def build_features_for_slice(self, s: int, h: int) -> pd.DataFrame:
        df = super().build_features_for_slice(s, h)
        ym_s = idx_to_ym(s)

        # Pair Migration (strictly AVAILABLE_YM <= ym_s)
        valid_mig = self.mig_avail_ym <= ym_s
        df["EXT_LOG_MIG_TOTAL"] = np.where(valid_mig, self.mig_total, np.nan).astype(np.float32)
        df["EXT_LOG_MIG_FWD"] = np.where(valid_mig, self.mig_fwd, np.nan).astype(np.float32)
        df["EXT_MIG_ASYM"] = np.where(valid_mig, self.mig_asym, np.nan).astype(np.float32)

        # Origin & Destination Census Demographics
        oi = self.org_ap_idx
        di = self.des_ap_idx
        df["EXT_ORG_CBSA_POP"] = self.ap_cbsa_pop[oi, s]
        df["EXT_DES_CBSA_POP"] = self.ap_cbsa_pop[di, s]
        df["EXT_GRAVITY_POP"] = (self.ap_cbsa_pop[oi, s] + self.ap_cbsa_pop[di, s]).astype(np.float32)
        df["EXT_ORG_CBSA_YOY"] = self.ap_cbsa_yoy[oi, s]
        df["EXT_DES_CBSA_YOY"] = self.ap_cbsa_yoy[di, s]
        df["EXT_ORG_CBSA_MIG"] = self.ap_cbsa_mig[oi, s]
        df["EXT_DES_CBSA_MIG"] = self.ap_cbsa_mig[di, s]
        df["EXT_ORG_CTY_DENS"] = self.ap_cty_dens[oi, s]
        df["EXT_DES_CTY_DENS"] = self.ap_cty_dens[di, s]
        df["EXT_ORG_ST_MIG"] = self.ap_st_mig[oi, s]
        df["EXT_DES_ST_MIG"] = self.ap_st_mig[di, s]

        # FAA Airport Enplanements & O&D capture ratio
        org_faa = self.ap_faa_enp[oi, s]
        des_faa = self.ap_faa_enp[di, s]
        df["EXT_ORG_FAA_ENP"] = org_faa
        df["EXT_DES_FAA_ENP"] = des_faa
        df["EXT_ORG_FAA_YOY"] = self.ap_faa_yoy[oi, s]
        df["EXT_DES_FAA_YOY"] = self.ap_faa_yoy[di, s]
        # Log ratio of OD airport total (annualized) vs FAA boardings -> connecting hub vs O&D profile
        df["EXT_ORG_OD_VS_FAA"] = (np.log1p(self.pair_org_r12[:, s] * 12.0) - org_faa).astype(np.float32)
        df["EXT_DES_OD_VS_FAA"] = (np.log1p(self.pair_des_r12[:, s] * 12.0) - des_faa).astype(np.float32)

        if self.include_macro:
            for j, col in enumerate(self.macro_cols):
                val = self.macro_mat[s, j]
                df[f"EXT_MACRO_{col}"] = np.full(self.n_pairs, val, dtype=np.float32)
            # Route-level macro interactions (oil * distance, unrate * domestic)
            df["EXT_OIL_X_DIST"] = (self.macro_mat[s, 4] * self.static_feats["LOG_DIST"]).astype(np.float32)

        return df
