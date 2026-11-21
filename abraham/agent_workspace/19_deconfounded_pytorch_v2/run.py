from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.external_feature_store import ExternalFoldFeatureStore
from shared_eval.runner import run_backtest_suite
from shared_eval.splits import FoldSpec, assert_origin_cutoff

_spec17 = importlib.util.spec_from_file_location(
    "deconf17",
    Path(__file__).resolve().parents[1] / "17_deconfounded_xgb_v2" / "run.py",
)
_mod17 = importlib.util.module_from_spec(_spec17)
_spec17.loader.exec_module(_mod17)
compute_structural_anchor_for_slice = _mod17.compute_structural_anchor_for_slice
_clean_deconfounded_features = _mod17._clean_deconfounded_features

_spec08 = importlib.util.spec_from_file_location(
    "pt08",
    Path(__file__).resolve().parents[1] / "08_pytorch_residual_panel" / "run.py",
)
_mod08 = importlib.util.module_from_spec(_spec08)
_spec08.loader.exec_module(_mod08)
get_device = _mod08.get_device
ResBlock = _mod08.ResBlock


class DeconfoundedPanelNet(nn.Module):
    """Deep Residual Network predicting bounded adjustment around anchor_v2 (log scale)."""

    def __init__(self, n_num: int, n_airports: int):
        super().__init__()
        self.org_emb = nn.Embedding(n_airports + 1, 12)
        self.des_emb = nn.Embedding(n_airports + 1, 12)
        self.h_emb = nn.Embedding(25, 6)
        self.m_emb = nn.Embedding(13, 6)

        in_dim = n_num + 12 + 12 + 6 + 6
        hid = 160
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.LayerNorm(hid),
            nn.SiLU(),
        )
        self.block1 = ResBlock(hid, dropout=0.15)
        self.block2 = ResBlock(hid, dropout=0.15)
        self.head = nn.Linear(hid, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_num, org_id, des_id, h_id, m_id):
        h = torch.cat(
            [
                x_num,
                self.org_emb(org_id),
                self.des_emb(des_id),
                self.h_emb(h_id),
                self.m_emb(m_id),
            ],
            dim=1,
        )
        h = self.block2(self.block1(self.in_proj(h)))
        # Tight trust region [-0.38, +0.38] around structural anchor_v2
        return 0.38 * torch.tanh(self.head(h).squeeze(-1))


def predict_fold(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    torch.manual_seed(42)
    np.random.seed(42)
    device = get_device()

    store = ExternalFoldFeatureStore(hist_panel, airport_df, include_macro=False)
    t_cut = store.t_cut

    frames, org_ids, des_ids, ys, ws, ancs = [], [], [], [], [], []
    p_org = store.org_ap_idx + 1
    p_des = store.des_ap_idx + 1
    for s in range(6, t_cut, 2):
        for h in range(1, min(24, t_cut - s) + 1):
            t_tgt = s + h
            if 14 <= t_tgt <= 25:
                continue
            df_s = store.build_features_for_slice(s, h)
            anc = compute_structural_anchor_for_slice(store, airport_df, s, h)
            df_clean = _clean_deconfounded_features(df_s, anc)
            shock_w = 0.20 if 26 <= t_tgt <= 28 else 1.0
            rec_w = float(np.exp(-(t_cut - t_tgt) / 18.0)) * shock_w

            frames.append(df_clean)
            org_ids.append(p_org)
            des_ids.append(p_des)
            ys.append(store.P[:, t_tgt].astype(np.float32))
            ws.append(np.full(store.n_pairs, rec_w, dtype=np.float32))
            ancs.append(anc.astype(np.float32))

    X_df = pd.concat(frames, ignore_index=True)
    org_arr = np.concatenate(org_ids)
    des_arr = np.concatenate(des_ids)
    y_true = np.concatenate(ys)
    w_rec = np.concatenate(ws)
    anc_arr = np.concatenate(ancs)
    del frames, org_ids, des_ids, ys, ws, ancs
    gc.collect()

    feat_cols = list(X_df.columns)
    X_raw = X_df[feat_cols].to_numpy(dtype=np.float32)
    medians = np.nanmedian(X_raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    nmask = ~np.isfinite(X_raw)
    if nmask.any():
        X_raw[nmask] = np.take(medians, np.where(nmask)[1])
    mean = X_raw.mean(axis=0)
    std = np.maximum(X_raw.std(axis=0), 1e-4)
    X_norm = np.clip((X_raw - mean) / std, -8.0, 8.0).astype(np.float32)

    h_id = X_df["HORIZON"].to_numpy(dtype=np.int64)
    m_id = X_df["DEP_MONTH"].to_numpy(dtype=np.int64)
    del X_df, X_raw
    gc.collect()

    ds = TensorDataset(
        torch.from_numpy(X_norm),
        torch.from_numpy(org_arr.astype(np.int64)),
        torch.from_numpy(des_arr.astype(np.int64)),
        torch.from_numpy(h_id),
        torch.from_numpy(m_id),
        torch.from_numpy(anc_arr),
        torch.from_numpy(y_true),
        torch.from_numpy(w_rec),
    )
    loader = DataLoader(ds, batch_size=16384, shuffle=True, drop_last=False)

    model = DeconfoundedPanelNet(n_num=len(feat_cols), n_airports=store.n_aps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1e-3)
    epochs = 6
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    mean_y_scale = float(np.mean(y_true)) + 1.0

    model.train()
    for _ in range(epochs):
        for bx, bo, bd, bh, bm, ba, by, bw in loader:
            bx, bo, bd = bx.to(device), bo.to(device), bd.to(device)
            bh, bm, ba = bh.to(device), bm.to(device), ba.to(device)
            by, bw = by.to(device), bw.to(device)

            opt.zero_grad()
            delta = model(bx, bo, bd, bh, bm)
            y_hat = (ba + 1.0) * torch.exp(delta) - 1.0
            y_hat = y_hat.clamp(min=0.0)
            l_abs = torch.nn.functional.smooth_l1_loss(y_hat / mean_y_scale, by / mean_y_scale, reduction="none", beta=0.05)
            l_log = torch.nn.functional.smooth_l1_loss(torch.log1p(y_hat), torch.log1p(by), reduction="none", beta=0.25)
            loss = (bw * (l_abs + 0.15 * l_log)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"device": [str(device)], "epochs": [epochs], "n_train": [len(y_true)]}).to_csv(
        mdata_dir / "train_info.csv", index=False
    )

    model.eval()
    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    o_te = torch.from_numpy((store.org_ap_idx + 1).astype(np.int64)).to(device)
    d_te = torch.from_numpy((store.des_ap_idx + 1).astype(np.int64)).to(device)

    with torch.no_grad():
        for h in range(1, 25):
            t_tgt = t_cut + h
            ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
            X_te_raw_df = store.build_features_for_slice(t_cut, h)
            anc_te = compute_structural_anchor_for_slice(store, airport_df, t_cut, h)
            X_te_df = _clean_deconfounded_features(X_te_raw_df, anc_te)
            X_te_raw = X_te_df[feat_cols].to_numpy(dtype=np.float32)
            nm = ~np.isfinite(X_te_raw)
            if nm.any():
                X_te_raw[nm] = np.take(medians, np.where(nm)[1])
            X_te_norm = np.clip((X_te_raw - mean) / std, -8.0, 8.0).astype(np.float32)

            bx = torch.from_numpy(X_te_norm).to(device)
            bh = torch.full((store.n_pairs,), h, dtype=torch.int64, device=device)
            bm = torch.full((store.n_pairs,), int(store.cal_df.loc[t_tgt, "DEP_MONTH"]), dtype=torch.int64, device=device)
            delta = model(bx, o_te, d_te, bh, bm).cpu().numpy().astype(np.float64)
            y_hat = np.clip((anc_te + 1.0) * np.exp(delta) - 1.0, 0.0, None)

            df_h = pairs.copy()
            df_h["YM"] = ym_tgt
            df_h["HORIZON"] = h
            df_h["y_pred"] = y_hat
            pred_frames.append(df_h)

    pred_all = pd.concat(pred_frames, ignore_index=True)
    return tgt_skeleton[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"]].merge(
        pred_all, on=["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"], how="left"
    )


if __name__ == "__main__":
    run_backtest_suite(Path(__file__).resolve().parent, predict_fold)
