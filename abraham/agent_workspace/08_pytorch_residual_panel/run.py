from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.feature_engine import FoldFeatureStore, assemble_training_dataset
from shared_eval.runner import run_backtest_suite
from shared_eval.splits import FoldSpec, assert_origin_cutoff


def get_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class AviationPanelNet(nn.Module):
    """Deep residual panel network with airport, horizon, and month embeddings."""

    def __init__(self, n_num: int, n_airports: int):
        super().__init__()
        self.org_emb = nn.Embedding(n_airports + 1, 16)
        self.des_emb = nn.Embedding(n_airports + 1, 16)
        self.h_emb = nn.Embedding(25, 8)
        self.m_emb = nn.Embedding(13, 6)

        in_dim = n_num + 16 + 16 + 8 + 6
        hid = 192
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.LayerNorm(hid),
            nn.SiLU(),
        )
        self.block1 = ResBlock(hid, dropout=0.1)
        self.block2 = ResBlock(hid, dropout=0.1)
        self.head = nn.Linear(hid, 1)
        # Initialize head near zero so initial predictions match damped seasonal anchor
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x_num: torch.Tensor,
        org_id: torch.Tensor,
        des_id: torch.Tensor,
        h_id: torch.Tensor,
        m_id: torch.Tensor,
    ) -> torch.Tensor:
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
        h = self.in_proj(h)
        h = self.block1(h)
        h = self.block2(h)
        # Bounded residual adjustment in log space [-2.0, +2.0]
        return 2.0 * torch.tanh(self.head(h).squeeze(-1) * 0.5)


def _assemble_with_ids(store: FoldFeatureStore, stride_s: int = 2):
    t_cut = store.t_cut
    frames, org_ids, des_ids, y_list, w_list = [], [], [], [], []
    p_org = store.org_ap_idx + 1  # 1-based; 0 reserved for unknown
    p_des = store.des_ap_idx + 1
    for s in range(6, t_cut, stride_s):
        max_h = min(24, t_cut - s)
        for h in range(1, max_h + 1):
            t_tgt = s + h
            if 14 <= t_tgt <= 25:
                continue
            df_slice = store.build_features_for_slice(s, h)
            rec_w = float(np.exp(-(t_cut - t_tgt) / 18.0))
            frames.append(df_slice)
            org_ids.append(p_org)
            des_ids.append(p_des)
            y_list.append(store.P[:, t_tgt].astype(np.float32))
            w_list.append(np.full(store.n_pairs, rec_w, dtype=np.float32))
    return (
        pd.concat(frames, ignore_index=True),
        np.concatenate(org_ids),
        np.concatenate(des_ids),
        np.concatenate(y_list),
        np.concatenate(w_list),
    )


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

    store = FoldFeatureStore(hist_panel, airport_df)
    X_df, org_id, des_id, y_true, w_rec = _assemble_with_ids(store, stride_s=2)

    feat_cols = [c for c in X_df.columns if c != "DAMPED_ANCHOR"]
    X_raw = X_df[feat_cols].to_numpy(dtype=np.float32)
    # Clip extreme ratios and impute medians strictly from training matrix
    medians = np.nanmedian(X_raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    nan_mask = ~np.isfinite(X_raw)
    if nan_mask.any():
        X_raw[nan_mask] = np.take(medians, np.where(nan_mask)[1])
    mean = X_raw.mean(axis=0)
    std = np.maximum(X_raw.std(axis=0), 1e-4)
    X_norm = np.clip((X_raw - mean) / std, -8.0, 8.0).astype(np.float32)

    log_anchor = X_df["LOG_DAMPED_ANCHOR"].to_numpy(dtype=np.float32)
    h_id = X_df["HORIZON"].to_numpy(dtype=np.int64)
    m_id = X_df["DEP_MONTH"].to_numpy(dtype=np.int64)

    ds = TensorDataset(
        torch.from_numpy(X_norm),
        torch.from_numpy(org_id.astype(np.int64)),
        torch.from_numpy(des_id.astype(np.int64)),
        torch.from_numpy(h_id),
        torch.from_numpy(m_id),
        torch.from_numpy(log_anchor),
        torch.from_numpy(y_true),
        torch.from_numpy(w_rec),
    )
    loader = DataLoader(ds, batch_size=16384, shuffle=True, drop_last=False)

    model = AviationPanelNet(n_num=len(feat_cols), n_airports=store.n_aps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    epochs = 6
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))

    mean_y_scale = float(np.mean(y_true)) + 1.0
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for bx, bo, bd, bh, bm, ba, by, bw in loader:
            bx = bx.to(device)
            bo = bo.to(device)
            bd = bd.to(device)
            bh = bh.to(device)
            bm = bm.to(device)
            ba = ba.to(device)
            by = by.to(device)
            bw = bw.to(device)

            opt.zero_grad()
            delta = model(bx, bo, bd, bh, bm)
            log_pred = ba + delta
            y_hat = torch.expm1(log_pred).clamp(min=0.0)
            # Composite WAPE-aligned + log-space Huber loss
            l_abs = torch.nn.functional.smooth_l1_loss(y_hat / mean_y_scale, by / mean_y_scale, reduction="none", beta=0.05)
            l_log = torch.nn.functional.smooth_l1_loss(log_pred, torch.log1p(by), reduction="none", beta=0.25)
            loss = (bw * (l_abs + 0.15 * l_log)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            total_loss += float(loss.item())

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"device": [str(device)], "epochs": [epochs], "n_train": [len(X_df)]}).to_csv(
        mdata_dir / "train_info.csv", index=False
    )

    model.eval()
    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    o_te = torch.from_numpy((store.org_ap_idx + 1).astype(np.int64)).to(device)
    d_te = torch.from_numpy((store.des_ap_idx + 1).astype(np.int64)).to(device)

    with torch.no_grad():
        for h in range(1, 25):
            t_tgt = store.t_cut + h
            ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
            X_te_df = store.build_features_for_slice(store.t_cut, h)
            X_te_raw = X_te_df[feat_cols].to_numpy(dtype=np.float32)
            nmask = ~np.isfinite(X_te_raw)
            if nmask.any():
                X_te_raw[nmask] = np.take(medians, np.where(nmask)[1])
            X_te_norm = np.clip((X_te_raw - mean) / std, -8.0, 8.0).astype(np.float32)

            bx = torch.from_numpy(X_te_norm).to(device)
            bh = torch.full((store.n_pairs,), h, dtype=torch.int64, device=device)
            bm = torch.full((store.n_pairs,), int(store.cal_df.loc[t_tgt, "DEP_MONTH"]), dtype=torch.int64, device=device)
            ba = X_te_df["LOG_DAMPED_ANCHOR"].to_numpy(dtype=np.float64)

            delta = model(bx, o_te, d_te, bh, bm).cpu().numpy().astype(np.float64)
            y_hat = np.clip(np.expm1(ba + delta), 0.0, None)

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
