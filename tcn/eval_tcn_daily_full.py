# eval_tcn_daily_full.py
# -*- coding: utf-8 -*-
"""
train_tcn_daily.py로 학습된 모델 평가용 (Stop Loss 추가)
"""
import re, sys, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

warnings.filterwarnings("ignore")
import os, random

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
random.seed(0);
np.random.seed(0);
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


# ========= CLI =========
def parse_args():
    ap = argparse.ArgumentParser(description="Daily eval for train_tcn_daily.py models")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--seq_len", type=int, required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--tp_bps", type=float, required=True)
    ap.add_argument("--sl_bps", type=float, default=0.0)  # ⭐ Stop Loss 추가
    ap.add_argument("--equity0", type=float, required=True)
    ap.add_argument("--equity_pct", type=float, required=True)
    ap.add_argument("--leverage", type=float, required=True)
    ap.add_argument("--taker_fee", type=float, required=True)
    ap.add_argument("--maker_fee", type=float, required=True)
    ap.add_argument("--slip_in_bps", type=float, required=True)
    ap.add_argument("--slip_out_bps", type=float, required=True)
    ap.add_argument("--max_open", type=int, required=True)
    ap.add_argument("--mode", choices=["taker", "maker"], default="taker")
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--flat_edge_min", type=float, default=0.0)
    return ap.parse_args()


# ========= TCN Model (train_tcn_daily.py와 동일) =========
class Chomp1d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):
        return x[:, :, :-self.c].contiguous() if self.c > 0 else x


def wconv(i, o, k, d):
    import torch.nn.utils as U
    pad = (k - 1) * d
    return U.weight_norm(nn.Conv1d(i, o, k, padding=pad, dilation=d))


class Block(nn.Module):
    def __init__(self, i, o, k, d, drop):
        super().__init__()
        self.c1 = wconv(i, o, k, d)
        self.h1 = Chomp1d((k - 1) * d)
        self.r1 = nn.ReLU()
        self.dr1 = nn.Dropout(drop)
        self.c2 = wconv(o, o, k, d)
        self.h2 = Chomp1d((k - 1) * d)
        self.r2 = nn.ReLU()
        self.dr2 = nn.Dropout(drop)
        self.ds = nn.Conv1d(i, o, 1) if i != o else None
        self.r = nn.ReLU()

    def forward(self, x):
        y = self.dr1(self.r1(self.h1(self.c1(x))))
        y = self.dr2(self.r2(self.h2(self.c2(y))))
        res = x if self.ds is None else self.ds(x)
        return self.r(y + res)


class TCN_Simple(nn.Module):
    def __init__(self, in_f, hidden=128, levels=6, k=3, drop=0.2):
        super().__init__()
        L = []
        ch = in_f
        for i in range(levels):
            L.append(Block(ch, hidden, k, 2 ** i, drop))
            ch = hidden
        self.tcn = nn.Sequential(*L)
        self.head_side = nn.Linear(hidden, 2)

    def forward(self, X):
        X = X.transpose(1, 2)
        H = self.tcn(X)[:, :, -1]
        return self.head_side(H)


# ========= Features (train_tcn_daily.py와 동일) =========
def make_features(df: pd.DataFrame):
    """train_tcn_daily.py의 make_features와 동일"""
    g = df.copy()
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g.groupby("symbol")["logc"].diff().fillna(0.0)

    def roll_std(s, w):
        return s.rolling(w, min_periods=max(2, w // 3)).std()

    for w in (5, 10, 20, 60):
        g[f"rv{w}"] = g.groupby("symbol")["ret1"].apply(lambda s: roll_std(s, w)).reset_index(level=0, drop=True)

    def mom(gp, w):
        ema = gp["close"].ewm(span=w, adjust=False).mean()
        return gp["close"] / ema - 1.0

    for w in (5, 10, 20, 60):
        g[f"mom{w}"] = g.groupby("symbol", group_keys=False).apply(lambda s: mom(s, w))

    for w in (10, 20, 60):
        mu = g.groupby("symbol")["volume"].apply(lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean()).reset_index(
            level=0, drop=True)
        sd = g.groupby("symbol")["volume"].apply(lambda s: s.rolling(w, min_periods=max(2, w // 3)).std()).reset_index(
            level=0, drop=True)
        sd = sd.replace(0, np.nan)
        g[f"vz{w}"] = (g["volume"] - mu) / sd.replace({0: np.nan}).fillna(1.0)

    prev_close = g.groupby("symbol")["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()], axis=1).max(axis=1)
    g["atr14"] = tr.groupby(g["symbol"]).transform(lambda s: s.rolling(14, min_periods=5).mean())

    feats = ["ret1", "rv5", "rv10", "rv20", "rv60", "mom5", "mom10", "mom20", "mom60", "vz10", "vz20", "vz60", "atr14"]
    g = g.dropna(subset=feats).reset_index(drop=True)
    return g, feats


# ========= Helpers =========
def tp_from_bps(px, bps, side):
    return px * (1 + bps / 1e4) if side == "Buy" else px * (1 - bps / 1e4)


def sl_from_bps(px, bps, side):
    """⭐ Stop Loss 가격 계산"""
    if bps <= 0:
        return None  # SL 미설정
    return px * (1 - bps / 1e4) if side == "Buy" else px * (1 + bps / 1e4)


def entry_notional(eq, pct, lev):
    margin = eq * max(min(pct, 1.0), 0.0)
    return margin * lev, margin


def apply_slip(px, bps, side, is_entry):
    sgn = +1 if (side == "Buy") else -1
    return px * (1.0 + sgn * (+1 if is_entry else -1) * (bps / 1e4))


def fee_cost(notional, fee_rate):
    return notional * fee_rate


def _ensure_dt_utc(s):
    if not is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, utc=True, errors="coerce")
    return s


def find_row_idx_by_ts(df_sym, ts_col, ts):
    col = _ensure_dt_utc(df_sym[ts_col])
    ts = pd.to_datetime(ts, utc=True)
    s = col.searchsorted(ts)
    if s < len(col) and col.iloc[s] == ts:
        return int(s)
    return None


def find_le_idx_by_ts(df_sym, ts_col, ts):
    col = _ensure_dt_utc(df_sym[ts_col])
    ts = pd.to_datetime(ts, utc=True)
    s = col.searchsorted(ts, side="right") - 1
    return int(s) if s >= 0 else None


def make_entry_candidates(ts, sym_frames, open_pos, model, seq_len, ts_col, args):
    cands = []
    for sym, pack in sym_frames.items():
        if sym in open_pos:
            continue
        g = pack["raw"]
        gf = pack["feat"]
        X = pack["X"]

        idx = find_le_idx_by_ts(gf, ts_col, ts)
        if (idx is None) or (idx < seq_len) or ((idx + 1) >= len(gf)):
            continue

        x = torch.from_numpy(X[idx - seq_len: idx]).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            p = torch.softmax(logits, dim=1).cpu().numpy()[0]
            p_short, p_long = float(p[0]), float(p[1])

            side = "Buy" if p_long >= p_short else "Sell"
            conf = abs(p_long - p_short)

        next_ts = gf.iloc[idx + 1][ts_col]
        nxt_g = find_row_idx_by_ts(g, ts_col, next_ts)
        if nxt_g is None:
            continue

        cands.append((conf, sym, side, idx))

    cands.sort(key=lambda z: (-float(z[0]), str(z[1])))
    return cands


# ========= Main =========
def main():
    args = parse_args()
    FEE_IN = args.taker_fee if args.mode == "taker" else args.maker_fee
    FEE_OUT = args.taker_fee if args.mode == "taker" else args.maker_fee

    # 체크포인트 로드
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    feat_cols = ckpt["feat_cols"]

    mu = np.asarray(ckpt["scaler_mu"], dtype=np.float32)
    sd = np.asarray(ckpt["scaler_sd"], dtype=np.float32)
    sd[sd == 0] = 1.0

    # 모델 생성 및 로드
    model = TCN_Simple(in_f=len(feat_cols))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    for m in model.modules():
        if hasattr(m, "train"):
            m.train(False)

    print(f"[INFO] Loaded model with {len(feat_cols)} features: {feat_cols}")
    if args.sl_bps > 0:
        print(f"[INFO] Stop Loss: {args.sl_bps} bps (TP: {args.tp_bps} bps, R:R = {args.sl_bps/args.tp_bps:.2f}:1)")

    # 데이터 로드
    if args.val.lower().endswith(".parquet"):
        df = pd.read_parquet(args.val)
    else:
        df = pd.read_csv(args.val)

    # 날짜 컬럼 처리
    ts_col = "date" if "date" in df.columns else "timestamp"
    if not is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    else:
        if is_datetime64tz_dtype(df[ts_col]):
            df[ts_col] = df[ts_col].dt.tz_convert("UTC")
        else:
            df[ts_col] = df[ts_col].dt.tz_localize("UTC")

    # 피처 생성
    df_feat, feat_built = make_features(df)

    if set(feat_cols) - set(feat_built):
        missing = list(sorted(set(feat_cols) - set(feat_built)))
        raise RuntimeError(f"Feature mismatch. Missing: {missing}")

    # 심볼별 데이터 준비
    sym_frames = {}
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values(ts_col).reset_index(drop=True).copy()
        g[ts_col] = pd.to_datetime(g[ts_col], utc=True, errors="coerce")
        g = g.dropna(subset=[ts_col]).reset_index(drop=True)

        gf = df_feat[df_feat["symbol"] == sym].reset_index(drop=True).copy()
        gf[ts_col] = pd.to_datetime(gf[ts_col], utc=True, errors="coerce")
        gf = gf.dropna(subset=[ts_col]).reset_index(drop=True)

        if len(gf) < args.seq_len + 1:
            continue

        X = (gf[feat_cols].to_numpy(np.float32) - mu) / sd
        sym_frames[sym] = dict(raw=g, feat=gf, X=X)

    if not sym_frames:
        print("[ERR] No symbol with enough history for seq_len")
        sys.exit(0)

    print(f"[INFO] Processing {len(sym_frames)} symbols")

    # 백테스트 실행
    all_ts = sorted(df[ts_col].unique())
    equity = args.equity0
    open_pos = {}
    trades = []

    for ts in all_ts:
        # 1) 포지션 관리 (TP/SL 체크)
        to_close = []
        for sym, st in open_pos.items():
            g = sym_frames[sym]["raw"]
            idx = find_row_idx_by_ts(g, ts_col, ts)
            if idx is None or idx <= st["i_entry"]:
                continue

            hi = float(g.loc[idx, "high"])
            lo = float(g.loc[idx, "low"])
            side = st["side"]
            tp_px = st["tp_px"]
            sl_px = st.get("sl_px", None)  # ⭐ SL 가격
            ent_px = st["entry_px"]
            exit_px = None
            reason = None

            # ⭐ Stop Loss 체크 (먼저 체크!)
            if sl_px is not None:
                if side == "Buy" and lo <= sl_px:
                    exit_px = apply_slip(sl_px, args.slip_out_bps, side, is_entry=False)
                    reason = "SL"
                elif side == "Sell" and hi >= sl_px:
                    exit_px = apply_slip(sl_px, args.slip_out_bps, side, is_entry=False)
                    reason = "SL"

            # Take Profit 체크
            if exit_px is None:
                if side == "Buy" and hi >= tp_px:
                    exit_px = apply_slip(tp_px, args.slip_out_bps, side, is_entry=False)
                    reason = "TP"
                elif side == "Sell" and lo <= tp_px:
                    exit_px = apply_slip(tp_px, args.slip_out_bps, side, is_entry=False)
                    reason = "TP"

            # Timeout 체크
            if exit_px is None and (idx - st["i_entry"]) >= args.horizon:
                exit_px = apply_slip(float(g.loc[idx, "close"]), args.slip_out_bps, side, is_entry=False)
                reason = "TIMEOUT"

            if exit_px is None:
                continue

            notional = st["notional"]
            pnl_dir = (exit_px / ent_px - 1.0) * (+1 if side == "Buy" else -1)
            pnl_usd = notional * pnl_dir
            fee_out = fee_cost(notional, FEE_OUT)
            fee_in = st["fee_in"]
            equity += (pnl_usd - fee_out)

            trades.append(dict(
                symbol=sym, ts_in=str(st["entry_ts"]), ts_out=str(ts),
                side=side, entry=ent_px, exit=exit_px,
                pnl_bps=pnl_dir * 1e4, pnl_usd=pnl_usd,
                fee_in=fee_in, fee_out=fee_out, reason=reason
            ))
            to_close.append(sym)

        for sym in to_close:
            open_pos.pop(sym, None)

        # 2) 신규 진입
        if len(open_pos) < args.max_open:
            cands = make_entry_candidates(ts, sym_frames, open_pos, model, args.seq_len, ts_col, args)
            take = max(0, args.max_open - len(open_pos))

            for conf, sym, side, idx_gf in cands[:take]:
                g = sym_frames[sym]["raw"]
                gf = sym_frames[sym]["feat"]

                next_ts = gf.iloc[idx_gf + 1][ts_col]
                nxt = find_row_idx_by_ts(g, ts_col, next_ts)
                if nxt is None:
                    continue

                raw_entry = float(g.loc[nxt, "open"])
                entry_px = apply_slip(raw_entry, args.slip_in_bps, side, is_entry=True)
                tp_px = tp_from_bps(entry_px, args.tp_bps, side)
                sl_px = sl_from_bps(entry_px, args.sl_bps, side)  # ⭐ SL 계산
                notional, margin = entry_notional(equity, args.equity_pct, args.leverage)

                if notional <= 0:
                    continue

                fee_in = fee_cost(notional, FEE_IN)

                open_pos[sym] = dict(
                    entry_ts=g.loc[nxt, ts_col],
                    side=side, entry_px=entry_px,
                    tp_px=tp_px, sl_px=sl_px,  # ⭐ SL 저장
                    notional=notional, margin=margin, fee_in=fee_in,
                    i_entry=nxt
                )
                equity -= fee_in

    # 3) 잔여 포지션 강제 청산
    for sym, st in list(open_pos.items()):
        g = sym_frames[sym]["raw"]
        last_i = len(g) - 1
        exit_px = apply_slip(float(g.loc[last_i, "close"]), args.slip_out_bps, st["side"], is_entry=False)
        notional = st["notional"]
        pnl_dir = (exit_px / st["entry_px"] - 1.0) * (+1 if st["side"] == "Buy" else -1)
        pnl_usd = notional * pnl_dir
        fee_out = fee_cost(notional, FEE_OUT)
        equity += (pnl_usd - fee_out)

        trades.append(dict(
            symbol=sym, ts_in=str(st["entry_ts"]), ts_out=str(g.loc[last_i, ts_col]),
            side=st["side"], entry=st["entry_px"], exit=exit_px,
            pnl_bps=pnl_dir * 1e4, pnl_usd=pnl_usd,
            fee_in=st["fee_in"], fee_out=fee_out,
            reason="FORCE_END"
        ))

    # 결과 출력
    df_tr = pd.DataFrame(trades)
    hit = (df_tr["pnl_bps"] > 0).mean() if len(df_tr) else 0.0
    sum_bps = df_tr["pnl_bps"].sum() if len(df_tr) else 0.0
    fees = (df_tr["fee_in"].sum() + df_tr["fee_out"].sum()) if len(df_tr) else 0.0

    print(f"[FULL-VAL] trades={len(df_tr)} | hit={hit * 100:.2f}% | sum_bps={sum_bps:.1f} | fees=${fees:.2f} | equity {args.equity0:.2f} -> {equity:.2f} (x{equity / args.equity0:.3f})")

    if args.out_csv:
        df_tr.to_csv(args.out_csv, index=False)
        print(f"[SAVED] {args.out_csv}")


if __name__ == "__main__":
    main()