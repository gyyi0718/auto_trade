# -*- coding: utf-8 -*-
"""
strategy_hooks.py
- 신호 방향/예측/점수화/TP구성 같은 '실험용' 로직을 분리
- 이 파일만 바꿔도 엔진/브로커 건드릴 필요 없음
"""

import math, numpy as np
from typing import Tuple, List

# ===== 공용 유틸 =====
def bps_from_to(a: float, b: float) -> float:
    if a <= 0: return 0.0
    return (b - a) / a * 10_000.0

def opp(side: str) -> str:
    return "Sell" if side == "Buy" else "Buy"

def roi(side: str, entry: float, price: float) -> float:
    return (price/entry - 1.0) if side=="Buy" else (entry/price - 1.0)

def phi(z: float)->float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ===== 전략: 방향/목표가 간단 모멘텀 =====
def predict_future_price(symbol: str, df, tp1_bps: float, use_deepar: bool,
                         ckpt_path: str, seq_len: int, pred_len: int) -> Tuple[float,int,float]:
    """
    return: (target_price, horizon_sec, confidence[0~1])
    - 기본: 모멘텀 (가벼움/의존성 無)
    - USE_DEEPAR=1 이면 DeepAR로 대체 (선택)
    """
    now = float(df["close"].iloc[-1])
    s = df["close"].values
    ma_fast = s[-10:].mean() if len(s) >= 10 else s.mean()
    ma_slow = s[-30:].mean() if len(s) >= 30 else s.mean()
    drift = np.sign(s[-1] - s[-5]) if len(s) >= 5 else 0.0
    bias = 1 if (ma_fast > ma_slow and drift >= 0) else -1
    step = 1.0 + (tp1_bps/10_000.0)
    target = now * (step if bias>0 else 1.0/step)
    conf = 0.80 if bias != 0 else 0.60

    if use_deepar:
        try:
            from pytorch_forecasting import DeepAR, TimeSeriesDataSet
            from pytorch_forecasting.data import GroupNormalizer
            import torch
            x = df.sort_values("timestamp").copy()
            x["time_idx"] = np.arange(len(x))
            x["series_id"] = symbol
            x["log_return"] = np.log(x["close"]).diff().fillna(0.0)
            enc = x.iloc[-seq_len:].copy()
            enc["time_idx"] -= enc["time_idx"].min()
            fut = {"time_idx": np.arange(seq_len, seq_len+pred_len),
                   "series_id": symbol, "log_return":[0.0]*pred_len}
            import pandas as pd
            comb = pd.concat([enc, pd.DataFrame(fut)], ignore_index=True)
            ds = TimeSeriesDataSet(
                comb, time_idx="time_idx", target="log_return", group_ids=["series_id"],
                max_encoder_length=seq_len, max_prediction_length=pred_len,
                time_varying_known_reals=["time_idx"],
                time_varying_unknown_reals=["log_return"],
                static_categoricals=["series_id"],
                target_normalizer=GroupNormalizer(groups=["series_id"]),
                add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
            )
            dl = ds.to_dataloader(train=False, batch_size=1)
            model = DeepAR.load_from_checkpoint(ckpt_path)
            q_raw = model.predict(dl, mode="quantiles",
                                  mode_kwargs={"quantiles":[0.1,0.5,0.9]},
                                  return_x=False)
            arr = _extract_quantile_array(q_raw)
            q10 = arr[0,:,0]; q50 = arr[0,:,1]; q90 = arr[0,:,2]

            def to_price_path(logret_vec):
                lr_cum = np.cumsum(logret_vec)
                return now * np.exp(lr_cum)

            p10 = to_price_path(q10); p50 = to_price_path(q50); p90 = to_price_path(q90)
            best_idx, best_gain, best_tp = 0, -1.0, None
            for i in range(len(p50)):
                raw_tp = float(p50[i])
                side = "Buy" if raw_tp > now else "Sell"
                # 최소 TP1 만큼은 움직이도록 보정
                adj = _adjust_tp_preview(side, now, raw_tp, tp1_bps)
                gain = abs(adj - now)
                if gain > best_gain:
                    best_gain, best_idx, best_tp = gain, i, adj
            band = float(p90[best_idx] - p10[best_idx])
            rel_unc = band / max(now, 1e-9)
            conf = float(np.clip(1.0 - (rel_unc / 0.01), 0.0, 1.0))
            return float(best_tp), int((best_idx + 1) * 60), conf
        except Exception:
            pass

    return float(target), 60, conf


def _extract_quantile_array(pred_out):
    import numpy as _np, torch as _torch
    if isinstance(pred_out, dict):
        for k in ("prediction","predictions","output"):
            if k in pred_out: pred_out = pred_out[k]; break
    if isinstance(pred_out, list):
        pred_out = _np.array(pred_out)
    if isinstance(pred_out, _torch.Tensor):
        arr = pred_out.detach().cpu().numpy()
    elif isinstance(pred_out, _np.ndarray):
        arr = pred_out
    else:
        arr = _np.array(pred_out)
    if arr.ndim == 2:
        if arr.shape[1] in (3,5): arr = arr[None,:,:]
        elif arr.shape[0] in (3,5): arr = arr.transpose(1,0)[None,:,:]
        else: arr = arr[None,:,:]
    elif arr.ndim == 3:
        B,A,C = arr.shape
        if C in (3,5): pass
        elif A in (3,5): arr = arr.transpose(0,2,1)
    else:
        raise RuntimeError(f"Unexpected prediction shape: {arr.shape}")
    return arr

def _adjust_tp_preview(side: str, now_price: float, predicted_target: float, tp1_bps: float)->float:
    step = 1.0 + (tp1_bps / 10_000.0)
    if side == "Buy":
        return max(predicted_target, round(now_price * step, 6))
    else:
        return min(predicted_target, round(now_price / step, 6))


# ===== 자질/체결 가드: 스프레드/갭 =====
def pass_quality_filters(bid: float, ask: float, mid: float,
                         last_close: float, spread_reject_bps: float, gap_reject_bps: float) -> bool:
    if bid > 0 and ask > 0 and mid > 0:
        spread_bps = (ask - bid) / mid * 10_000.0
        if spread_bps > spread_reject_bps:
            return False
    gap_bps = abs(mid - last_close) / max(last_close, 1e-9) * 10_000.0
    if gap_bps > gap_reject_bps:
        return False
    return True


# ===== 점수화(Kelly/Sharpe) & 포지션 크기 =====
def score_signal_with_kelly_sharpe(now_price: float, target_price: float, side: str,
                                   horizon_sec: int, est_qty: float, sigma_per_min: float,
                                   confidence: float, conf_min: float, pnl_min_usd: float,
                                   sharpe_min: float, kelly_cap: float) -> dict:
    STOP_RATIO = 1.0
    sign = 1.0 if side=="Buy" else -1.0
    dist = (target_price - now_price) * sign
    exp_usd = max(0.0, dist) * float(est_qty)
    if dist <= 0:
        return {"ok": False, "reason": "target not favorable", "exp_usd": 0.0, "sharpe": 0.0, "kelly_frac": 0.0, "score": 0.0}

    horizon_min = max(1.0, horizon_sec / 60.0)
    exp_ret_pct = dist / max(now_price, 1e-9)
    sigma_h = max(1e-9, sigma_per_min * math.sqrt(horizon_min))
    sharpe = exp_ret_pct / sigma_h

    p_win = float(np.clip(phi(sharpe), 0.0, 1.0))
    R = 1.0 / max(1e-9, STOP_RATIO)
    q = 1.0 - p_win
    kelly_raw = p_win - (q / R)
    kelly_frac = float(np.clip(kelly_raw, 0.0, kelly_cap))

    sharpe_norm = float(np.clip(sharpe / 1.0, 0.0, 1.0))
    pnl_norm = float(np.clip(exp_usd / max(1e-9, 5.0 * pnl_min_usd), 0.0, 1.0))
    score = 0.4*sharpe_norm + 0.4*float(np.clip(confidence,0.0,1.0)) + 0.2*pnl_norm

    if confidence < conf_min:
        return {"ok": False, "reason": f"conf<{conf_min}", "exp_usd": exp_usd, "sharpe": sharpe, "kelly_frac": kelly_frac, "score": score}
    if exp_usd < pnl_min_usd:
        return {"ok": False, "reason": f"pnl<{pnl_min_usd}$", "exp_usd": exp_usd, "sharpe": sharpe, "kelly_frac": kelly_frac, "score": score}
    if (sharpe_min is not None) and sharpe < sharpe_min:
        return {"ok": False, "reason": f"sharpe<{sharpe_min}", "exp_usd": exp_usd, "sharpe": sharpe, "kelly_frac": kelly_frac, "score": score}

    return {"ok": True, "reason": "ok", "exp_usd": exp_usd,
            "sharpe": sharpe, "kelly_frac": kelly_frac, "score": score}


# ===== TP/SL 티어 구성 =====
def build_tp_tiers(entry: float, side: str,
                   tp1_bps: float, tp2_bps: float, tp3_bps: float,
                   tp1_ratio: float, tp2_ratio: float) -> Tuple[List[float], List[float]]:
    def tp_from_bps(bps: float) -> float:
        return entry * (1.0 + bps/10000.0) if side=="Buy" else entry * (1.0 - bps/10000.0)
    tp_prices = [tp_from_bps(tp1_bps), tp_from_bps(tp2_bps), tp_from_bps(tp3_bps)]
    tp_ratios = [tp1_ratio, tp2_ratio, 1.0]   # TP3는 남은 전량
    return tp_prices, tp_ratios


# ===== 모드에 따른 방향 뒤집기 =====
def pick_side_with_mode(base_side: str, model_mode: str) -> str:
    import random
    if model_mode == "model":
        return base_side
    if model_mode == "inverse":
        return opp(base_side)
    if model_mode == "random":
        return random.choice(["Buy","Sell"])
    return base_side
