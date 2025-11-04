# bybit_test_tcn_daily_gap.py
# -*- coding: utf-8 -*-
"""
Daily paper trader with GAP TRADING strategy.

Gap Trading Strategy:
- Buy signal with gap > threshold → Short first, then Long at limit
- Sell signal with gap > threshold → Long first, then Sell at limit
- Captures profit from price movement to limit, plus original signal profit
"""

import os, math, json, time, requests, certifi
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ===================== utils/env =====================
def env_str(k, d=""): v = os.getenv(k); return d if v is None or str(v).strip() == "" else str(v).strip()


def env_int(k, d=0):
    v = os.getenv(k)
    try:
        return int(str(v).strip()) if v is not None and str(v).strip() != "" else d
    except:
        return d


def env_float(k, d=0.0):
    v = os.getenv(k)
    try:
        return float(str(v).strip()) if v is not None and str(v).strip() != "" else d
    except:
        return d


def env_bool(k, d=False):
    v = os.getenv(k)
    if v is None: return d
    return str(v).strip().lower() in ("1", "true", "y", "yes", "on")


os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

USE_TESTNET = env_bool("USE_TESTNET", False)
BASE_URL = "https://api-testnet.bybit.com" if USE_TESTNET else "https://api.bybit.com"
CATEGORY = "linear"
INTERVAL_1D = "D"
POLL_SEC = env_float("POLL_SEC", 2.0)
DAY_CUTOFF = env_str("DAY_CUTOFF_UTC", "23:55")
ENTRY_ATR_K = env_float("ENTRY_ATR_K", 0.5)

START_EQUITY = env_float("START_EQUITY", 10000.0)
LEVERAGE = env_float("LEVERAGE", 20.0)  # Paper Trading이니 자유롭게!
ENTRY_EQUITY_PCT = env_float("ENTRY_EQUITY_PCT", 0.20)
TAKER_FEE = env_float("TAKER_FEE", 0.0006)
SLIPPAGE_BPS = env_float("SLIPPAGE_BPS", 1.0)
ENTRY_TOL_BPS = env_float("ENTRY_TOL_BPS", 5.0)

# ===== Gap Trading 설정 =====
ENABLE_GAP_TRADING = env_bool("ENABLE_GAP_TRADING", True)  # Gap Trading 활성화
GAP_MIN_BPS = env_float("GAP_MIN_BPS", 300.0)  # 최소 gap (300 bps = 3%)
GAP_EQUITY_PCT = env_float("GAP_EQUITY_PCT", 0.10)  # Gap 거래 증거금 비율 (10%)
GAP_SL_BPS = env_float("GAP_SL_BPS", 50.0)  # Gap 포지션 Stop Loss (50 bps = 0.5%)


def now_utc_ts() -> int: return int(time.time())


# ===================== Bybit public (minimal) =====================
_ses = requests.Session()
_ses.verify = certifi.where()
_ses.headers.update({"User-Agent": "bybit-daily-gap/1.0", "Accept": "application/json"})


def _get(path: str, params: dict, tries=3, backoff=0.3) -> dict:
    url = BASE_URL + path
    for i in range(tries):
        try:
            r = _ses.get(url, params=params, timeout=8)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff * (i + 1));
                continue
            return r.json()
        except Exception:
            time.sleep(backoff * (i + 1))
    return {}


def get_quote(symbol: str) -> Tuple[float, float, float]:
    r = _get("/v5/market/tickers", {"category": CATEGORY, "symbol": symbol})
    row = ((r.get("result") or {}).get("list") or [{}])[0]
    bid = float(row.get("bid1Price") or 0.0)
    ask = float(row.get("ask1Price") or 0.0)
    last = float(row.get("lastPrice") or 0.0)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else (last or bid or ask)
    return bid, ask, float(mid or 0.0)


def get_kline_daily(symbol: str, limit: int = 400) -> pd.DataFrame:
    r = _get("/v5/market/kline",
             {"category": CATEGORY, "symbol": symbol, "interval": INTERVAL_1D, "limit": min(int(limit), 1000)})
    rows = ((r.get("result") or {}).get("list") or [])
    if not rows: return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    rows = sorted(rows, key=lambda x: int(x[0]))
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", "open", "high", "low", "close", "volume", "turnover"]].dropna().reset_index(drop=True)


# ===================== Model (same as original) =====================
class Chomp1d(nn.Module):
    def __init__(self, c): super().__init__(); self.c = c

    def forward(self, x): return x[:, :, :-self.c].contiguous() if self.c > 0 else x


def wconv(i, o, k, d):
    import torch.nn.utils as U
    pad = (k - 1) * d
    return U.weight_norm(nn.Conv1d(i, o, k, padding=pad, dilation=d))


class Block(nn.Module):
    def __init__(self, i, o, k, d, drop):
        super().__init__()
        self.c1 = wconv(i, o, k, d);
        self.h1 = Chomp1d((k - 1) * d);
        self.r1 = nn.ReLU();
        self.dr1 = nn.Dropout(drop)
        self.c2 = wconv(o, o, k, d);
        self.h2 = Chomp1d((k - 1) * d);
        self.r2 = nn.ReLU();
        self.dr2 = nn.Dropout(drop)
        self.ds = nn.Conv1d(i, o, 1) if i != o else None;
        self.r = nn.ReLU()

    def forward(self, x):
        y = self.dr1(self.r1(self.h1(self.c1(x))))
        y = self.dr2(self.r2(self.h2(self.c2(y))))
        res = x if self.ds is None else self.ds(x)
        return self.r(y + res)


class TCN_MT(nn.Module):
    def __init__(self, in_f, hidden=128, levels=6, k=3, drop=0.2):
        super().__init__()
        L = [];
        ch = in_f
        for i in range(levels):
            L.append(Block(ch, hidden, k, 2 ** i, drop));
            ch = hidden
        self.tcn = nn.Sequential(*L)
        self.head_side = nn.Linear(hidden, 2)
        self.head_tp = nn.Linear(hidden, 1)
        self.head_ttt = nn.Linear(hidden, 1)

    def forward(self, X):
        X = X.transpose(1, 2)
        H = self.tcn(X)[:, :, -1]
        return self.head_side(H), self.head_tp(H), self.head_ttt(H)


def make_daily_feats(df: pd.DataFrame):
    g = df.copy().sort_values("date").reset_index(drop=True)
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g["logc"].diff().fillna(0.0)

    def roll_std(s, w):
        return s.rolling(w, min_periods=max(2, w // 3)).std()

    for w in (5, 10, 20, 60): g[f"rv{w}"] = roll_std(g["ret1"], w)

    def mom_arr(price, w):
        ema = pd.Series(price).ewm(span=w, adjust=False).mean().values
        return price / np.maximum(ema, 1e-12) - 1.0

    for w in (5, 10, 20, 60): g[f"mom{w}"] = mom_arr(g["close"].values, w)
    for w in (10, 20, 60):
        mu = g["volume"].rolling(w, min_periods=max(2, w // 3)).mean()
        sd = g["volume"].rolling(w, min_periods=max(2, w // 3)).std().replace(0, np.nan)
        g[f"vz{w}"] = (g["volume"] - mu) / sd.replace({0: np.nan}).fillna(1.0)
    prev_close = g["close"].shift(1)
    tr = pd.concat([(g["high"] - g["low"]).abs(), (g["high"] - prev_close).abs(), (g["low"] - prev_close).abs()],
                   axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14, min_periods=5).mean()
    feats = ["ret1", "rv5", "rv10", "rv20", "rv60", "mom5", "mom10", "mom20", "mom60", "vz10", "vz20", "vz60", "atr14"]
    g = g.dropna(subset=feats).reset_index(drop=True)
    return g, feats


def load_daily_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck.get("model", ck)
    feat_cols = ck.get("feat_cols", ck.get("features", []))
    meta = ck.get("meta", {})
    seq_len = int(meta.get("seq_len", 60))
    mu = ck.get("scaler_mu", None)
    sd = ck.get("scaler_sd", None)
    in_f = len(feat_cols) if feat_cols else 13
    model = TCN_MT(in_f=in_f).eval()
    model.load_state_dict(state, strict=False)
    return model, feat_cols, seq_len, mu, sd


def _apply_scaler(X: np.ndarray, mu, sd):
    if mu is None or sd is None: return X
    mu = np.asarray(mu, dtype=np.float32);
    sd = np.asarray(sd, dtype=np.float32)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


@torch.no_grad()
def infer_symbol_plan(symbol: str, model, feat_cols: List[str], seq_len: int, mu, sd) -> Optional[dict]:
    df = get_kline_daily(symbol, limit=max(120, seq_len + 30))
    if df.empty: return None
    gf, _ = make_daily_feats(df)
    if len(gf) < seq_len + 1: return None
    X = gf[feat_cols].to_numpy(np.float32) if feat_cols else gf.drop(columns=["date"]).to_numpy(np.float32)
    X = _apply_scaler(X, mu, sd)
    if len(X) < seq_len: return None
    Xseq = X[-seq_len:]
    Xt = torch.from_numpy(Xseq[None, ...])
    logits_side, raw_tp, _ = model(Xt)
    pred = int(logits_side.argmax(dim=1).item())
    side = "Sell" if pred == 0 else "Buy"
    tp_bps = float(raw_tp[0, 0].item())
    cl = float(gf["close"].iloc[-1])
    atr = float(gf["atr14"].iloc[-1])
    limit = cl - ENTRY_ATR_K * atr if side == "Buy" else cl + ENTRY_ATR_K * atr
    if tp_bps > 0:
        tp = limit * (1.0 + tp_bps / 1e4) if side == "Buy" else limit * (1.0 - tp_bps / 1e4)
    else:
        tp = limit
    return {"side": side, "limit": float(limit), "tp": float(tp), "tp_bps": float(tp_bps),
            "close": float(cl), "atr": float(atr), "atr_ratio": (atr / max(cl, 1e-12))}


# ===================== Paper Broker with Gap Trading =====================
class Paper:
    def __init__(self, equity: float, lev: float, pct: float, fee: float, slip: float):
        self.base_equity = equity
        self.equity = equity  # Available Balance
        self.lev = lev
        self.pct = pct
        self.fee = fee
        self.slip = slip
        self.pos: Dict[str, dict] = {}
        self.gap_pos: Dict[str, dict] = {}
        self.trades = []
        self.gap_trades = []

    def get_locked_margin(self) -> float:
        """현재 locked된 담보금 계산"""
        locked = 0.0
        for p in self.pos.values():
            locked += p["alloc"]
        for p in self.gap_pos.values():
            locked += p["alloc"]
        return locked

    def get_margin_info(self, prices: Dict[str, float]) -> dict:
        """거래소 스타일 자산 정보"""
        locked_margin = self.get_locked_margin()
        unrealized_pnl = 0.0

        for sym, p in self.pos.items():
            m = prices.get(sym, 0.0)
            if m > 0:
                if p["side"] == "Buy":
                    unrealized_pnl += (m - p["entry"]) * p["qty"]
                else:
                    unrealized_pnl += (p["entry"] - m) * p["qty"]

        for sym, p in self.gap_pos.items():
            m = prices.get(sym, 0.0)
            if m > 0:
                if p["side"] == "Buy":
                    unrealized_pnl += (m - p["entry"]) * p["qty"]
                else:
                    unrealized_pnl += (p["entry"] - m) * p["qty"]

        # equity = Available (담보금이 차감된 상태)
        # Total = Available + Locked + uPnL
        total_equity = self.equity + locked_margin + unrealized_pnl
        roi = ((total_equity - self.base_equity) / max(self.base_equity, 1e-12)) * 100.0

        return {
            "available": self.equity,
            "locked_margin": locked_margin,
            "unrealized_pnl": unrealized_pnl,
            "total_equity": total_equity,
            "roi": roi
        }

    def try_open(self, sym: str, side: str, limit: float, mid: float, is_gap: bool = False) -> bool:
        """포지션 진입 - 거래소 방식"""
        # 담보금 계산 (전체 equity 기준)
        alloc = (GAP_EQUITY_PCT if is_gap else self.pct) * self.equity

        if alloc <= 0:
            print(f"\n⚠️ [SKIP] {sym} | No equity available")
            return False

        notional = alloc * self.lev
        qty = notional / max(limit, 1e-12)
        fee_cost = notional * self.fee
        slip_cost = notional * (self.slip / 1e4)
        cost = fee_cost + slip_cost

        # 담보금 + 수수료 확인
        total_required = alloc + cost
        if total_required > self.equity:
            print(f"\n⚠️ [SKIP] {sym} | Insufficient equity: ${total_required:.2f} > ${self.equity:.2f}")
            return False

        # 담보금 + 수수료 실제 차감
        self.equity -= total_required

        pos_dict = self.gap_pos if is_gap else self.pos
        pos_dict[sym] = {
            "side": side, "entry": limit, "qty": qty,
            "alloc": alloc, "notional": notional, "fee": fee_cost, "slip": slip_cost
        }

        pos_type = "Gap" if is_gap else "Main"
        print(
            f"    └─ [{pos_type}] Alloc: ${alloc:.2f} | Notional: ${notional:.2f} (Lev {self.lev}x) | Qty: {qty:.6f} | Fee: ${cost:.2f}")
        return True

    def try_close(self, sym: str, exit_price: float, is_gap: bool = False) -> Optional[dict]:
        """포지션 청산 - 거래소 방식"""
        pos_dict = self.gap_pos if is_gap else self.pos
        p = pos_dict.get(sym)
        if not p:
            return None

        side = p["side"]
        if side == "Buy":
            pnl = (exit_price - p["entry"]) * p["qty"]
        else:
            pnl = (p["entry"] - exit_price) * p["qty"]

        exit_notional = exit_price * p["qty"]
        exit_fee = exit_notional * self.fee
        exit_slip = exit_notional * (self.slip / 1e4)
        pnl -= (exit_fee + exit_slip)

        # 담보금 반환 + PnL 적용
        self.equity += (p["alloc"] + pnl)

        bps = (pnl / max(p["notional"], 1e-12)) * 1e4
        roi_alloc = (pnl / max(p["alloc"], 1e-12)) * 100.0
        roi_equity = (pnl / max(self.base_equity, 1e-12)) * 100.0

        result = {
            "symbol": sym, "side": side, "entry": p["entry"], "exit": exit_price,
            "qty": p["qty"], "pnl": pnl, "bps": bps,
            "roi_alloc_pct": roi_alloc, "roi_equity_pct": roi_equity, "is_gap": is_gap
        }

        if is_gap:
            self.gap_trades.append(result)
        else:
            self.trades.append(result)

        del pos_dict[sym]
        return result

    def mtm(self, prices: Dict[str, float]) -> Tuple[float, float]:
        """Mark-to-Market 계산"""
        info = self.get_margin_info(prices)
        return info["total_equity"], info["roi"]


# ===================== Gap Trading Logic =====================
def should_use_gap_trading(plan: dict, current_price: float) -> Tuple[bool, float, str]:
    """Gap Trading 사용 여부 결정"""
    if not ENABLE_GAP_TRADING:
        return False, 0.0, ""

    limit = plan["limit"]
    gap_bps = abs((limit - current_price) / max(current_price, 1e-12)) * 1e4

    if gap_bps < GAP_MIN_BPS:
        return False, gap_bps, "gap too small"

    # Gap 방향 결정
    if plan["side"] == "Buy":
        # Buy 신호 → limit이 현재가보다 낮음 → Short 먼저
        gap_side = "Sell"
    else:
        # Sell 신호 → limit이 현재가보다 높음 → Long 먼저
        gap_side = "Buy"

    return True, gap_bps, gap_side


# ===================== helpers =====================
def within_tol(mid: float, target: float, tol_bps: float) -> Tuple[bool, float]:
    if tol_bps <= 0 or mid <= 0 or target <= 0: return (False, 0.0)
    dev_bps = ((mid / target) - 1.0) * 1e4
    return (abs(dev_bps) <= tol_bps, dev_bps)


# ===================== Daily run with Gap Trading =====================
def run_day_once(symbols: List[str], plan: Dict[str, dict]):
    print(f"\n{'=' * 100}")
    print(f"🚀 GAP TRADING STRATEGY")
    print(f"{'=' * 100}")
    print(f"[CONFIG] Gap Trading: {'ENABLED ✅' if ENABLE_GAP_TRADING else 'DISABLED ❌'}")
    print(f"[CONFIG] Gap Min: {GAP_MIN_BPS:.0f} bps ({GAP_MIN_BPS / 100:.1f}%)")
    print(f"[CONFIG] Gap Equity: {GAP_EQUITY_PCT * 100:.0f}%")
    print(f"[CONFIG] Gap SL: {GAP_SL_BPS:.0f} bps ({GAP_SL_BPS / 100:.1f}%)")
    print(f"[START] Daily run on UTC {time.strftime('%Y-%m-%d')} | Symbols: {symbols}")
    print(f"{'=' * 100}\n")

    cutoff = utc_seconds_until(DAY_CUTOFF)
    brok = Paper(START_EQUITY, LEVERAGE, ENTRY_EQUITY_PCT, TAKER_FEE, SLIPPAGE_BPS)
    filled: Dict[str, dict] = {}
    gap_active: Dict[str, dict] = {}  # Gap 포지션 추적
    closed_results = []

    while True:
        now = int(time.time())
        if now >= cutoff:
            # 모든 포지션 청산
            for sym in list(brok.pos.keys()):
                _, _, m = get_quote(sym)
                if m > 0:
                    res = brok.try_close(sym, m)
                    if res: closed_results.append(res)

            for sym in list(brok.gap_pos.keys()):
                _, _, m = get_quote(sym)
                if m > 0:
                    res = brok.try_close(sym, m, is_gap=True)
                    if res: closed_results.append(res)

            print("\n[END] Cutoff reached. Closing all positions.")
            break

        prices = {}
        for s in symbols:
            bid, ask, mid = get_quote(s)
            prices[s] = mid
            pl = plan.get(s)
            if not pl: continue

            limit = float(pl["limit"])
            tp = float(pl["tp"])
            side = pl["side"]

            # === Gap Trading 로직 ===
            if s not in brok.gap_pos and s not in brok.pos:
                use_gap, gap_bps, gap_side = should_use_gap_trading(pl, mid)

                if use_gap and isinstance(gap_side, str):
                    # Gap 포지션 진입
                    if brok.try_open(s, gap_side, mid, mid, is_gap=True):
                        # SL 방향 수정: Buy는 아래, Sell은 위
                        if gap_side == "Buy":
                            # Long → 하락 시 손실 → SL은 아래
                            sl_price = mid * (1.0 - GAP_SL_BPS / 1e4)
                        else:  # gap_side == "Sell"
                            # Short → 상승 시 손실 → SL은 위
                            sl_price = mid * (1.0 + GAP_SL_BPS / 1e4)

                        gap_active[s] = {
                            "gap_side": gap_side,
                            "gap_entry": mid,
                            "target_limit": limit,
                            "original_side": side,
                            "original_tp": tp,
                            "sl": sl_price
                        }

                        sl_dist = abs((sl_price - mid) / mid) * 100
                        print(
                            f"\n🎯 [GAP] {s} | {gap_side} @ ${mid:.6f} | Gap: {gap_bps:.1f} bps ({gap_bps / 100:.2f}%) | Target: ${limit:.6f} | SL: ${sl_price:.6f} ({sl_dist:.2f}%)")

            # === Gap Stop Loss 체크 ===
            if s in brok.gap_pos and s in gap_active:
                ga = gap_active[s]
                gap_side = ga["gap_side"]
                sl = ga["sl"]
                gap_entry = ga["gap_entry"]

                hit_sl = False
                if gap_side == "Buy":
                    # Long → 하락하면 손실 → mid <= sl
                    if mid <= sl:
                        hit_sl = True
                        print(f"\n⚠️ [GAP SL CHECK] {s} | Long | Mid ${mid:.6f} <= SL ${sl:.6f} → HIT!")
                elif gap_side == "Sell":
                    # Short → 상승하면 손실 → mid >= sl
                    if mid >= sl:
                        hit_sl = True
                        print(f"\n⚠️ [GAP SL CHECK] {s} | Short | Mid ${mid:.6f} >= SL ${sl:.6f} → HIT!")

                if hit_sl:
                    res = brok.try_close(s, mid, is_gap=True)
                    if res:
                        loss_pct = res['roi_alloc_pct']
                        print(
                            f"❌ [GAP SL] {s} | {gap_side} | Entry: ${gap_entry:.6f} → Exit: ${mid:.6f} | PnL: ${res['pnl']:.2f} ({loss_pct:+.2f}% of alloc)")
                        closed_results.append(res)
                        del gap_active[s]
                        plan.pop(s, None)
                    continue

            # === Gap Limit 도달 체크 (전환) ===
            if s in brok.gap_pos and s in gap_active:
                ga = gap_active[s]
                gap_side = ga["gap_side"]
                target_limit = ga["target_limit"]

                limit_reached = False
                if gap_side == "Sell" and mid <= target_limit:  # Short → 하락
                    limit_reached = True
                elif gap_side == "Buy" and mid >= target_limit:  # Long → 상승
                    limit_reached = True

                if limit_reached:
                    # Gap 청산
                    res = brok.try_close(s, mid, is_gap=True)
                    if res:
                        print(
                            f"\n✅ [GAP CLOSE] {s} | {gap_side} | Entry: ${ga['gap_entry']:.6f} → Exit: ${mid:.6f} | PnL: ${res['pnl']:.2f} ({res['bps']:.1f} bps)")
                        closed_results.append(res)

                    # 원래 신호로 전환
                    original_side = ga["original_side"]
                    original_tp = ga["original_tp"]
                    if brok.try_open(s, original_side, target_limit, mid):
                        filled[s] = {"side": original_side, "limit": target_limit, "tp": original_tp}
                        print(f"🔄 [FLIP] {s} | {original_side} @ ${target_limit:.6f} | TP: ${original_tp:.6f}")

                    del gap_active[s]
                    continue

            # === 기존 Limit 진입 로직 (Gap 없이) ===
            if s not in brok.pos and s not in brok.gap_pos:
                trig = False
                if side == "Buy":
                    tol_ok, dev = within_tol(mid, limit, ENTRY_TOL_BPS)
                    trig = (bid <= limit) or (mid <= limit) or tol_ok
                    if trig:
                        if brok.try_open(s, "Buy", limit, mid):
                            print(f"\n📈 [ENTRY] {s} | Buy @ ${limit:.6f} | Mid: ${mid:.6f} | TP: ${tp:.6f}")
                            filled[s] = {"side": "Buy", "limit": limit, "tp": tp}
                else:
                    tol_ok, dev = within_tol(mid, limit, ENTRY_TOL_BPS)
                    trig = (ask >= limit) or (mid >= limit) or tol_ok
                    if trig:
                        if brok.try_open(s, "Sell", limit, mid):
                            print(f"\n📉 [ENTRY] {s} | Sell @ ${limit:.6f} | Mid: ${mid:.6f} | TP: ${tp:.6f}")
                            filled[s] = {"side": "Sell", "limit": limit, "tp": tp}

            # === TP 모니터 ===
            if s in brok.pos:
                fs = filled.get(s)
                if fs:
                    tp_reached = False
                    if fs["side"] == "Buy" and mid >= fs["tp"]:
                        tp_reached = True
                    elif fs["side"] == "Sell" and mid <= fs["tp"]:
                        tp_reached = True

                    if tp_reached:
                        res = brok.try_close(s, mid)
                        if res:
                            print(
                                f"\n💰 [TP CLOSE] {s} | {fs['side']} | Entry: ${fs['limit']:.6f} → Exit: ${mid:.6f} | PnL: ${res['pnl']:.2f} ({res['bps']:.1f} bps)")
                            closed_results.append(res)
                            plan.pop(s, None)

        mtm_eq, mtm_roi = brok.mtm(prices)
        margin_info = brok.get_margin_info(prices)
        gap_count = len(brok.gap_pos)
        main_count = len(brok.pos)

        # 실시간 포지션 상태 (10초마다 상세 출력)
        if int(time.time()) % 10 == 0 and (gap_count > 0 or main_count > 0):
            print(f"\n\n{'=' * 100}")
            print(f"📊 LIVE POSITIONS")
            print(f"{'=' * 100}")
            print(f"💰 Total Equity: ${margin_info['total_equity']:.2f} | ROI: {margin_info['roi']:+.2f}%")
            print(f"   ├─ Available: ${margin_info['available']:.2f}")
            print(f"   ├─ Locked Margin: ${margin_info['locked_margin']:.2f}")
            print(f"   └─ Unrealized PnL: ${margin_info['unrealized_pnl']:+.2f}")
            print(f"{'=' * 100}")

            if gap_count > 0:
                print(f"\n💎 GAP POSITIONS ({gap_count}):")
                print(f"{'-' * 100}")
                for sym, p in brok.gap_pos.items():
                    mid = prices.get(sym, 0)
                    if mid > 0:
                        if p['side'] == 'Buy':
                            pnl = (mid - p['entry']) * p['qty']
                            pnl_pct = ((mid / p['entry']) - 1) * 100
                        else:
                            pnl = (p['entry'] - mid) * p['qty']
                            pnl_pct = ((p['entry'] / mid) - 1) * 100

                        ga = gap_active.get(sym, {})
                        target = ga.get('target_limit', 0)
                        sl = ga.get('sl', 0)

                        if target > 0:
                            if p['side'] == 'Buy':
                                target_dist = ((target / mid) - 1) * 100
                            else:
                                target_dist = ((mid / target) - 1) * 100
                        else:
                            target_dist = 0

                        emoji = "🟢" if pnl > 0 else "🔴"
                        print(f"{emoji} {sym:12} {p['side']:4} | Entry: ${p['entry']:.6f} → Now: ${mid:.6f} | "
                              f"PnL: ${pnl:+7.2f} ({pnl_pct:+.2f}%) | To Target: {target_dist:+.2f}%")
                print(f"{'-' * 100}\n")

            if main_count > 0:
                print(f"📈 MAIN POSITIONS ({main_count}):")
                print(f"{'-' * 100}")
                for sym, p in brok.pos.items():
                    mid = prices.get(sym, 0)
                    if mid > 0:
                        if p['side'] == 'Buy':
                            pnl = (mid - p['entry']) * p['qty']
                            pnl_pct = ((mid / p['entry']) - 1) * 100
                        else:
                            pnl = (p['entry'] - mid) * p['qty']
                            pnl_pct = ((p['entry'] / mid) - 1) * 100

                        fs = filled.get(sym, {})
                        tp = fs.get('tp', 0)

                        if tp > 0:
                            if p['side'] == 'Buy':
                                tp_dist = ((tp / mid) - 1) * 100
                            else:
                                tp_dist = ((mid / tp) - 1) * 100
                        else:
                            tp_dist = 0

                        emoji = "🟢" if pnl > 0 else "🔴"
                        print(f"{emoji} {sym:12} {p['side']:4} | Entry: ${p['entry']:.6f} → Now: ${mid:.6f} | "
                              f"PnL: ${pnl:+7.2f} ({pnl_pct:+.2f}%) | To TP: {tp_dist:+.2f}%")
                print(f"{'-' * 100}\n")

            print(f"{'=' * 100}\n")

        # 간단한 상태 표시 (매 루프)
        print(
            f"\r[LIVE] {time.strftime('%H:%M:%S')} | Total: ${margin_info['total_equity']:.2f} (Avail: ${margin_info['available']:.2f} | Margin: ${margin_info['locked_margin']:.2f} | uPnL: ${margin_info['unrealized_pnl']:+.2f}) | ROI: {margin_info['roi']:+.2f}% | Gap: {gap_count} | Main: {main_count}      ",
            end="", flush=True)
        time.sleep(max(0.5, POLL_SEC))

    # === 결과 출력 ===
    # 최종 가격으로 마진 정보 계산
    final_prices = {}
    for s in symbols:
        _, _, mid = get_quote(s)
        final_prices[s] = mid

    final_info = brok.get_margin_info(final_prices)

    print(f"\n\n{'=' * 100}")
    print(f"📊 FINAL RESULTS")
    print(f"{'=' * 100}")
    print(f"Start Equity: ${START_EQUITY:.2f}")
    print(f"Final Total Equity: ${final_info['total_equity']:.2f}")
    print(f"   ├─ Available: ${final_info['available']:.2f}")
    print(f"   ├─ Locked Margin: ${final_info['locked_margin']:.2f}")
    print(f"   └─ Unrealized PnL: ${final_info['unrealized_pnl']:+.2f}")
    print(f"ROI: {final_info['roi']:+.2f}%")
    print(f"{'=' * 100}\n")

    if brok.gap_trades:
        print(f"💎 GAP TRADES ({len(brok.gap_trades)})")
        print(f"{'-' * 100}")
        gap_pnl = 0.0
        for r in brok.gap_trades:
            gap_pnl += r['pnl']
            emoji = "✅" if r['pnl'] > 0 else "❌"
            print(
                f"{emoji} {r['symbol']:12} {r['side']:4} | Entry: ${r['entry']:.6f} → Exit: ${r['exit']:.6f} | PnL: ${r['pnl']:+7.2f} ({r['bps']:+6.1f} bps)")
        print(f"{'-' * 100}")
        print(f"Gap Total PnL: ${gap_pnl:+.2f}\n")

    if brok.trades:
        print(f"📈 MAIN TRADES ({len(brok.trades)})")
        print(f"{'-' * 100}")
        main_pnl = 0.0
        for r in brok.trades:
            main_pnl += r['pnl']
            emoji = "✅" if r['pnl'] > 0 else "❌"
            print(
                f"{emoji} {r['symbol']:12} {r['side']:4} | Entry: ${r['entry']:.6f} → Exit: ${r['exit']:.6f} | PnL: ${r['pnl']:+7.2f} ({r['bps']:+6.1f} bps)")
        print(f"{'-' * 100}")
        print(f"Main Total PnL: ${main_pnl:+.2f}\n")

    print(f"{'=' * 100}")


def utc_seconds_until(hhmm: str) -> int:
    try:
        hh, mm = [int(x) for x in hhmm.split(":")]
    except Exception:
        hh, mm = 23, 55
    t = time.gmtime()
    cutoff = time.struct_time((t.tm_year, t.tm_mon, t.tm_mday, hh, mm, 0, t.tm_wday, t.tm_yday, t.tm_isdst))
    cutoff_ts = int(time.mktime(cutoff))
    if cutoff_ts <= int(time.time()): cutoff_ts += 24 * 3600
    return cutoff_ts


# ===================== main =====================
if __name__ == "__main__":
    ckpt = env_str("DAILY_CKPT", "./models_daily_v2/daily_simple_best.ckpt")
    symbols = [s.strip().upper() for s in
               env_str("SYMBOLS", "BTCUSDT,BNBUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,XRPUSDT,ADAUSDT").split(",") if s.strip()]

    print(f"\n🎯 GAP TRADING DAILY PLAN")
    print(f"Date: {time.strftime('%Y-%m-%d')}")
    print(f"Checkpoint: {ckpt}")
    print(f"Symbols: {symbols}\n")

    # Build plan
    model, feat_cols, seq_len, mu, sd = load_daily_model(ckpt)
    plan = {}

    print("=" * 100)
    print("📋 PLAN GENERATION")
    print("=" * 100)

    for s in symbols:
        try:
            pl = infer_symbol_plan(s, model, feat_cols, seq_len, mu, sd)
            if pl:
                plan[s] = pl
                gap_bps = abs((pl["limit"] - pl["close"]) / max(pl["close"], 1e-12)) * 1e4
                gap_side = "Short first" if pl["side"] == "Buy" else "Long first"
                gap_status = f"✅ {gap_side} (Gap: {gap_bps:.1f} bps)" if gap_bps >= GAP_MIN_BPS else f"❌ No gap ({gap_bps:.1f} bps)"

                print(
                    f"{s:12} | {pl['side']:4} @ ${pl['limit']:.6f} | Current: ${pl['close']:.6f} | TP: ${pl['tp']:.6f} ({pl['tp_bps']:.0f} bps) | {gap_status}")
        except Exception as e:
            print(f"{s:12} | ERROR: {e}")

    print("=" * 100 + "\n")

    run_day_once(symbols, plan)