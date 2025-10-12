# bybit_test_tcn_daily.py
# -*- coding: utf-8 -*-
"""
Daily one-shot paper trader (limit entry + TP) driven by a trained daily TCN model.

What it does
------------
1) Loads your daily checkpoint (from train_daily_limit_tp.py).
2) For each symbol, infers:
   - side (LONG/SHORT)
   - target TP in bps (converted to price from the planned limit price)
   - a reasonable LIMIT price using ATR (configurable).
3) Two modes:
   - WRITE_PLAN_JSON=1  → just write daily_plan.json and exit.
   - WRITE_PLAN_JSON=0  → place the (paper) limit, poll real-time quotes, fill when touched,
                          then close at TP or by day cutoff (UTC) and print PnL.

Environment (PowerShell example)
--------------------------------
$env:DAILY_CKPT=".\models_daily\daily_limit_tp_best.ckpt"
$env:SYMBOLS="BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT"
$env:WRITE_PLAN_JSON="0"
python .\bybit_test_tcn_daily.py

Key env vars
------------
DAILY_CKPT:      path to ckpt
SYMBOLS:         comma list
USE_TESTNET:     0/1
POLL_SEC:        quote poll seconds (default 2)
ENTRY_ATR_K:     fraction of ATR for limit offset (default 0.5)
DAY_CUTOFF_UTC:  HH:MM (default 23:55)
START_EQUITY:    paper equity (default 10000)
LEVERAGE:        (default 25)
ENTRY_EQUITY_PCT:(default 0.20)   ← 20%로 변경
TAKER_FEE:       (default 0.0006)
SLIPPAGE_BPS:    bps per fill assumed (default 1.0)
"""

import os, math, json, time, requests, certifi
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ===================== utils/env =====================
def env_str(k, d=""): v=os.getenv(k); return d if v is None or str(v).strip()=="" else str(v).strip()
def env_int(k, d=0):
    v=os.getenv(k)
    try: return int(str(v).strip()) if v is not None and str(v).strip()!="" else d
    except: return d
def env_float(k, d=0.0):
    v=os.getenv(k)
    try: return float(str(v).strip()) if v is not None and str(v).strip()!="" else d
    except: return d
def env_bool(k, d=False):
    v=os.getenv(k)
    if v is None: return d
    return str(v).strip().lower() in ("1","true","y","yes","on")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

USE_TESTNET   = env_bool("USE_TESTNET", False)
BASE_URL      = "https://api-testnet.bybit.com" if USE_TESTNET else "https://api.bybit.com"
CATEGORY      = "linear"
INTERVAL_1D   = "D"
POLL_SEC      = env_float("POLL_SEC", 2.0)
DAY_CUTOFF    = env_str("DAY_CUTOFF_UTC", "23:55")   # HH:MM (UTC)
ENTRY_ATR_K   = env_float("ENTRY_ATR_K", 0.5)        # fraction of ATR used for limit offset

START_EQUITY     = env_float("START_EQUITY", 10_000.0)
LEVERAGE         = env_float("LEVERAGE", 25.0)
ENTRY_EQUITY_PCT = env_float("ENTRY_EQUITY_PCT", 0.20)  # ← 기본 20%
TAKER_FEE        = env_float("TAKER_FEE", 0.0006)
SLIPPAGE_BPS     = env_float("SLIPPAGE_BPS", 1.0)    # per trade side

def now_utc_ts() -> int: return int(time.time())

# ===================== Bybit public (minimal) =====================
_ses = requests.Session()
_ses.verify = certifi.where()
_ses.headers.update({"User-Agent":"bybit-daily-paper/1.0","Accept":"application/json"})

def _get(path: str, params: dict, tries=3, backoff=0.3) -> dict:
    url = BASE_URL + path
    for i in range(tries):
        try:
            r = _ses.get(url, params=params, timeout=8)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff*(i+1)); continue
            return r.json()
        except Exception:
            time.sleep(backoff*(i+1))
    return {}

def get_quote(symbol: str) -> Tuple[float,float,float]:
    r = _get("/v5/market/tickers", {"category": CATEGORY, "symbol": symbol})
    row = ((r.get("result") or {}).get("list") or [{}])[0]
    bid = float(row.get("bid1Price") or 0.0)
    ask = float(row.get("ask1Price") or 0.0)
    last= float(row.get("lastPrice") or 0.0)
    mid = (bid+ask)/2.0 if bid>0 and ask>0 else (last or bid or ask)
    return bid, ask, float(mid or 0.0)

def get_kline_daily(symbol: str, limit: int = 400) -> pd.DataFrame:
    r = _get("/v5/market/kline", {"category": CATEGORY, "symbol": symbol, "interval": INTERVAL_1D, "limit": min(int(limit),1000)})
    rows = ((r.get("result") or {}).get("list") or [])
    if not rows: return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    rows = sorted(rows, key=lambda x: int(x[0]))
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume","turnover"])
    df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    for c in ["open","high","low","close","volume","turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date","open","high","low","close","volume","turnover"]].dropna().reset_index(drop=True)

# ===================== Model (must match train_daily_limit_tp.py) =====================
class Chomp1d(nn.Module):
    def __init__(self, c): super().__init__(); self.c=c
    def forward(self,x): return x[:,:,:-self.c].contiguous() if self.c>0 else x
def wconv(i,o,k,d):
    import torch.nn.utils as U
    pad=(k-1)*d
    return U.weight_norm(nn.Conv1d(i,o,k,padding=pad,dilation=d))
class Block(nn.Module):
    def __init__(self,i,o,k,d,drop):
        super().__init__()
        self.c1=wconv(i,o,k,d); self.h1=Chomp1d((k-1)*d); self.r1=nn.ReLU(); self.dr1=nn.Dropout(drop)
        self.c2=wconv(o,o,k,d); self.h2=Chomp1d((k-1)*d); self.r2=nn.ReLU(); self.dr2=nn.Dropout(drop)
        self.ds=nn.Conv1d(i,o,1) if i!=o else None; self.r=nn.ReLU()
    def forward(self,x):
        y=self.dr1(self.r1(self.h1(self.c1(x))))
        y=self.dr2(self.r2(self.h2(self.c2(y))))
        res=x if self.ds is None else self.ds(x)
        return self.r(y+res)
class TCN_MT(nn.Module):
    def __init__(self, in_f, hidden=128, levels=6, k=3, drop=0.2):
        super().__init__()
        L=[]; ch=in_f
        for i in range(levels):
            L.append(Block(ch, hidden, k, 2**i, drop)); ch=hidden
        self.tcn=nn.Sequential(*L)
        self.head_side=nn.Linear(hidden, 2)
        self.head_tp  =nn.Linear(hidden, 1)
        self.head_ttt =nn.Linear(hidden, 1)
    def forward(self, X):      # X: [B,L,F]
        X=X.transpose(1,2)
        H=self.tcn(X)[:,:,-1]
        return self.head_side(H), self.head_tp(H), self.head_ttt(H)

def make_daily_feats(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    g = df.copy()
    g = g.sort_values("date").reset_index(drop=True)
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g["logc"].diff().fillna(0.0)

    def roll_std(s, w): return s.rolling(w, min_periods=max(2,w//3)).std()

    for w in (5,10,20,60):
        g[f"rv{w}"] = roll_std(g["ret1"], w)

    def mom_arr(price, w):
        ema = pd.Series(price).ewm(span=w, adjust=False).mean().values
        return price/np.maximum(ema,1e-12) - 1.0
    for w in (5,10,20,60):
        g[f"mom{w}"] = mom_arr(g["close"].values, w)

    for w in (10,20,60):
        mu = g["volume"].rolling(w, min_periods=max(2,w//3)).mean()
        sd = g["volume"].rolling(w, min_periods=max(2,w//3)).std().replace(0, np.nan)
        g[f"vz{w}"] = (g["volume"] - mu) / sd.replace({0:np.nan}).fillna(1.0)

    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (g["high"]-g["low"]).abs(),
        (g["high"]-prev_close).abs(),
        (g["low"] -prev_close).abs()], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14, min_periods=5).mean()

    feats = ["ret1","rv5","rv10","rv20","rv60","mom5","mom10","mom20","mom60","vz10","vz20","vz60","atr14"]
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
    mu = np.asarray(mu, dtype=np.float32); sd = np.asarray(sd, dtype=np.float32)
    sd = np.where(sd>0, sd, 1.0)
    return (X - mu) / sd

@torch.no_grad()
def infer_symbol_plan(symbol: str, model, feat_cols: List[str], seq_len: int, mu, sd) -> Optional[dict]:
    df = get_kline_daily(symbol, limit=max(120, seq_len+30))
    if df.empty: return None
    gf, _ = make_daily_feats(df)
    if len(gf) < seq_len+1: return None

    X = gf[feat_cols].to_numpy(np.float32) if feat_cols else gf.drop(columns=["date"]).to_numpy(np.float32)
    X = _apply_scaler(X, mu, sd)
    x = torch.from_numpy(X[-seq_len:][None, ...])  # [1,L,F]
    s, tp, _ = model(x)                            # s:[1,2], tp:[1,1]
    side_idx = int(torch.argmax(s, dim=1).item())  # 0=SHORT, 1=LONG
    side = "Buy" if side_idx==1 else "Sell"

    # invert log1p scaling: tp_bps = (exp(pred)-1)*1e4
    tp_log = float(tp.item())
    tp_bps = max(0.0, (math.exp(max(min(tp_log, 10.0), -10.0)) - 1.0) * 1e4)

    last_close = float(gf["close"].iloc[-1])
    atr = float(gf["atr14"].iloc[-1])
    atr_ratio = (atr / max(last_close, 1e-12))

    if side == "Buy":
        limit = last_close * (1.0 - ENTRY_ATR_K * atr_ratio)
        takep = limit * (1.0 + tp_bps/1e4)
    else:
        limit = last_close * (1.0 + ENTRY_ATR_K * atr_ratio)
        takep = limit * (1.0 - tp_bps/1e4)

    if not np.isfinite(limit) or not np.isfinite(takep): return None
    return {"side": side, "limit": float(limit), "tp": float(takep), "tp_bps": float(tp_bps),
            "close": last_close, "atr": atr, "atr_ratio": atr_ratio}

# ===================== Paper broker (with returns) =====================
def bps(a: float, b: float) -> float:
    return ((b - a) / max(a, 1e-12)) * 1e4  # signed (Buy:+, Sell:-)

class Paper:
    """
    - 사이징: 시작증거금의 ENTRY_EQUITY_PCT × 레버리지 기준(진입마다 동일)
    - equity: 시작증거금에서 수수료/실현손익만 반영(MTM은 별도로 계산)
    """
    def __init__(self, start_equity: float, leverage: float, entry_pct: float,
                 taker_fee: float, slip_bps: float):
        self.base_equity = float(start_equity)    # 고정 기준
        self.equity      = float(start_equity)    # 실시간(수수료/실현손익 반영)
        self.lev         = float(leverage)
        self.pct         = float(entry_pct)
        self.fee         = float(taker_fee)
        self.slip_bps    = float(slip_bps)
        self.pos: Dict[str, dict] = {}  # sym -> {side, qty, entry, alloc}

    def alloc_notional(self, price: float) -> float:
        # 고정: 시작증거금 × 20% × 레버리지
        return self.base_equity * self.pct * self.lev

    def _qty_for(self, price: float) -> float:
        notional = self.alloc_notional(price)
        return max(0.0, notional / max(price, 1e-9))

    def try_open(self, sym: str, side: str, limit_px: float, mark_px: float) -> bool:
        if sym in self.pos: return False
        qty = self._qty_for(limit_px)
        if qty <= 0: return False

        # 슬리피지 반영한 체결가
        if side=="Buy":
            fill = limit_px * (1.0 + self.slip_bps/1e4)
        else:
            fill = limit_px * (1.0 - self.slip_bps/1e4)

        entry_fee = (fill * qty) * self.fee
        self.equity -= entry_fee

        alloc_margin = self.base_equity * self.pct  # 배분 증거금(20%)
        self.pos[sym] = {"side": side, "qty": qty, "entry": fill, "alloc": alloc_margin}

        print(f"[FILLED][ENTRY] {sym} {side} qty={qty:.6f} at {fill:.6f} "
              f"fee={entry_fee:.2f} eq={self.equity:.2f}")
        return True

    def try_close(self, sym: str, exit_px: float) -> Optional[dict]:
        p = self.pos.get(sym)
        if not p: return None
        qty, side, entry, alloc = p["qty"], p["side"], p["entry"], p["alloc"]

        # 슬리피지 반영한 체결가
        if side=="Buy":
            fill = exit_px * (1.0 - self.slip_bps/1e4)
            gross = (fill - entry) * qty
            signed_bps = bps(entry, fill)  # +
        else:
            fill = exit_px * (1.0 + self.slip_bps/1e4)
            gross = (entry - fill) * qty
            signed_bps = -bps(entry, fill) # +

        exit_fee = (fill * qty) * self.fee
        pnl = gross - exit_fee
        self.equity += pnl  # 실현손익만 반영(진입 수수료는 이미 차감됨)

        # 수익률 계산
        pos_roi_alloc = (pnl / max(alloc, 1e-12)) * 100.0
        eq_roi = (pnl / max(self.base_equity, 1e-12)) * 100.0

        out = {
            "symbol": sym, "side": side, "pnl": pnl, "pnl_fee": exit_fee,
            "exit_fill": fill, "bps": signed_bps,
            "roi_alloc_pct": pos_roi_alloc, "roi_equity_pct": eq_roi
        }
        del self.pos[sym]

        print(f"[FILLED][EXIT ] {sym} pnl={pnl:.2f} bps={signed_bps:.1f} "
              f"roi_alloc={pos_roi_alloc:.3f}% roi_equity={eq_roi:.3f}% "
              f"exit_fill={fill:.6f} fee={exit_fee:.2f} eq={self.equity:.2f}")
        return out

    def mtm(self, sym_prices: Dict[str,float]) -> Tuple[float,float]:
        upnl = 0.0
        base = self.equity
        for s, p in self.pos.items():
            m = sym_prices.get(s, 0.0)
            if m<=0: continue
            if p["side"]=="Buy":
                upnl += (m - p["entry"]) * p["qty"]
            else:
                upnl += (p["entry"] - m) * p["qty"]
        mtm_eq = base + upnl
        mtm_roi_pct = ((mtm_eq - self.base_equity) / max(self.base_equity,1e-12)) * 100.0
        return mtm_eq, mtm_roi_pct

# ===================== Risk helpers (optional, used by UI/log) =====================
def estimate_liq_price(entry: float, side: str, lev: float, mmr_bps: float = 50.0) -> float:
    mmr = mmr_bps / 1e4
    if side == "Buy":   # 롱
        return entry * (1.0 - (1.0/lev) + mmr)
    else:               # 숏
        return entry * (1.0 + (1.0/lev) - mmr)

def safe_sl_from_limit(limit_px: float, side: str, liq_px: float,
                       sl_roi_pct: float, liq_gap_bps: float) -> float:
    base_sl = limit_px*(1.0 - sl_roi_pct) if side=="Buy" else limit_px*(1.0 + sl_roi_pct)
    if side=="Buy":
        min_safe = liq_px * (1.0 + liq_gap_bps/1e4)
        return max(base_sl, min_safe)
    else:
        min_safe = liq_px * (1.0 - liq_gap_bps/1e4)
        return min(base_sl, min_safe)

MIN_LIQ_BUFFER_BPS = float(os.getenv("MIN_LIQ_BUFFER_BPS","150"))
LIQ_SAFE_GAP_BPS   = float(os.getenv("LIQ_SAFE_GAP_BPS","20"))
MMR_BPS            = float(os.getenv("MMR_BPS","50"))
ATR_LIQ_MULT       = float(os.getenv("ATR_LIQ_MULT","1.0"))
USE_SL             = int(os.getenv("USE_SL","1"))
SL_ROI_PCT         = float(os.getenv("SL_ROI_PCT","0.015"))

def _pre_entry_risk_ok(symbol, side, limit_px, plan_atr):
    liq = estimate_liq_price(limit_px, side, LEVERAGE, mmr_bps=MMR_BPS)
    liq_buf_bps = abs(bps(limit_px, liq))
    if liq_buf_bps < MIN_LIQ_BUFFER_BPS:
        print(f"[RISK][{symbol}] liq_buf {liq_buf_bps:.1f}bps < min {MIN_LIQ_BUFFER_BPS}bps → SKIP")
        return False, liq, None
    if plan_atr and ATR_LIQ_MULT>0:
        atr_bps = (plan_atr / max(limit_px,1e-12))*1e4
        if liq_buf_bps < atr_bps*ATR_LIQ_MULT:
            print(f"[RISK][{symbol}] liq_buf {liq_buf_bps:.1f}bps < ATR*{ATR_LIQ_MULT}({atr_bps*ATR_LIQ_MULT:.1f}bps) → SKIP")
            return False, liq, None
    sl_px = None
    if USE_SL:
        sl_px = safe_sl_from_limit(limit_px, side, liq, SL_ROI_PCT, LIQ_SAFE_GAP_BPS)
        if (side=="Buy" and not (sl_px > liq)) or (side=="Sell" and not (sl_px < liq)):
            print(f"[RISK][{symbol}] SL not safe vs liq (sl={sl_px:.6f}, liq={liq:.6f}) → SKIP")
            return False, liq, sl_px
    return True, liq, sl_px

# ===================== Daily run =====================
def parse_symbols() -> List[str]:
    raw = env_str("SYMBOLS", "BTCUSDT,ETHUSDT")
    return [s.strip().upper() for s in raw.split(",") if s.strip()]

def utc_seconds_until(hhmm: str) -> int:
    try:
        hh, mm = [int(x) for x in hhmm.split(":")]
    except Exception:
        hh, mm = 23, 55
    t = time.gmtime()
    cutoff = time.struct_time((t.tm_year,t.tm_mon,t.tm_mday, hh, mm, 0, t.tm_wday,t.tm_yday,t.tm_isdst))
    cutoff_ts = int(time.mktime(cutoff))
    if cutoff_ts <= int(time.time()):
        cutoff_ts += 24*3600
    return cutoff_ts

def build_plan_from_model(symbols: List[str], ckpt: str) -> Dict[str,dict]:
    model, feat_cols, seq_len, mu, sd = load_daily_model(ckpt)
    plan = {}
    for s in symbols:
        try:
            pl = infer_symbol_plan(s, model, feat_cols, seq_len, mu, sd)
            if pl: plan[s] = pl
            print(f"[PLAN] {s} -> {pl if pl else 'skip'}")
        except Exception as e:
            print(f"[PLAN] {s} error: {e}")
    return plan

def run_day_once(symbols: List[str], plan: Dict[str,dict]):
    print(f"[START] daily run on UTC {time.strftime('%Y-%m-%d')} symbols={symbols}")
    cutoff = utc_seconds_until(DAY_CUTOFF)
    brok = Paper(START_EQUITY, LEVERAGE, ENTRY_EQUITY_PCT, TAKER_FEE, SLIPPAGE_BPS)
    filled: Dict[str,dict] = {}  # sym -> {"side","limit","tp"}

    closed_results = []  # for summary

    while True:
        now = int(time.time())
        if now >= cutoff:
            # Close all at last mid
            for sym, p in list(brok.pos.items()):
                _,_,m = get_quote(sym)
                if m>0:
                    res = brok.try_close(sym, m)
                    if res: closed_results.append(res)
            print("\n[END] cutoff reached.")
            break

        prices = {}
        for s in symbols:
            bid, ask, mid = get_quote(s)
            prices[s] = mid
            pl = plan.get(s)
            if not pl:  # no plan or already removed after TP
                continue
            limit = float(pl["limit"]); tp = float(pl["tp"]); side = pl["side"]

            # Risk pre-check (optional; 로그용)
            # ok, liq, sl_px = _pre_entry_risk_ok(s, side, limit, pl.get("atr"))

            # Not yet filled -> watch for touch
            if s not in brok.pos:
                if side=="Buy" and mid>0 and bid>0 and (bid <= limit or mid <= limit):
                    if brok.try_open(s, "Buy", limit, mid):
                        filled[s] = {"side":"Buy","limit":limit,"tp":tp}
                elif side=="Sell" and mid>0 and ask>0 and (ask >= limit or mid >= limit):
                    if brok.try_open(s, "Sell", limit, mid):
                        filled[s] = {"side":"Sell","limit":limit,"tp":tp}
            else:
                # TP monitor
                fs = filled.get(s)
                if not fs: continue
                if fs["side"]=="Buy" and mid>0 and mid >= fs["tp"]:
                    res = brok.try_close(s, mid)
                    if res: closed_results.append(res)
                    plan.pop(s, None)   # once-a-day
                elif fs["side"]=="Sell" and mid>0 and mid <= fs["tp"]:
                    res = brok.try_close(s, mid)
                    if res: closed_results.append(res)
                    plan.pop(s, None)

        # live line with ROI
        mtm_eq, mtm_roi = brok.mtm(prices)
        print(f"\r[LIVE] {time.strftime('%H:%M:%S', time.gmtime())} "
              f"MTM={mtm_eq:.2f}  ROI={mtm_roi:.3f}%  open={list(brok.pos.keys())}", end="", flush=True)
        time.sleep(max(0.5, POLL_SEC))

    # ===== summary =====
    print(f"[EQUITY] start={START_EQUITY:.2f}  end={brok.equity:.2f}  ROI={(brok.equity-START_EQUITY)/max(START_EQUITY,1e-12)*100:.3f}%")
    if closed_results:
        print("[TRADES]")
        for r in closed_results:
            print(f" - {r['symbol']} {r['side']}  pnl={r['pnl']:.2f}  bps={r['bps']:.1f}  "
                  f"roi_alloc={r['roi_alloc_pct']:.3f}%  roi_equity={r['roi_equity_pct']:.3f}%")

# ===================== main =====================
if __name__ == "__main__":
    ckpt = env_str("DAILY_CKPT", "./models_daily/daily_limit_tp_ep009_validation.ckpt")
    symbols = parse_symbols()
    write_plan = env_bool("WRITE_PLAN_JSON", False)

    print(f"[START] daily plan on {time.strftime('%Y-%m-%d')}  symbols={symbols}")

    plan = build_plan_from_model(symbols, ckpt)

    if write_plan:
        with open("daily_plan.json","w",encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
        print("[PLAN] saved -> daily_plan.json")
    else:
        run_day_once(symbols, plan)
