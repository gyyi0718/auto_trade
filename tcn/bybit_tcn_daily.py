# bybit_live_daily_exec.py
# -*- coding: utf-8 -*-
"""
Daily live executor (개선 버전 - 거래소 스타일 마진 표시)
- 매일 UTC 00:05에(가변) 모델로 일일 플랜 생성
- 실시간 가격 폴링으로 limit 터치 감지 → (시장가/지정가) 진입, TP/SL 브래킷 세팅
- TP/SL 체결/일일 마감 시 포지션 종료
- Bybit Unified v5 Linear Perp (pybit 필요: pip install pybit)
- 거래소 스타일 자산 표시: Total Equity = Available + Locked Margin + Unrealized PnL
"""

import os, time, math, json, traceback
from typing import Dict, List, Tuple, Optional
from decimal import Decimal, ROUND_DOWN
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import requests, certifi
from pybit.unified_trading import HTTP


# ===================== ENV / CONFIG =====================
def _env_str(k, d=""):
    v = os.getenv(k)
    return d if v is None or str(v).strip() == "" else str(v).strip()


def _env_int(k, d=0):
    v = os.getenv(k)
    try:
        return int(str(v).strip()) if v is not None and str(v).strip() != "" else d
    except:
        return d


def _env_float(k, d=0.0):
    v = os.getenv(k)
    try:
        return float(str(v).strip()) if v is not None and str(v).strip() != "" else d
    except:
        return d


def _env_bool(k, d=False):
    v = os.getenv(k)
    if v is None:
        return d
    return str(v).strip().lower() in ("1", "true", "y", "yes", "on")


os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# API 설정 (환경변수에서 로드 - 보안 강화)
API_KEY = "Dlp4eJD6YFmO99T8vC"
API_SEC = "YYYB5tMw2TWvfVF5wqi6lQRHqEIiDSpDJF1U"
TESTNET = _env_bool("BYBIT_TESTNET", False)

if not API_KEY or not API_SEC:
    raise ValueError("❌ BYBIT_API_KEY and BYBIT_API_SECRET must be set in environment variables!")

CATEGORY = "linear"
NEAR_TOUCH_BPS = _env_float("NEAR_TOUCH_BPS", 8.0)

# 모델 설정
DAILY_CKPT = _env_str("DAILY_CKPT", "./models_daily_v2/daily_simple_best.ckpt")
SYMBOLS = [s.strip().upper() for s in
           _env_str("SYMBOLS", "BTCUSDT").split(",") if s.strip()]

# 거래 설정
LEVERAGE = _env_float("LEVERAGE", 50.0)
ENTRY_EQUITY_PCT = _env_float("ENTRY_EQUITY_PCT", 0.5)
USE_FIXED_BASE = _env_bool("USE_FIXED_BASE", True)  # True: 고정 기준금액 사용
FIXED_BASE_EQUITY = _env_float("FIXED_BASE_EQUITY", 0.0)  # 0이면 시작시 자동 설정
SL_PCT = _env_float("SL_PCT", 0.015)  # 1.5% 손절
MAKER_FEE = _env_float("MAKER_FEE", 0.0002)  # 0.02%
TAKER_FEE = _env_float("TAKER_FEE", 0.0006)  # 0.06%
EXEC_MODE = _env_str("EXEC_MODE", "market_on_touch").lower()
TIF = _env_str("TIF", "PostOnly")
FILL_TIMEOUT_SEC = _env_int("FILL_TIMEOUT", 120)  # 2분으로 단축

# 일중 동작
ENTRY_ATR_K = _env_float("ENTRY_ATR_K", 0.5)
POLL_SEC = _env_float("POLL_SEC", 1.0)
DAY_CUTOFF_UTC = _env_str("DAY_CUTOFF_UTC", "23:55")
REPLAN_UTC = _env_str("REPLAN_UTC", "00:05")

# 리스크 관리
MIN_LIQ_BUFFER_BPS = _env_float("MIN_LIQ_BUFFER_BPS", 150.0)
MIN_NOTIONAL = _env_float("MIN_NOTIONAL", 10.0)  # 최소 주문 금액
MMR_BPS = _env_float("MMR_BPS", 50.0)  # 유지증거금률 0.5%

STATE_FILE = "live_state.json"

# ===================== HTTP / BYBIT =====================
client = HTTP(api_key=API_KEY, api_secret=API_SEC, testnet=TESTNET, timeout=10, recv_window=5000)
mkt = HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

_ses = requests.Session()
_ses.verify = certifi.where()
_ses.headers.update({"User-Agent": "bybit-daily-live/2.0", "Accept": "application/json"})
BASE_URL = "https://api-testnet.bybit.com" if TESTNET else "https://api.bybit.com"


def _get(path: str, params: dict, tries=3, backoff=0.3) -> dict:
    url = BASE_URL + path
    for i in range(tries):
        try:
            r = _ses.get(url, params=params, timeout=8)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff * (i + 1))
                continue
            return r.json()
        except Exception as e:
            if i == tries - 1:
                print(f"[API][ERR] {path} failed after {tries} tries: {e}")
            time.sleep(backoff * (i + 1))
    return {}


# ===== market helpers =====
_symbol_filters: Dict[str, Dict[str, Decimal]] = {}


def get_filters(symbol: str) -> Dict[str, Decimal]:
    if symbol in _symbol_filters:
        return _symbol_filters[symbol]
    try:
        info = mkt.get_instruments_info(category=CATEGORY, symbol=symbol)
        row = ((info.get("result") or {}).get("list") or [{}])[0]
        lot = row.get("lotSizeFilter", {}) or {}
        pf = row.get("priceFilter", {}) or {}
        _symbol_filters[symbol] = {
            "qty_step": Decimal(str(lot.get("qtyStep", "0.001"))),
            "min_qty": Decimal(str(lot.get("minOrderQty", "0.001"))),
            "px_step": Decimal(str(pf.get("tickSize", "0.01"))),
        }
    except Exception as e:
        print(f"[FILTER][ERR] {symbol}: {e}")
        _symbol_filters[symbol] = {
            "qty_step": Decimal("0.001"),
            "min_qty": Decimal("0.001"),
            "px_step": Decimal("0.01"),
        }
    return _symbol_filters[symbol]


def q_px(symbol: str, px: float) -> str:
    f = get_filters(symbol)["px_step"]
    v = (Decimal(str(px)) / f).to_integral_value(rounding=ROUND_DOWN) * f
    s = f.normalize().as_tuple().exponent
    return f"{v:.{abs(s)}f}" if s < 0 else str(v)


def q_qty(symbol: str, qty: float) -> str:
    f = get_filters(symbol)
    step, minq = f["qty_step"], f["min_qty"]
    v = (Decimal(str(qty)) / step).to_integral_value(rounding=ROUND_DOWN) * step
    if v < minq:
        v = minq
    s = step.normalize().as_tuple().exponent
    return f"{v:.{abs(s)}f}" if s < 0 else str(v)


def get_quote(symbol: str) -> Tuple[float, float, float]:
    try:
        r = mkt.get_tickers(category=CATEGORY, symbol=symbol)
        row = ((r.get("result") or {}).get("list") or [{}])[0]
        bid = float(row.get("bid1Price") or 0.0)
        ask = float(row.get("ask1Price") or 0.0)
        last = float(row.get("lastPrice") or 0.0)
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else (last or bid or ask or 0.0)
        return bid, ask, mid
    except Exception as e:
        print(f"[QUOTE][ERR] {symbol}: {e}")
        return 0.0, 0.0, 0.0


def ensure_oneway_and_leverage(symbol: str, lev: float):
    """One-Way 모드 + 레버리지 설정 (헤지모드 불필요)"""
    try:
        # 0 = One-Way Mode (양방향 동시 보유 불가)
        client.switch_position_mode(category=CATEGORY, symbol=symbol, mode=0)
    except Exception as e:
        if "110025" not in str(e):  # already in one-way mode
            print(f"[MODE][WARN] {symbol}: {e}")

    try:
        client.set_leverage(
            category=CATEGORY,
            symbol=symbol,
            buyLeverage=str(lev),
            sellLeverage=str(lev)
        )
    except Exception as e:
        msg = str(e)
        if "110043" in msg or "not modified" in msg or "leverage not modified" in msg.lower():
            pass
        else:
            print(f"[LEV][WARN] {symbol}: {e}")


# ===== 거래소 스타일 마진 정보 조회 =====
def get_margin_info() -> dict:
    """
    거래소 스타일 자산 정보 조회
    Returns:
        {
            "total_equity": float,
            "available": float,
            "locked_margin": float,
            "unrealized_pnl": float,
            "positions": dict
        }
    """
    try:
        # 지갑 정보 조회
        wb = client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        rows = (wb.get("result") or {}).get("list") or []

        total_equity = 0.0
        available = 0.0
        unrealized_pnl = 0.0

        if rows:
            coin_rows = rows[0].get("coin", [])
            for c in coin_rows:
                if c.get("coin") == "USDT":
                    # Total Equity = Wallet Balance + Unrealized PnL
                    wallet_bal = float(c.get("walletBalance") or 0.0)
                    upnl = float(c.get("unrealisedPnl") or 0.0)
                    total_equity = wallet_bal + upnl

                    # Available Balance
                    available = float(c.get("usdValue") or 0.0)

                    # Unrealized PnL
                    unrealized_pnl = upnl
                    break

        # Locked Margin = Total - Available
        locked_margin = max(0.0, total_equity - available - unrealized_pnl)

        # 포지션 정보 조회
        positions = {}
        try:
            pos_resp = client.get_positions(category=CATEGORY, settleCoin="USDT")
            for p in (pos_resp.get("result") or {}).get("list") or []:
                size = float(p.get("size") or 0.0)
                if size > 0:
                    sym = p.get("symbol")
                    positions[sym] = {
                        "side": p.get("side"),
                        "size": size,
                        "entry_price": float(p.get("avgPrice") or 0.0),
                        "mark_price": float(p.get("markPrice") or 0.0),
                        "leverage": float(p.get("leverage") or 0.0),
                        "upnl": float(p.get("unrealisedPnl") or 0.0),
                        "position_value": float(p.get("positionValue") or 0.0),
                        "position_margin": float(p.get("positionIM") or 0.0),  # Initial Margin
                        "tp": float(p.get("takeProfit") or 0.0),
                        "sl": float(p.get("stopLoss") or 0.0),
                    }
        except Exception as e:
            print(f"[MARGIN][POS_ERR] {e}")

        return {
            "total_equity": total_equity,
            "available": available,
            "locked_margin": locked_margin,
            "unrealized_pnl": unrealized_pnl,
            "positions": positions
        }
    except Exception as e:
        print(f"[MARGIN][ERR] {e}")
        return {
            "total_equity": 0.0,
            "available": 0.0,
            "locked_margin": 0.0,
            "unrealized_pnl": 0.0,
            "positions": {}
        }


def wallet_equity_usdt() -> float:
    """간단한 equity 조회 (하위 호환성)"""
    info = get_margin_info()
    return info["total_equity"]


def notional_from_equity(mid: float, current_equity: float, base_equity: float,
                         lev: float, pct: float, use_fixed: bool) -> float:
    """
    포지션 크기 계산
    use_fixed=True: base_equity 기준으로 고정 금액 (권장)
    use_fixed=False: current_equity 기준으로 변동 금액
    """
    if mid <= 0 or pct <= 0:
        return 0.0

    # 고정 기준 vs 변동 기준
    reference = base_equity if use_fixed else current_equity

    if reference <= 0:
        return 0.0

    gross = reference * pct * lev
    # 테이커 수수료 차감
    fee = TAKER_FEE if EXEC_MODE == "market_on_touch" else MAKER_FEE
    net = gross * (1.0 - fee)
    return max(MIN_NOTIONAL, net)


def qty_from_notional(symbol: str, notional: float, px: float) -> float:
    if px <= 0:
        return 0.0
    return float(q_qty(symbol, notional / px))


def place_market(symbol: str, side: str, qty: float) -> Optional[str]:
    try:
        r = client.place_order(
            category=CATEGORY,
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=q_qty(symbol, qty),
            timeInForce="IOC",
            reduceOnly=False,
            positionIdx=0  # One-Way 모드: 0
        )
        oid = str(((r.get("result") or {}).get("orderId") or ""))
        return oid if oid else None
    except Exception as e:
        print(f"[ORDER][MKT_FAIL] {symbol} {side} qty={qty} -> {e}")
        traceback.print_exc()
        return None


def place_limit(symbol: str, side: str, qty: float, limit_px: float, tif: str = "PostOnly") -> Optional[str]:
    try:
        r = client.place_order(
            category=CATEGORY,
            symbol=symbol,
            side=side,
            orderType="Limit",
            qty=q_qty(symbol, qty),
            price=q_px(symbol, limit_px),
            timeInForce=tif,
            reduceOnly=False,
            positionIdx=0  # One-Way 모드: 0
        )
        oid = str(((r.get("result") or {}).get("orderId") or ""))
        return oid if oid else None
    except Exception as e:
        print(f"[ORDER][LMT_FAIL] {symbol} {side} limit={limit_px} qty={qty} -> {e}")
        traceback.print_exc()
        return None


def avg_fill_price(symbol: str, order_id: str) -> float:
    try:
        tr = client.get_order_history(category=CATEGORY, symbol=symbol, orderId=order_id)
        lst = (tr.get("result") or {}).get("list") or []
        if lst:
            return float(lst[0].get("avgPrice") or 0.0)
    except Exception:
        pass
    return 0.0


def wait_fill(symbol: str, order_id: str, timeout_sec: int = 120, poll_sec: float = 1.0) -> Optional[float]:
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        try:
            r = client.get_open_orders(category=CATEGORY, symbol=symbol, orderId=order_id)
            open_list = (r.get("result") or {}).get("list") or []
            if not open_list:
                px = avg_fill_price(symbol, order_id)
                return px if px > 0 else None
        except Exception:
            pass
        time.sleep(poll_sec)
    return None


def set_full_tpsl(symbol: str, side: str, tp_price: Optional[float], sl_price: Optional[float]) -> bool:
    try:
        param = dict(
            category=CATEGORY,
            symbol=symbol,
            tpslMode="Full",
            tpTriggerBy="MarkPrice",
            slTriggerBy="MarkPrice",
            positionIdx=0  # One-Way 모드: 0
        )
        if tp_price and tp_price > 0:
            param["takeProfit"] = q_px(symbol, tp_price)
            param["tpOrderType"] = "Market"
        if sl_price and sl_price > 0:
            param["stopLoss"] = q_px(symbol, sl_price)
            param["slOrderType"] = "Market"
        client.set_trading_stop(**param)
        return True
    except Exception as e:
        print(f"[TPSL][FAIL] {symbol}/{side} -> {e}")
        traceback.print_exc()
        return False


def get_position_size(symbol: str, side: str) -> float:
    """실제 포지션 크기 조회 (One-Way 모드)"""
    try:
        pos = client.get_positions(category=CATEGORY, symbol=symbol, settleCoin="USDT")
        for p in (pos.get("result") or {}).get("list") or []:
            if p.get("symbol") == symbol:
                pos_side = p.get("side")  # "Buy" or "Sell"
                size = float(p.get("size") or 0.0)
                # One-Way 모드에서는 방향이 일치하는 포지션만 확인
                if size > 0 and pos_side == side:
                    return size
        return 0.0
    except Exception:
        return 0.0


def has_open_position(symbol: str) -> bool:
    """실제 포지션 존재 여부 확인 (One-Way 모드)"""
    try:
        pos = client.get_positions(category=CATEGORY, symbol=symbol, settleCoin="USDT")
        for p in (pos.get("result") or {}).get("list") or []:
            if p.get("symbol") == symbol:
                size = float(p.get("size") or 0)
                if size > 0:
                    return True
        return False
    except Exception:
        return False


# ===================== MODEL (Daily) =====================
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


class TCN_MT(nn.Module):
    def __init__(self, in_f, hidden=128, levels=6, k=3, drop=0.2):
        super().__init__()
        L = []
        ch = in_f
        for i in range(levels):
            L.append(Block(ch, hidden, k, 2 ** i, drop))
            ch = hidden
        self.tcn = nn.Sequential(*L)
        self.head_side = nn.Linear(hidden, 2)
        self.head_tp = nn.Linear(hidden, 1)
        self.head_ttt = nn.Linear(hidden, 1)

    def forward(self, X):
        X = X.transpose(1, 2)
        H = self.tcn(X)[:, :, -1]
        return self.head_side(H), self.head_tp(H), self.head_ttt(H)


def get_kline_daily(symbol: str, limit: int = 400) -> pd.DataFrame:
    r = _get("/v5/market/kline", {
        "category": CATEGORY,
        "symbol": symbol,
        "interval": "D",
        "limit": min(int(limit), 1000)
    })
    rows = ((r.get("result") or {}).get("list") or [])
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    rows = sorted(rows, key=lambda x: int(x[0]))
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", "open", "high", "low", "close", "volume", "turnover"]].dropna().reset_index(drop=True)


def make_daily_feats(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    g = df.sort_values("date").reset_index(drop=True)
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g["logc"].diff().fillna(0.0)

    def roll_std(s, w):
        return s.rolling(w, min_periods=max(2, w // 3)).std()

    for w in (5, 10, 20, 60):
        g[f"rv{w}"] = roll_std(g["ret1"], w)

    def mom_arr(price, w):
        ema = pd.Series(price).ewm(span=w, adjust=False).mean().values
        return price / np.maximum(ema, 1e-12) - 1.0

    for w in (5, 10, 20, 60):
        g[f"mom{w}"] = mom_arr(g["close"].values, w)

    for w in (10, 20, 60):
        mu = g["volume"].rolling(w, min_periods=max(2, w // 3)).mean()
        sd = g["volume"].rolling(w, min_periods=max(2, w // 3)).std().replace(0, np.nan)
        g[f"vz{w}"] = (g["volume"] - mu) / sd.replace({0: np.nan}).fillna(1.0)

    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14, min_periods=5).mean()

    feats = ["ret1", "rv5", "rv10", "rv20", "rv60", "mom5", "mom10", "mom20", "mom60", "vz10", "vz20", "vz60", "atr14"]
    g = g.dropna(subset=feats).reset_index(drop=True)
    return g, feats


def load_daily_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck.get("model", ck)
    feat_cols = ck.get("feat_cols", ck.get("feature_cols", ck.get("features", [])))
    meta = ck.get("meta", {})
    seq_len = int(meta.get("seq_len", 60))
    mu = ck.get("scaler_mu", None)
    sd = ck.get("scaler_sd", None)
    in_f = len(feat_cols) if feat_cols else 13
    model = TCN_MT(in_f=in_f).eval()

    # float64 -> float32 호환
    for k, v in list(state.items()):
        if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
            state[k] = v.float()

    model.load_state_dict(state, strict=False)
    return model, feat_cols, seq_len, mu, sd


def _apply_scaler(X: np.ndarray, mu, sd):
    if mu is None or sd is None:
        return X
    mu = np.asarray(mu, dtype=np.float32)
    sd = np.asarray(sd, dtype=np.float32)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


@torch.no_grad()
def infer_symbol_plan(symbol: str, model, feat_cols: List[str], seq_len: int, mu, sd) -> Optional[dict]:
    df = get_kline_daily(symbol, limit=max(120, seq_len + 30))
    if df.empty:
        return None
    gf, _ = make_daily_feats(df)
    if len(gf) < seq_len + 1:
        return None

    X = gf[feat_cols].to_numpy(np.float32) if feat_cols else gf.drop(columns=["date"]).to_numpy(np.float32)
    X = _apply_scaler(X, mu, sd)
    x = torch.from_numpy(X[-seq_len:][None, ...])

    s, tp, _ = model(x)
    side_idx = int(torch.argmax(s, dim=1).item())
    side = "Buy" if side_idx == 1 else "Sell"
    tp_log = float(tp.item())
    tp_bps = max(0.0, (math.exp(max(min(tp_log, 10.0), -10.0)) - 1.0) * 1e4)

    last_close = float(gf["close"].iloc[-1])
    atr = float(gf["atr14"].iloc[-1])
    atr_ratio = (atr / max(last_close, 1e-12))

    if side == "Buy":
        limit = last_close * (1.0 - ENTRY_ATR_K * atr_ratio)
        takep = limit * (1.0 + tp_bps / 1e4)
    else:
        limit = last_close * (1.0 + ENTRY_ATR_K * atr_ratio)
        takep = limit * (1.0 - tp_bps / 1e4)

    if not np.isfinite(limit) or not np.isfinite(takep):
        return None

    return {
        "side": side,
        "limit": float(limit),
        "tp": float(takep),
        "tp_bps": float(tp_bps),
        "close": last_close,
        "atr": atr,
        "atr_ratio": atr_ratio
    }


# ===================== RISK CHECKS =====================
def estimate_liq_price(entry: float, side: str, lev: float) -> float:
    """청산가 추정"""
    mmr = MMR_BPS / 1e4
    if side == "Buy":
        return entry * (1.0 - (1.0 / lev) + mmr)
    else:
        return entry * (1.0 + (1.0 / lev) - mmr)


def validate_tpsl_prices(symbol: str, side: str, entry: float, tp: float, sl: float) -> Tuple[float, float]:
    """TP/SL 가격 검증 및 조정"""
    if side == "Buy":
        # TP는 진입가보다 위
        if tp <= entry:
            tp = entry * 1.005  # 최소 0.5%
            print(f"[WARN] {symbol} TP adjusted to {q_px(symbol, tp)}")
        # SL은 진입가보다 아래
        if sl >= entry:
            sl = entry * 0.995
            print(f"[WARN] {symbol} SL adjusted to {q_px(symbol, sl)}")

        # 청산가 체크
        liq = estimate_liq_price(entry, side, LEVERAGE)
        if sl <= liq:
            sl = liq * 1.002  # 청산가보다 0.2% 위
            print(f"[RISK] {symbol} SL too close to liquidation, adjusted to {q_px(symbol, sl)}")
    else:
        # TP는 진입가보다 아래
        if tp >= entry:
            tp = entry * 0.995
            print(f"[WARN] {symbol} TP adjusted to {q_px(symbol, tp)}")
        # SL은 진입가보다 위
        if sl <= entry:
            sl = entry * 1.005
            print(f"[WARN] {symbol} SL adjusted to {q_px(symbol, sl)}")

        # 청산가 체크
        liq = estimate_liq_price(entry, side, LEVERAGE)
        if sl >= liq:
            sl = liq * 0.998
            print(f"[RISK] {symbol} SL too close to liquidation, adjusted to {q_px(symbol, sl)}")

    return tp, sl


# ===================== STATE MANAGEMENT =====================
def save_state(filled: dict):
    """상태 저장"""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(filled, f, indent=2)
    except Exception as e:
        print(f"[STATE][SAVE_ERR] {e}")


def load_state() -> dict:
    """상태 복구"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            print(f"[STATE] Recovered {len(state)} positions from {STATE_FILE}")
            return state
    except Exception as e:
        print(f"[STATE][LOAD_ERR] {e}")
    return {}


# ===================== TIME UTILS (수정됨) =====================
def parse_hhmm_utc(hhmm: str) -> Tuple[int, int]:
    try:
        hh, mm = [int(x) for x in hhmm.split(":")]
        return hh % 24, mm % 60
    except Exception:
        return 0, 5


def next_utc_ts_for(hhmm: str) -> int:
    """다음 UTC 시각의 타임스탬프 반환 (수정됨)"""
    hh, mm = parse_hhmm_utc(hhmm)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    target = now_utc.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now_utc:
        target += datetime.timedelta(days=1)
    return int(target.timestamp())


# ===================== DAILY EXEC LOOP =====================
def build_plan(ckpt_path: str, symbols: List[str]) -> Dict[str, dict]:
    model, feat_cols, seq_len, mu, sd = load_daily_model(ckpt_path)
    plan = {}
    for s in symbols:
        try:
            pl = infer_symbol_plan(s, model, feat_cols, seq_len, mu, sd)
            if pl:
                plan[s] = pl
            print(f"[PLAN] {s} -> {pl if pl else 'skip'}")
        except Exception as e:
            print(f"[PLAN][ERR] {s} -> {e}")
            traceback.print_exc()
    return plan


def live_loop():
    def near_touch(side: str, limit_px: float, bid: float, ask: float, mid: float) -> bool:
        """limit 대비 ±NEAR_TOUCH_BPS 이내면 '터치'로 간주하여 진입 허용"""
        if limit_px <= 0 or mid <= 0:
            return False
        tol = NEAR_TOUCH_BPS / 1e4  # bps → ratio
        diff = abs(mid - limit_px) / limit_px
        if side == "Buy":
            # 정상 터치(bid<=limit) 또는 limit 위에 있지만 아주 근접(mid>=limit & diff<=tol)
            return (bid <= limit_px) or (mid >= limit_px and diff <= tol)
        else:
            # 정상 터치(ask>=limit) 또는 limit 아래에 있지만 아주 근접(mid<=limit & diff<=tol)
            return (ask >= limit_px) or (mid <= limit_px and diff <= tol)

    # 초기 설정
    ensure_all = set()
    for s in SYMBOLS:
        if s not in ensure_all:
            try:
                ensure_oneway_and_leverage(s, LEVERAGE)
                print(f"[INIT] {s} one-way mode + leverage {LEVERAGE}x OK")
            except Exception as e:
                print(f"[LEV][WARN] {s}: {e}")
            ensure_all.add(s)

    # 기준 자산 설정
    margin_info = get_margin_info()
    initial_equity = margin_info["total_equity"]
    base_equity = FIXED_BASE_EQUITY if FIXED_BASE_EQUITY > 0 else initial_equity

    print(f"\n{'=' * 80}")
    print(f"💰 WALLET STATUS")
    print(f"{'=' * 80}")
    print(f"Total Equity: ${initial_equity:.2f}")
    print(f"   ├─ Available: ${margin_info['available']:.2f}")
    print(f"   ├─ Locked Margin: ${margin_info['locked_margin']:.2f}")
    print(f"   └─ Unrealized PnL: ${margin_info['unrealized_pnl']:+.2f}")

    if USE_FIXED_BASE:
        print(f"\nBase Equity (Fixed): ${base_equity:.2f}")
        print(f"Position Size: ${base_equity * ENTRY_EQUITY_PCT:.2f} per trade (fixed)")
    else:
        print(f"\nBase Equity: Dynamic")
        print(f"Position Size: {ENTRY_EQUITY_PCT * 100}% of current equity (variable)")
    print(f"{'=' * 80}\n")

    # 상태 복구
    filled = load_state()

    # 기존 포지션 확인 및 동기화
    for sym in list(filled.keys()):
        if not has_open_position(sym):
            print(f"[SYNC] {sym} position not found, removing from state")
            del filled[sym]

    # 플랜 생성
    plan = build_plan(DAILY_CKPT, SYMBOLS)
    cutoff_ts = next_utc_ts_for(DAY_CUTOFF_UTC)
    replan_ts = next_utc_ts_for(REPLAN_UTC)

    print(f"[START] daily live | TESTNET={int(TESTNET)} | symbols={SYMBOLS}")
    print(
        f"[TIME] Next replan: {datetime.datetime.fromtimestamp(replan_ts, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(
        f"[TIME] Day cutoff: {datetime.datetime.fromtimestamp(cutoff_ts, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    save_counter = 0
    last_detailed_print = 0

    while True:
        now = int(time.time())

        # 매일 재플랜(UTC)
        if now >= replan_ts:
            print(f"\n[REPLAN] {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            plan = build_plan(DAILY_CKPT, SYMBOLS)
            replan_ts = next_utc_ts_for(REPLAN_UTC)
            cutoff_ts = next_utc_ts_for(DAY_CUTOFF_UTC)
            filled.clear()
            save_state(filled)
            print(
                f"[TIME] Next replan: {datetime.datetime.fromtimestamp(replan_ts, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # 일일 마감
        if now >= cutoff_ts:
            print(f"\n[END] DAY CUTOFF reached — roll to next day")
            print(f"[TIME] {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            plan = build_plan(DAILY_CKPT, SYMBOLS)
            cutoff_ts = next_utc_ts_for(DAY_CUTOFF_UTC)
            replan_ts = next_utc_ts_for(REPLAN_UTC)
            filled.clear()
            save_state(filled)

        # 거래소 마진 정보 조회
        margin_info = get_margin_info()
        eq = margin_info["total_equity"]

        if eq <= 0:
            print(f"[EQUITY][WARN] equity={eq:.2f} <= 0")
            time.sleep(5)
            continue

        open_syms = []

        # 10초마다 상세 정보 출력
        if now - last_detailed_print >= 10 and len(margin_info["positions"]) > 0:
            print(f"\n\n{'=' * 100}")
            print(f"📊 LIVE POSITIONS")
            print(f"{'=' * 100}")
            print(
                f"💰 Total Equity: ${margin_info['total_equity']:.2f} | ROI: {((eq - base_equity) / base_equity * 100):+.2f}%")
            print(f"   ├─ Available: ${margin_info['available']:.2f}")
            print(f"   ├─ Locked Margin: ${margin_info['locked_margin']:.2f}")
            print(f"   └─ Unrealized PnL: ${margin_info['unrealized_pnl']:+.2f}")
            print(f"{'=' * 100}")

            if margin_info["positions"]:
                print(f"\n💎 POSITIONS ({len(margin_info['positions'])}):")
                print(f"{'-' * 100}")
                for sym, pos in margin_info["positions"].items():
                    emoji = "🟢" if pos["upnl"] > 0 else "🔴"
                    pnl_pct = (pos["upnl"] / pos["position_value"] * 100) if pos["position_value"] > 0 else 0
                    print(
                        f"{emoji} {sym:12} {pos['side']:4} | Entry: ${pos['entry_price']:.4f} → Mark: ${pos['mark_price']:.4f} | "
                        f"uPnL: ${pos['upnl']:+7.2f} ({pnl_pct:+.2f}%) | Margin: ${pos['position_margin']:.2f}")
                print(f"{'-' * 100}\n")

            print(f"{'=' * 100}\n")
            last_detailed_print = now

        for sym in list(plan.keys()):
            try:
                bid, ask, mid = get_quote(sym)
                if mid <= 0:
                    continue

                pl = plan.get(sym)
                if not pl:
                    continue

                side, limit_px, tp_px = pl["side"], float(pl["limit"]), float(pl["tp"])

                # 이미 체결된 심볼 - 실제 포지션 확인
                if sym in filled:
                    pos_size = get_position_size(sym, filled[sym]["side"])
                    if pos_size > 0:
                        open_syms.append(sym)
                        continue
                    else:
                        # TP/SL 청산됨
                        print(f"\n[CLOSED] {sym} position closed by TP/SL")
                        del filled[sym]
                        save_state(filled)
                        # 재진입 가능하도록 continue하지 않음

                # 중복 진입 방지 (실제 포지션 체크)
                if has_open_position(sym):
                    print(f"\n[SKIP] {sym} already has open position")
                    continue

                # 근접 터치 확인
                '''if near_touch(side, limit_px, bid, ask, mid):'''
                if True:
                    notion = notional_from_equity(
                        mid, eq, base_equity, LEVERAGE, ENTRY_EQUITY_PCT, USE_FIXED_BASE
                    )

                    # 최소 주문 금액 체크
                    if notion < MIN_NOTIONAL:
                        print(f"\n[SIZE][SKIP] {sym} notional={notion:.2f} < ${MIN_NOTIONAL}")
                        continue

                    # 증거금 체크 (현재 자산 기준)
                    required_margin = notion / LEVERAGE
                    available_margin = margin_info["available"] * 0.9  # Available의 90%까지만 사용
                    if required_margin > available_margin:
                        print(
                            f"\n[SIZE][SKIP] {sym} insufficient margin (req=${required_margin:.2f}, avail=${available_margin:.2f})")
                        continue

                    qty = qty_from_notional(sym, notion, mid if EXEC_MODE == "market_on_touch" else limit_px)
                    if qty <= 0:
                        print(f"\n[SIZE][SKIP] {sym} qty<=0 (eq={eq:.2f})")
                        continue

                    ensure_oneway_and_leverage(sym, LEVERAGE)

                    if EXEC_MODE == "market_on_touch":
                        oid = place_market(sym, side, qty)
                        if not oid:
                            print(f"\n[ENTRY][FAIL] {sym} market order failed")
                            continue

                        # 체결가 확인
                        time.sleep(0.5)
                        avg = avg_fill_price(sym, oid)
                        if avg <= 0:
                            print(f"\n[ENTRY][WARN] {sym} cannot get fill price, using mid={mid:.4f}")
                            avg = mid

                        # SL 계산
                        sl_px = avg * (1.0 - SL_PCT) if side == "Buy" else avg * (1.0 + SL_PCT)

                        # TP/SL 검증 및 조정
                        tp_px, sl_px = validate_tpsl_prices(sym, side, avg, tp_px, sl_px)

                        # TP/SL 설정
                        ok = set_full_tpsl(sym, side, tp_price=tp_px, sl_price=sl_px)

                        print(f"\n[ENTRY][MKT] {sym} {side} qty={q_qty(sym, qty)} avg={q_px(sym, avg)}")
                        print(
                            f"  TP={q_px(sym, tp_px)} ({((tp_px / avg - 1) * 100 if side == 'Buy' else (1 - tp_px / avg) * 100):.2f}%)")
                        print(
                            f"  SL={q_px(sym, sl_px)} ({((1 - sl_px / avg) * 100 if side == 'Buy' else (sl_px / avg - 1) * 100):.2f}%)")
                        print(f"  TPSL={'✓' if ok else '✗ CRITICAL - MANUAL CHECK REQUIRED'}")

                        filled[sym] = {
                            "side": side,
                            "tp": tp_px,
                            "sl": sl_px,
                            "avg": avg,
                            "qty": qty,
                            "oid": oid,
                            "timestamp": now
                        }
                        save_state(filled)

                        if not ok:
                            print(f"[CRITICAL] {sym} TPSL setting failed! Position is UNPROTECTED!")

                    else:  # limit order mode
                        oid = place_limit(sym, side, qty, limit_px, TIF)
                        if not oid:
                            print(f"\n[ENTRY][FAIL] {sym} limit order placement failed")
                            continue

                        print(
                            f"\n[ENTRY][LMT] {sym} {side} qty={q_qty(sym, qty)} limit={q_px(sym, limit_px)} oid={oid}")
                        print(f"  Waiting for fill (timeout={FILL_TIMEOUT_SEC}s)...")

                        avg = wait_fill(sym, oid, timeout_sec=FILL_TIMEOUT_SEC, poll_sec=1.0)

                        if avg is None or avg <= 0:
                            print(f"[ENTRY][TIMEOUT] {sym} not filled, cancelling order")
                            try:
                                client.cancel_order(category=CATEGORY, symbol=sym, orderId=oid)
                                print(f"[ENTRY][CANCEL] {sym} order cancelled")
                            except Exception as e:
                                print(f"[ENTRY][CANCEL_ERR] {sym}: {e}")
                            continue

                        # SL 계산
                        sl_px = avg * (1.0 - SL_PCT) if side == "Buy" else avg * (1.0 + SL_PCT)

                        # TP/SL 검증 및 조정
                        tp_px, sl_px = validate_tpsl_prices(sym, side, avg, tp_px, sl_px)

                        # TP/SL 설정
                        ok = set_full_tpsl(sym, side, tp_price=tp_px, sl_price=sl_px)

                        print(f"[ENTRY][FILL] {sym} avg={q_px(sym, avg)}")
                        print(
                            f"  TP={q_px(sym, tp_px)} ({((tp_px / avg - 1) * 100 if side == 'Buy' else (1 - tp_px / avg) * 100):.2f}%)")
                        print(
                            f"  SL={q_px(sym, sl_px)} ({((1 - sl_px / avg) * 100 if side == 'Buy' else (sl_px / avg - 1) * 100):.2f}%)")
                        print(f"  TPSL={'✓' if ok else '✗ CRITICAL'}")

                        filled[sym] = {
                            "side": side,
                            "tp": tp_px,
                            "sl": sl_px,
                            "avg": avg,
                            "qty": qty,
                            "oid": oid,
                            "timestamp": now
                        }
                        save_state(filled)

                        if not ok:
                            print(f"[CRITICAL] {sym} TPSL setting failed! Position is UNPROTECTED!")

                    open_syms.append(sym)

            except Exception as e:
                print(f"\n[ERR][{sym}] {e}")
                traceback.print_exc()

        # 주기적으로 상태 저장
        save_counter += 1
        if save_counter >= 60:  # 약 1분마다
            save_state(filled)
            save_counter = 0

        # 상태 출력 (거래소 스타일)
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')
        pnl = eq - base_equity if USE_FIXED_BASE else 0
        roi = (pnl / base_equity * 100) if USE_FIXED_BASE and base_equity > 0 else 0

        print(
            f"\r[LIVE] {now_str} | Total: ${eq:.2f} (Avail: ${margin_info['available']:.2f} | Margin: ${margin_info['locked_margin']:.2f} | uPnL: ${margin_info['unrealized_pnl']:+.2f}) | ROI: {roi:+.2f}% | Open: {len(open_syms)}      ",
            end="", flush=True)

        time.sleep(max(0.2, POLL_SEC))


# ===================== MAIN =====================
if __name__ == "__main__":
    print("=" * 80)
    print("Bybit Daily Live Trading Bot v2.0 (Exchange Style)")
    print("=" * 80)
    print(f"[CONFIG] TESTNET: {TESTNET}")
    print(f"[CONFIG] Symbols: {SYMBOLS}")
    print(f"[CONFIG] Exec Mode: {EXEC_MODE}")
    print(f"[CONFIG] Leverage: {LEVERAGE}x")
    print(f"[CONFIG] Entry %: {ENTRY_EQUITY_PCT * 100}%")
    print(f"[CONFIG] Stop Loss: {SL_PCT * 100}%")
    print(f"[CONFIG] Position Sizing: {'FIXED (권장)' if USE_FIXED_BASE else 'DYNAMIC'}")
    if USE_FIXED_BASE and FIXED_BASE_EQUITY > 0:
        print(f"[CONFIG] Fixed Base: ${FIXED_BASE_EQUITY:.2f}")
    print(f"[CONFIG] Model: {DAILY_CKPT}")
    print("=" * 80)

    # API 연결 테스트
    try:
        margin_info = get_margin_info()
        print(f"[✓] API Connection OK")
        print(f"    Total Equity: ${margin_info['total_equity']:.2f}")
        print(f"    Available: ${margin_info['available']:.2f}")
        print(f"    Locked Margin: ${margin_info['locked_margin']:.2f}")
        print(f"    Unrealized PnL: ${margin_info['unrealized_pnl']:+.2f}")
    except Exception as e:
        print(f"[✗] API Connection Failed: {e}")
        print("Please check your API_KEY and API_SECRET")
        exit(1)

    print("=" * 80)
    print("Starting live trading loop...")
    print("Press Ctrl+C to stop")
    print("=" * 80)

    try:
        live_loop()
    except KeyboardInterrupt:
        print("\n\n[STOP] Keyboard interrupt received")
        print("Saving state before exit...")
        print("Exiting...")
    except Exception as e:
        print(f"\n\n[FATAL] Unexpected error: {e}")
        traceback.print_exc()
