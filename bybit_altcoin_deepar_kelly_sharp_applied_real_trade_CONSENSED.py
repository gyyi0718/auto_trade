# -*- coding: utf-8 -*-
"""
Bybit Live Trading (Unified v5, Linear Perp)
- 실거래용: test3.py의 조건(멀티 TP/BE/트레일/글쿨다운/보수적 슬리피지 등) 그대로 적용
- 안전장치: reduceOnly TP1/TP2 예약, TP3/SL은 set_trading_stop 사용 + 주기적 BE/트레일 SL 갱신
- 핵심 포인트:
  * TP1=+0.50%, TP2=+1.00%, TP3=+2.00% (기본값)
  * TP1_RATIO=0.35, TP2_RATIO=0.35 (잔여는 TP3)
  * SL_ROI_PCT=1.0% (가격 기준)
  * TP2 체결 후 BE(+ε=0.02%) 이동
  * TP1 이후 트레일링 SL(0.40%)
  * 손실 청산 시 글로벌 쿨다운 180초
!!! 실제 실거래 코드입니다. 테스트넷으로 먼저 검증하세요 !!!
"""

import os, time, math, random, threading
from typing import Dict, Tuple, List, Optional

try:
    from pybit.unified_trading import HTTP
except Exception as e:
    raise RuntimeError("pybit.unified_trading 모듈을 설치하세요: pip install pybit") from e

import sys

# --------- 출력 필터 (불필요 로그 억제) ---------
BLOCKED_KEYWORDS = [
    "GPU available", "TPU available", "IPU available", "HPU available",
    "CUDA_VISIBLE_DEVICES", "LOCAL_RANK", "predict_dataloader",
    "FutureWarning", "lightning.pytorch","[SKIP]","[WARN]","[SIG]"
]
class FilteredStdout:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout

    def write(self, message):
        # --- 에러 라벨은 무조건 통과 ---
        if "[MONITOR ERR]" in message or "[SCANNER ERR]" in message:
            self.original_stdout.write(message)
            return
        # --- 나머지 평소 필터 ---
        if not any(keyword in message for keyword in BLOCKED_KEYWORDS):
            self.original_stdout.write(message)

    def flush(self):
        self.original_stdout.flush()

sys.stdout = FilteredStdout(sys.stdout)
sys.stderr = FilteredStdout(sys.stderr)  # 선택 적용

# ===================== 설정 =====================
CATEGORY = "linear"

# 진입/리스크
ENTRY_MODE = os.getenv("ENTRY_MODE", "model")   # "model" | "inverse" | "random"
LEVERAGE   = float(os.getenv("LEVERAGE", "100"))
TAKER_FEE  = float(os.getenv("TAKER_FEE", "0.0006"))
MAX_OPEN   = int(os.getenv("MAX_OPEN", "10"))

# 손절 (둘 중 하나, SL_ABS_USD가 0이면 SL_ROI_PCT 사용)
SL_ROI_PCT = float(os.getenv("SL_ROI_PCT", "0.01"))   # 1.0% 손절 (가격 기준)
SL_ABS_USD = float(os.getenv("SL_ABS_USD", "0.0"))    # 예: 5.0 → 포지션당 -$5 손절, 0이면 비활성

# TP/BE/Trail (test3 조건)
TP1_BPS     = float(os.getenv("TP1_BPS", "35.0"))     # +0.50%
TP2_BPS     = float(os.getenv("TP2_BPS", "100.0"))    # +1.00%
TP3_BPS     = float(os.getenv("TP3_BPS", "200.0"))    # +2.00%
TP1_RATIO   = float(os.getenv("TP1_RATIO", "0.40"))
TP2_RATIO   = float(os.getenv("TP2_RATIO", "0.35"))
BE_EPS_BPS  = float(os.getenv("BE_EPS_BPS", "2.0"))
BE_AFTER_TIER = int(os.getenv("BE_AFTER_TIER", "2"))
TRAIL_BPS   = float(os.getenv("TRAIL_BPS", "50.0"))
TRAIL_AFTER_TIER = int(os.getenv("TRAIL_AFTER_TIER", "2"))

# ---- Trend/Volatility Filter (hotfix) ----
TREND_GAP_MIN_BPS = float(os.getenv("TREND_GAP_MIN_BPS", "6.0"))   # MA20-60 gap ≥ 0.15%
MIN_SIGMA_PCT     = float(os.getenv("MIN_SIGMA_PCT", "0.0003"))     # recent σ ≥ 0.06%

# 시간/쿨다운
MAX_HOLD_SEC = int(os.getenv("MAX_HOLD_SEC", "3600"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "10"))
NEG_LOSS_GLOBAL_COOLDOWN_SEC = int(os.getenv("NEG_LOSS_GLOBAL_COOLDOWN_SEC", "180"))
GLOBAL_COOLDOWN_UNTIL = 0.0
PEAK_EQUITY = 0.0
MAX_DRAWDOWN_PCT = float(os.getenv('MAX_DRAWDOWN_PCT','0.02'))
AUTO_FLATTEN_ON_DD = (str(os.getenv('AUTO_FLATTEN_ON_DD','false')).lower() in ('1','true','yes'))
DAILY_PNL_TARGET_USD = float(os.getenv('DAILY_PNL_TARGET_USD','0.0'))
_DAILY_DAY = -1
_DAILY_PNL = 0.0


RISK_SCALE_LOSS2 = float(os.getenv("RISK_SCALE_LOSS2","0.85"))
RISK_SCALE_LOSS3 = float(os.getenv("RISK_SCALE_LOSS3","0.7"))



# 사이징
RISK_PCT_OF_EQUITY         = float(os.getenv("RISK_PCT_OF_EQUITY", "0.5"))
ENTRY_PORTION              = float(os.getenv("ENTRY_PORTION", "0.75"))
MIN_NOTIONAL               = float(os.getenv("MIN_NOTIONAL", "25.0"))
MAX_NOTIONAL_ABS           = float(os.getenv("MAX_NOTIONAL_ABS", "2000.0"))
MAX_NOTIONAL_PCT_OF_EQUITY = float(os.getenv("MAX_NOTIONAL_PCT_OF_EQUITY", "20.0"))

# 심볼 화이트리스트
SYMBOLS = [
    "ETHUSDT", "BTCUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "LINKUSDT", "ADAUSDT", "SUIUSDT", "1000PEPEUSDT", "MNTUSDT",
    "AVAXUSDT", "APTUSDT", "BNBUSDT", "UNIUSDT", "LTCUSDT"
]

# ===================== 환경변수/HTTP =====================
def _to_bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "y", "yes")

API_KEY    = ""
API_SECRET = ""
TESTNET    = _to_bool(os.getenv("BYBIT_TESTNET", "false"))

client = HTTP(
    api_key=API_KEY,
    api_secret=API_SECRET,
    testnet=TESTNET,
    timeout=10,
    recv_window=5000,  # ✅ v5에서 안전 상한: 5000ms
)
# ====== Server time offset patch (for ErrCode:10002) ======
import time, threading

RECV_WINDOW_MS = 5000   # v5 안전값
TIME_OFFSET_MS = 0      # 서버시각(ms) - 로컬시각(ms)

def _fetch_server_ms():
    try:
        srv = client.get_server_time()
        # v5 응답: {"retCode":0,"result":{"timeSecond":"...","timeNano":"..."}}
        if "result" in srv and "timeNano" in srv["result"]:
            return int(int(srv["result"]["timeNano"]) // 1_000_000)
        # 예비 (일부 버전): srv["time"]
        return int(srv.get("time", int(time.time()*1000)))
    except Exception:
        return int(time.time()*1000)

def _measure_offset_ms():
    # 간단한 왕복지연 보정: 요청 전후를 반영해 중앙값 근사
    t0 = int(time.time()*1000)
    s  = _fetch_server_ms()
    t1 = int(time.time()*1000)
    rtt2 = (t1 - t0) // 2
    return (s - t1) + rtt2  # 서버-로컬 추정 오프셋(ms)

# 최초 측정
TIME_OFFSET_MS = _measure_offset_ms()
print(f"[TIME] server-local diff ≈ {TIME_OFFSET_MS} ms (corrected)")

def _keep_offset():
    global TIME_OFFSET_MS
    while True:
        try:
            TIME_OFFSET_MS = _measure_offset_ms()
        except Exception:
            pass
        time.sleep(120)  # 2분마다 재측정(정책/지연 변동 대비)

threading.Thread(target=_keep_offset, daemon=True).start()

def _ts_ms():
    """서버 기준으로 보정된 timestamp(ms) 생성"""
    return int(time.time()*1000 + TIME_OFFSET_MS)

# ---- 래퍼: 서명 필요한 private 엔드포인트는 전부 이걸로 ----
def _place_order(**kw):
    kw.setdefault("timestamp", _ts_ms())
    kw.setdefault("recv_window", RECV_WINDOW_MS)
    return client.place_order(**kw)

def _cancel_all_orders(**kw):
    kw.setdefault("timestamp", _ts_ms())
    kw.setdefault("recv_window", RECV_WINDOW_MS)
    return client.cancel_all_orders(**kw)

def _set_trading_stop(**kw):
    kw.setdefault("timestamp", _ts_ms())
    kw.setdefault("recv_window", RECV_WINDOW_MS)
    return client.set_trading_stop(**kw)

def _set_leverage(**kw):
    kw.setdefault("timestamp", _ts_ms())
    kw.setdefault("recv_window", RECV_WINDOW_MS)
    return client.set_leverage(**kw)

def _get_wallet_balance(**kw):
    kw.setdefault("timestamp", _ts_ms())
    kw.setdefault("recv_window", RECV_WINDOW_MS)
    return client.get_wallet_balance(**kw)

def _get_positions(**kw):
    kw.setdefault("timestamp", _ts_ms())
    kw.setdefault("recv_window", RECV_WINDOW_MS)
    return client.get_positions(**kw)
# ==========================================================

# 스모크 체크: 서버시각-로컬시각 편차 확인(선택)
srv = client.get_server_time()
import time
diff_ms = int(srv["time"] if "time" in srv else srv["result"]["timeNano"]//1_000_000) - int(time.time()*1000)
print(f"[TIME] server-local diff ≈ {diff_ms} ms")
def _dbg(*a, **k):
    print(*a, **k, flush=True)

# ===================== 유틸 =====================
def _round_down(x: float, step: float) -> float:
    if step <= 0: return x
    return math.floor(x / step) * step

def _round_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return x
    return math.floor(x / tick) * tick

def _choose_side(model_side: Optional[str], mode: str) -> Optional[str]:
    m = (mode or "model").lower()
    if m == "model":  return model_side if model_side in ("Buy", "Sell") else None
    if m == "inverse":
        if model_side == "Buy": return "Sell"
        if model_side == "Sell": return "Buy"
        return None
    if m == "random": return "Buy" if random.random() < 0.5 else "Sell"
    return None

def get_quote(symbol: str) -> Tuple[float, float, float]:
    res = client.get_tickers(category=CATEGORY, symbol=symbol)
    row = (res.get("result") or {}).get("list") or [{}]
    row = row[0] if row else {}
    f = lambda k: float(row.get(k) or 0.0)
    bid, ask, last = f("bid1Price"), f("ask1Price"), f("lastPrice")
    mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else (last or bid or ask)
    return bid, ask, mid or 0.0

def get_instrument_rule(symbol: str) -> Dict[str, float]:
    info = client.get_instruments_info(category=CATEGORY, symbol=symbol)
    lst = (info.get("result") or {}).get("list") or []
    if not lst:
        return {"tickSize": 0.0001, "lotSize": 0.001, "minOrderQty": 0.001}
    it = lst[0]
    tick = float((it.get("priceFilter") or {}).get("tickSize") or 0.0001)
    lot  = float((it.get("lotSizeFilter") or {}).get("qtyStep") or 0.001)
    minq = float((it.get("lotSizeFilter") or {}).get("minOrderQty") or lot)
    return {"tickSize": tick, "lotSize": lot, "minOrderQty": minq}

def get_wallet_equity() -> float:
    try:
        wb = client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        rows = (wb.get("result") or {}).get("list") or []
        if not rows: return 0.0
        total = 0.0
        for c in rows[0].get("coin", []):
            if c.get("coin") == "USDT":
                total += float(c.get("walletBalance") or 0.0)
                total += float(c.get("unrealisedPnl") or 0.0)
        return float(total)
    except Exception:
        return 0.0

def get_positions() -> Dict[str, dict]:
    """v5: symbol 또는 settleCoin 필요 → USDT Perp 전체 보기"""
    res = client.get_positions(category=CATEGORY, settleCoin="USDT")
    lst = (res.get("result") or {}).get("list") or []
    out = {}
    for p in lst:
        sym  = p.get("symbol")
        size = float(p.get("size") or 0.0)
        if not sym or size <= 0:
            continue
        out[sym] = {
            "side": p.get("side"),
            "qty": float(p.get("size") or 0.0),
            "entry": float(p.get("avgPrice") or 0.0),
            "liq": float(p.get("liqPrice") or 0.0),
            "positionIdx": int(p.get("positionIdx") or 0),
            "tp": float(p.get("takeProfit") or 0.0),
            "sl": float(p.get("stopLoss") or 0.0),
        }
    return out

def get_top_symbols_by_volume(whitelist: List[str], top_n: int = MAX_OPEN) -> List[str]:
    res = client.get_tickers(category=CATEGORY)
    data = (res.get("result") or {}).get("list") or []
    white = set(whitelist or [])
    flt = [x for x in data if x.get("symbol") in white]
    try:
        flt.sort(key=lambda x: float(x.get("turnover24h") or 0.0), reverse=True)
    except Exception:
        pass
    return [x.get("symbol") for x in flt[:top_n]]

# ===================== 신호(합의) 훅 =====================
def get_consensus_side(symbol: str) -> Optional[str]:
    """
    실제 합의/모델 신호가 있다면 여기에서 ("Buy"/"Sell"/None) 을 반환하세요.
    현재는 placeholder로 None을 반환합니다.
    """
    return None


def _ma_gap_bps(closes, n_fast=20, n_slow=60):
    import numpy as np
    if len(closes) < max(n_fast, n_slow):
        return 0.0
    ma_f = float(np.mean(closes[-n_fast:]))
    ma_s = float(np.mean(closes[-n_slow:]))
    if ma_s <= 0: return 0.0
    return (ma_f - ma_s) / ma_s * 10_000.0

def _sigma_pct(closes, lookback=120):
    import numpy as np
    if len(closes) < lookback+1: return 0.0
    x = np.array(closes[-(lookback+1):], dtype=float)
    r = np.diff(np.log(np.clip(x, 1e-12, None)))
    return float(np.std(r))

def _ma_side_from_kline(symbol: str, interval: str = "1", limit: int = 120) -> Optional[str]:
    """간단 MA20/MA60 신호 (디버그/검증용)"""
    try:
        k = client.get_kline(category=CATEGORY, symbol=symbol, interval=interval, limit=min(int(limit), 1000))
        rows = (k.get("result") or {}).get("list") or []
        if len(rows) < 60:
            return None
        closes = [float(r[4]) for r in rows][::-1]
        ma20 = sum(closes[-20:]) / 20.0
        ma60 = sum(closes[-60:]) / 60.0
        if ma20 > ma60:  return "Buy"
        if ma20 < ma60:  return "Sell"
        return None
    except Exception as e:
        _dbg("[WARN] _ma_side_from_kline:", symbol, repr(e))
        return None

def choose_entry_side(symbol: str) -> Optional[str]:
    side = get_consensus_side(symbol)
    if side not in ("Buy", "Sell"):
        side = _ma_side_from_kline(symbol)
    return _choose_side(side, ENTRY_MODE)

# ===================== 주문/SL/TP =====================
def place_market(symbol: str, side: str, qty: float) -> bool:
    try:
        client.place_order(
            category=CATEGORY, symbol=symbol, side=side,
            orderType="Market", qty=str(qty), reduceOnly=False
        )
        return True
    except Exception as e:
        _dbg(f"[ERR] place_market {symbol} {side} {qty}:", repr(e))
        return False

def place_reduce_limit(symbol: str, side: str, qty: float, price: float) -> bool:
    try:
        client.place_order(
            category=CATEGORY, symbol=symbol, side=side,
            orderType="Limit", qty=str(qty), price=str(price),
            timeInForce="PostOnly", reduceOnly=True
        )
        return True
    except Exception as e:
        _dbg(f"[WARN] place_reduce_limit {symbol} {side} {qty}@{price}:", repr(e))
        return False

def cancel_all(symbol: str) -> None:
    try:
        client.cancel_all_orders(category=CATEGORY, symbol=symbol)
    except Exception:
        pass

def ensure_leverage(symbol: str):
    try:
        client.set_leverage(category=CATEGORY, symbol=symbol,
                            buyLeverage=str(LEVERAGE), sellLeverage=str(LEVERAGE))
    except Exception as e:
        _dbg(f"[WARN] set_leverage({symbol}):", repr(e))

def compute_sl(entry: float, qty: float, side: str) -> float:
    """SL_ABS_USD > 0 우선, 아니면 SL_ROI_PCT"""
    if SL_ABS_USD and SL_ABS_USD > 0:
        loss = float(SL_ABS_USD)
        if side == "Buy":
            return max(1e-9, entry - (loss / max(qty, 1e-12)))
        else:
            return entry + (loss / max(qty, 1e-12))
    return entry * (1.0 - SL_ROI_PCT) if side == "Buy" else entry * (1.0 + SL_ROI_PCT)

def tp_from_bps(entry: float, bps: float, side: str) -> float:
    return entry * (1.0 + bps/10000.0) if side == "Buy" else entry * (1.0 - bps/10000.0)

def set_stop(symbol: str, sl_price: float = None, tp_price: float = None) -> bool:
    """
    - 포지션이 존재해야 함 (체결 후 호출)
    - 헤지모드면 positionIdx 필요: Long=1, Short=2
    - 가격은 tickSize에 맞춰 반올림
    """
    # 현재 포지션(심볼 기준)
    pos_all = get_positions()
    pos = pos_all.get(symbol) or {}
    position_idx = int(pos.get("positionIdx") or 0)

    # 심볼 규격 → tick 반올림
    rule = get_instrument_rule(symbol)
    tick = rule["tickSize"]
    if sl_price is not None:
        sl_price = _round_to_tick(float(sl_price), tick)
    if tp_price is not None:
        tp_price = _round_to_tick(float(tp_price), tick)

    params = dict(category=CATEGORY, symbol=symbol, tpslMode="Full",
                  slTriggerBy="LastPrice", tpTriggerBy="LastPrice")
    if position_idx:
        params["positionIdx"] = position_idx
    if sl_price is not None:
        params["stopLoss"] = str(sl_price)
        params["slOrderType"] = "Market"
    if tp_price is not None:
        params["takeProfit"] = str(tp_price)
        params["tpOrderType"] = "Market"

    try:
        res = client.set_trading_stop(**params)
        _dbg("[SET_TPSL]", symbol, "retCode=", res.get("retCode"), "retMsg=", res.get("retMsg"))
        return (res.get("retCode") == 0)
    except Exception as e:
        _dbg(f"[ERR] set_trading_stop {symbol}:", repr(e))
        return False

def get_effective_leverage(symbol: str, requested_lev: float) -> float:
    """심볼별 허용 레버리지 상한/스텝 조회해 실제 적용 레버리지 계산."""
    try:
        r = client.get_instruments_info(category=CATEGORY, symbol=symbol)
        info = r["result"]["list"][0]
        lev_filter = info.get("leverageFilter", {})
        max_lev = float(lev_filter.get("maxLeverage", 1))
        step = float(lev_filter.get("leverageStep", 1))
        capped = min(float(requested_lev), max_lev)
        eff = max(step, (int(capped / step)) * step)
        return eff
    except Exception:
        return float(requested_lev)

def price_targets_from_roi(entry_price: float, side: str, tp_roi_pct: float, sl_roi_pct: float, eff_leverage: float):
    """
    ROI(계정 기준)를 목표로 삼아, 가격 변동률 = ROI / 레버리지 로 환산
    tp_roi_pct, sl_roi_pct 는 0.02 (=2%) 같은 소수 비율 기대
    """
    tp_px_delta = tp_roi_pct / eff_leverage
    sl_px_delta = sl_roi_pct / eff_leverage

    if side == "Buy":
        tp = entry_price * (1.0 + tp_px_delta)
        sl = entry_price * (1.0 - sl_px_delta)
    else:  # "Sell"
        tp = entry_price * (1.0 - tp_px_delta)
        sl = entry_price * (1.0 + sl_px_delta)
    return tp, sl

# ===================== 전략/진입 =====================
SymbolState = Dict[str, dict]   # 상태 저장(초기 수량/피크/트로프/BE 이동여부/TP체결여부 등)

STATE: SymbolState = {}
COOLDOWN_UNTIL: Dict[str, float] = {}
LOSS_STREAK = 0
WIN_STREAK = 0

def _init_state(symbol: str, side: str, entry: float, qty: float, lot_step: float):
    STATE[symbol] = {
        "side": side,
        "entry": float(entry),
        "init_qty": float(qty),
        "tp_done": [False, False, False],
        "be_moved": False,
        "peak": float(entry),
        "trough": float(entry),
        "entry_ts": time.time(),
        "lot_step": float(lot_step),
    }

def _mark_tp_flags_by_qty(symbol: str, live_qty: float):
    """현재 수량 기준으로 TP1/TP2 체결 여부 추정"""
    st = STATE.get(symbol, {})
    init_qty = float(st.get("init_qty", 0.0))
    if init_qty <= 0: return
    remain_ratio = live_qty / init_qty
    # TP1 fill: 잔여 비율 <= 1 - TP1_RATIO
    if not st["tp_done"][0] and remain_ratio <= (1.0 - TP1_RATIO + 1e-8):
        st["tp_done"][0] = True
    # TP2 fill: 잔여 비율 <= 1 - TP1_RATIO - TP2_RATIO
    if not st["tp_done"][1] and remain_ratio <= (1.0 - TP1_RATIO - TP2_RATIO + 1e-8):
        st["tp_done"][1] = True
    # All out (TP3 또는 기타 사유)
    if remain_ratio <= 1e-9:
        st["tp_done"][2] = True

def try_enter(symbol: str):
    global _DAILY_PNL
    if DAILY_PNL_TARGET_USD > 0 and _DAILY_PNL >= DAILY_PNL_TARGET_USD:
        return
    # 전역 쿨다운
    if time.time() < GLOBAL_COOLDOWN_UNTIL:
        return
    # 심볼 쿨다운
    if COOLDOWN_UNTIL.get(symbol, 0.0) > time.time():
        return
    # 이미 포지션 있으면 skip
    pos_now = get_positions()
    if symbol in pos_now:
        return

    # trend/vol gate
    try:
        k = client.get_kline(category=CATEGORY, symbol=symbol, interval="1", limit=200)
        rows = (k.get("result") or {}).get("list") or []
        closes = [float(r[4]) for r in rows][::-1]
        gap_bps = _ma_gap_bps(closes, 20, 60)
        sig_pct = _sigma_pct(closes, 120)
        if gap_bps < TREND_GAP_MIN_BPS or sig_pct < MIN_SIGMA_PCT:
            return
    except Exception as _e:
        pass

    # 신호 결정
    side = choose_entry_side(symbol)
    _dbg(f"[SIG] {symbol} side={side}")
    if side not in ("Buy", "Sell"):
        return

    # 견적
    bid, ask, mid = get_quote(symbol)
    if not mid or mid <= 0:
        _dbg(f"[SKIP] {symbol} invalid mid={mid}")
        return

    # 심볼 규격
    rule = get_instrument_rule(symbol)
    lot  = rule["lotSize"]; minq = rule["minOrderQty"]; tick = rule["tickSize"]

    # 사이징
    eq = get_wallet_equity()
    if eq <= 0:
        _dbg("[WARN] equity<=0")
    use_cash_cap = eq * RISK_PCT_OF_EQUITY
    use_cash = min(eq * ENTRY_PORTION, use_cash_cap)
    notional_raw = use_cash * LEVERAGE
    # streak-aware risk scaling
    if LOSS_STREAK >= 3:
        notional_raw *= RISK_SCALE_LOSS3
    elif LOSS_STREAK >= 2:
        notional_raw *= RISK_SCALE_LOSS2
    notional_cap = min(MAX_NOTIONAL_ABS, eq * (MAX_NOTIONAL_PCT_OF_EQUITY/100.0))
    notional = max(MIN_NOTIONAL, min(notional_raw, notional_cap))
    qty_raw = notional / mid
    qty = _round_down(qty_raw, lot)
    if qty < minq:
        _dbg(f"[SKIP] {symbol} qty<{minq} (qty={qty})")
        return

    # 주문
    ensure_leverage(symbol)
    cancel_all(symbol)
    ok = place_market(symbol, side, qty)
    if not ok:
        return
    time.sleep(0.6)

    # 체결 후 포지션 확정
    pos2 = get_positions().get(symbol)
    if not pos2:
        _dbg(f"[WARN] {symbol} no position after market? (exchange delay)")
        return
    qty_live  = float(pos2["qty"])
    entry_live= float(pos2["entry"])

    # 상태 초기화
    _init_state(symbol, side, entry_live, qty_live, lot)

    # SL/TP 계산 및 배치
    eff_leverage = get_effective_leverage(symbol, LEVERAGE)
    # TP3/SL은 계정 ROI 기반 환산 사용(슬리피지/수수료는 별도)
    tp3_price, sl_price = price_targets_from_roi(entry_live, side, tp_roi_pct=TP3_BPS/10000.0,
                                                 sl_roi_pct=SL_ROI_PCT, eff_leverage=eff_leverage)

    # test3 고정 TP 가격(엔트리 기준 BPS)
    tp1 = tp_from_bps(entry_live, TP1_BPS, side)
    tp2 = tp_from_bps(entry_live, TP2_BPS, side)
    tp3 = tp_from_bps(entry_live, TP3_BPS, side)
    # 안전: TP3은 set_trading_stop에도 반영
    tp_final = max(tp3, tp3_price) if side=="Buy" else min(tp3, tp3_price)

    # reduceOnly 수량 계산
    qty1 = _round_down(qty_live * TP1_RATIO, lot)
    qty2 = _round_down(qty_live * TP2_RATIO, lot)

    # 스탑/테이크 (TP3 & SL 등록)
    set_stop(symbol, sl_price=sl_price, tp_price=tp_final)

    close_side = "Sell" if side == "Buy" else "Buy"
    if qty1 > 0: place_reduce_limit(symbol, close_side, qty1, _round_to_tick(tp1, tick))
    if qty2 > 0: place_reduce_limit(symbol, close_side, qty2, _round_to_tick(tp2, tick))

    _dbg(f"[ENTRY] {symbol} {side} qty={qty_live:.6f} entry≈{entry_live:.6f} TP1={tp1:.6f} TP2={tp2:.6f} TP3={tp3:.6f} SL={sl_price:.6f}")

# ===================== 모니터/관리 루프 =====================
def _update_trailing_and_be(symbol: str):
    """TP 체결 추정/BE 이동/트레일링 SL 갱신"""
    st = STATE.get(symbol);  pos = get_positions().get(symbol)
    if not st or not pos:
        return
    side = st["side"]; entry = st["entry"]; lot_step = st["lot_step"]
    qty_live = float(pos["qty"]); bid, ask, mid = get_quote(symbol)

    # 피크/트로프 갱신
    if side == "Buy":
        st["peak"] = max(st.get("peak", entry), mid)
    else:
        st["trough"] = min(st.get("trough", entry), mid)

    # TP 체결 추정 (잔여 수량 기준)
    _mark_tp_flags_by_qty(symbol, qty_live)

    # BE 이동: 설정된 티어 이후
    tiers_done = sum(1 for x in st["tp_done"] if x)
    if (not st["be_moved"]) and (tiers_done >= BE_AFTER_TIER):
        eps = BE_EPS_BPS/10000.0
        be_px = entry * (1.0 + eps) if side=="Buy" else entry * (1.0 - eps)
        set_stop(symbol, sl_price=be_px)   # BE로 SL 상향/하향
        st["be_moved"] = True
        _dbg(f"[MOVE->BE] {symbol} SL→{be_px:.6f} after TP{BE_AFTER_TIER}")

    # 트레일링: 특정 티어 이후 시작
    if tiers_done >= max(1, TRAIL_AFTER_TIER):
        if side=="Buy":
            trail = float(st.get("peak", mid)) * (1.0 - TRAIL_BPS/10000.0)
            if st["be_moved"]:
                trail = max(trail, entry * (1.0 + BE_EPS_BPS/10000.0))
            set_stop(symbol, sl_price=trail)
        else:
            trail = float(st.get("trough", mid)) * (1.0 + TRAIL_BPS/10000.0)
            if st["be_moved"]:
                trail = min(trail, entry * (1.0 - BE_EPS_BPS/10000.0))
            set_stop(symbol, sl_price=trail)

def _check_time_stop_and_cooldowns(symbol: str):
    """타임스탑 및 포지션 종료/쿨다운 처리"""
    st = STATE.get(symbol);  pos = get_positions().get(symbol)
    if not st or not pos:
        return
    now = time.time()
    if MAX_HOLD_SEC and (now - float(st.get("entry_ts", now))) >= MAX_HOLD_SEC:
        # 시장가 강제 청산
        side = pos["side"]; qty = float(pos["qty"])
        if qty > 0:
            close_side = "Sell" if side=="Buy" else "Buy"
            try:
                client.place_order(category=CATEGORY, symbol=symbol, side=close_side,
                                   orderType="Market", qty=str(qty), reduceOnly=True)
                _dbg(f"[FORCE-CLOSE] {symbol} by TIMESTOP")
            except Exception as e:
                _dbg(f"[ERR] force-close {symbol}:", repr(e))
        COOLDOWN_UNTIL[symbol] = time.time() + COOLDOWN_SEC

def monitor_loop(poll_sec: float = 1.0):
    """오픈 포지션 슬/TP 관리 루프"""
    last_pos_keys = set()
    while True:
        try:
            # equity & daily checks
            global PEAK_EQUITY, _DAILY_DAY, _DAILY_PNL, GLOBAL_COOLDOWN_UNTIL
            eq = get_wallet_equity()
            if eq > 0:
                if PEAK_EQUITY <= 0: PEAK_EQUITY = eq
                PEAK_EQUITY = max(PEAK_EQUITY, eq)
                if PEAK_EQUITY > 0 and (PEAK_EQUITY - eq) / PEAK_EQUITY >= MAX_DRAWDOWN_PCT:
                    GLOBAL_COOLDOWN_UNTIL = time.time() + max(NEG_LOSS_GLOBAL_COOLDOWN_SEC, 180)
                    if AUTO_FLATTEN_ON_DD:
                        for s in list(get_positions().keys()):
                            try:
                                side = get_positions()[s]['side']; qty = float(get_positions()[s]['qty'])
                                if qty>0:
                                    client.place_order(category=CATEGORY, symbol=s, side=('Sell' if side=='Buy' else 'Buy'), orderType='Market', qty=str(qty), reduceOnly=True)
                            except Exception:
                                pass
            day_now = time.gmtime().tm_yday
            if _DAILY_DAY != day_now:
                _DAILY_DAY = day_now; _DAILY_PNL = 0.0
            pos = get_positions()
            pos_keys = set(pos.keys())

            # 신규 종료 감지 (사이즈 0으로 사라진 심볼)
            closed_syms = last_pos_keys - pos_keys
            for sym in closed_syms:
                st = STATE.pop(sym, None)
                COOLDOWN_UNTIL[sym] = time.time() + COOLDOWN_SEC
                # 손익 추정(대략): mid vs entry
                try:
                    _, _, mid = get_quote(sym)
                except Exception:
                    mid = None
                if st and mid is not None:
                    side = st.get("side"); entry = float(st.get("entry", mid))
                    pnl_val = (mid - entry) if side=="Buy" else (entry - mid)
                    _DAILY_PNL += float(pnl_val * float(st.get('init_qty',0)))
                    if pnl_val < 0:
                        GLOBAL_COOLDOWN_UNTIL = time.time() + NEG_LOSS_GLOBAL_COOLDOWN_SEC
                        _dbg(f"[COOLDOWN][GLOBAL] negative close on {sym}: block entries {NEG_LOSS_GLOBAL_COOLDOWN_SEC}s")
                # streak update
                pnl_sign_val = float(pnl_val) if 'pnl_val' in locals() else 0.0
                if pnl_sign_val < 0:
                    globals()['LOSS_STREAK'] = globals().get('LOSS_STREAK',0) + 1
                    globals()['WIN_STREAK'] = 0
                else:
                    globals()['WIN_STREAK'] = globals().get('WIN_STREAK',0) + 1
                    globals()['LOSS_STREAK'] = 0
                _dbg(f"[EXIT DETECTED] {sym} closed (streak L{LOSS_STREAK}/W{WIN_STREAK})")

            # 오픈 포지션 관리
            for sym in pos_keys:
                _update_trailing_and_be(sym)
                _check_time_stop_and_cooldowns(sym)

            last_pos_keys = pos_keys
            time.sleep(poll_sec)
        except KeyboardInterrupt:
            _dbg("[MONITOR] stopped by user")
            break
        except Exception as e:
            _dbg("[MONITOR ERR]", repr(e))
            time.sleep(2.0)

# ===================== 엔트리 스캐너 루프 =====================
def scanner_loop(iter_delay: float = 2.5):
    while True:
        try:
            # 전역 쿨다운
            if time.time() < GLOBAL_COOLDOWN_UNTIL:
                time.sleep(iter_delay);  continue

            syms = get_top_symbols_by_volume(SYMBOLS, top_n=min(MAX_OPEN, len(SYMBOLS)))
            # 현재 보유 포지션
            pos = get_positions()
            if len(pos) >= MAX_OPEN:
                time.sleep(iter_delay);  continue

            for s in syms:
                if len(get_positions()) >= MAX_OPEN:
                    break
                try_enter(s)

            time.sleep(iter_delay)
        except KeyboardInterrupt:
            _dbg("[SCANNER] stopped by user")
            break
        except Exception as e:
            _dbg("[SCANNER ERR]", repr(e))
            time.sleep(2.0)

# ===================== 실행부 =====================
def main():
    _dbg(f"[START] Bybit Live Trading MODE={ENTRY_MODE} TESTNET={TESTNET}")
    # 모니터/스캐너 병렬 실행
    t1 = threading.Thread(target=monitor_loop, args=(1.0,), daemon=True)
    t2 = threading.Thread(target=scanner_loop, args=(2.5,), daemon=True)
    t1.start(); t2.start()
    # 포그라운드 유지
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
