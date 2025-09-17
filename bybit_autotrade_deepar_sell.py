# -*- coding: utf-8 -*-
"""
Bybit Live Trading (Unified v5, Linear Perp) — MAKER/TAKER + DeepAR 신호 진입
- DeepAR ckpt 기반 방향 예측 + 수수료/슬리피지 임계 적용
- PostOnly 진입 지원, TP1/TP2는 RO 지정가, TP3/SL은 trading_stop
- BE/트레일, 타임아웃, 쿨다운 유지
"""

import os, time, math, csv, threading
from typing import Optional, Dict, Tuple, List

try:
    from pybit.unified_trading import HTTP
except Exception as e:
    raise RuntimeError("pip install pybit") from e

# ===== basic params =====
CATEGORY = "linear"
ENTRY_MODE = os.getenv("ENTRY_MODE", "model")  # model|inverse|random
LEVERAGE   = float(os.getenv("LEVERAGE", "100"))
TAKER_FEE  = float(os.getenv("TAKER_FEE", "0.0006"))
MAKER_FEE  = float(os.getenv("MAKER_FEE", "0.0"))
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "maker").lower()   # maker|taker
EXIT_EXEC_MODE = os.getenv("EXIT_EXEC_MODE", "taker").lower()   # maker|taker
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER", "1.0"))
SLIPPAGE_BPS_MAKER = float(os.getenv("SLIPPAGE_BPS_MAKER", "0.0"))
MIN_EXPECTED_ROI_BPS = float(os.getenv("MIN_EXPECTED_ROI_BPS", "20.0"))
WAIT_FOR_MAKER_FILL_SEC = float(os.getenv("WAIT_FOR_MAKER_FILL_SEC", "30"))
CANCEL_UNFILLED_MAKER = str(os.getenv("CANCEL_UNFILLED_MAKER","1")).lower() in ("1","y","yes","true")
MAX_OPEN   = int(os.getenv("MAX_OPEN", "5"))

# SL/TP
SL_ROI_PCT      = float(os.getenv("SL_ROI_PCT", "0.01"))  # 1% dyn SL
HARD_SL_PCT     = 0.01
TP1_BPS,TP2_BPS,TP3_BPS = 50.0, 100.0, 200.0
TP1_RATIO,TP2_RATIO     = 0.40, 0.35
BE_EPS_BPS = 2.0
BE_AFTER_TIER = 2
TRAIL_BPS = 50.0
TRAIL_AFTER_TIER = 2

# sizing & re-entry
ENTRY_EQUITY_PCT   = float(os.getenv("ENTRY_EQUITY_PCT", "0.20"))
REENTRY_ON_DIP     = str(os.getenv("REENTRY_ON_DIP","1")).lower() in ("1","y","yes","true")
REENTRY_PCT        = float(os.getenv("REENTRY_PCT", "0.20"))
REENTRY_SIZE_PCT   = float(os.getenv("REENTRY_SIZE_PCT","0.20"))

RISK_PCT_OF_EQUITY = float(os.getenv("RISK_PCT_OF_EQUITY","1.0"))
ENTRY_PORTION = float(os.getenv("ENTRY_PORTION","0.75"))
MIN_NOTIONAL = float(os.getenv("MIN_NOTIONAL","10.0"))
MAX_NOTIONAL_ABS = float(os.getenv("MAX_NOTIONAL_ABS","2000.0"))
MAX_NOTIONAL_PCT_OF_EQUITY = float(os.getenv("MAX_NOTIONAL_PCT_OF_EQUITY","20.0"))
KELLY_MAX_ON_CASH = float(os.getenv("KELLY_MAX_ON_CASH","0.5"))

# control
MAX_HOLD_SEC = int(os.getenv("MAX_HOLD_SEC","3600"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC","10"))
NEG_LOSS_GLOBAL_COOLDOWN_SEC=int(os.getenv("NEG_LOSS_GLOBAL_COOLDOWN_SEC","180"))
GLOBAL_COOLDOWN_UNTIL = 0.0

# universe
SYMBOLS = [
    "ETHUSDT","BTCUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
    "BNBUSDT","ADAUSDT","LINKUSDT","UNIUSDT","TRXUSDT",
    "LTCUSDT","MNTUSDT","SUIUSDT","1000PEPEUSDT",
    "XLMUSDT","ARBUSDT","APTUSDT","OPUSDT","AVAXUSDT"
]

# io
TRADES_CSV = os.getenv("TRADES_CSV","live_trades.csv")
EQUITY_CSV = os.getenv("EQUITY_CSV","live_equity.csv")
TRADES_FIELDS=["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","reason","mode"]
EQUITY_FIELDS=["ts","equity"]


TIMEOUT_MODE = os.getenv("TIMEOUT_MODE", "atr")  # atr|fixed
VOL_WIN = int(os.getenv("VOL_WIN", "60"))        # 분
TIMEOUT_K = float(os.getenv("TIMEOUT_K", "1.5")) # 여유 배수
TIMEOUT_MIN = int(os.getenv("TIMEOUT_MIN", "3")) # 최소 분
TIMEOUT_MAX = int(os.getenv("TIMEOUT_MAX", "20"))# 최대 분
SCAN_PARALLEL = str(os.getenv("SCAN_PARALLEL","1")).lower() in ("1","y","yes","true")
SCAN_WORKERS  = int(os.getenv("SCAN_WORKERS","5"))

# 전역 락(동시 주문 슬롯 보호)
PLACE_LOCK = threading.Lock()

from concurrent.futures import ThreadPoolExecutor

def _scan_symbol(s):
    if s in PENDING_MAKER: return
    if len(get_positions()) >= MAX_OPEN: return
    if COOLDOWN_UNTIL.get(s,0.0) > time.time(): return
    if s in get_positions(): return

    re = LAST_EXIT.get(s)
    if REENTRY_ON_DIP and re:
        last_side, last_px = re.get("side"), float(re.get("px") or 0.0)
        b,a,m = get_quote(s)
        if last_side=="Buy" and m>0 and last_px>0 and m <= last_px*(1.0 - REENTRY_PCT):
            try_enter(s, side_override="Buy", size_pct=REENTRY_SIZE_PCT); return
        if last_side=="Sell" and m>0 and last_px>0 and m >= last_px*(1.0 + REENTRY_PCT):
            try_enter(s, side_override="Sell", size_pct=REENTRY_SIZE_PCT); return
    try_enter(s)

def scanner_loop(iter_delay:float=2.5):
    while True:
        try:
            if time.time() < GLOBAL_COOLDOWN_UNTIL: time.sleep(iter_delay); continue
            syms = _top_symbols_by_volume(SYMBOLS, top_n=min(MAX_OPEN, len(SYMBOLS)))
            if len(get_positions()) >= MAX_OPEN: time.sleep(iter_delay); continue

            if SCAN_PARALLEL:
                with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
                    ex.map(_scan_symbol, syms)
            else:
                for s in syms: _scan_symbol(s)

            time.sleep(iter_delay)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("[SCANNER ERR]", e); time.sleep(2.0)


def _getenv_float_optional(name):
    v = os.getenv(name)
    if v is None: return None
    v = v.strip()
    if v == "": return None
    try:
        return float(v)
    except ValueError:
        return None

TIMEOUT_TARGET_BPS = _getenv_float_optional("TIMEOUT_TARGET_BPS")  # None이면 TP1_BPS 사용


def _recent_vol_bps(symbol: str, minutes: int = None) -> float:
    """최근 1분봉 절대 로그수익 중앙값을 bps로 환산."""
    minutes = minutes or VOL_WIN
    r = mkt.get_kline(category=CATEGORY, symbol=symbol, interval="1", limit=min(minutes, 1000))
    rows = (r.get("result") or {}).get("list") or []
    if len(rows) < 10:
        return 10.0  # 보수적 기본값 bps
    closes = [float(z[4]) for z in rows][::-1]
    import numpy as _np
    lr = _np.diff(_np.log(_np.asarray(closes)))
    v_bps = float(_np.median(_np.abs(lr)) * 1e4)
    return max(v_bps, 1.0)  # 0 방지

def _calc_timeout_minutes(symbol: str) -> int:
    """TP1_BPS 도달 예상 시간 × K, [MIN,MAX] 클리핑."""
    if TIMEOUT_MODE == "fixed":
        # 기존 고정 보유시간을 그대로 사용
        return max(1, int(math.ceil(max(1, MAX_HOLD_SEC) / 60.0)))
    target_bps = TIMEOUT_TARGET_BPS if TIMEOUT_TARGET_BPS is not None else TP1_BPS
    v_bps = _recent_vol_bps(symbol, VOL_WIN)
    need = int(math.ceil((target_bps / v_bps) * TIMEOUT_K))
    return int(min(max(need, TIMEOUT_MIN), TIMEOUT_MAX))


def _ensure_csv():
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(TRADES_FIELDS)
    if not os.path.exists(EQUITY_CSV):
        with open(EQUITY_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(EQUITY_FIELDS)

def _log_trade(row:dict):
    _ensure_csv()
    with open(TRADES_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(k,"") for k in TRADES_FIELDS])

def _log_equity(eq:float):
    _ensure_csv()
    with open(EQUITY_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([int(time.time()), f"{eq:.6f}"])

# ===== http clients =====
TESTNET = str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")
API_KEY    = os.getenv("BYBIT_API_KEY","")
API_SECRET = os.getenv("BYBIT_API_SECRET","")
client = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=TESTNET, timeout=10, recv_window=5000)
mkt    = HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

# ===== market utils =====
LEV_CAPS: Dict[str, Tuple[float, float]] = {}

def get_symbol_leverage_caps(symbol: str) -> Tuple[float, float]:
    if symbol in LEV_CAPS:
        return LEV_CAPS[symbol]
    info = client.get_instruments_info(category=CATEGORY, symbol=symbol)
    lst = (info.get("result") or {}).get("list") or []
    if not lst:
        caps = (1.0, float(LEVERAGE))
    else:
        lf = (lst[0].get("leverageFilter") or {})
        minL = float(lf.get("minLeverage") or 1.0)
        maxL = float(lf.get("maxLeverage") or LEVERAGE)
        caps = (minL, maxL)
    LEV_CAPS[symbol] = caps
    return caps

def get_instrument_rule(symbol:str)->Dict[str,float]:
    info=client.get_instruments_info(category=CATEGORY, symbol=symbol)
    lst=(info.get("result") or {}).get("list") or []
    if not lst: return {"tickSize":0.0001,"qtyStep":0.001,"minOrderQty":0.001}
    it=lst[0]
    tick=float((it.get("priceFilter") or {}).get("tickSize") or 0.0001)
    lot =float((it.get("lotSizeFilter") or {}).get("qtyStep") or 0.001)
    minq=float((it.get("lotSizeFilter") or {}).get("minOrderQty") or lot)
    return {"tickSize":tick,"qtyStep":lot,"minOrderQty":minq}

def _round_down(x:float, step:float)->float:
    if step<=0: return x
    return math.floor(x/step)*step

def _round_to_tick(x:float, tick:float)->float:
    if tick<=0: return x
    return math.floor(x/tick)*tick

def get_quote(symbol:str)->Tuple[float,float,float]:
    r=mkt.get_tickers(category=CATEGORY, symbol=symbol)
    row=((r.get("result") or {}).get("list") or [{}])[0]
    f=lambda k: float(row.get(k) or 0.0)
    bid,ask,last=f("bid1Price"),f("ask1Price"),f("lastPrice")
    mid=(bid+ask)/2.0 if (bid>0 and ask>0) else (last or bid or ask)
    return bid,ask,mid or 0.0

# ===== wallet/positions =====
def get_wallet_equity()->float:
    try:
        wb=client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        rows=(wb.get("result") or {}).get("list") or []
        if not rows: return 0.0
        total=0.0
        for c in rows[0].get("coin",[]):
            if c.get("coin")=="USDT":
                total+=float(c.get("walletBalance") or 0.0)
                total+=float(c.get("unrealisedPnl") or 0.0)
        return float(total)
    except Exception:
        return 0.0

def get_positions()->Dict[str,dict]:
    res=client.get_positions(category=CATEGORY, settleCoin="USDT")
    lst=(res.get("result") or {}).get("list") or []
    out={}
    for p in lst:
        sym=p.get("symbol"); size=float(p.get("size") or 0.0)
        if not sym or size<=0: continue
        out[sym]={
            "side":p.get("side"),
            "qty":float(p.get("size") or 0.0),
            "entry":float(p.get("avgPrice") or 0.0),
            "liq":float(p.get("liqPrice") or 0.0),
            "positionIdx":int(p.get("positionIdx") or 0),
        }
    return out

# ===== leverage / orders / stops =====
def ensure_isolated_and_leverage(symbol: str):
    try:
        minL, maxL = get_symbol_leverage_caps(symbol)
        useL = max(min(float(LEVERAGE), maxL), minL)
        kw = dict(category=CATEGORY, symbol=symbol, buyLeverage=str(useL), sellLeverage=str(useL), tradeMode=1)
        try:
            client.set_leverage(**kw)
        except Exception as e:
            msg=str(e)
            if "ErrCode: 110043" in msg or "leverage not modified" in msg:
                pass
            else:
                raise
    except Exception as e:
        print(f"[WARN] set_leverage {symbol}: {e}")

def _order_with_mode_retry(params: dict, side: Optional[str] = None):
    try:
        return client.place_order(**params)
    except Exception as e:
        msg=str(e)
        if "position idx not match position mode" in msg and side in ("Buy","Sell"):
            try:
                p2=dict(params); p2["positionIdx"]=1 if side=="Buy" else 2
                return client.place_order(**p2)
            except Exception as e2:
                raise e2
        raise e

def place_market(symbol: str, side: str, qty: float) -> bool:
    try:
        rule=get_instrument_rule(symbol)
        q=_round_down(qty, rule["qtyStep"])
        if q<rule["minOrderQty"]: return False
        p=dict(category=CATEGORY, symbol=symbol, side=side, orderType="Market", qty=str(q), timeInForce="IOC", reduceOnly=False)
        _order_with_mode_retry(p, side=side)
        return True
    except Exception as e:
        print(f"[ERR] place_market {symbol} {side} {qty} -> {e}")
        return False

def place_limit_postonly(symbol: str, side: str, qty: float, price: float) -> Optional[str]:
    try:
        rule=get_instrument_rule(symbol)
        q=_round_down(qty, rule["qtyStep"])
        px=_round_to_tick(price, rule["tickSize"])
        if q<rule["minOrderQty"]: return None
        p=dict(category=CATEGORY, symbol=symbol, side=side, orderType="Limit", qty=str(q), price=str(px), timeInForce="PostOnly", reduceOnly=False)
        ret=_order_with_mode_retry(p, side=side)
        oid=((ret or {}).get("result") or {}).get("orderId")
        print(f"[ORDER][POST_ONLY] {symbol} {side} {q}@{px} oid={oid}")
        return oid
    except Exception as e:
        print(f"[ERR] place_limit_postonly {symbol} {side} {qty}@{price} -> {e}")
        return None

def place_reduce_limit(symbol:str, side:str, qty:float, price:float)->bool:
    try:
        rule=get_instrument_rule(symbol)
        q=_round_down(qty, rule["qtyStep"])
        px=_round_to_tick(price, rule["tickSize"])
        if q<rule["minOrderQty"]: return False
        p=dict(category=CATEGORY, symbol=symbol, side=side, orderType="Limit", qty=str(q), price=str(px), timeInForce="PostOnly", reduceOnly=True)
        _order_with_mode_retry(p, side=side)
        return True
    except Exception as e:
        print(f"[WARN] reduce_limit {symbol} {side} {qty}@{price}: {e}")
        return False

def compute_sl(entry:float, qty:float, side:str)->float:
    dyn = entry*(1.0 - SL_ROI_PCT) if side=="Buy" else entry*(1.0 + SL_ROI_PCT)
    hard = entry*(1.0 - HARD_SL_PCT) if side=="Buy" else entry*(1.0 + HARD_SL_PCT)
    return max(dyn, hard) if side=="Buy" else min(dyn, hard)

def tp_from_bps(entry:float, bps:float, side:str)->float:
    return entry*(1.0 + bps/10000.0) if side=="Buy" else entry*(1.0 - bps/10000.0)

def set_stop_with_retry(symbol: str, sl_price: Optional[float] = None, tp_price: Optional[float] = None, position_idx: Optional[int] = None) -> bool:
    rule=get_instrument_rule(symbol); tick=float(rule.get("tickSize") or 0.0)
    if sl_price is not None: sl_price=_round_to_tick(float(sl_price), tick)
    if tp_price is not None: tp_price=_round_to_tick(float(tp_price), tick)
    params=dict(category=CATEGORY, symbol=symbol, tpslMode="Full", slTriggerBy="MarkPrice", tpTriggerBy="MarkPrice")
    if sl_price is not None:
        params["stopLoss"]=f"{sl_price:.10f}".rstrip("0").rstrip(".")
        params["slOrderType"]="Market"
    if tp_price is not None:
        params["takeProfit"]=f"{tp_price:.10f}".rstrip("0").rstrip(".")
        params["tpOrderType"]="Market"
    if position_idx in (1,2): params["positionIdx"]=position_idx
    try:
        ret=client.set_trading_stop(**params)
        return (ret.get("retCode")==0)
    except Exception:
        return False

# ===== qty helpers =====
def compute_qty_by_equity_pct(symbol: str, mid: float, equity_now: float, pct: float) -> Tuple[float, float]:
    rule = get_instrument_rule(symbol)
    qty_step = float(rule["qtyStep"]); min_qty = float(rule["minOrderQty"])
    MIN_NOTIONAL_USDT = float(os.getenv("MIN_NOTIONAL_USDT","10.0"))
    if mid <= 0 or equity_now <= 0 or pct <= 0:
        return 0.0, 0.0
    minL, maxL = get_symbol_leverage_caps(symbol)
    lv_used = max(min(float(LEVERAGE), maxL), minL)
    notional_raw = max(equity_now * pct, MIN_NOTIONAL_USDT) * lv_used
    cap_abs = MAX_NOTIONAL_ABS * lv_used
    cap_pct = equity_now * lv_used * (MAX_NOTIONAL_PCT_OF_EQUITY / 100.0)
    notional_cap = min(cap_abs, cap_pct)
    notional = max(MIN_NOTIONAL, min(notional_raw, notional_cap))
    qty = notional / mid
    qty = math.ceil(qty/qty_step)*qty_step
    if qty < min_qty: qty = math.ceil(min_qty/qty_step)*qty_step
    notional = qty * mid
    return qty, notional

# ===== MODEL SIGNAL (DeepAR) =====
MODEL_CKPT = os.getenv("MODEL_CKPT", "models/multi_deepar_model.ckpt")
SEQ_LEN    = int(os.getenv("SEQ_LEN","60"))
PRED_LEN   = int(os.getenv("PRED_LEN","60"))
FEE_SAFETY = float(os.getenv("FEE_SAFETY", "1.20"))  # 1.1~1.3 권장

def _fee_threshold(taker_fee=TAKER_FEE, slip_bps=SLIPPAGE_BPS_TAKER, safety=FEE_SAFETY):
    rt = 2.0*taker_fee + 2.0*(slip_bps/1e4)
    return math.log(1.0 + safety*rt)

import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.data import GroupNormalizer

DEEPar = DeepAR.load_from_checkpoint(MODEL_CKPT).eval()
_THR = _fee_threshold()

def _build_infer_dataset(df_sym: pd.DataFrame):
    df = df_sym.sort_values("timestamp").copy()
    df["time_idx"] = np.arange(len(df), dtype=np.int64)
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="log_return",
        group_ids=["series_id"],
        max_encoder_length=SEQ_LEN,
        max_prediction_length=PRED_LEN,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["log_return"],
        static_categoricals=["series_id"],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

def _recent_1m_df(symbol: str, minutes: int = 200):
    r = mkt.get_kline(category="linear", symbol=symbol, interval="1", limit=min(minutes,1000))
    rows = (r.get("result") or {}).get("list") or []
    if not rows: return None
    rows = rows[::-1]
    df = pd.DataFrame([{
        "timestamp": pd.to_datetime(int(z[0]), unit="ms", utc=True),
        "series_id": symbol,
        "open": float(z[1]), "high": float(z[2]), "low": float(z[3]),
        "close": float(z[4]), "volume": float(z[5]), "turnover": float(z[6]),
    } for z in rows])
    df["log_return"] = np.log(df["close"]).diff().fillna(0.0)
    return df

@torch.no_grad()
def deepar_direction(symbol: str):
    df = _recent_1m_df(symbol, minutes=SEQ_LEN+PRED_LEN+10)
    if df is None or len(df) < (SEQ_LEN+1):
        return None, 0.0
    tsd = _build_infer_dataset(df)
    dl = tsd.to_dataloader(train=False, batch_size=1, num_workers=0)
    mu = float(DEEPar.predict(dl, mode="prediction")[0, 0])  # 다음 1스텝 기대 log-return
    if   mu >  _THR: return "Buy",  1.0
    elif mu < -_THR: return "Sell", 1.0
    else:            return None,   0.0

# MA 백업 신호
def _ma_side_from_kline(symbol:str, interval:str="1", limit:int=120)->Optional[str]:
    try:
        k=mkt.get_kline(category=CATEGORY, symbol=symbol, interval=interval, limit=min(int(limit),1000))
        rows=(k.get("result") or {}).get("list") or []
        if len(rows)<60: return None
        closes=[float(r[4]) for r in rows][::-1]
        ma20=sum(closes[-20:])/20.0; ma60=sum(closes[-60:])/60.0
        if ma20>ma60: return "Buy"
        if ma20<ma60: return "Sell"
        return None
    except Exception:
        return None

def choose_entry(symbol:str)->Tuple[Optional[str], float]:
    side, conf = deepar_direction(symbol)  # 1순위: 모델
    if side is None:
        side = _ma_side_from_kline(symbol)  # 백업
        conf = 0.60 if side else 0.0
    m=(ENTRY_MODE or "model").lower()
    if m=="inverse":
        if side=="Buy": side="Sell"
        elif side=="Sell": side="Buy"
    elif m=="random":
        side = "Buy" if (time.time()*1000)%2<1 else "Sell"
    return side, float(conf or 0.0)

# ===== per-position state & pending post-only =====
STATE: Dict[str,dict] = {}
COOLDOWN_UNTIL: Dict[str,float] = {}
PENDING_MAKER: Dict[str,dict] = {}  # symbol -> {orderId, ts, side, qty, price}
LAST_EXIT: Dict[str,dict] = {}      # symbol -> {"side": "...", "px": float, "ts": float}

def _init_state(symbol:str, side:str, entry:float, qty:float, lot_step:float):
    STATE[symbol] = {
        "side":side, "entry":float(entry), "init_qty":float(qty),
        "tp_done":[False,False,False], "be_moved":False,
        "peak":float(entry), "trough":float(entry),
        "entry_ts":time.time(), "lot_step":float(lot_step),
    }

def _bps_between(p1:float, p2:float)->float:
    return abs((p2 - p1) / max(p1, 1e-12)) * 10_000.0

def _entry_cost_bps()->float:
    fee = (0.0 if EXECUTION_MODE=="maker" else TAKER_FEE) * 10_000.0
    slip= (SLIPPAGE_BPS_MAKER if EXECUTION_MODE=="maker" else SLIPPAGE_BPS_TAKER)
    return max(0.0, fee + max(0.0, slip))

def _exit_cost_bps()->float:
    fee = (0.0 if EXIT_EXEC_MODE=="maker" else TAKER_FEE) * 10_000.0
    slip= (SLIPPAGE_BPS_MAKER if EXIT_EXEC_MODE=="maker" else SLIPPAGE_BPS_TAKER)
    return max(0.0, fee + max(0.0, slip))

def try_enter(symbol:str, side_override: Optional[str]=None, size_pct: Optional[float]=None):
    if time.time() < GLOBAL_COOLDOWN_UNTIL: return
    if COOLDOWN_UNTIL.get(symbol,0.0) > time.time(): return
    if symbol in PENDING_MAKER: return
    if symbol in get_positions(): return

    # 신호
    side, conf = choose_entry(symbol)
    if side_override in ("Buy","Sell"):
        side = side_override; conf = max(conf, 0.60)
    if side not in ("Buy","Sell") or conf < 0.55: return

    # 호가/ROI 필터
    bid,ask,mid = get_quote(symbol)
    if mid <= 0: return
    rule = get_instrument_rule(symbol)
    entry_ref = float(ask if side=="Buy" else bid)
    tp1_px = _round_to_tick(tp_from_bps(entry_ref, TP1_BPS, side), rule["tickSize"])
    edge_bps = _bps_between(entry_ref, tp1_px)
    cost_bps = _entry_cost_bps() + _exit_cost_bps()
    need_bps = cost_bps + MIN_EXPECTED_ROI_BPS
    if edge_bps < need_bps: return

    # 수량
    eq = get_wallet_equity()
    use_pct = float(size_pct if size_pct is not None else ENTRY_EQUITY_PCT)
    qty, notional = compute_qty_by_equity_pct(symbol, mid, eq, use_pct)
    if qty <= 0: return

    # 동시 주문 구간
    with PLACE_LOCK:
        if len(get_positions()) >= MAX_OPEN: return
        if COOLDOWN_UNTIL.get(symbol,0.0) > time.time(): return
        if symbol in PENDING_MAKER or symbol in get_positions(): return

        ensure_isolated_and_leverage(symbol)

        if EXECUTION_MODE == "maker":
            limit_px = bid if side=="Buy" else ask
            oid = place_limit_postonly(symbol, side, qty, limit_px)
            if oid:
                PENDING_MAKER[symbol] = {"orderId": oid, "ts": time.time(), "side": side, "qty": qty, "price": limit_px}
                _log_trade({"ts":int(time.time()),"event":"ENTRY_POSTONLY","symbol":symbol,"side":side,
                            "qty":f"{qty:.8f}","entry":f"{limit_px:.6f}","exit":"",
                            "tp1":f"{tp1_px:.6f}","tp2":"","tp3":"","sl":"",
                            "reason":"open","mode":ENTRY_MODE})
            return

        # taker
        ok = place_market(symbol, side, qty)
        if not ok: return
        pos = get_positions().get(symbol)
        if not pos: return

        entry = float(pos.get("avgPrice") or pos.get("entry") or mid)
        lot = rule["qtyStep"]
        sl  = compute_sl(entry, qty, side)
        tp1 = tp_from_bps(entry, TP1_BPS, side)
        tp2 = tp_from_bps(entry, TP2_BPS, side)
        tp3 = tp_from_bps(entry, TP3_BPS, side)
        qty_tp1 = _round_down(qty * TP1_RATIO, lot)
        qty_tp2 = _round_down(qty * TP2_RATIO, lot)
        close_side = "Sell" if side=="Buy" else "Buy"
        if qty_tp1>0: place_reduce_limit(symbol, close_side, qty_tp1, tp1)
        if qty_tp2>0: place_reduce_limit(symbol, close_side, qty_tp2, tp2)
        set_stop_with_retry(symbol, sl_price=sl, tp_price=tp3, position_idx=pos.get("positionIdx"))
        _init_state(symbol, side, entry, qty, lot)
        # 가변 타임아웃 데드라인
        _min_hold = _calc_timeout_minutes(symbol)
        STATE[symbol]["deadline_ts"] = time.time() + _min_hold * 60

        _log_trade({"ts":int(time.time()),"event":"ENTRY","symbol":symbol,"side":side,
                    "qty":f"{qty:.8f}","entry":f"{entry:.6f}","exit":"",
                    "tp1":f"{tp1:.6f}","tp2":f"{tp2:.6f}","tp3":f"{tp3:.6f}",
                    "sl":f"{sl:.6f}","reason":"open","mode":ENTRY_MODE})

def _update_trailing_and_be(symbol:str):
    st = STATE.get(symbol)
    pos = get_positions().get(symbol)
    if not st or not pos: return
    side=st["side"]; entry=st["entry"]
    qty=float(pos.get("qty") or 0.0)
    if qty<=0: return
    _,_,mark = get_quote(symbol)
    st["peak"]   = max(st["peak"], mark)
    st["trough"] = min(st["trough"], mark)
    init_qty = float(st.get("init_qty",0.0))
    rem = qty / max(init_qty, 1e-12)
    if not st["tp_done"][0] and rem <= (1.0 - TP1_RATIO + 1e-8):
        st["tp_done"][0]=True; _log_trade({"ts":int(time.time()),"event":"TP1_FILL","symbol":symbol,"side":side,"qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"","reason":"","mode":ENTRY_MODE})
    if not st["tp_done"][1] and rem <= (1.0 - TP1_RATIO - TP2_RATIO + 1e-8):
        st["tp_done"][1]=True; _log_trade({"ts":int(time.time()),"event":"TP2_FILL","symbol":symbol,"side":side,"qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"","reason":"","mode":ENTRY_MODE})
    be_px = entry*(1.0 + BE_EPS_BPS/10000.0) if side=="Buy" else entry*(1.0 - BE_EPS_BPS/10000.0)
    if st["tp_done"][1] and not st["be_moved"]:
        set_stop_with_retry(symbol, sl_price=be_px, tp_price=None, position_idx=pos.get("positionIdx"))
        st["be_moved"]=True
        _log_trade({"ts":int(time.time()),"event":"MOVE_SL","symbol":symbol,"side":side,"qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":f"{be_px:.6f}","reason":"BE","mode":ENTRY_MODE})
    if st["tp_done"][TRAIL_AFTER_TIER-1]:
        if side=="Buy":
            new_sl = mark*(1.0 - TRAIL_BPS/10000.0)
        else:
            new_sl = mark*(1.0 + TRAIL_BPS/10000.0)
        set_stop_with_retry(symbol, sl_price=new_sl, tp_price=None, position_idx=pos.get("positionIdx"))

def _check_time_stop_and_cooldowns(symbol:str):
    st = STATE.get(symbol)
    pos = get_positions().get(symbol)
    if not st or not pos:
        return

    # 데드라인이 없으면 계산해서 부여
    ddl = st.get("deadline_ts")
    if ddl is None:
        _min_hold = _calc_timeout_minutes(symbol)
        ddl = time.time() + _min_hold * 60
        st["deadline_ts"] = ddl

    if time.time() >= ddl:
        side = pos["side"]
        qty = float(pos["qty"])
        close_side = "Sell" if side == "Buy" else "Buy"
        _order_with_mode_retry(dict(
            category=CATEGORY, symbol=symbol, side=close_side,
            orderType="Market", qty=str(qty), reduceOnly=True, timeInForce="IOC"
        ), side=close_side)
        COOLDOWN_UNTIL[symbol] = time.time() + COOLDOWN_SEC
        _log_trade({"ts": int(time.time()), "event": "EXIT", "symbol": symbol, "side": side,
                    "qty": f"{qty:.8f}", "entry": "", "exit": "", "tp1": "", "tp2": "", "tp3": "",
                    "sl": "", "reason": "HORIZON_TIMEOUT", "mode": ENTRY_MODE})

# ===== pending PostOnly 관리 =====
def _get_open_orders(symbol: Optional[str]=None)->List[dict]:
    try:
        res=client.get_open_orders(category=CATEGORY, symbol=symbol)
        return (res.get("result") or {}).get("list") or []
    except Exception:
        return []

def _cancel_order(symbol:str, order_id:str)->bool:
    try:
        client.cancel_order(category=CATEGORY, symbol=symbol, orderId=order_id)
        return True
    except Exception as e:
        print(f"[CANCEL ERR] {symbol} {order_id}: {e}")
        return False

def process_pending_postonly():
    now=time.time()
    if not PENDING_MAKER: return
    open_map: Dict[str, List[dict]] = {}
    for sym in list(PENDING_MAKER.keys()):
        if sym not in open_map:
            open_map[sym]=_get_open_orders(sym)
        oinfo=PENDING_MAKER.get(sym) or {}
        oid=oinfo.get("orderId"); side=oinfo.get("side")
        open_list=open_map.get(sym) or []
        still_open = any((x.get("orderId")==oid) for x in open_list)
        if not still_open:
            PENDING_MAKER.pop(sym, None)
            continue
        if CANCEL_UNFILLED_MAKER and (now - float(oinfo.get("ts", now)) > WAIT_FOR_MAKER_FILL_SEC):
            ok=_cancel_order(sym, oid)
            PENDING_MAKER.pop(sym, None)
            if ok:
                COOLDOWN_UNTIL[sym]=time.time()+COOLDOWN_SEC
                _log_trade({"ts":int(time.time()),"event":"CANCEL_POSTONLY","symbol":sym,"side":side,"qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"","reason":"TIMEOUT","mode":ENTRY_MODE})

# ===== loops =====
def monitor_loop(poll_sec: float = 1.0):
    last_pos_keys = set()
    while True:
        try:
            _log_equity(get_wallet_equity())
            process_pending_postonly()
            pos = get_positions()
            pos_keys = set(pos.keys())

            # 신규 포지션 세팅
            new_syms = pos_keys - last_pos_keys
            for sym in new_syms:
                p = pos[sym]
                side = p["side"]
                entry = float(p.get("entry") or 0.0)
                qty = float(p.get("qty") or 0.0)
                if entry <= 0 or qty <= 0: continue
                lot = get_instrument_rule(sym)["qtyStep"]
                sl  = compute_sl(entry, qty, side)
                tp1 = tp_from_bps(entry, TP1_BPS, side)
                tp2 = tp_from_bps(entry, TP2_BPS, side)
                tp3 = tp_from_bps(entry, TP3_BPS, side)
                qty_tp1 = _round_down(qty * TP1_RATIO, lot)
                qty_tp2 = _round_down(qty * TP2_RATIO, lot)
                close_side = "Sell" if side == "Buy" else "Buy"
                if qty_tp1 > 0: place_reduce_limit(sym, close_side, qty_tp1, tp1)
                if qty_tp2 > 0: place_reduce_limit(sym, close_side, qty_tp2, tp2)
                set_stop_with_retry(sym, sl_price=sl, tp_price=tp3, position_idx=p.get("positionIdx"))
                _init_state(sym, side, entry, qty, lot)
                # === 가변 타임아웃 데드라인 설정 ===
                _min_hold = _calc_timeout_minutes(sym)
                STATE[sym]["deadline_ts"] = time.time() + _min_hold * 60
                _log_trade({"ts": int(time.time()), "event": "ENTRY","symbol": sym, "side": side, "qty": f"{qty:.8f}",
                            "entry": f"{entry:.6f}", "exit": "", "tp1": f"{tp1:.6f}", "tp2": f"{tp2:.6f}", "tp3": f"{tp3:.6f}",
                            "sl": f"{sl:.6f}", "reason": "postonly_fill_or_detect", "mode": ENTRY_MODE})

            # 닫힌 포지션
            closed_syms = last_pos_keys - pos_keys
            for sym in closed_syms:
                b, a, m = get_quote(sym)
                st = STATE.pop(sym, None)
                side = (st or {}).get("side", "")
                LAST_EXIT[sym] = {"side": side, "px": float(m or a or b or 0.0), "ts": time.time()}
                COOLDOWN_UNTIL[sym] = time.time() + COOLDOWN_SEC
                _log_trade({"ts": int(time.time()), "event": "EXIT","symbol": sym, "side": side, "qty": "",
                            "entry": "", "exit": f"{(m or 0.0):.6f}","tp1": "", "tp2": "", "tp3": "", "sl": "",
                            "reason": "CLOSED", "mode": ENTRY_MODE})

            # 관리
            for sym in pos_keys:
                _update_trailing_and_be(sym)
                _check_time_stop_and_cooldowns(sym)

            last_pos_keys = pos_keys
            time.sleep(poll_sec)

        except KeyboardInterrupt:
            print("[MONITOR] stop"); break
        except Exception as e:
            print("[MONITOR ERR]", e); time.sleep(2.0)

def _top_symbols_by_volume(whitelist:List[str], top_n:int=MAX_OPEN)->List[str]:
    res=mkt.get_tickers(category=CATEGORY)
    data=(res.get("result") or {}).get("list") or []
    white=set(whitelist or [])
    flt=[x for x in data if x.get("symbol") in white]
    try:
        flt.sort(key=lambda x: float(x.get("turnover24h") or 0.0), reverse=True)
    except Exception:
        pass
    return [x.get("symbol") for x in flt[:top_n]]

def scanner_loop(iter_delay:float=2.5):
    while True:
        try:
            if time.time() < GLOBAL_COOLDOWN_UNTIL:
                time.sleep(iter_delay); continue
            syms = _top_symbols_by_volume(SYMBOLS, top_n=min(MAX_OPEN, len(SYMBOLS)))
            if len(get_positions()) >= MAX_OPEN:
                time.sleep(iter_delay); continue
            for s in syms:
                if s in PENDING_MAKER: continue
                if len(get_positions()) >= MAX_OPEN: break
                if COOLDOWN_UNTIL.get(s,0.0) > time.time(): continue
                if s in get_positions(): continue
                # 재진입
                re = LAST_EXIT.get(s)
                if REENTRY_ON_DIP and re:
                    last_side, last_px = re.get("side"), float(re.get("px") or 0.0)
                    b,a,m = get_quote(s)
                    if last_side=="Buy" and m>0 and last_px>0 and m <= last_px*(1.0 - REENTRY_PCT):
                        try_enter(s, side_override="Buy", size_pct=REENTRY_SIZE_PCT); continue
                    if last_side=="Sell" and m>0 and last_px>0 and m >= last_px*(1.0 + REENTRY_PCT):
                        try_enter(s, side_override="Sell", size_pct=REENTRY_SIZE_PCT); continue
                # 일반 진입
                try_enter(s)
            time.sleep(iter_delay)
        except KeyboardInterrupt:
            print("[SCANNER] stop"); break
        except Exception as e:
            print("[SCANNER ERR]", e); time.sleep(2.0)

# ===== run =====
def main():
    print(f"[START] MODE={ENTRY_MODE} EXEC={EXECUTION_MODE}/{EXIT_EXEC_MODE} TESTNET={TESTNET}")
    t1=threading.Thread(target=monitor_loop, args=(1.0,), daemon=True)
    t2=threading.Thread(target=scanner_loop, args=(2.5,), daemon=True)
    t1.start(); t2.start()
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
