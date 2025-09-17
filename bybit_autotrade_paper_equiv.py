# -*- coding: utf-8 -*-
"""
Bybit Live Trading (Unified v5, Linear Perp) — PAPER-Equivalent
- DeepAR 신호 통과 시 즉시 진입 (ROI 필터/TP 라더/SL 없음)
- 청산: (1) 절대익절 USD(ABS_TP_USD) 도달, (2) 변동성 기반 타임아웃(HORIZON_TIMEOUT)
- 병렬 스캔 옵션 지원
"""
import os, time, math, threading
from typing import Optional, Dict, Tuple, List

# ========= ENV =========
CATEGORY = "linear"
ENTRY_MODE = os.getenv("ENTRY_MODE","model")  # model|inverse|random
SYMBOLS = os.getenv("SYMBOLS","ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT,DOGEUSDT").split(",")

LEVERAGE = float(os.getenv("LEVERAGE","20"))
ENTRY_EQUITY_PCT = float(os.getenv("ENTRY_EQUITY_PCT","0.20"))
TAKER_FEE = float(os.getenv("TAKER_FEE","0.0006"))
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER","1.0"))

ABS_TP_USD = float(os.getenv("ABS_TP_USD","1.0"))
TP1_BPS = float(os.getenv("TP1_BPS","50.0"))  # 타임아웃 계산의 기본 목표
PRED_HORIZ_MIN = int(os.getenv("PRED_HORIZ_MIN","1"))  # fixed 모드 fallback

TIMEOUT_MODE = os.getenv("TIMEOUT_MODE","atr")  # atr|fixed
VOL_WIN = int(os.getenv("VOL_WIN","60"))
TIMEOUT_K = float(os.getenv("TIMEOUT_K","1.5"))
TIMEOUT_MIN = int(os.getenv("TIMEOUT_MIN","3"))
TIMEOUT_MAX = int(os.getenv("TIMEOUT_MAX","20"))
_v = os.getenv("TIMEOUT_TARGET_BPS")
TIMEOUT_TARGET_BPS = float(_v) if (_v is not None and _v.strip()!="") else None

MAX_OPEN = int(os.getenv("MAX_OPEN","5"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC","5"))
SCAN_PARALLEL = str(os.getenv("SCAN_PARALLEL","1")).lower() in ("1","y","yes","true")
SCAN_WORKERS = int(os.getenv("SCAN_WORKERS","8"))
VERBOSE = str(os.getenv("VERBOSE","1")).lower() in ("1","y","yes","true")

TESTNET = str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")
API_KEY = os.getenv("BYBIT_API_KEY","")
API_SECRET = os.getenv("BYBIT_API_SECRET","")

TRADES_CSV = os.getenv("TRADES_CSV","live_trades_paper_equiv.csv")
EQUITY_CSV = os.getenv("EQUITY_CSV","live_equity_paper_equiv.csv")

# ========= IO =========
import csv
TRADES_FIELDS=["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","reason","mode"]
EQUITY_FIELDS=["ts","equity"]

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
    if VERBOSE:
        print(f"[EQUITY] {eq:.2f}")

# ========= Bybit HTTP =========
try:
    from pybit.unified_trading import HTTP
except Exception as e:
    raise RuntimeError("pip install pybit") from e

client = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=TESTNET, timeout=10, recv_window=5000)
mkt    = HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

def get_quote(symbol:str)->Tuple[float,float,float]:
    r=mkt.get_tickers(category=CATEGORY, symbol=symbol)
    row=((r.get("result") or {}).get("list") or [{}])[0]
    f=lambda k: float(row.get(k) or 0.0)
    bid,ask,last=f("bid1Price"),f("ask1Price"),f("lastPrice")
    mid=(bid+ask)/2.0 if (bid>0 and ask>0) else (last or bid or ask)
    return bid,ask,mid or 0.0

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
            "positionIdx":int(p.get("positionIdx") or 0),
        }
    return out

def get_instrument_rule(symbol:str)->Dict[str,float]:
    info=client.get_instruments_info(category=CATEGORY, symbol=symbol)
    lst=(info.get("result") or {}).get("list") or []
    if not lst:
        return {"tickSize":0.0001,"qtyStep":0.001,"minOrderQty":0.001}
    it=lst[0]
    tick=float((it.get("priceFilter") or {}).get("tickSize") or 0.0001)
    lot =float((it.get("lotSizeFilter") or {}).get("qtyStep") or 0.001)
    minq=float((it.get("lotSizeFilter") or {}).get("minOrderQty") or lot)
    return {"tickSize":tick,"qtyStep":lot,"minOrderQty":minq}

def _round_down(x:float, step:float)->float:
    if step<=0: return x
    return math.floor(x/step)*step

# ========= DeepAR =========
import numpy as np, pandas as pd, torch
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.data import GroupNormalizer

MODEL_CKPT = os.getenv("MODEL_CKPT","models/multi_deepar_model.ckpt")
SEQ_LEN = int(os.getenv("SEQ_LEN","60"))
PRED_LEN= int(os.getenv("PRED_LEN","60"))
FEE_SAFETY = float(os.getenv("FEE_SAFETY","1.20"))

DEEPar = None
try:
    DEEPar = DeepAR.load_from_checkpoint(MODEL_CKPT, map_location="cpu").eval()
except Exception:
    DEEPar = None

def _fee_threshold(taker_fee=TAKER_FEE, slip_bps=SLIPPAGE_BPS_TAKER, safety=FEE_SAFETY):
    rt = 2.0*taker_fee + 2.0*(slip_bps/1e4)
    return math.log(1.0 + safety*rt)

_THR = _fee_threshold()

def _recent_1m_df(symbol: str, minutes: int = 200):
    r = mkt.get_kline(category=CATEGORY, symbol=symbol, interval="1", limit=min(minutes,1000))
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
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )

@torch.no_grad()
def deepar_direction(symbol: str):
    # 1순위: 모델
    if DEEPar is not None:
        df = _recent_1m_df(symbol, minutes=SEQ_LEN+PRED_LEN+10)
        if df is not None and len(df) >= (SEQ_LEN+1):
            tsd = _build_infer_dataset(df)
            dl = tsd.to_dataloader(train=False, batch_size=1, num_workers=0)
            mu = float(DEEPar.predict(dl, mode="prediction")[0, 0])
            if   mu >  _THR: base="Buy"
            elif mu < -_THR: base="Sell"
            else:            base=None
        else:
            base=None
    else:
        base=None

    # 백업: MA20/60
    if base is None:
        try:
            k=mkt.get_kline(category=CATEGORY, symbol=symbol, interval="1", limit=120)
            rows=(k.get("result") or {}).get("list") or []
            if len(rows)>=60:
                closes=[float(r[4]) for r in rows][::-1]
                ma20=sum(closes[-20:])/20.0; ma60=sum(closes[-60:])/60.0
                base = "Buy" if ma20>ma60 else ("Sell" if ma20<ma60 else None)
        except Exception:
            base=None

    side=base
    m=(ENTRY_MODE or "model").lower()
    if m=="inverse":
        if side=="Buy": side="Sell"
        elif side=="Sell": side="Buy"
    elif m=="random":
        side = "Buy" if (time.time()*1000)%2<1 else "Sell"
    return side

# ========= Timeout =========
def _recent_vol_bps(symbol: str, minutes: int = None) -> float:
    minutes = minutes or VOL_WIN
    r = mkt.get_kline(category=CATEGORY, symbol=symbol, interval="1", limit=min(minutes, 1000))
    rows = (r.get("result") or {}).get("list") or []
    if len(rows) < 10:
        return 10.0
    closes = [float(z[4]) for z in rows][::-1]
    import numpy as _np
    lr = _np.diff(_np.log(_np.asarray(closes)))
    v_bps = float(_np.median(_np.abs(lr)) * 1e4)
    return max(v_bps, 1.0)

def _calc_timeout_minutes(symbol: str) -> int:
    if TIMEOUT_MODE == "fixed":
        return max(1, int(PRED_HORIZ_MIN))
    target_bps = TIMEOUT_TARGET_BPS if TIMEOUT_TARGET_BPS is not None else TP1_BPS
    v_bps = _recent_vol_bps(symbol, VOL_WIN)
    need = int(math.ceil((target_bps / v_bps) * TIMEOUT_K))
    return int(min(max(need, TIMEOUT_MIN), TIMEOUT_MAX))

# ========= Trading helpers =========
LEV_CAPS: Dict[str, Tuple[float,float]] = {}
def get_symbol_leverage_caps(symbol: str) -> Tuple[float,float]:
    if symbol in LEV_CAPS:
        return LEV_CAPS[symbol]
    info=client.get_instruments_info(category=CATEGORY, symbol=symbol)
    lst=(info.get("result") or {}).get("list") or []
    if not lst:
        caps=(1.0,float(LEVERAGE))
    else:
        lf=(lst[0].get("leverageFilter") or {})
        caps=(float(lf.get("minLeverage") or 1.0), float(lf.get("maxLeverage") or LEVERAGE))
    LEV_CAPS[symbol]=caps
    return caps

def ensure_isolated_and_leverage(symbol: str):
    try:
        minL,maxL=get_symbol_leverage_caps(symbol)
        useL=max(min(float(LEVERAGE),maxL),minL)
        kw=dict(category=CATEGORY, symbol=symbol, buyLeverage=str(useL), sellLeverage=str(useL), tradeMode=1)
        try:
            client.set_leverage(**kw)
        except Exception as e:
            msg=str(e)
            if "110043" in msg or "not modified" in msg:
                pass
            else:
                raise
    except Exception as e:
        print(f"[WARN] set_leverage {symbol}: {e}")

def compute_qty(symbol:str, mid:float, equity:float)->float:
    rule=get_instrument_rule(symbol)
    qty_step=min(max(float(rule["qtyStep"]),1e-12),1e9)
    min_qty=float(rule["minOrderQty"])
    if mid<=0 or equity<=0: return 0.0
    minL,maxL=get_symbol_leverage_caps(symbol)
    lv=max(min(float(LEVERAGE),maxL),minL)
    notional=max(equity*ENTRY_EQUITY_PCT, 10.0)*lv
    qty = notional / mid
    qty = math.floor(qty/qty_step)*qty_step
    if qty < min_qty: qty = math.floor(min_qty/qty_step)*qty_step
    return qty

def place_market(symbol:str, side:str, qty:float)->bool:
    try:
        client.place_order(category=CATEGORY, symbol=symbol, side=side, orderType="Market",
                           qty=str(qty), timeInForce="IOC", reduceOnly=False)
        return True
    except Exception as e:
        print(f"[ERR] market {symbol} {side} {qty} -> {e}")
        return False

def close_market(symbol:str, side:str, qty:float):
    close_side = "Sell" if side=="Buy" else "Buy"
    try:
        client.place_order(category=CATEGORY, symbol=symbol, side=close_side, orderType="Market",
                           qty=str(qty), timeInForce="IOC", reduceOnly=True)
    except Exception as e:
        print(f"[ERR] close {symbol} {close_side} {qty} -> {e}")

# ========= State =========
COOLDOWN_UNTIL: Dict[str,float] = {}
DEADLINE: Dict[str,float] = {}

# ========= Core =========
def try_enter(symbol:str):
    if COOLDOWN_UNTIL.get(symbol,0.0) > time.time(): return
    if symbol in get_positions(): return
    if len(get_positions()) >= MAX_OPEN: return

    side = deepar_direction(symbol)
    if side not in ("Buy","Sell"): return

    bid,ask,mid = get_quote(symbol)
    if mid <= 0: return

    eq = get_wallet_equity()
    qty = compute_qty(symbol, mid, eq)
    if qty <= 0: return

    ensure_isolated_and_leverage(symbol)

    ok = place_market(symbol, side, qty)
    if not ok: return

    # 데드라인
    horizon_min = _calc_timeout_minutes(symbol)
    DEADLINE[symbol] = time.time() + horizon_min*60

    # 로그
    _log_trade({"ts":int(time.time()),"event":"ENTRY","symbol":symbol,"side":side,
                "qty":f"{qty:.8f}","entry":f"{mid:.6f}","exit":"",
                "tp1":"","tp2":"","tp3":"","sl":"","reason":"open","mode":ENTRY_MODE})
    if VERBOSE:
        print(f"[OPEN] {symbol} {side} entry={mid:.4f} timeout={horizon_min}m")

def monitor_loop():
    last_pos=set()
    while True:
        try:
            _log_equity(get_wallet_equity())
            pos=get_positions()
            now=time.time()

            # HORIZON_TIMEOUT 또는 ABS_TP_USD
            for sym,p in pos.items():
                side=p["side"]; qty=float(p["qty"]); entry=float(p["entry"])
                _,_,mark = get_quote(sym)
                if mark<=0: continue
                pnl = (mark-entry)*qty if side=="Buy" else (entry-mark)*qty
                hit_abs = pnl >= ABS_TP_USD
                hit_time= (now >= DEADLINE.get(sym, now+1e9))
                if hit_abs or hit_time:
                    close_market(sym, side, qty)
                    COOLDOWN_UNTIL[sym]=time.time()+COOLDOWN_SEC
                    _log_trade({"ts":int(time.time()),"event":"EXIT","symbol":sym,"side":side,
                                "qty":f"{qty:.8f}","entry":f"{entry:.6f}","exit":f"{mark:.6f}",
                                "tp1":"","tp2":"","tp3":"","sl":"",
                                "reason": "ABS_TP" if hit_abs else "HORIZON_TIMEOUT",
                                "mode":ENTRY_MODE})
                    DEADLINE.pop(sym, None)

            time.sleep(1.0)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("[MONITOR ERR]", e); time.sleep(2.0)

from concurrent.futures import ThreadPoolExecutor
def scanner_loop():
    while True:
        try:
            if len(get_positions()) >= MAX_OPEN:
                time.sleep(1.0); continue
            syms = SYMBOLS[:MAX_OPEN]  # paper와 유사: 상위 N만 본다고 가정
            if SCAN_PARALLEL:
                with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
                    ex.map(try_enter, syms)
            else:
                for s in syms: try_enter(s)
            time.sleep(2.0)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("[SCANNER ERR]", e); time.sleep(2.0)

def main():
    print(f"[START] PAPER-Equivalent LIVE | MODE={ENTRY_MODE} TESTNET={TESTNET}")
    t1=threading.Thread(target=monitor_loop, daemon=True)
    t2=threading.Thread(target=scanner_loop, daemon=True)
    t1.start(); t2.start()
    while True:
        time.sleep(60)

if __name__=="__main__":
    main()
