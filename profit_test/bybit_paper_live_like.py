# -*- coding: utf-8 -*-
"""
Paper Trading — LIVE-Equivalent
- 진입: DeepAR 신호 + 기대 ROI 필터(edge_bps >= entry/exit 비용 + MIN_EXPECTED_ROI_BPS)
- 청산: TP1/TP2 분할(RO 지정가 시뮬레이션), TP3/SL(시장가 시뮬), TP2 후 BE, 이후 트레일링
- 변동성 기반 타임아웃 추가
- 병렬 스캔 옵션
"""
import os, time, math, csv, threading
from typing import Optional, Dict, Tuple, List

# ===== ENV =====
# ===== ENV =====
CATEGORY = "linear"
ENTRY_MODE = os.getenv("ENTRY_MODE","model")
SYMBOLS = os.getenv("SYMBOLS","ASTERUSDT,AIAUSDT,0GUSDT,STBLUSDT,WLFIUSDT,LINEAUSDT,AVMTUSDT,BARDUSDT,SOMIUSDT,UBUSDT,OPENUSDT").split(",")

START_EQUITY = float(os.getenv("START_EQUITY","1000.0"))
LEVERAGE = float(os.getenv("LEVERAGE","20"))
ENTRY_EQUITY_PCT = float(os.getenv("ENTRY_EQUITY_PCT","0.20"))

TAKER_FEE = float(os.getenv("TAKER_FEE","0.0006"))
MAKER_FEE = float(os.getenv("MAKER_FEE","0.0"))
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER","1.0"))
SLIPPAGE_BPS_MAKER = float(os.getenv("SLIPPAGE_BPS_MAKER","0.0"))

MIN_EXPECTED_ROI_BPS = float(os.getenv("MIN_EXPECTED_ROI_BPS","20.0"))
ABS_TP_USD = float(os.getenv("ABS_TP_USD","5.0"))  # ← 추가: 절대 이익 달러 기준 익절


MIN_EXPECTED_ROI_BPS = float(os.getenv("MIN_EXPECTED_ROI_BPS","20.0"))

TP1_BPS,TP2_BPS,TP3_BPS = [float(os.getenv(k,v)) for k,v in [
    ("TP1_BPS","50.0"),("TP2_BPS","100.0"),("TP3_BPS","200.0")
]]
TP1_RATIO = float(os.getenv("TP1_RATIO","0.40"))
TP2_RATIO = float(os.getenv("TP2_RATIO","0.35"))
SL_ROI_PCT = float(os.getenv("SL_ROI_PCT","0.01"))
BE_EPS_BPS = float(os.getenv("BE_EPS_BPS","2.0"))
TRAIL_BPS = float(os.getenv("TRAIL_BPS","50.0"))
TRAIL_AFTER_TIER = int(os.getenv("TRAIL_AFTER_TIER","2"))  # 1->TP1후, 2->TP2후

# Timeout(ATR)
TIMEOUT_MODE = os.getenv("TIMEOUT_MODE","atr")  # atr|fixed
VOL_WIN = int(os.getenv("VOL_WIN","60"))
TIMEOUT_K = float(os.getenv("TIMEOUT_K","1.5"))
TIMEOUT_MIN = int(os.getenv("TIMEOUT_MIN","3"))
TIMEOUT_MAX = int(os.getenv("TIMEOUT_MAX","20"))
_v = os.getenv("TIMEOUT_TARGET_BPS")
TIMEOUT_TARGET_BPS = float(_v) if (_v is not None and _v.strip()!="") else None
PRED_HORIZ_MIN = int(os.getenv("PRED_HORIZ_MIN","1"))

MAX_OPEN = int(os.getenv("MAX_OPEN","10"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC","5"))
SCAN_PARALLEL = str(os.getenv("SCAN_PARALLEL","1")).lower() in ("1","y","yes","true")
SCAN_WORKERS  = int(os.getenv("SCAN_WORKERS","8"))
VERBOSE = str(os.getenv("VERBOSE","1")).lower() in ("1","y","yes","true")

# ===== IO =====
TRADES_CSV = os.getenv("TRADES_CSV",f"paper_trades_live_like_{ENTRY_MODE}_{START_EQUITY}_{LEVERAGE}.csv")
EQUITY_CSV = os.getenv("EQUITY_CSV",f"paper_equity_live_like_{ENTRY_MODE}_{START_EQUITY}_{LEVERAGE}.csv")
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

# ===== Market (public only) =====
try:
    from pybit.unified_trading import HTTP
except Exception as e:
    raise RuntimeError("pip install pybit") from e

TESTNET = str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")
mkt = HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

def get_quote(symbol: str) -> Tuple[float,float,float]:
    r = mkt.get_tickers(category=CATEGORY, symbol=symbol)
    row = ((r.get("result") or {}).get("list") or [{}])[0]
    bid = float(row.get("bid1Price") or 0.0)
    ask = float(row.get("ask1Price") or 0.0)
    last = float(row.get("lastPrice") or 0.0)
    mid = (bid+ask)/2.0 if bid>0 and ask>0 else (last or bid or ask)
    return bid, ask, float(mid or 0.0)

# ===== Equity =====
_equity = START_EQUITY
def get_wallet_equity()->float:
    return float(_equity)

def apply_pnl_and_fee(side:str, entry:float, exitp:float, qty:float,
                      entry_exec="taker", exit_exec="taker"):
    global _equity
    gross = (exitp-entry)*qty if side=="Buy" else (entry-exitp)*qty
    fee_in  = (entry*qty) * (TAKER_FEE if entry_exec=="taker" else MAKER_FEE)
    fee_out = (exitp*qty) * (TAKER_FEE if exit_exec =="taker" else MAKER_FEE)
    _equity = _equity + gross - fee_in - fee_out
    _log_equity(_equity)
    return gross, (fee_in+fee_out), _equity

# ===== MODEL =====
import numpy as np, pandas as pd, torch
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.data import GroupNormalizer

MODEL_CKPT = os.getenv("MODEL_CKPT","models/multi_deepar_best.ckpt")
SEQ_LEN = int(os.getenv("SEQ_LEN","60"))
PRED_LEN= int(os.getenv("PRED_LEN","60"))
FEE_SAFETY = float(os.getenv("FEE_SAFETY","1.20"))

DEEPar=None
try:
    DEEPar = DeepAR.load_from_checkpoint(MODEL_CKPT, map_location="cpu").eval()
except Exception:
    DEEPar=None

def _fee_threshold(taker_fee=TAKER_FEE, slip_bps=SLIPPAGE_BPS_TAKER, safety=FEE_SAFETY):
    rt = 2.0*taker_fee + 2.0*(slip_bps/1e4)
    return math.log(1.0 + safety*rt)
_THR=_fee_threshold()

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
    base=None
    if DEEPar is not None:
        df = _recent_1m_df(symbol, minutes=SEQ_LEN+PRED_LEN+10)
        if df is not None and len(df) >= (SEQ_LEN+1):
            tsd = _build_infer_dataset(df)
            dl  = tsd.to_dataloader(train=False, batch_size=1, num_workers=0)
            mu  = float(DEEPar.predict(dl, mode="prediction")[0, 0])
            if   mu >  _THR: base="Buy"
            elif mu < -_THR: base="Sell"
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
    m=(ENTRY_MODE or "model").lower()
    if m=="inverse":
        if base=="Buy": base="Sell"
        elif base=="Sell": base="Buy"
    elif m=="random":
        base = "Buy" if (time.time()*1000)%2<1 else "Sell"
    return base

# ===== helpers =====
def tp_from_bps(entry:float, bps:float, side:str)->float:
    return entry*(1.0 + bps/10000.0) if side=="Buy" else entry*(1.0 - bps/10000.0)

def _bps_between(p1:float,p2:float)->float:
    return abs((p2 - p1) / max(p1, 1e-12)) * 10_000.0

def _entry_cost_bps(exec_mode="taker")->float:
    fee = (TAKER_FEE if exec_mode=="taker" else MAKER_FEE)*10_000.0
    slip= (SLIPPAGE_BPS_TAKER if exec_mode=="taker" else SLIPPAGE_BPS_MAKER)
    return max(0.0, fee + max(0.0, slip))

def _exit_cost_bps(exec_mode="taker")->float:
    fee = (TAKER_FEE if exec_mode=="taker" else MAKER_FEE)*10_000.0
    slip= (SLIPPAGE_BPS_TAKER if exec_mode=="taker" else SLIPPAGE_BPS_MAKER)
    return max(0.0, fee + max(0.0, slip))

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

def calc_timeout_minutes(symbol: str) -> int:
    if TIMEOUT_MODE == "fixed":
        return max(1, int(PRED_HORIZ_MIN))
    target_bps = TIMEOUT_TARGET_BPS if TIMEOUT_TARGET_BPS is not None else TP1_BPS
    v_bps = _recent_vol_bps(symbol, VOL_WIN)
    need = int(math.ceil((target_bps / v_bps) * TIMEOUT_K))
    return int(min(max(need, TIMEOUT_MIN), TIMEOUT_MAX))

# ===== Paper Broker =====
class PaperBroker:
    def __init__(self):
        self.pos: Dict[str,dict] = {}  # symbol -> {side, entry, qty, opened, sl, tp1,tp2,tp3, be_moved, deadline}
        self.qty_step = 0.001
        self.min_qty = 0.001

    def try_open(self, symbol: str, side: str, entry: float, qty: float, timeout_min: int):
        if qty < self.min_qty: return False
        self.pos[symbol] = {
            "side": side, "entry": entry, "qty": qty, "opened": int(time.time()),
            "tp1": tp_from_bps(entry, TP1_BPS, side),
            "tp2": tp_from_bps(entry, TP2_BPS, side),
            "tp3": tp_from_bps(entry, TP3_BPS, side),
            "sl" : entry*(1.0 - SL_ROI_PCT) if side=="Buy" else entry*(1.0 + SL_ROI_PCT),
            "tp1_filled": False, "tp2_filled": False, "be_moved": False,
            "deadline": int(time.time()) + timeout_min*60
        }
        return True

    def positions(self): return self.pos

    def close_all(self, symbol: str, exitp: float, reason: str):
        if symbol not in self.pos: return None
        p = self.pos.pop(symbol)
        side=p["side"]; entry=p["entry"]; qty=p["qty"]
        gross, fee, new_eq = apply_pnl_and_fee(side, entry, exitp, qty, "taker", "taker")
        _log_trade({"ts":int(time.time()),"event":"EXIT","symbol":symbol,
                    "side":side,"qty":f"{qty:.8f}","entry":f"{entry:.6f}","exit":f"{exitp:.6f}",
                    "tp1":"","tp2":"","tp3":"","sl":"","reason":reason,"mode":"paper"})
        return gross, fee, new_eq

BROKER = PaperBroker()
COOLDOWN_UNTIL: Dict[str,float] = {}
ENTRY_LOCK = threading.Lock()

# ===== qty =====
def compute_qty(symbol:str, mid:float)->float:
    if mid<=0: return 0.0
    notional = max(get_wallet_equity()*ENTRY_EQUITY_PCT, 10.0) * LEVERAGE
    qty = notional / mid
    step = BROKER.qty_step
    qty = math.floor(qty/step)*step
    if qty < BROKER.min_qty:
        qty = math.floor(BROKER.min_qty/step)*step
    return qty

# ===== try_enter (LIVE-Equivalent) =====
def try_enter(symbol: str):
    if symbol in BROKER.positions(): return
    if COOLDOWN_UNTIL.get(symbol,0.0) > time.time(): return

    side = deepar_direction(symbol)
    if side not in ("Buy","Sell"): return

    bid,ask,mid = get_quote(symbol)
    if mid <= 0: return

    # ROI 필터: edge_bps >= 비용(entry+exit) + MIN_EXPECTED_ROI_BPS
    entry_ref = float(ask if side=="Buy" else bid)
    tp1_px = tp_from_bps(entry_ref, TP1_BPS, side)
    edge_bps = _bps_between(entry_ref, tp1_px)
    cost_bps = _entry_cost_bps("taker") + _exit_cost_bps("taker")
    if edge_bps < (cost_bps + MIN_EXPECTED_ROI_BPS):
        return

    qty = compute_qty(symbol, mid)
    if qty <= 0: return

    timeout_min = calc_timeout_minutes(symbol)

    # 경쟁구간 보호
    with ENTRY_LOCK:
        if symbol in BROKER.positions(): return
        if not BROKER.try_open(symbol, side, mid, qty, timeout_min): return

        p = BROKER.positions()[symbol]
        _log_trade({"ts":int(time.time()),"event":"ENTRY","symbol":symbol,"side":side,
                    "qty":f"{qty:.8f}","entry":f"{mid:.6f}","exit":"",
                    "tp1":f"{p['tp1']:.6f}","tp2":f"{p['tp2']:.6f}","tp3":f"{p['tp3']:.6f}",
                    "sl":f"{p['sl']:.6f}","reason":"open","mode":"paper"})
        if VERBOSE:
            print(f"[OPEN] {symbol} {side} entry={mid:.4f} timeout={timeout_min}m")

# ===== monitor: TP/SL/BE/Trail/Timeout 시뮬 =====
def monitor_loop(poll_sec: float = 1.0):
    while True:
        try:
            _log_equity(get_wallet_equity())
            now=int(time.time())
            for sym,p in list(BROKER.positions().items()):
                side=p["side"]; entry=p["entry"]; qty=p["qty"]
                tp1=p["tp1"]; tp2=p["tp2"]; tp3=p["tp3"]; sl=p["sl"]
                _,_,mark = get_quote(sym)
                if mark<=0: continue

                # TP1/TP2 분할 시뮬: 가격이 목표 통과 시 해당 비율만 청산한 것으로 처리하여 평균가 조정
                if (not p["tp1_filled"]) and ((mark>=tp1 and side=="Buy") or (mark<=tp1 and side=="Sell")):
                    part = qty*TP1_RATIO
                    apply_pnl_and_fee(side, entry, tp1, part, "taker","taker")
                    qty = qty - part
                    p["tp1_filled"]=True
                    _log_trade({"ts":int(time.time()),"event":"TP1_FILL","symbol":sym,"side":side,
                                "qty":f"{part:.8f}","entry":f"{entry:.6f}","exit":f"{tp1:.6f}",
                                "tp1":"","tp2":"","tp3":"","sl":"","reason":"","mode":"paper"})
                    p["qty"]=qty  # 잔량 갱신

                if (qty>0) and (not p["tp2_filled"]) and ((mark>=tp2 and side=="Buy") or (mark<=tp2 and side=="Sell")):
                    part = qty*TP2_RATIO/(1.0-TP1_RATIO) * (1.0-TP1_RATIO)  # 남은 잔량 기준
                    part = min(part, qty)
                    apply_pnl_and_fee(side, entry, tp2, part, "taker","taker")
                    qty = qty - part
                    p["tp2_filled"]=True
                    _log_trade({"ts":int(time.time()),"event":"TP2_FILL","symbol":sym,"side":side,
                                "qty":f"{part:.8f}","entry":f"{entry:.6f}","exit":f"{tp2:.6f}",
                                "tp1":"","tp2":"","tp3":"","sl":"","reason":"","mode":"paper"})
                    p["qty"]=qty

                # BE 이동
                if p["tp2_filled"] and not p["be_moved"]:
                    be_px = entry*(1.0 + BE_EPS_BPS/10000.0) if side=="Buy" else entry*(1.0 - BE_EPS_BPS/10000.0)
                    p["sl"] = be_px
                    p["be_moved"]=True
                    _log_trade({"ts":int(time.time()),"event":"MOVE_SL","symbol":sym,"side":side,
                                "qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"",
                                "sl":f"{be_px:.6f}","reason":"BE","mode":"paper"})

                # 트레일링 SL
                if (p["tp1_filled"] if TRAIL_AFTER_TIER==1 else p["tp2_filled"]):
                    if side=="Buy":
                        new_sl = mark*(1.0 - TRAIL_BPS/10000.0)
                        p["sl"] = max(p["sl"], new_sl)
                    else:
                        new_sl = mark*(1.0 + TRAIL_BPS/10000.0)
                        p["sl"] = min(p["sl"], new_sl)

                # SL/TP3 도달 → 잔량 전량 종료
                # SL/TP3 도달 → 잔량 전량 종료
                if qty > 0:
                    # 추가: 절대익절(달러) 조건
                    pnl_unreal = (mark - entry) * qty if side == "Buy" else (entry - mark) * qty
                    hit_abs = pnl_unreal >= ABS_TP_USD

                    hit_tp3 = (mark >= tp3 and side == "Buy") or (mark <= tp3 and side == "Sell")
                    hit_sl = (mark <= p["sl"] and side == "Buy") or (mark >= p["sl"] and side == "Sell")
                    hit_to = (now >= p["deadline"])
                    if hit_tp3 or hit_sl or hit_to or hit_abs:
                        reason = "ABS_TP" if hit_abs else (
                            "TP3" if hit_tp3 else ("SL" if hit_sl else "HORIZON_TIMEOUT"))
                        apply_pnl_and_fee(side, entry, mark, qty, "taker", "taker")
                        _log_trade({"ts": int(time.time()), "event": "EXIT", "symbol": sym, "side": side,
                                    "qty": f"{qty:.8f}", "entry": f"{entry:.6f}", "exit": f"{mark:.6f}",
                                    "tp1": "", "tp2": "", "tp3": "", "sl": "", "reason": reason, "mode": "paper"})
                        BROKER.positions().pop(sym, None)
                        COOLDOWN_UNTIL[sym] = time.time() + COOLDOWN_SEC

            time.sleep(poll_sec)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("[MONITOR ERR]", e); time.sleep(2.0)

# ===== scanner =====
from concurrent.futures import ThreadPoolExecutor
def scanner_loop(iter_delay:float=2.5):
    while True:
        try:
            if len(BROKER.positions()) >= MAX_OPEN:
                time.sleep(iter_delay); continue
            syms = SYMBOLS[:MAX_OPEN]
            if SCAN_PARALLEL:
                with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
                    ex.map(try_enter, syms)
            else:
                for s in syms: try_enter(s)
            time.sleep(iter_delay)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("[SCANNER ERR]", e); time.sleep(2.0)

# ===== main =====
def main():
    print(f"[START] PAPER LIVE-Equivalent | MODE={ENTRY_MODE} TESTNET={TESTNET}")
    _log_equity(get_wallet_equity())
    t1=threading.Thread(target=monitor_loop, daemon=True)
    t2=threading.Thread(target=scanner_loop, daemon=True)
    t1.start(); t2.start()
    while True:
        time.sleep(60)

if __name__=="__main__":
    main()
