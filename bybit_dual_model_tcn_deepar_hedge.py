# -*- coding: utf-8 -*-
"""
Bybit Live Trading (Unified v5, Linear Perp) — DeepAR(main) + TCN(alt) Consensus, HEDGE enabled + META GATE
- 헤지: 동일 심볼 Buy/Sell 동시 운용. 상태키 = (symbol, side)
- 진입: DeepAR μ와 TCN μ가 같은 방향이고 임계치 충족일 때만
- META GATE: 최근 성과가 기준 미달이면 해당 '심볼'을 일정 시간 쉬게 함(자동 재개). inverse 전환 없음.
- TP1/TP2: reduce-only PostOnly 지정가, TP3/SL: trading_stop 유지
- BE/트레일/타임아웃 포함
- 메이커/테이커 선택, 미체결 만료/취소, 최소 기대 ROI 필터
"""
import os, time, math, csv, threading, warnings, logging, collections
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.serialization import add_safe_globals
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.deepar import DeepAR
import torch.nn as nn

# ===== lightning noise off =====
warnings.filterwarnings("ignore", module="lightning")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
os.environ.setdefault("PL_DISABLE_FORK", "1")
os.environ.setdefault("PYTORCH_LIGHTNING_DISABLE_PBAR", "1")

# ===== allowlist for checkpoint load =====
class SignAwareNormalLoss(NormalDistributionLoss):
    pass
add_safe_globals([GroupNormalizer, NormalDistributionLoss, SignAwareNormalLoss, DataFrame])

# ===== config =====
CATEGORY = "linear"
ENTRY_MODE = os.getenv("ENTRY_MODE", "model").lower()
LEVERAGE   = float(os.getenv("LEVERAGE", "20"))
TAKER_FEE  = float(os.getenv("TAKER_FEE", "0.0006"))
MAKER_FEE  = float(os.getenv("MAKER_FEE", "0.0"))
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "taker").lower()   # maker|taker
EXIT_EXEC_MODE = os.getenv("EXIT_EXEC_MODE", "taker").lower()   # maker|taker
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER", "1.0"))
SLIPPAGE_BPS_MAKER = float(os.getenv("SLIPPAGE_BPS_MAKER", "0.0"))
FEE_SAFETY = float(os.getenv("FEE_SAFETY","1.2"))  # fee 모드 안전계수
MIN_EXPECTED_ROI_BPS = float(os.getenv("MIN_EXPECTED_ROI_BPS", "40.0"))
WAIT_FOR_MAKER_FILL_SEC = float(os.getenv("WAIT_FOR_MAKER_FILL_SEC", "30"))
CANCEL_UNFILLED_MAKER = str(os.getenv("CANCEL_UNFILLED_MAKER", "1")).lower() in ("1","y","yes","true")
MAX_OPEN   = int(os.getenv("MAX_OPEN", "8"))

SL_AMEND_MIN_TICKS = int(os.getenv("SL_AMEND_MIN_TICKS","1"))   # SL 변경 최소 틱
TP_AMEND_MIN_TICKS = int(os.getenv("TP_AMEND_MIN_TICKS","1"))   # TP 변경 최소 틱

# SL/TP
SL_ROI_PCT      = float(os.getenv("SL_ROI_PCT", "0.01"))
HARD_SL_PCT     = 0.01
TP1_BPS, TP2_BPS, TP3_BPS = 50.0, 80.0, 130.0
TP1_RATIO, TP2_RATIO      = 0.30, 0.45
BE_EPS_BPS = 2.0
TRAIL_BPS = 50.0
TRAIL_AFTER_TIER = 2

# sizing & re-entry
ENTRY_EQUITY_PCT   = float(os.getenv("ENTRY_EQUITY_PCT", "0.2"))
REENTRY_ON_DIP     = str(os.getenv("REENTRY_ON_DIP","1")).lower() in ("1","y","yes","true")
REENTRY_PCT        = float(os.getenv("REENTRY_PCT", "0.5"))
REENTRY_SIZE_PCT   = float(os.getenv("REENTRY_SIZE_PCT","0.8"))

# signals cfg
SEQ_LEN  = int(os.getenv("SEQ_LEN", "240"))
PRED_LEN = int(os.getenv("PRED_LEN", "60"))
MODEL_CKPT_MAIN = os.getenv("MODEL_CKPT_MAIN", "./models/multi_deepar_best_main.ckpt")
TCN_CKPT        = os.getenv("TCN_CKPT", "tcn_best_new_coin.pt")
THR_MODE = os.getenv("THR_MODE", "fixed").lower()  # fixed|fee
MODEL_THR_BPS = float(os.getenv("MODEL_THR_BPS", "5.0"))
DUAL_RULE = os.getenv("DUAL_RULE", "loose").lower() # loose|strict
DEBUG_SIGNALS = str(os.getenv("DEBUG_SIGNALS","1")).lower() in ("1","y","yes","true")
ALLOW_TCN_FALLBACK = str(os.getenv("ALLOW_TCN_FALLBACK","0")).lower() in ("1","y","yes","true")

# ===== META GATE =====
META_ON = int(os.getenv("META_ON","1"))
META_WIN = int(os.getenv("META_WIN","30"))                 # 최근 n건(심볼 단위)
META_MIN_HIT = float(os.getenv("META_MIN_HIT","0.5"))      # 적중률 최소
META_MIN_EVR = float(os.getenv("META_MIN_EVR","0.0"))      # 평균 bps 최소
META_MAX_DD_BPS = float(os.getenv("META_MAX_DD_BPS","-300.0"))  # 허용 최대 낙폭(bps, 음수)
META_COOLDOWN = int(os.getenv("META_COOLDOWN","60"))      # 쉬는 시간(초)
META_DEBUG = str(os.getenv("META_DEBUG","1")).lower() in ("1","y","yes","true")

# runtime state (hedge-aware)
STATE: Dict[Tuple[str,str],dict] = {}             # (sym, side)->state
COOLDOWN_UNTIL: Dict[Tuple[str,str],float] = {}
PENDING_MAKER: Dict[Tuple[str,str],dict] = {}
LAST_EXIT: Dict[Tuple[str,str],dict] = {}
GLOBAL_COOLDOWN_UNTIL = 0.0
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC","10"))
MAX_HOLD_SEC = int(os.getenv("MAX_HOLD_SEC","3600"))

# deepar fault gating
SKIP_SYMBOLS = set()
FAIL_CNT = collections.defaultdict(int)
FAIL_MAX = int(os.getenv("DEEPar_FAIL_MAX","3"))

# META state (심볼 단위)
from collections import deque
META_TRADES: Dict[str, deque] = collections.defaultdict(lambda: deque(maxlen=max(5, META_WIN)))
META_BLOCK_UNTIL: Dict[str, float] = {}

# io
TRADES_CSV = os.getenv("TRADES_CSV","live_trades.csv")
EQUITY_CSV = os.getenv("EQUITY_CSV","live_equity.csv")
TRADES_FIELDS=["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","reason","mode"]
EQUITY_FIELDS=["ts","equity"]
def _ensure_csv():
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(TRADES_FIELDS)
    if not os.path.exists(EQUITY_CSV):
        with open(EQUITY_CSV,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(EQUITY_FIELDS)
def _log_trade(row:dict):
    _ensure_csv()
    with open(TRADES_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(k,"") for k in TRADES_FIELDS])
def _log_equity(eq:float):
    _ensure_csv()
    with open(EQUITY_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([int(time.time()), f"{float(eq):.6f}"])

# ===== http =====
try:
    from pybit.unified_trading import HTTP
except Exception as e:
    raise RuntimeError("pip install pybit") from e
TESTNET = str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")
API_KEY = os.getenv("BYBIT_API_KEY","Dlp4eJD6YFmO99T8vC")
API_SEC = os.getenv("BYBIT_API_SECRET","YYYB5tMw2TWvfVF5wqi6lQRHqEIiDSpDJF1U")
client = HTTP(api_key=API_KEY, api_secret=API_SEC, testnet=TESTNET, timeout=10, recv_window=5000)
mkt    = HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

# ===== market utils =====
LEV_CAPS: Dict[str, Tuple[float,float]] = {}
def get_symbol_leverage_caps(symbol: str) -> Tuple[float,float]:
    if symbol in LEV_CAPS: return LEV_CAPS[symbol]
    info=client.get_instruments_info(category=CATEGORY, symbol=symbol)
    lst=(info.get("result") or {}).get("list") or []
    if not lst: caps=(1.0, float(LEVERAGE))
    else:
        lf=(lst[0].get("leverageFilter") or {})
        caps=(float(lf.get("minLeverage") or 1.0), float(lf.get("maxLeverage") or LEVERAGE))
    LEV_CAPS[symbol]=caps; return caps


# ==== get_instrument_rule에 stop 필드 유지 ====
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

def _round_to_tick(x: float, tick: float) -> float:
    if tick<=0: return x
    return math.floor(x/tick)*tick

def get_quote(symbol:str)->Tuple[float,float,float]:
    r=mkt.get_tickers(category=CATEGORY, symbol=symbol)
    row=((r.get("result") or {}).get("list") or [{}])[0]
    f=lambda k: float(row.get(k) or 0.0)
    bid,ask,last=f("bid1Price"),f("ask1Price"),f("lastPrice")
    mid=(bid+ask)/2.0 if (bid>0 and ask>0) else (last or bid or ask)
    return bid,ask,mid or 0.0

# ===== wallet/positions (hedge-aware) =====
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
def get_positions()->Dict[Tuple[str,str],dict]:
    res=client.get_positions(category=CATEGORY, settleCoin="USDT")
    lst=(res.get("result") or {}).get("list") or []
    out={}
    for p in lst:
        sym=p.get("symbol"); size=float(p.get("size") or 0.0)
        if not sym or size<=0: continue
        side=p.get("side")
        out[(sym,side)] = {
            "side":side,
            "qty":float(p.get("size") or 0.0),
            "entry":float(p.get("avgPrice") or 0.0),
            "liq":float(p.get("liqPrice") or 0.0),
            "positionIdx":int(p.get("positionIdx") or (1 if side=="Buy" else 2)),
            "stopLoss": float(p.get("stopLoss") or 0.0),
            "takeProfit": float(p.get("takeProfit") or 0.0),
        }
    return out

def _need_amend(old_px: float, new_px: float, tick: float, min_ticks: int) -> bool:
    if new_px is None or new_px<=0: return False
    if old_px is None or old_px<=0: return True
    return abs(new_px - old_px) >= (tick * max(1, int(min_ticks)))

def ensure_isolated_and_leverage(symbol: str):
    try:
        minL, maxL = get_symbol_leverage_caps(symbol)
        useL = max(min(float(LEVERAGE), maxL), minL)
        kw=dict(category=CATEGORY, symbol=symbol, buyLeverage=str(useL), sellLeverage=str(useL), tradeMode=1)
        try: client.set_leverage(**kw)
        except Exception as e:
            msg=str(e)
            if "110043" in msg or "not modified" in msg: pass
            else: raise
    except Exception as e:
        print(f"[WARN] set_leverage {symbol}: {e}")

def _order_with_mode_retry(params: dict, side: Optional[str] = None):
    if "positionIdx" not in params and side in ("Buy", "Sell"):
        params = dict(params, positionIdx=(1 if side == "Buy" else 2))
    return client.place_order(**params)

def place_market(symbol: str, side: str, qty: float) -> bool:
    try:
        rule=get_instrument_rule(symbol)
        q=_round_down(qty, rule["qtyStep"])
        if q<rule["minOrderQty"]: return False
        p=dict(category=CATEGORY, symbol=symbol, side=side, orderType="Market",
               qty=str(q), timeInForce="IOC", reduceOnly=False)
        _order_with_mode_retry(p, side=side)
        return True
    except Exception as e:
        print(f"[ERR] place_market {symbol} {side} {qty} -> {e}")
        return False

def place_limit_postonly(symbol: str, side: str, qty: float, price: float) -> Optional[str]:
    try:
        rule=get_instrument_rule(symbol)
        q=_round_down(qty, rule["qtyStep"]); px=_round_to_tick(price, rule["tickSize"])
        if q<rule["minOrderQty"]: return None
        p=dict(category=CATEGORY, symbol=symbol, side=side, orderType="Limit",
               qty=str(q), price=str(px), timeInForce="PostOnly", reduceOnly=False)
        ret=_order_with_mode_retry(p, side=side)
        oid=((ret or {}).get("result") or {}).get("orderId")
        print(f"[ORDER][POST_ONLY] {symbol} {side} {q}@{px} oid={oid}")
        return oid
    except Exception as e:
        print(f"[ERR] place_limit_postonly {symbol} {side} {qty}@{price} -> {e}")
        return None

def place_reduce_limit(symbol: str, close_side: str, qty: float, price: float, position_idx: int) -> bool:
    try:
        rule = get_instrument_rule(symbol)
        q = _round_down(qty, rule["qtyStep"]); px = _round_to_tick(price, rule["tickSize"])
        if q < rule["minOrderQty"]: return False
        p = dict(category=CATEGORY, symbol=symbol, side=close_side, orderType="Limit",
                 qty=str(q), price=str(px), timeInForce="PostOnly", reduceOnly=True,
                 positionIdx=int(position_idx))
        _order_with_mode_retry(p)
        return True
    except Exception as e:
        print(f"[WARN] reduce_limit {symbol} {close_side} {qty}@{price}: {e}")
        return False

# ==== set_stop_with_retry 교체: 동일값이면 호출 생략, 34040 무시 ====
def set_stop_with_retry(symbol: str, side: str, sl_price: Optional[float] = None, tp_price: Optional[float] = None) -> bool:
    rule=get_instrument_rule(symbol); tick=float(rule.get("tickSize") or 0.0)
    if sl_price is not None: sl_price=_round_to_tick(float(sl_price), tick)
    if tp_price is not None: tp_price=_round_to_tick(float(tp_price), tick)

    pos = get_positions().get((symbol, side))
    cur_sl = float((pos or {}).get("stopLoss") or 0.0)
    cur_tp = float((pos or {}).get("takeProfit") or 0.0)

    # 변경 필요 없으면 스킵
    if sl_price is not None and not _need_amend(cur_sl, sl_price, tick, SL_AMEND_MIN_TICKS):
        sl_price = None
    if tp_price is not None and not _need_amend(cur_tp, tp_price, tick, TP_AMEND_MIN_TICKS):
        tp_price = None
    if sl_price is None and tp_price is None:
        return True

    params=dict(category=CATEGORY, symbol=symbol, tpslMode="Full",
                slTriggerBy="MarkPrice", tpTriggerBy="MarkPrice",
                positionIdx=(1 if side=="Buy" else 2))
    if sl_price is not None:
        params["stopLoss"]=f"{sl_price:.10f}".rstrip("0").rstrip(".")
        params["slOrderType"]="Market"
    if tp_price is not None:
        params["takeProfit"]=f"{tp_price:.10f}".rstrip("0").rstrip(".")
        params["tpOrderType"]="Market"
    try:
        ret=client.set_trading_stop(**params)
        return (ret.get("retCode")==0)
    except Exception as e:
        msg=str(e)
        # 동일값 갱신 시 34040 → 성공으로 처리
        if "34040" in msg or "not modified" in msg.lower():
            return True
        print(f"[STOP ERR] {symbol}/{side}: {e}")
        return False
# ===== sizing =====
def compute_qty_by_equity_pct(symbol: str, mid: float, equity_now: float, pct: float) -> Tuple[float,float]:
    rule = get_instrument_rule(symbol)
    qty_step, min_qty = float(rule["qtyStep"]), float(rule["minOrderQty"])
    if mid<=0 or equity_now<=0 or pct<=0: return 0.0, 0.0
    MIN_NOTIONAL_USDT = float(os.getenv("MIN_NOTIONAL_USDT","10.0"))
    MAX_NOTIONAL_ABS  = float(os.getenv("MAX_NOTIONAL_ABS","2000.0"))
    MAX_NOTIONAL_PCT  = float(os.getenv("MAX_NOTIONAL_PCT_OF_EQUITY","20.0"))
    minL, maxL = get_symbol_leverage_caps(symbol)
    lv_used = max(min(float(LEVERAGE), maxL), minL)
    target_notional = max(equity_now * pct, MIN_NOTIONAL_USDT) * lv_used
    cap_abs  = MAX_NOTIONAL_ABS * lv_used
    cap_pct  = equity_now * (MAX_NOTIONAL_PCT/100.0) * lv_used
    capped_notional = min(max(target_notional, MIN_NOTIONAL_USDT), cap_abs, cap_pct)
    qty = math.ceil((capped_notional / mid) / qty_step) * qty_step
    if qty < min_qty: qty = math.ceil(min_qty/qty_step)*qty_step
    return qty, qty*mid

# ===== helpers =====
def tp_from_bps(entry:float, bps:float, side:str)->float:
    return entry*(1.0 + bps/10000.0) if side=="Buy" else entry*(1.0 - bps/10000.0)
def compute_sl(entry:float, side:str)->float:
    dyn = entry*(1.0 - SL_ROI_PCT) if side=="Buy" else entry*(1.0 + SL_ROI_PCT)
    hard = entry*(1.0 - HARD_SL_PCT) if side=="Buy" else entry*(1.0 + HARD_SL_PCT)
    return max(dyn, hard) if side=="Buy" else min(dyn, hard)
def _bps_between(p1:float,p2:float)->float:
    return abs((p2 - p1) / max(p1, 1e-12)) * 10_000.0
def _entry_cost_bps()->float:
    fee = (0.0 if EXECUTION_MODE=="maker" else TAKER_FEE) * 10_000.0
    slip= (SLIPPAGE_BPS_MAKER if EXECUTION_MODE=="maker" else SLIPPAGE_BPS_TAKER)
    return max(0.0, fee + max(0.0, slip))
def _exit_cost_bps()->float:
    fee = (0.0 if EXIT_EXEC_MODE=="maker" else TAKER_FEE) * 10_000.0
    slip= (SLIPPAGE_BPS_MAKER if EXIT_EXEC_MODE=="maker" else SLIPPAGE_BPS_TAKER)
    return max(0.0, fee + max(0.0, slip))
def _total_cost_bps()->float:
    return _entry_cost_bps() + _exit_cost_bps()

# ===== META helpers =====
def _bps_from_prices(entry: float, exit_px: float, side: str) -> float:
    if entry <= 0 or exit_px <= 0: return 0.0
    r = (exit_px / entry - 1.0)
    sgn = +1.0 if side == "Buy" else -1.0
    return r * 10_000.0 * sgn

def _meta_push(symbol: str, pnl_bps: float):
    META_TRADES[symbol].append(float(pnl_bps))

def _meta_stats(symbol: str) -> Tuple[float,float,float]:
    arr = list(META_TRADES.get(symbol, []))
    if not arr: return 1.0, 0.0, 0.0
    n = len(arr)
    hit = sum(1 for x in arr if x > 0) / n
    mean = sum(arr)/n
    peak = -1e18; mdd = 0.0; cum = 0.0
    for x in arr:
        cum += x
        if cum > peak: peak = cum
        mdd = min(mdd, cum - peak)
    return hit, mean, mdd  # mdd는 음수

def _meta_allows(symbol: str) -> bool:
    if not META_ON: return True
    if META_BLOCK_UNTIL.get(symbol, 0.0) > time.time():
        if META_DEBUG: print(f"[META][BLOCK] {symbol} cooldown")
        return False
    hit, mean_bps, mdd = _meta_stats(symbol)
    ok = (hit >= META_MIN_HIT) and (mean_bps >= META_MIN_EVR) and (mdd > META_MAX_DD_BPS)
    if not ok:
        META_BLOCK_UNTIL[symbol] = time.time() + META_COOLDOWN
        if META_DEBUG:
            print(f"[META][PAUSE] {symbol} hit={hit:.2f} evr={mean_bps:.1f} mdd={mdd:.1f} -> rest {META_COOLDOWN}s")
        return False
    return True

# ===== data cache =====
KLINE_CACHE = {}; KLINE_TTL_SEC = float(os.getenv("KLINE_TTL_SEC","10"))
PRED_CACHE  = {}; PRED_TTL_SEC  = float(os.getenv("PRED_TTL_SEC","20"))

def _recent_1m_df(symbol: str, minutes: int = None):
    minutes = minutes or (SEQ_LEN + PRED_LEN + 10)
    now = time.time()
    c = KLINE_CACHE.get(symbol)
    if c and now - c["ts"] < KLINE_TTL_SEC: return c["df"]
    r = mkt.get_kline(category=CATEGORY, symbol=symbol, interval="1", limit=min(minutes,1000))
    rows = (r.get("result") or {}).get("list") or []
    if not rows: return c["df"] if c else None
    rows = rows[::-1]
    df = pd.DataFrame([{
        "timestamp": pd.to_datetime(int(z[0]), unit="ms", utc=True),
        "series_id": symbol,
        "open": float(z[1]), "high": float(z[2]), "low": float(z[3]),
        "close": float(z[4]), "volume": float(z[5]), "turnover": float(z[6]),
    } for z in rows])
    df["log_return"] = np.log(df["close"]).diff().fillna(0.0)
    KLINE_CACHE[symbol] = {"ts": now, "df": df}
    return df

# ===== DeepAR =====
DEEPar_MAIN=None
try:
    DEEPar_MAIN = DeepAR.load_from_checkpoint(MODEL_CKPT_MAIN, map_location="cpu").eval()
    print("[DEEPar] loaded")
except Exception as _e:
    print(f".[WARN] DeepAR(main) load failed: {_e}"); DEEPar_MAIN=None

@torch.no_grad()
def _build_infer_dataset(df_sym: pd.DataFrame):
    df = df_sym.sort_values("timestamp").copy()
    df["time_idx"] = np.arange(len(df), dtype=np.int64)
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx", target="log_return", group_ids=["series_id"],
        max_encoder_length=SEQ_LEN, max_prediction_length=max(1,PRED_LEN),
        time_varying_known_reals=["time_idx"], time_varying_unknown_reals=["log_return"],
        static_categoricals=["series_id"], target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )

@torch.no_grad()
def _deepar_mu(symbol: str, df: pd.DataFrame) -> Optional[float]:
    if DEEPar_MAIN is None: return None
    now=time.time(); c=PRED_CACHE.get(symbol)
    if c and now-c["ts"]<PRED_TTL_SEC: return c["mu"]
    dl = _build_infer_dataset(df).to_dataloader(train=False, batch_size=1, num_workers=0)
    try:
        mu = float(DEEPar_MAIN.predict(dl, mode="prediction")[0,0])
    except Exception as e:
        if DEBUG_SIGNALS: print(f"[SIG][{symbol}] deepar-error: {e}")
        return None
    PRED_CACHE[symbol]={"ts":now,"mu":mu}
    return mu

# ===== TCN =====
TCN_FEATS = ["ret","rv","mom","vz"]; TCN_SEQ_LEN_FALLBACK=240
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size=chomp_size
    def forward(self,x): return x[:,:, :-self.chomp_size].contiguous()
def weight_norm_conv(in_c,out_c,k,dil):
    pad=(k-1)*dil
    return nn.utils.weight_norm(nn.Conv1d(in_c,out_c,kernel_size=k,padding=pad,dilation=dil))
class TemporalBlock(nn.Module):
    def __init__(self,in_c,out_c,k,dil,drop):
        super().__init__()
        self.conv1=weight_norm_conv(in_c,out_c,k,dil); self.chomp1=Chomp1d((k-1)*dil)
        self.relu1=nn.ReLU(); self.drop1=nn.Dropout(drop)
        self.conv2=weight_norm_conv(out_c,out_c,k,dil); self.chomp2=Chomp1d((k-1)*dil)
        self.relu2=nn.ReLU(); self.drop2=nn.Dropout(drop)
        self.down=nn.Conv1d(in_c,out_c,1) if in_c!=out_c else None; self.relu=nn.ReLU()
    def forward(self,x):
        y=self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        y=self.drop2(self.relu2(self.chomp2(self.conv2(y))))
        res=x if self.down is None else self.down(x)
        return self.relu(y+res)
class TCNBinary(nn.Module):
    def __init__(self,in_feat,hidden=128,levels=6,k=3,drop=0.1,out_dim=2):
        super().__init__(); layers=[]; ch=in_feat
        for i in range(levels):
            layers.append(TemporalBlock(ch,hidden,k,2**i,drop)); ch=hidden
        self.tcn=nn.Sequential(*layers); self.head=nn.Linear(hidden,out_dim)
    def forward(self,x):
        x=x.transpose(1,2); h=self.tcn(x)[:,:,-1]; o=self.head(h)
        return (o[:,0], (o[:,1] if o.shape[1]>1 else torch.zeros_like(o[:,0])))

def _build_tcn_features(df: pd.DataFrame) -> pd.DataFrame:
    g=df.sort_values("timestamp").copy()
    c=g["close"].astype(float).values; v=g["volume"].astype(float).values
    ret=np.zeros_like(c); ret[1:]=np.log(c[1:]/np.clip(c[:-1],1e-12,None))
    win=60
    rv=pd.Series(ret).rolling(win, min_periods=win//3).std().bfill().values
    ema_win=60; alpha=2/(ema_win+1); ema=np.zeros_like(c); ema[0]=c[0]
    for i in range(1,len(c)): ema[i]=alpha*c[i]+(1-alpha)*ema[i-1]
    mom=(c/np.maximum(ema,1e-12)-1.0)
    v_ma=pd.Series(v).rolling(win, min_periods=win//3).mean().bfill().values
    v_sd=pd.Series(v).rolling(win, min_periods=win//3).std().replace(0,np.nan).bfill().values
    vz=(v - v_ma) / np.where(v_sd>0, v_sd, 1.0)
    return pd.DataFrame({"timestamp":g["timestamp"].values,"ret":ret,"rv":rv,"mom":mom,"vz":vz})

def _apply_tcn_scaler(X: np.ndarray, mu, sd):
    if mu is None or sd is None: return X
    mu=np.asarray(mu,dtype=np.float32); sd=np.asarray(sd,dtype=np.float32); sd=np.where(sd>0,sd,1.0)
    return (X - mu)/sd

def _infer_tcn_arch_from_state(state_dict):
    keys=[k for k in state_dict.keys() if k.endswith("conv1.weight_v")]
    if not keys: return dict(hidden=128,levels=6,k=3,out_dim=2)
    head_w=state_dict.get("head.weight",None); out_dim=int(head_w.shape[0]) if head_w is not None else 2
    sample=state_dict[keys[0]]; hidden=int(sample.shape[0]); ksz=int(sample.shape[2])
    lvl=-1
    for k in state_dict.keys():
        if k.startswith("tcn.") and ".conv1.weight_v" in k:
            try: lvl=max(lvl,int(k.split(".")[1]))
            except: pass
    return dict(hidden=hidden, levels=(lvl+1 if lvl>=0 else 6), k=ksz, out_dim=out_dim)

TCN_MODEL=None; TCN_CFG={}; TCN_MU=None; TCN_SD=None; TCN_SEQ_LEN=TCN_SEQ_LEN_FALLBACK
try:
    ckpt=torch.load(TCN_CKPT, map_location="cpu")
    raw_state=ckpt["model"] if isinstance(ckpt,dict) and "model" in ckpt else ckpt
    for k,v in list(raw_state.items()):
        if isinstance(v,torch.Tensor) and v.dtype==torch.float64: raw_state[k]=v.float()
    TCN_CFG=ckpt.get("cfg",{}) if isinstance(ckpt,dict) else {}
    feats=TCN_CFG.get("FEATS", TCN_FEATS); TCN_FEATS=feats
    arch=_infer_tcn_arch_from_state(raw_state)
    TCN_SEQ_LEN=int(TCN_CFG.get("SEQ_LEN", TCN_SEQ_LEN_FALLBACK))
    TCN_MODEL=TCNBinary(in_feat=len(feats), hidden=arch["hidden"], levels=arch["levels"], k=arch["k"], out_dim=arch["out_dim"]).eval()
    TCN_MODEL.load_state_dict(raw_state, strict=False)
    TCN_MU=ckpt.get("scaler_mu",None) if isinstance(ckpt,dict) else None
    TCN_SD=ckpt.get("scaler_sd",None) if isinstance(ckpt,dict) else None
    print(f"[TCN] loaded: feats={feats} seq_len={TCN_SEQ_LEN}")
except Exception as e:
    print(f".[WARN] TCN load failed: {e}"); TCN_MODEL=None

@torch.no_grad()
def _tcn_mu(symbol: str, df: pd.DataFrame) -> Optional[float]:
    if TCN_MODEL is None: return None
    if df is None or len(df)<(TCN_SEQ_LEN+1): return None
    feats=_build_tcn_features(df)
    if len(feats)<TCN_SEQ_LEN: return None
    X=feats[TCN_FEATS].tail(TCN_SEQ_LEN).to_numpy().astype(np.float32)
    X=_apply_tcn_scaler(X, TCN_MU, TCN_SD)
    mu,_=TCN_MODEL(torch.from_numpy(X[None,...]))
    return float(mu.item())

def _resolve_thr():
    if THR_MODE=="fixed": return MODEL_THR_BPS/1e4
    tf=TAKER_FEE; slip=(SLIPPAGE_BPS_TAKER/1e4)
    return np.log(1.0 + FEE_SAFETY*(2*tf + 2*slip))

@torch.no_grad()
def dual_deepar_tcn_direction(symbol: str) -> Tuple[Optional[str], float]:
    if symbol in SKIP_SYMBOLS: return (None, 0.0)
    df=_recent_1m_df(symbol, minutes=SEQ_LEN+PRED_LEN+10)
    if df is None or len(df)<(SEQ_LEN+1): return (None,0.0)
    mu1=_deepar_mu(symbol, df) if DEEPar_MAIN is not None else None
    mu2=_tcn_mu(symbol, df)
    if (mu1 is None) and (mu2 is None): return (None,0.0)
    mu1=float(mu1 or 0.0); mu2=float(mu2 or 0.0); thr=_resolve_thr()
    dir1="Buy" if mu1>+thr else ("Sell" if mu1<-thr else "None")
    dir2="Buy" if mu2>+thr else ("Sell" if mu2<-thr else "None")
    same_sign=(mu1>0 and mu2>0) or (mu1<0 and mu2<0)
    mag1=abs(mu1)>=thr; mag2=abs(mu2)>=thr
    side=None
    if same_sign and ((DUAL_RULE=="strict" and (mag1 and mag2)) or (DUAL_RULE!="strict" and (mag1 or mag2))):
        side="Buy" if mu1>0 else "Sell"
    if side is None and ALLOW_TCN_FALLBACK and (mu2 is not None) and (abs(mu2)>=thr):
        side="Buy" if mu2>0 else "Sell"
    if side is not None and ENTRY_MODE=="inverse": side=("Sell" if side=="Buy" else "Buy")
    if DEBUG_SIGNALS:
        print(f"[SIG][{symbol}] deepar={mu1:.5g}({dir1}) tcn={mu2:.5g}({dir2}) thr={thr:.5g} -> {side or 'None'}")
    return (side, 1.0 if side else 0.0)

# ===== entry =====
SIG_MIN_INTERVAL_SEC = float(os.getenv("SIG_MIN_INTERVAL_SEC","5"))
LAST_SIG_TS: Dict[Tuple[str,str], float] = {}

def _init_state(key:Tuple[str,str], entry:float, qty:float, lot_step:float):
    sym,side=key
    STATE[key] = {"side":side,"entry":float(entry),"init_qty":float(qty),
                  "tp_done":[False,False,False],"be_moved":False,
                  "peak":float(entry),"trough":float(entry),
                  "entry_ts":time.time(),"lot_step":float(lot_step)}

def _roi_filter(sym: str, side: str, entry_ref: float) -> bool:
    tp1_px = tp_from_bps(entry_ref, TP1_BPS, side)
    edge_bps = _bps_between(entry_ref, tp1_px)
    return edge_bps >= (_total_cost_bps() + MIN_EXPECTED_ROI_BPS)

# ==== 2) 마켓 전용으로 _try_open 교체 ====
# ==== 2) 마켓 전용으로 _try_open 교체 ====
def _try_open(sym: str, side: str, size_pct: Optional[float]=None):
    # per-(symbol, side) rate limit
    if time.time() - LAST_SIG_TS.get((sym,side), 0.0) < SIG_MIN_INTERVAL_SEC:
        return
    LAST_SIG_TS[(sym,side)] = time.time()

    if len(get_positions()) >= MAX_OPEN:
        return
    if COOLDOWN_UNTIL.get((sym,side),0.0) > time.time():
        return
    if (sym,side) in PENDING_MAKER:
        return
    if (sym,side) in get_positions():
        return

    # 시그널 합의 확인
    side_sig, conf = dual_deepar_tcn_direction(sym)
    if side_sig != side or conf < 0.55:
        return

    bid,ask,mid = get_quote(sym)
    if mid <= 0:
        return
    rule = get_instrument_rule(sym)

    # 기대 ROI 필터
    entry_ref = (ask if side=="Buy" else bid)
    tp1_px = _round_to_tick(tp_from_bps(entry_ref, TP1_BPS, side), rule["tickSize"])
    edge_bps = _bps_between(entry_ref, tp1_px)
    cost_bps = _entry_cost_bps() + _exit_cost_bps()
    if edge_bps < (cost_bps + MIN_EXPECTED_ROI_BPS):
        return

    # 사이징
    eq = get_wallet_equity()
    use_pct = float(size_pct if size_pct is not None else ENTRY_EQUITY_PCT)
    qty, _ = compute_qty_by_equity_pct(sym, mid, eq, use_pct)
    qty *= 3
    if qty <= 0:
        return

    ensure_isolated_and_leverage(sym)

    # === 마켓 진입(강제) ===
    ok = place_market(sym, side, qty)
    if not ok:
        return

    # 체결 후 포지션 조회(약간의 지연 허용)
    pos = None
    for _ in range(5):
        pos = get_positions().get((sym, side))
        if pos:
            break
        time.sleep(0.1)
    if not pos:
        return

    entry = float(pos.get("entry") or mid)
    lot   = rule["qtyStep"]
    sl    = compute_sl(entry, side)
    tp1   = tp_from_bps(entry, TP1_BPS, side)
    tp2   = tp_from_bps(entry, TP2_BPS, side)
    tp3   = tp_from_bps(entry, TP3_BPS, side)

    pos_idx = int(pos.get("positionIdx") or (1 if side=="Buy" else 2))
    close_side = "Sell" if side=="Buy" else "Buy"
    qty_tp1 = _round_down(qty*TP1_RATIO, lot)
    qty_tp2 = _round_down(qty*TP2_RATIO, lot)
    if qty_tp1 > 0:
        place_reduce_limit(sym, close_side, qty_tp1, tp1, pos_idx)
    if qty_tp2 > 0:
        place_reduce_limit(sym, close_side, qty_tp2, tp2, pos_idx)
    set_stop_with_retry(sym, side, sl_price=sl, tp_price=tp3)

    _init_state((sym,side), entry, qty, lot)
    _log_trade({
        "ts": int(time.time()), "event":"ENTRY", "symbol":sym, "side":side,
        "qty": f"{qty:.8f}", "entry": f"{entry:.6f}", "exit":"",
        "tp1": f"{tp1:.6f}", "tp2": f"{tp2:.6f}", "tp3": f"{tp3:.6f}",
        "sl": f"{sl:.6f}", "reason":"open_market", "mode":ENTRY_MODE
    })

# ===== manage =====
def _update_trailing_and_be(sym:str, side:str):
    key=(sym,side); st=STATE.get(key); pos=get_positions().get(key)
    if not st or not pos: return
    entry=st["entry"]; qty=float(pos.get("qty") or 0.0)
    if qty<=0: return
    _,_,mark=get_quote(sym)
    st["peak"]=max(st["peak"], mark); st["trough"]=min(st["trough"], mark)
    init_qty=float(st.get("init_qty",0.0)); rem=qty/max(init_qty,1e-12)
    if not st["tp_done"][0] and rem <= (1.0 - TP1_RATIO + 1e-8):
        st["tp_done"][0]=True; _log_trade({"ts":int(time.time()),"event":"TP1_FILL","symbol":sym,"side":side,
                                           "qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"",
                                           "reason":"","mode":ENTRY_MODE})
    if not st["tp_done"][1] and rem <= (1.0 - TP1_RATIO - TP2_RATIO + 1e-8):
        st["tp_done"][1]=True; _log_trade({"ts":int(time.time()),"event":"TP2_FILL","symbol":sym,"side":side,
                                           "qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"",
                                           "reason":"","mode":ENTRY_MODE})
    if st["tp_done"][1] and not st["be_moved"] and entry>0:
        be_px = entry*(1.0 + BE_EPS_BPS/10000.0) if side=="Buy" else entry*(1.0 - BE_EPS_BPS/10000.0)
        set_stop_with_retry(sym, side, sl_price=be_px, tp_price=None)
        st["be_moved"]=True
        _log_trade({"ts":int(time.time()),"event":"MOVE_SL","symbol":sym,"side":side,
                    "qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"",
                    "sl":f"{be_px:.6f}","reason":"BE","mode":ENTRY_MODE})
    if st["tp_done"][TRAIL_AFTER_TIER-1]:
        new_sl = (mark*(1.0 - TRAIL_BPS/10000.0)) if side=="Buy" else (mark*(1.0 + TRAIL_BPS/10000.0))
        set_stop_with_retry(sym, side, sl_price=new_sl, tp_price=None)

def _check_time_stop(sym:str, side:str):
    key=(sym,side); st=STATE.get(key); pos=get_positions().get(key)
    if not st or not pos: return
    if MAX_HOLD_SEC>0 and (time.time()-st.get("entry_ts",time.time()))>MAX_HOLD_SEC:
        qty=float(pos["qty"]); close_side=("Sell" if side=="Buy" else "Buy")
        _order_with_mode_retry(dict(category=CATEGORY, symbol=sym, side=close_side, orderType="Market",
                                    qty=str(qty), reduceOnly=True, timeInForce="IOC"), side=close_side)
        COOLDOWN_UNTIL[key]=time.time()+COOLDOWN_SEC
        _log_trade({"ts":int(time.time()),"event":"EXIT","symbol":sym,"side":side,"qty":f"{qty:.8f}",
                    "entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"","reason":"TIMEOUT","mode":ENTRY_MODE})

# ===== pending PostOnly =====
def _get_open_orders(symbol: Optional[str]=None)->List[dict]:
    try:
        res=client.get_open_orders(category=CATEGORY, symbol=symbol)
        return (res.get("result") or {}).get("list") or []
    except Exception:
        return []
def _cancel_order(symbol:str, order_id:str)->bool:
    try:
        client.cancel_order(category=CATEGORY, symbol=symbol, orderId=order_id); return True
    except Exception as e:
        print(f"[CANCEL ERR] {symbol} {order_id}: {e}"); return False

def process_pending_postonly():
    now=time.time()
    if not PENDING_MAKER: return
    open_cache: Dict[str,List[dict]] = {}
    for key in list(PENDING_MAKER.keys()):
        sym, side = key
        if sym not in open_cache: open_cache[sym]=_get_open_orders(sym)
        info=PENDING_MAKER.get(key) or {}
        oid=info.get("orderId"); ol=open_cache.get(sym) or []
        still_open=any((x.get("orderId")==oid) for x in ol)
        if not still_open:
            PENDING_MAKER.pop(key, None); continue
        if CANCEL_UNFILLED_MAKER and (now - float(info.get("ts",now)) > WAIT_FOR_MAKER_FILL_SEC):
            if _cancel_order(sym, oid):
                PENDING_MAKER.pop(key, None)
                COOLDOWN_UNTIL[key]=time.time()+COOLDOWN_SEC
                _log_trade({"ts":int(time.time()),"event":"CANCEL_POSTONLY","symbol":sym,"side":side,
                            "qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"",
                            "reason":"TIMEOUT","mode":ENTRY_MODE})

# ===== loops =====
def _top_symbols_by_volume(whitelist:List[str], top_n:int)->List[str]:
    res=mkt.get_tickers(category=CATEGORY)
    data=(res.get("result") or {}).get("list") or []
    white=set(whitelist or [])
    flt=[x for x in data if x.get("symbol") in white]
    try: flt.sort(key=lambda x: float(x.get("turnover24h") or 0.0), reverse=True)
    except Exception: pass
    return [x.get("symbol") for x in flt[:top_n]]

def _has_enough_history_symbol(symbol: str) -> bool:
    need = SEQ_LEN + max(1, PRED_LEN)
    df=_recent_1m_df(symbol, minutes=need+10)
    return (df is not None) and (len(df) >= need)

RUNNABLE: List[str]=[]

def monitor_loop(poll_sec: float = 1.0):
    last_pos_keys=set()
    while True:
        try:
            _log_equity(get_wallet_equity())
            process_pending_postonly()

            pos_map=get_positions(); pos_keys=set(pos_map.keys())

            # 새 포지션 감지
            new_keys = pos_keys - last_pos_keys
            for key in new_keys:
                sym,side=key; p=pos_map[key]
                entry=float(p.get("entry") or 0.0); qty=float(p.get("qty") or 0.0)
                if entry<=0 or qty<=0: continue
                rule=get_instrument_rule(sym); lot=rule["qtyStep"]
                sl=compute_sl(entry, side)
                tp1=tp_from_bps(entry, TP1_BPS, side)
                tp2=tp_from_bps(entry, TP2_BPS, side)
                tp3=tp_from_bps(entry, TP3_BPS, side)
                qty_tp1=_round_down(qty*TP1_RATIO, lot); qty_tp2=_round_down(qty*TP2_RATIO, lot)
                pos_idx = int(p.get("positionIdx") or (1 if side == "Buy" else 2))
                close_side = "Sell" if side == "Buy" else "Buy"
                if qty_tp1 > 0: place_reduce_limit(sym, close_side, qty_tp1, tp1, pos_idx)
                if qty_tp2 > 0: place_reduce_limit(sym, close_side, qty_tp2, tp2, pos_idx)
                set_stop_with_retry(sym, side, sl_price=sl, tp_price=tp3)
                _init_state(key, entry, qty, lot)
                _log_trade({"ts":int(time.time()),"event":"ENTRY","symbol":sym,"side":side,
                            "qty":f"{qty:.8f}","entry":f"{entry:.6f}","exit":"",
                            "tp1":f"{tp1:.6f}","tp2":f"{tp2:.6f}","tp3":f"{tp3:.6f}",
                            "sl":f"{sl:.6f}","reason":"postonly_fill_or_detect","mode":ENTRY_MODE})

            # 닫힘 감지 + META 기록
            closed_keys = last_pos_keys - pos_keys
            for key in closed_keys:
                sym,side=key
                st = STATE.get(key, {})
                entry_px = float(st.get("entry") or 0.0)
                b,a,m=get_quote(sym)
                exit_px = float(m or a or b or 0.0)
                # META: bps 산출(수수료/슬리피지 총합 차감)
                pnl_bps = _bps_from_prices(entry_px, exit_px, side) - _total_cost_bps()
                if META_ON and entry_px>0 and exit_px>0:
                    _meta_push(sym, pnl_bps)
                    if META_DEBUG:
                        hit, mean_bps, mdd = _meta_stats(sym)
                        print(f"[META][FEED] {sym} bps={pnl_bps:.1f} | hit={hit:.2f} evr={mean_bps:.1f} mdd={mdd:.1f}")
                STATE.pop(key, None)
                LAST_EXIT[key]={"px":exit_px,"ts":time.time()}
                COOLDOWN_UNTIL[key]=time.time()+COOLDOWN_SEC
                _log_trade({"ts":int(time.time()),"event":"EXIT","symbol":sym,"side":side,
                            "qty":"","entry":"","exit":f"{(exit_px or 0.0):.6f}",
                            "tp1":"","tp2":"","tp3":"","sl":"","reason":"CLOSED","mode":ENTRY_MODE})

            # 관리
            for key in pos_keys:
                sym,side=key
                _update_trailing_and_be(sym, side)
                _check_time_stop(sym, side)

            last_pos_keys = pos_keys
            time.sleep(poll_sec)
        except KeyboardInterrupt:
            print("[MONITOR] stop"); break
        except Exception as e:
            print("[MONITOR ERR]", e); time.sleep(2.0)

def scanner_loop(iter_delay:float=2.5):
    while True:
        try:
            if time.time() < GLOBAL_COOLDOWN_UNTIL: time.sleep(iter_delay); continue
            syms=_top_symbols_by_volume(RUNNABLE, top_n=min(MAX_OPEN, len(RUNNABLE)))
            if len(get_positions()) >= MAX_OPEN: time.sleep(iter_delay); continue
            for s in syms:
                for side in ("Buy","Sell"):
                    key=(s,side)
                    if len(get_positions()) >= MAX_OPEN: break
                    if key in PENDING_MAKER: continue
                    if COOLDOWN_UNTIL.get(key,0.0) > time.time(): continue
                    if key in get_positions(): continue

                    re = LAST_EXIT.get(key)
                    if REENTRY_ON_DIP and re:
                        last_px=float(re.get("px") or 0.0); b,a,m=get_quote(s)
                        if side=="Buy" and m>0 and last_px>0 and m <= last_px*(1.0 - REENTRY_PCT):
                            _try_open(s, "Buy", REENTRY_SIZE_PCT); continue
                        if side=="Sell" and m>0 and last_px>0 and m >= last_px*(1.0 + REENTRY_PCT):
                            _try_open(s, "Sell", REENTRY_SIZE_PCT); continue

                    _try_open(s, side)
            time.sleep(iter_delay)
        except KeyboardInterrupt:
            print("[SCANNER] stop"); break
        except Exception as e:
            print("[SCANNER ERR]", e); time.sleep(2.0)

# ===== run =====
def main():
    syms_env=os.getenv("SYMBOLS","HEMIUSDT,ASTERUSDT,AIAUSDT,0GUSDT,STBLUSDT,WLFIUSDT,LINEAUSDT,BARDUSDT,SOMIUSDT,UBUSDT,OPENUSDT")
    symbols=[s.strip() for s in syms_env.split(",") if s.strip()]
    print(f"[START] Bybit Live (HEDGE+META) MODE={ENTRY_MODE} EXEC={EXECUTION_MODE}/{EXIT_EXEC_MODE} TESTNET={TESTNET}")
    need=SEQ_LEN + max(1,PRED_LEN)
    runnable=[]
    for s in symbols:
        if _has_enough_history_symbol(s): runnable.append(s)
        else:
            SKIP_SYMBOLS.add(s)
            if DEBUG_SIGNALS: print(f"[DISABLE] {s} insufficient history (need {need})")
    if not runnable:
        print("[ABORT] no runnable symbols"); return
    global RUNNABLE; RUNNABLE=runnable
    t1=threading.Thread(target=monitor_loop, args=(1.0,), daemon=True)
    t2=threading.Thread(target=scanner_loop, args=(2.5,), daemon=True)
    t1.start(); t2.start()
    while True: time.sleep(60)

if __name__ == "__main__":
    main()
