# -*- coding: utf-8 -*-
"""
Bybit Live Trading (Unified v5, Linear Perp) — DeepAR(main) + TCN(alt) Consensus
- 진입: DeepAR μ와 TCN μ가 같은 방향이고 임계치 충족일 때만
- TP1/TP2: reduce-only PostOnly 지정가, TP3/SL: trading_stop 유지
- BE/트레일/타임아웃 포함
- 메이커(PostOnly) 또는 테이커 즉시체결 선택
- 미체결 만료/취소와 최소 기대 ROI 필터

Env:
  BYBIT_TESTNET=0|1
  BYBIT_API_KEY=...
  BYBIT_API_SECRET=...

  SEQ_LEN=240
  PRED_LEN=60
  MODEL_CKPT_MAIN=models/multi_deepar_best_main.ckpt
  TCN_CKPT=models/tcn_best.pt

  DUAL_RULE=loose|strict
  THR_MODE=fixed|fee
  MODEL_THR_BPS=5.0

  ENTRY_MODE=model|inverse|random
  EXECUTION_MODE=maker|taker
  EXIT_EXEC_MODE=taker|maker

  LEVERAGE=15
  TAKER_FEE=0.0006
  MAKER_FEE=0.0
  SLIPPAGE_BPS_TAKER=1.0
  SLIPPAGE_BPS_MAKER=0.0
  MIN_EXPECTED_ROI_BPS=30.0

  ENTRY_EQUITY_PCT=0.2
  MAX_OPEN=5
  COOLDOWN_SEC=10
  MAX_HOLD_SEC=3600

  WAIT_FOR_MAKER_FILL_SEC=30
  CANCEL_UNFILLED_MAKER=1

  SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
"""
from pandas import DataFrame
import os, time, math, csv, threading, warnings, logging, collections
from typing import Optional, Dict, Tuple, List
# --- 추가 import 및 allowlist ---
from pandas import DataFrame
from torch.serialization import add_safe_globals
from pytorch_forecasting.metrics import NormalDistributionLoss, DistributionLoss

# 사용자 정의 손실 스텁(이름만 맞추면 됨. 추론시 손실은 안 씀)
class SignAwareNormalLoss(NormalDistributionLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# 이미 있던 GroupNormalizer allowlist에 함께 등록
from pytorch_forecasting.data.encoders import GroupNormalizer
add_safe_globals([GroupNormalizer, NormalDistributionLoss, SignAwareNormalLoss, DataFrame])

# ===== misc =====
warnings.filterwarnings("ignore", module="lightning")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
os.environ.setdefault("PL_DISABLE_FORK","1")
os.environ.setdefault("PYTORCH_LIGHTNING_DISABLE_PBAR","1")

ALLOW_TCN_FALLBACK = str(os.getenv("ALLOW_TCN_FALLBACK","0")).lower() in ("1","y","yes","true")

# ===== config =====
CATEGORY = "linear"
ENTRY_MODE = os.getenv("ENTRY_MODE", "model").lower()          # model|inverse|random
LEVERAGE   = float(os.getenv("LEVERAGE", "10"))
TAKER_FEE  = float(os.getenv("TAKER_FEE", "0.0006"))
MAKER_FEE  = float(os.getenv("MAKER_FEE", "0.0"))
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "maker").lower()  # maker|taker
EXIT_EXEC_MODE = os.getenv("EXIT_EXEC_MODE", "taker").lower()  # maker|taker
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER", "1.0"))
SLIPPAGE_BPS_MAKER = float(os.getenv("SLIPPAGE_BPS_MAKER", "0.0"))
MIN_EXPECTED_ROI_BPS = float(os.getenv("MIN_EXPECTED_ROI_BPS", "40.0"))
WAIT_FOR_MAKER_FILL_SEC = float(os.getenv("WAIT_FOR_MAKER_FILL_SEC", "30"))
CANCEL_UNFILLED_MAKER = str(os.getenv("CANCEL_UNFILLED_MAKER","1")).lower() in ("1","y","yes","true")
MAX_OPEN   = int(os.getenv("MAX_OPEN", "5"))

# SL/TP
SL_ROI_PCT      = float(os.getenv("SL_ROI_PCT", "0.01"))
HARD_SL_PCT     = 0.01
TP1_BPS,TP2_BPS,TP3_BPS = 50.0, 80.0, 120.0
TP1_RATIO,TP2_RATIO     = 0.30, 0.45
BE_EPS_BPS = 2.0
TRAIL_BPS = 50.0
TRAIL_AFTER_TIER = 2

# sizing & re-entry
ENTRY_EQUITY_PCT   = float(os.getenv("ENTRY_EQUITY_PCT", "0.2"))
REENTRY_ON_DIP     = str(os.getenv("REENTRY_ON_DIP","1")).lower() in ("1","y","yes","true")
REENTRY_PCT        = float(os.getenv("REENTRY_PCT", "0.5"))
REENTRY_SIZE_PCT   = float(os.getenv("REENTRY_SIZE_PCT","0.8"))

# state
STATE: Dict[str,dict] = {}
COOLDOWN_UNTIL: Dict[str,float] = {}
PENDING_MAKER: Dict[str,dict] = {}
LAST_EXIT: Dict[str,dict] = {}
GLOBAL_COOLDOWN_UNTIL = 0.0
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC","10"))
MAX_HOLD_SEC = int(os.getenv("MAX_HOLD_SEC","3600"))
DEBUG_SIGNALS = str(os.getenv("DEBUG_SIGNALS","1")).lower() in ("1","y","yes","true")

# deepar 실패 차단
SKIP_SYMBOLS = set()
FAIL_CNT = collections.defaultdict(int)
FAIL_MAX = int(os.getenv("DEEPar_FAIL_MAX", "3"))

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
        csv.writer(f).writerow([int(time.time()), f"{eq:.6f}"])

# ===== http =====
try:
    from pybit.unified_trading import HTTP
except Exception as e:
    raise RuntimeError("pip install pybit") from e

TESTNET = str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY","Dlp4eJD6YFmO99T8vC")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET","YYYB5tMw2TWvfVF5wqi6lQRHqEIiDSpDJF1U")
client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET, testnet=TESTNET, timeout=10, recv_window=5000)
mkt    = HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

# ===== market utils =====
LEV_CAPS: Dict[str, Tuple[float, float]] = {}

def get_symbol_leverage_caps(symbol: str) -> Tuple[float, float]:
    if symbol in LEV_CAPS: return LEV_CAPS[symbol]
    info=client.get_instruments_info(category=CATEGORY, symbol=symbol)
    lst=(info.get("result") or {}).get("list") or []
    if not lst: caps=(1.0, float(LEVERAGE))
    else:
        lf=(lst[0].get("leverageFilter") or {})
        caps=(float(lf.get("minLeverage") or 1.0), float(lf.get("maxLeverage") or LEVERAGE))
    LEV_CAPS[symbol]=caps; return caps

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

# ===== leverage / orders =====
def ensure_isolated_and_leverage(symbol: str):
    try:
        minL, maxL = get_symbol_leverage_caps(symbol)
        useL = max(min(float(LEVERAGE), maxL), minL)
        curL=None
        try:
            pr=client.get_positions(category=CATEGORY, symbol=symbol)
            lst=(pr.get("result") or {}).get("list") or []
            if lst: curL=float(lst[0].get("leverage") or 0.0)
        except Exception: pass
        if curL is not None and abs(curL-useL)<1e-9: return
        kw=dict(category=CATEGORY, symbol=symbol, buyLeverage=str(useL), sellLeverage=str(useL), tradeMode=1)
        try: client.set_leverage(**kw)
        except Exception as e:
            msg=str(e)
            if "ErrCode: 110043" in msg or "leverage not modified" in msg: pass
            else: raise
    except Exception as e:
        print(f"[WARN] set_leverage {symbol}: {e}")

def _order_with_mode_retry(params: dict, side: Optional[str] = None):
    try:
        return client.place_order(**params)
    except Exception as e:
        msg=str(e)
        if "position idx not match position mode" in msg and side in ("Buy","Sell"):
            p2=dict(params); p2["positionIdx"]=1 if side=="Buy" else 2
            return client.place_order(**p2)
        raise

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
        q=_round_down(qty, rule["qtyStep"])
        px=_round_to_tick(price, rule["tickSize"])
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

def place_reduce_limit(symbol:str, side:str, qty:float, price:float)->bool:
    try:
        rule=get_instrument_rule(symbol)
        q=_round_down(qty, rule["qtyStep"])
        px=_round_to_tick(price, rule["tickSize"])
        if q<rule["minOrderQty"]: return False
        p=dict(category=CATEGORY, symbol=symbol, side=side, orderType="Limit",
               qty=str(q), price=str(px), timeInForce="PostOnly", reduceOnly=True)
        _order_with_mode_retry(p, side=side)
        return True
    except Exception as e:
        print(f"[WARN] reduce_limit {symbol} {side} {qty}@{price}: {e}")
        return False

# ===== sizing =====
def compute_qty_by_equity_pct(symbol: str, mid: float, equity_now: float, pct: float) -> Tuple[float, float]:
    rule = get_instrument_rule(symbol)
    qty_step = float(rule["qtyStep"]); min_qty = float(rule["minOrderQty"])
    if mid <= 0 or equity_now <= 0 or pct <= 0:
        return 0.0, 0.0

    MIN_NOTIONAL_USDT = float(os.getenv("MIN_NOTIONAL_USDT","10.0"))
    MIN_NOTIONAL      = float(os.getenv("MIN_NOTIONAL","10.0"))
    MAX_NOTIONAL_ABS  = float(os.getenv("MAX_NOTIONAL_ABS","2000.0"))
    MAX_NOTIONAL_PCT  = float(os.getenv("MAX_NOTIONAL_PCT_OF_EQUITY","20.0"))

    minL, maxL = get_symbol_leverage_caps(symbol)
    lv_used = max(min(float(LEVERAGE), maxL), minL)

    target_notional = max(equity_now * pct, MIN_NOTIONAL_USDT) * lv_used
    cap_abs  = MAX_NOTIONAL_ABS * lv_used
    cap_pct  = equity_now * (MAX_NOTIONAL_PCT / 100.0) * lv_used

    if pct <= (MAX_NOTIONAL_PCT / 100.0):
        capped_notional = min(max(target_notional, MIN_NOTIONAL), cap_abs)
    else:
        capped_notional = min(max(target_notional, MIN_NOTIONAL), cap_abs, cap_pct)

    qty = math.ceil((capped_notional / mid) / qty_step) * qty_step
    if qty < min_qty:
        qty = math.ceil(min_qty / qty_step) * qty_step

    return qty, qty * mid

# ===== stops/tp =====
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

    def _call_once(use_idx: Optional[int]):
        params=dict(category=CATEGORY, symbol=symbol, tpslMode="Full", slTriggerBy="MarkPrice", tpTriggerBy="MarkPrice")
        if sl_price is not None:
            params["stopLoss"]=f"{sl_price:.10f}".rstrip("0").rstrip("."); params["slOrderType"]="Market"
        if tp_price is not None:
            params["takeProfit"]=f"{tp_price:.10f}".rstrip("0").rstrip("."); params["tpOrderType"]="Market"
        if use_idx in (1,2): params["positionIdx"]=use_idx
        return client.set_trading_stop(**params)

    ok=False
    for _ in range(3):
        try:
            ret=_call_once(use_idx=None); ok=(ret.get("retCode")==0)
            if ok: break
        except Exception as e1:
            if "position idx not match position mode" in str(e1):
                pos=get_positions().get(symbol) or {}; s=pos.get("side","")
                use_idx=1 if s=="Buy" else 2 if s=="Sell" else None
                try:
                    ret=_call_once(use_idx); ok=(ret.get("retCode")==0)
                    if ok: break
                except Exception: pass
        time.sleep(0.3)
    return ok

# ===== signals =====
import numpy as np, pandas as pd, torch
from torch.serialization import add_safe_globals
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.deepar import DeepAR
add_safe_globals([GroupNormalizer, NormalDistributionLoss, DataFrame])

SEQ_LEN  = int(os.getenv("SEQ_LEN","240"))
PRED_LEN = int(os.getenv("PRED_LEN","60"))
MODEL_CKPT_MAIN = os.getenv("MODEL_CKPT_MAIN", "models/multi_deepar_best_main.ckpt")

# 전역
KLINE_CACHE = {}  # symbol -> {"ts": float, "df": pd.DataFrame}
KLINE_TTL_SEC = float(os.getenv("KLINE_TTL_SEC", "10"))

# 전역
PRED_CACHE = {}  # symbol -> {"ts": float, "mu": float}
PRED_TTL_SEC = float(os.getenv("PRED_TTL_SEC", "20"))

@torch.no_grad()
def _deepar_mu(symbol: str, df: pd.DataFrame) -> Optional[float]:
    if DEEPar_MAIN is None: return None
    now = time.time()
    c = PRED_CACHE.get(symbol)
    if c and now - c["ts"] < PRED_TTL_SEC:
        return c["mu"]
    tsd = _build_infer_dataset(df)
    dl  = tsd.to_dataloader(train=False, batch_size=1, num_workers=0)
    try:
        mu = float(DEEPar_MAIN.predict(dl, mode="prediction")[0, 0])
    except Exception as e:
        if DEBUG_SIGNALS: print(f"[SIG][{symbol}] deepar-error: {e}", flush=True)
        return None
    PRED_CACHE[symbol] = {"ts": now, "mu": mu}
    return mu


def _recent_1m_df(symbol: str, minutes: int = None):
    minutes = minutes or (SEQ_LEN + PRED_LEN + 10)
    now = time.time()
    c = KLINE_CACHE.get(symbol)
    if c and now - c["ts"] < KLINE_TTL_SEC:
        return c["df"]

    r = mkt.get_kline(category=CATEGORY, symbol=symbol, interval="1", limit=min(minutes, 1000))
    rows = (r.get("result") or {}).get("list") or []
    if not rows:
        return c["df"] if c else None

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

def _build_infer_dataset(df_sym: pd.DataFrame):
    df = df_sym.sort_values("timestamp").copy()
    df["time_idx"] = np.arange(len(df), dtype=np.int64)
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="log_return",
        group_ids=["series_id"],
        max_encoder_length=SEQ_LEN,
        max_prediction_length=max(1, PRED_LEN),  # 빈 축 방지
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["log_return"],
        static_categoricals=["series_id"],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )

DEEPar_MAIN = None
try:
    # allowlist 추가 후 로드 (Lightning 내부 torch.load의 weights_only 경로에서도 통과)
    DEEPar_MAIN = DeepAR.load_from_checkpoint(MODEL_CKPT_MAIN, map_location="cpu").eval()
    print("[DEEPar] loaded")
except Exception as _e:
    print(f"[WARN] DeepAR(main) load failed (disabled): {_e}")
    DEEPar_MAIN = None

# ===== TCN =====
import torch.nn as nn

TCN_CKPT = os.getenv("TCN_CKPT", "tcn_best.pt")
TCN_FEATS = ["ret","rv","mom","vz"]
TCN_SEQ_LEN_FALLBACK = 240

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

def weight_norm_conv(in_c, out_c, k, dilation):
    pad = (k-1)*dilation
    return nn.utils.weight_norm(nn.Conv1d(in_c, out_c, kernel_size=k, padding=pad, dilation=dilation))

class TemporalBlock(nn.Module):
    def __init__(self, in_c, out_c, k, dilation, dropout):
        super().__init__()
        self.conv1 = weight_norm_conv(in_c, out_c, k, dilation)
        self.chomp1 = Chomp1d((k-1)*dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = weight_norm_conv(out_c, out_c, k, dilation)
        self.chomp2 = Chomp1d((k-1)*dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNBinary(nn.Module):
    def __init__(self, in_feat, hidden=128, levels=6, k=3, dropout=0.1, out_dim=2):
        super().__init__()
        layers=[]; ch_in=in_feat
        for i in range(levels):
            layers += [TemporalBlock(ch_in, hidden, k, 2**i, dropout)]
            ch_in = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x):
        x = x.transpose(1,2)        # (B,T,F)->(B,F,T)
        h = self.tcn(x)[:, :, -1]   # (B,C)
        o = self.head(h)
        if o.shape[1] == 1:
            return o[:,0], torch.zeros_like(o[:,0])
        return o[:,0], o[:,1]

def _build_tcn_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.sort_values("timestamp").copy()
    c = g["close"].astype(float).values
    v = g["volume"].astype(float).values
    ret = np.zeros_like(c, dtype=np.float64); ret[1:] = np.log(c[1:] / np.clip(c[:-1], 1e-12, None))
    win = 60
    rv = pd.Series(ret).rolling(win, min_periods=win//3).std().bfill().values
    ema_win=60; alpha=2/(ema_win+1)
    ema=np.zeros_like(c, dtype=np.float64); ema[0]=c[0]
    for i in range(1,len(c)): ema[i] = alpha*c[i] + (1-alpha)*ema[i-1]
    mom = (c/np.maximum(ema,1e-12) - 1.0)
    v_ma = pd.Series(v).rolling(win, min_periods=win//3).mean().bfill().values
    v_sd = pd.Series(v).rolling(win, min_periods=win//3).std().replace(0, np.nan).bfill().values
    vz  = (v - v_ma) / np.where(v_sd>0, v_sd, 1.0)
    return pd.DataFrame({"timestamp": g["timestamp"].values,
                         "ret": ret, "rv": rv, "mom": mom, "vz": vz})

def _apply_tcn_scaler(X: np.ndarray, mu, sd):
    if mu is None or sd is None: return X
    mu = np.asarray(mu, dtype=np.float32); sd = np.asarray(sd, dtype=np.float32)
    sd = np.where(sd>0, sd, 1.0)
    return (X - mu) / sd

def _infer_tcn_arch_from_state(state_dict):
    keys = [k for k in state_dict.keys() if k.endswith("conv1.weight_v")]
    if not keys:  # fallback
        return dict(hidden=128, levels=6, k=3, out_dim=2)
    def blk_idx(k):
        try: return int(k.split(".")[1])
        except: return 0
    keys.sort(key=blk_idx)
    w = state_dict[keys[0]]
    out_c, in_feat_ckpt, k = w.shape
    hidden = int(out_c)
    lvl = -1
    for kname in state_dict.keys():
        if kname.startswith("tcn.") and ".conv1.weight_v" in kname:
            try:
                idx = int(kname.split(".")[1]); lvl = max(lvl, idx)
            except: pass
    levels = (lvl + 1) if lvl >= 0 else 6
    head_w = state_dict.get("head.weight", None)
    out_dim = int(head_w.shape[0]) if head_w is not None else 2
    return dict(hidden=hidden, levels=levels, k=int(k), out_dim=out_dim)

TCN_MODEL = None; TCN_CFG={}; TCN_MU=None; TCN_SD=None; TCN_SEQ_LEN=TCN_SEQ_LEN_FALLBACK
try:
    _ckpt = torch.load(TCN_CKPT, map_location="cpu")   # 호환성 유지
    raw_state = _ckpt["model"] if isinstance(_ckpt, dict) and "model" in _ckpt else _ckpt
    TCN_CFG   = _ckpt.get("cfg", {}) if isinstance(_ckpt, dict) else {}
    TCN_FEATS = TCN_CFG.get("FEATS", TCN_FEATS)
    arch = _infer_tcn_arch_from_state(raw_state)
    hidden  = int(arch["hidden"]); levels=int(arch["levels"]); ksz=int(arch["k"]); out_dim=int(arch["out_dim"])
    TCN_SEQ_LEN = int(TCN_CFG.get("SEQ_LEN", TCN_SEQ_LEN_FALLBACK))
    TCN_MODEL = TCNBinary(in_feat=len(TCN_FEATS), hidden=hidden, levels=levels, k=ksz, dropout=0.1, out_dim=out_dim).eval()
    # fp64 파라미터 방지
    for k,v in list(raw_state.items()):
        if isinstance(v, torch.Tensor) and v.dtype==torch.float64:
            raw_state[k] = v.float()
    TCN_MODEL.load_state_dict(raw_state, strict=False)
    TCN_MU = _ckpt.get("scaler_mu", None) if isinstance(_ckpt, dict) else None
    TCN_SD = _ckpt.get("scaler_sd", None) if isinstance(_ckpt, dict) else None
    print(f"[TCN] loaded: hidden={hidden} levels={levels} k={ksz} out_dim={out_dim} feats={TCN_FEATS} seq_len={TCN_SEQ_LEN}")
except Exception as e:
    print(f"[WARN] TCN load failed: {e}")
    TCN_MODEL = None

@torch.no_grad()
def _tcn_mu(symbol: str, df: pd.DataFrame) -> Optional[float]:
    if TCN_MODEL is None: return None
    if df is None or len(df) < (TCN_SEQ_LEN+1): return None
    feats = _build_tcn_features(df)
    if len(feats) < TCN_SEQ_LEN: return None
    X = feats[TCN_FEATS].tail(TCN_SEQ_LEN).to_numpy().astype(np.float32)
    X = _apply_tcn_scaler(X, TCN_MU, TCN_SD)
    x_t = torch.from_numpy(X[None, ...])
    mu, _ = TCN_MODEL(x_t)
    return float(mu.item())

def _resolve_thr():
    mode = os.getenv("THR_MODE", "fixed").lower()
    if mode == "fixed":
        bps = float(os.getenv("MODEL_THR_BPS","5.0"))
        return bps / 1e4
    tf = float(os.getenv("TAKER_FEE","0.0006"))
    slip = float(os.getenv("SLIPPAGE_BPS_TAKER","1.0"))/1e4
    safety = float(os.getenv("FEE_SAFETY","1.2"))
    return np.log(1.0 + safety*(2*tf + 2*slip))

@torch.no_grad()
def dual_deepar_tcn_direction(symbol: str) -> Tuple[Optional[str], float]:
    if symbol in SKIP_SYMBOLS:
        return (None, 0.0)

    need = SEQ_LEN + max(1, PRED_LEN)
    df = _recent_1m_df(symbol, minutes=SEQ_LEN + PRED_LEN + 10)
    if df is None or len(df) < (SEQ_LEN + 1): return (None, 0.0)
    if df is None or len(df) < need:
        if DEBUG_SIGNALS: print(f"[SIG][{symbol}] no-data len<{need}", flush=True)
        return (None, 0.0)

    mu1 = None
    if DEEPar_MAIN is not None:
        try:
            tsd = _build_infer_dataset(df)
            dl  = tsd.to_dataloader(train=False, batch_size=1, num_workers=0)
            if getattr(dl, "dataset", None) is None or len(dl.dataset)==0:
                return (None, 0.0)
            pred = DEEPar_MAIN.predict(dl, mode="prediction")
            if isinstance(pred, (list, tuple)) and len(pred)>0: pred = pred[0]
            if (pred is None) or (not torch.is_tensor(pred)) or pred.ndim<2 or pred.shape[0]<1 or pred.shape[1]<1:
                return (None, 0.0)
            mu1 = _deepar_mu(symbol, df)  # 캐시 사용
            FAIL_CNT[symbol] = 0
        except Exception as e:
            FAIL_CNT[symbol] += 1
            if DEBUG_SIGNALS: print(f"[SIG][{symbol}] deepar-error: {e} (fail {FAIL_CNT[symbol]}/{FAIL_MAX})", flush=True)
            if FAIL_CNT[symbol] >= FAIL_MAX:
                SKIP_SYMBOLS.add(symbol)
                if DEBUG_SIGNALS: print(f"[DISABLE][DEEPar] {symbol} disabled this session", flush=True)
            mu1 = None

    mu2 = _tcn_mu(symbol, df)

    if (mu1 is None) and (mu2 is None):
        if DEBUG_SIGNALS: print(f"[SIG][{symbol}] both-mu=None", flush=True)
        return (None, 0.0)

    mu1 = float(mu1 or 0.0)
    mu2 = float(mu2 or 0.0)
    thr = _resolve_thr()

    dir1 = "Buy" if mu1 > +thr else ("Sell" if mu1 < -thr else "None")
    dir2 = "Buy" if mu2 > +thr else ("Sell" if mu2 < -thr else "None")

    same_sign = (mu1 > 0 and mu2 > 0) or (mu1 < 0 and mu2 < 0)
    mag1 = abs(mu1) >= thr
    mag2 = abs(mu2) >= thr
    rule = os.getenv("DUAL_RULE","loose").lower()

    side = None
    if same_sign and ((rule=="strict" and (mag1 and mag2)) or (rule!="strict" and (mag1 or mag2))):
        side = "Buy" if mu1 > 0 else "Sell"

    if DEBUG_SIGNALS:
        print(f"[SIG][{symbol}] deepar(mu={mu1:.6g},{dir1}) | tcn(mu={mu2:.6g},{dir2}) | thr={thr:.6g} rule={rule} -> {side or 'None'}", flush=True)

    if side is None:
        return (None, 0.0)

    if ENTRY_MODE == "inverse":
        side = "Sell" if side=="Buy" else "Buy"
    elif ENTRY_MODE == "random":
        pass

    if side is None and ALLOW_TCN_FALLBACK and (mu2 is not None) and (abs(mu2) >= thr):
        side = "Buy" if mu2 > 0 else "Sell"

    return side, 1.0

# ===== entry =====
def _init_state(symbol:str, side:str, entry:float, qty:float, lot_step:float):
    STATE[symbol] = {
        "side":side, "entry":float(entry), "init_qty":float(qty),
        "tp_done":[False,False,False], "be_moved":False,
        "peak":float(entry), "trough":float(entry),
        "entry_ts":time.time(), "lot_step":float(lot_step),
    }

def _bps_between_local(p1:float, p2:float)->float:
    return abs((p2 - p1) / max(p1, 1e-12)) * 10_000.0

def _entry_cost_bps_local()->float:
    fee = (0.0 if EXECUTION_MODE=="maker" else TAKER_FEE) * 10_000.0
    slip= (SLIPPAGE_BPS_MAKER if EXECUTION_MODE=="maker" else SLIPPAGE_BPS_TAKER)
    return max(0.0, fee + max(0.0, slip))

def _exit_cost_bps_local()->float:
    fee = (0.0 if EXIT_EXEC_MODE=="maker" else TAKER_FEE) * 10_000.0
    slip= (SLIPPAGE_BPS_MAKER if EXIT_EXEC_MODE=="maker" else SLIPPAGE_BPS_TAKER)
    return max(0.0, fee + max(0.0, slip))

def choose_entry(symbol:str)->Tuple[Optional[str], float]:
    side, conf = dual_deepar_tcn_direction(symbol)
    return (side, float(conf or 0.0)) if side in ("Buy","Sell") else (None, 0.0)
LAST_SIG_TS = {}  # symbol -> float
SIG_MIN_INTERVAL_SEC = float(os.getenv("SIG_MIN_INTERVAL_SEC", "5"))

def try_enter(symbol: str, side_override: Optional[str]=None, size_pct: Optional[float]=None):
    if time.time() - LAST_SIG_TS.get(symbol, 0.0) < SIG_MIN_INTERVAL_SEC:
        return
    LAST_SIG_TS[symbol] = time.time()
    if len(get_positions()) >= MAX_OPEN: return
    if COOLDOWN_UNTIL.get(symbol,0.0) > time.time(): return
    if symbol in PENDING_MAKER: return

    side, conf = choose_entry(symbol)
    if side_override in ("Buy","Sell"):
        side = side_override; conf = max(conf, 0.60)
    if side not in ("Buy","Sell") or conf < 0.55: return

    bid,ask,mid = get_quote(symbol)
    if mid <= 0: return

    rule = get_instrument_rule(symbol)
    entry_ref = float(ask if side=="Buy" else bid)
    tp1_px = _round_to_tick(tp_from_bps(entry_ref, TP1_BPS, side), rule["tickSize"])
    edge_bps = _bps_between_local(entry_ref, tp1_px)
    cost_bps = _entry_cost_bps_local() + _exit_cost_bps_local()
    need_bps = cost_bps + MIN_EXPECTED_ROI_BPS
    if edge_bps < need_bps: return

    eq = get_wallet_equity()
    use_pct = float(size_pct if size_pct is not None else ENTRY_EQUITY_PCT)
    qty, _ = compute_qty_by_equity_pct(symbol, mid, eq, use_pct)
    qty *= 4
    if qty <= 0: return

    ensure_isolated_and_leverage(symbol)

    if EXECUTION_MODE == "maker":
        limit_px = bid if side=="Buy" else ask
        oid = place_limit_postonly(symbol, side, qty, limit_px)
        if oid:
            PENDING_MAKER[symbol] = {"orderId": oid, "ts": time.time(), "side": side, "qty": qty, "price": limit_px}
            _log_trade({"ts":int(time.time()),"event":"ENTRY_POSTONLY","symbol":symbol,"side":side,
                        "qty":f"{qty:.8f}","entry":f"{limit_px:.6f}","exit":"","tp1":f"{tp1_px:.6f}",
                        "tp2":"","tp3":"","sl":"","reason":"open","mode":ENTRY_MODE})
        return

    # taker
    ok = place_market(symbol, side, qty)
    if not ok: return
    pos = get_positions().get(symbol)
    if not pos: return

    entry = float(pos.get("entry") or mid)
    lot = rule["qtyStep"]
    sl = compute_sl(entry, qty, side)
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
    _log_trade({"ts":int(time.time()),"event":"ENTRY","symbol":symbol,"side":side,
                "qty":f"{qty:.8f}","entry":f"{entry:.6f}","exit":"","tp1":f"{tp1:.6f}",
                "tp2":f"{tp2:.6f}","tp3":f"{tp3:.6f}","sl":f"{sl:.6f}","reason":"open","mode":ENTRY_MODE})

# ===== manage =====
def _update_trailing_and_be(symbol:str):
    st = STATE.get(symbol)
    if not st: return
    pos = get_positions().get(symbol)
    if not pos: return
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

    if st["tp_done"][1] and not st["be_moved"]:
        be_px = entry*(1.0 + BE_EPS_BPS/10000.0) if side=="Buy" else entry*(1.0 - BE_EPS_BPS/10000.0)
        set_stop_with_retry(symbol, sl_price=be_px, tp_price=None, position_idx=pos.get("positionIdx"))
        st["be_moved"]=True
        _log_trade({"ts":int(time.time()),"event":"MOVE_SL","symbol":symbol,"side":side,"qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":f"{be_px:.6f}","reason":"BE","mode":ENTRY_MODE})

    if st["tp_done"][TRAIL_AFTER_TIER-1]:
        if side=="Buy": new_sl = mark*(1.0 - TRAIL_BPS/10000.0)
        else:           new_sl = mark*(1.0 + TRAIL_BPS/10000.0)
        set_stop_with_retry(symbol, sl_price=new_sl, tp_price=None, position_idx=pos.get("positionIdx"))

def _check_time_stop_and_cooldowns(symbol:str):
    st=STATE.get(symbol); pos=get_positions().get(symbol)
    if not st or not pos: return
    if MAX_HOLD_SEC>0 and (time.time()-st.get("entry_ts",time.time()))>MAX_HOLD_SEC:
        side=pos["side"]; qty=float(pos["qty"])
        close_side = "Sell" if side=="Buy" else "Buy"
        _order_with_mode_retry(dict(category=CATEGORY, symbol=symbol, side=close_side, orderType="Market",
                                    qty=str(qty), reduceOnly=True, timeInForce="IOC"), side=close_side)
        COOLDOWN_UNTIL[symbol]=time.time()+COOLDOWN_SEC
        _log_trade({"ts":int(time.time()),"event":"EXIT","symbol":symbol,"side":side,"qty":f"{qty:.8f}","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"","reason":"TIMEOUT","mode":ENTRY_MODE})

# ===== pending PostOnly =====
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

def _has_enough_history_symbol(symbol: str) -> bool:
    need = SEQ_LEN + max(1, PRED_LEN)
    df = _recent_1m_df(symbol, minutes=need + 10)
    return (df is not None) and (len(df) >= need)

RUNNABLE: List[str] = []

def monitor_loop(poll_sec: float = 1.0):
    last_pos_keys = set()
    while True:
        try:
            _log_equity(get_wallet_equity())
            process_pending_postonly()

            pos = get_positions()
            pos_keys = set(pos.keys())

            # 새 포지션
            new_syms = pos_keys - last_pos_keys
            for sym in new_syms:
                p = pos[sym]
                side = p["side"]; entry = float(p.get("entry") or 0.0); qty = float(p.get("qty") or 0.0)
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
                _log_trade({"ts": int(time.time()), "event": "ENTRY","symbol": sym, "side": side,
                            "qty": f"{qty:.8f}","entry": f"{entry:.6f}", "exit": "",
                            "tp1": f"{tp1:.6f}", "tp2": f"{tp2:.6f}", "tp3": f"{tp3:.6f}",
                            "sl": f"{sl:.6f}", "reason": "postonly_fill_or_detect", "mode": ENTRY_MODE})

            # 닫힘
            closed_syms = last_pos_keys - pos_keys
            for sym in closed_syms:
                b, a, m = get_quote(sym)
                st = STATE.pop(sym, None)
                side = (st or {}).get("side", "")
                LAST_EXIT[sym] = {"side": side, "px": float(m or a or b or 0.0), "ts": time.time()}
                COOLDOWN_UNTIL[sym] = time.time() + COOLDOWN_SEC
                _log_trade({"ts": int(time.time()), "event": "EXIT","symbol": sym, "side": side, "qty": "",
                            "entry": "", "exit": f"{(m or 0.0):.6f}",
                            "tp1": "", "tp2": "", "tp3": "", "sl": "",
                            "reason": "CLOSED", "mode": ENTRY_MODE})

            # 관리
            for sym in pos_keys:
                try:
                    _update_trailing_and_be(sym)
                    _check_time_stop_and_cooldowns(sym)
                except Exception as _e:
                    print(f"[MANAGE ERR] {sym}", _e)

            last_pos_keys = pos_keys
            time.sleep(poll_sec)

        except KeyboardInterrupt:
            print("[MONITOR] stop"); break
        except Exception as e:
            print("[MONITOR ERR]", e); time.sleep(2.0)

def scanner_loop(iter_delay:float=2.5):
    global RUNNABLE
    while True:
        try:
            if time.time() < GLOBAL_COOLDOWN_UNTIL:
                time.sleep(iter_delay); continue

            syms = _top_symbols_by_volume(RUNNABLE, top_n=min(MAX_OPEN, len(RUNNABLE)))
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
    global RUNNABLE
    syms_env = os.getenv("SYMBOLS","ASTERUSDT")
    SYMBOLS = [s.strip() for s in syms_env.split(",") if s.strip()]
    print(f"[START] Bybit Live Trading (DeepAR+TCN) MODE={ENTRY_MODE} EXEC={EXECUTION_MODE}/{EXIT_EXEC_MODE} TESTNET={TESTNET}")
    # 초기 이력 필터
    need = SEQ_LEN + max(1, PRED_LEN)
    runnable = []
    for s in SYMBOLS:
        if _has_enough_history_symbol(s):
            runnable.append(s)
        else:
            SKIP_SYMBOLS.add(s)
            if DEBUG_SIGNALS: print(f"[DISABLE] {s} insufficient history at start (need {need} bars)")
    if not runnable:
        print("[ABORT] no runnable symbols")
        return
    RUNNABLE = runnable
    # 루프
    t1=threading.Thread(target=monitor_loop, args=(1.0,), daemon=True)
    t2=threading.Thread(target=scanner_loop, args=(2.5,), daemon=True)
    t1.start(); t2.start()
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
