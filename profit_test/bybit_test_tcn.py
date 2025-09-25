# -*- coding: utf-8 -*-
"""
TCN(BINARY) paper trader for Bybit USDT-Perp 1m
- 이진 분류 체크포인트(tcn_best.pt) 로드: 출력 1개(logit)
- 시그모이드 확률로 방향 결정: p>=SIG_THR → Buy, p<=1-SIG_THR → Sell, 그 외 skip
- 나머지 진입 필터/TP/SL/ABS/타임아웃 로직 동일
"""

import os, time, math, csv, random, threading, collections
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pybit.unified_trading import HTTP
from concurrent.futures import ThreadPoolExecutor

# =================== ENV ===================
CATEGORY = "linear"
INTERVAL = "1"  # 1m

SYMBOLS = os.getenv("SYMBOLS","ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT,ADAUSDT,LINKUSDT,UNIUSDT,TRXUSDT,LTCUSDT,MNTUSDT,SUIUSDT,1000PEPEUSDT,XLMUSDT,ARBUSDT,APTUSDT,OPUSDT,AVAXUSDT").split(",")
SYMBOLS = [
    "ASTERUSDT"
]
ENTRY_MODE = os.getenv("ENTRY_MODE", "model")  # model|inverse|random

# ===== Binary decision =====
SIG_THR = float(os.getenv("SIG_THR","0.5"))        # p>=SIG_THR → Buy, p<=1-SIG_THR → Sell
NO_TRADE_BAND = float(os.getenv("NO_TRADE_BAND","0.005"))  # |p-0.5|<band → skip

# Account and fees
START_EQUITY = float(os.getenv("START_EQUITY","1000.0"))
LEVERAGE = float(os.getenv("LEVERAGE","20"))
ENTRY_EQUITY_PCT = float(os.getenv("ENTRY_EQUITY_PCT","0.20"))
TAKER_FEE = float(os.getenv("TAKER_FEE","0.0006"))
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER","1.0"))
FEE_SAFETY = float(os.getenv("FEE_SAFETY","1.2"))

# TP/SL
TP1_BPS = float(os.getenv("TP1_BPS","50.0"))
TP2_BPS = float(os.getenv("TP2_BPS","100.0"))
TP3_BPS = float(os.getenv("TP3_BPS","200.0"))
SL_ROI_PCT = float(os.getenv("SL_ROI_PCT","0.01"))
TP1_RATIO = float(os.getenv("TP1_RATIO","0.40"))
TP2_RATIO = float(os.getenv("TP2_RATIO","0.35"))
BE_EPS_BPS = float(os.getenv("BE_EPS_BPS","2.0"))
TRAIL_BPS = float(os.getenv("TRAIL_BPS","50.0"))
TRAIL_AFTER_TIER = int(os.getenv("TRAIL_AFTER_TIER","2"))

# ABS TP
ABS_TP_USD = float(os.getenv("ABS_TP_USD","0"))
ABS_K = float(os.getenv("ABS_K","1.0"))
ABS_TP_USD_FLOOR = float(os.getenv("ABS_TP_USD_FLOOR","5.0"))

# Timeout
TIMEOUT_MODE = os.getenv("TIMEOUT_MODE", "atr")  # fixed|atr
VOL_WIN = int(os.getenv("VOL_WIN", "60"))
TIMEOUT_K = float(os.getenv("TIMEOUT_K", "1.5"))
TIMEOUT_MIN = int(os.getenv("TIMEOUT_MIN", "3"))
TIMEOUT_MAX = int(os.getenv("TIMEOUT_MAX", "20"))
PRED_HORIZ_MIN = int(os.getenv("PRED_HORIZ_MIN","1"))
_v = os.getenv("TIMEOUT_TARGET_BPS")
TIMEOUT_TARGET_BPS = float(_v) if (_v is not None and _v.strip() != "") else None

# Filters / session
MIN_EXPECTED_ROI_BPS = float(os.getenv("MIN_EXPECTED_ROI_BPS","30.0"))
V_BPS_FLOOR = float(os.getenv("V_BPS_FLOOR","5.0"))
N_CONSEC_SIGNALS = int(os.getenv("N_CONSEC_SIGNALS","1"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC","30"))
SESSION_MAX_TRADES = int(os.getenv("SESSION_MAX_TRADES","1000"))
SESSION_MAX_MINUTES = int(os.getenv("SESSION_MAX_MINUTES","5"))

# Scan
SCAN_PARALLEL = str(os.getenv("SCAN_PARALLEL","1")).lower() in ("1","y","yes","true")
SCAN_WORKERS  = int(os.getenv("SCAN_WORKERS","8"))

# Logs
TRADES_CSV = os.getenv("TRADES_CSV",f"paper_trades_tcn_bin_{START_EQUITY}_{LEVERAGE}.csv")
EQUITY_CSV = os.getenv("EQUITY_CSV",f"paper_equity_tcn_bin_{START_EQUITY}_{LEVERAGE}.csv")
TRADES_FIELDS = ["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","reason","mode"]
EQUITY_FIELDS = ["ts","equity"]
VERBOSE = str(os.getenv("VERBOSE","1")).lower() in ("1","y","yes","true")

# Model
TCN_CKPT = os.getenv("TCN_CKPT",r"D:\ygy_work\coin\multimodel\tcn_best.pt")   # binary classifier checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =================== STATE ===================
_equity_cache = START_EQUITY
POSITION_DEADLINE: Dict[str,int] = {}
STATE: Dict[str,dict] = {}
ENTRY_LOCK = threading.Lock()
ENTRY_HISTORY: Dict[str,collections.deque] = {s: collections.deque(maxlen=max(1,N_CONSEC_SIGNALS)) for s in SYMBOLS}
COOLDOWN_UNTIL: Dict[str,float] = {}
SESSION_START_TS = time.time()
SESSION_TRADES = 0

# =================== MARKET ===================
TESTNET = str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")
mkt = HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

def _ensure_csv():
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(TRADES_FIELDS)
    if not os.path.exists(EQUITY_CSV):
        with open(EQUITY_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(EQUITY_FIELDS)

def _log_trade(row: dict):
    _ensure_csv()
    with open(TRADES_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(k,"") for k in TRADES_FIELDS])
    if VERBOSE: print(f"[TRADE] {row}")

def _log_equity(eq: float):
    global _equity_cache
    _equity_cache = float(eq)
    _ensure_csv()
    with open(EQUITY_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([int(time.time()), f"{_equity_cache:.6f}"])
    if VERBOSE: print(f"[EQUITY] {_equity_cache:.2f}")

def get_wallet_equity() -> float: return float(_equity_cache)

def apply_trade_pnl(side: str, entry: float, exit: float, qty: float, taker_fee: float = TAKER_FEE):
    gross = (exit - entry) * qty if side=="Buy" else (entry - exit) * qty
    fee = (entry * qty + exit * qty) * taker_fee
    new_eq = get_wallet_equity() + gross - fee
    _log_equity(new_eq); return gross, fee, new_eq

def get_quote(symbol: str) -> Tuple[float,float,float]:
    r = mkt.get_tickers(category=CATEGORY, symbol=symbol)
    row = ((r.get("result") or {}).get("list") or [{}])[0]
    bid = float(row.get("bid1Price") or 0.0)
    ask = float(row.get("ask1Price") or 0.0)
    last = float(row.get("lastPrice") or 0.0)
    mid = (bid+ask)/2.0 if bid>0 and ask>0 else (last or bid or ask)
    return bid, ask, float(mid or 0.0)

def get_recent_1m(symbol: str, minutes: int = 260) -> Optional[pd.DataFrame]:
    r = mkt.get_kline(category=CATEGORY, symbol=symbol, interval=INTERVAL, limit=min(minutes,1000))
    rows = (r.get("result") or {}).get("list") or []
    if not rows: return None
    rows = rows[::-1]
    df = pd.DataFrame([{
        "timestamp": pd.to_datetime(int(z[0]), unit="ms", utc=True),
        "symbol": symbol,
        "open": float(z[1]), "high": float(z[2]), "low": float(z[3]),
        "close": float(z[4]), "volume": float(z[5]), "turnover": float(z[6]),
    } for z in rows])
    return df

def recent_vol_bps(symbol: str, minutes: int = None) -> float:
    minutes = minutes or VOL_WIN
    df = get_recent_1m(symbol, minutes=minutes)
    if df is None or len(df) < 5: return 10.0
    ret = np.log(df["close"]).diff().abs().fillna(0.0).to_numpy()
    v_bps = float(np.median(ret) * 1e4)
    return max(v_bps, 1.0)

def fee_threshold(taker_fee=TAKER_FEE, slip_bps=SLIPPAGE_BPS_TAKER, safety=FEE_SAFETY):
    rt = 2.0*taker_fee + 2.0*(slip_bps/1e4)
    return math.log(1.0 + safety*rt)

def tp_from_bps(entry: float, bps: float, side: str) -> float:
    return entry*(1.0 + bps/10000.0) if side=="Buy" else entry*(1.0 - bps/10000.0)

def _bps_between(p1:float,p2:float)->float:
    return abs((p2 - p1) / max(p1, 1e-12)) * 10_000.0

def _entry_cost_bps()->float: return TAKER_FEE*1e4 + SLIPPAGE_BPS_TAKER
def _exit_cost_bps()->float:  return TAKER_FEE*1e4 + SLIPPAGE_BPS_TAKER

def calc_timeout_minutes(symbol: str) -> int:
    if TIMEOUT_MODE == "fixed": return max(1, int(PRED_HORIZ_MIN))
    target_bps = TIMEOUT_TARGET_BPS if TIMEOUT_TARGET_BPS is not None else TP1_BPS
    v_bps = recent_vol_bps(symbol, VOL_WIN)
    need = math.ceil((target_bps / v_bps) * TIMEOUT_K)
    return int(min(max(need, TIMEOUT_MIN), TIMEOUT_MAX))

def _dynamic_abs_tp_usd(symbol:str, mid:float, qty:float)->float:
    if ABS_TP_USD > 0: return ABS_TP_USD
    v_bps = recent_vol_bps(symbol, VOL_WIN)
    atr_usd = (v_bps/1e4) * mid * qty
    return max(ABS_TP_USD_FLOOR, ABS_K * atr_usd)

# =================== TCN MODEL (Binary) ===================
FEATS = ["ret","rv","mom","vz"]  # trainer와 동일해야 함

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
        out = self.conv1(x); out = self.chomp1(out); out = self.relu1(out); out = self.drop1(out)
        out = self.conv2(out); out = self.chomp2(out); out = self.relu2(out); out = self.drop2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNBinary(nn.Module):
    def __init__(self, in_feat, hidden=128, levels=6, k=3, dropout=0.1):
        super().__init__()
        layers = []
        ch_in = in_feat
        for i in range(levels):
            ch_out = hidden
            dilation = 2**i
            layers += [TemporalBlock(ch_in, ch_out, k, dilation, dropout)]
            ch_in = ch_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)   # single logit
    def forward(self, x):             # x: [B,L,F]
        x = x.transpose(1,2)          # [B,F,L]
        h = self.tcn(x)               # [B,H,L]
        h = h[:, :, -1]               # last step
        logit = self.head(h).squeeze(1)  # [B]
        return logit

def load_tcn(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"]
    cfg = ckpt.get("cfg", {"H":3,"SEQ_LEN":240,"FEATS":FEATS})
    mu = ckpt.get("scaler_mu", None)
    sd = ckpt.get("scaler_sd", None)

    # ---- 구조 자동 추론 ----
    # hidden: head.weight shape = [1, hidden]
    hidden = state["head.weight"].shape[1]

    # levels: tcn.<idx>.conv1.* 의 최대 idx + 1
    blk_ids = []
    for k in state.keys():
        if k.startswith("tcn.") and ".conv1." in k:
            blk_ids.append(int(k.split(".")[1]))
    levels = (max(blk_ids) + 1) if blk_ids else 6

    # kernel size k: 첫 블록 conv1 weight_v의 마지막 차원
    kname = f"tcn.0.conv1.weight_v"
    k = state[kname].shape[-1] if kname in state else 3

    # in_feat: 첫 블록 downsample.weight 입력 채널
    in_feat = state.get("tcn.0.downsample.weight", None)
    if in_feat is not None:
        in_feat = in_feat.shape[1]
    else:
        in_feat = len(cfg.get("FEATS", FEATS))

    # ---- 모델 생성 & 로드 ----
    model = TCNBinary(in_feat=in_feat, hidden=hidden, levels=levels, k=k, dropout=0.1)
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval(), cfg, mu, sd

TCN_MODEL, TCN_CFG, SCALER_MU, SCALER_SD = load_tcn(TCN_CKPT)
SEQ_LEN = int(TCN_CFG.get("SEQ_LEN", 240))
H = int(TCN_CFG.get("H", 3))

# =================== FEATURES ===================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.sort_values("timestamp").copy()
    c = g["close"].astype(float).values
    v = g["volume"].astype(float).values

    ret = np.zeros_like(c, dtype=np.float64); ret[1:] = np.log(c[1:] / np.clip(c[:-1], 1e-12, None))
    win = 60
    rv = pd.Series(ret).rolling(win, min_periods=win//3).std().bfill().values
    ema_win = 60; alpha = 2/(ema_win+1)
    ema = np.zeros_like(c, dtype=np.float64); ema[0] = c[0]
    for i in range(1,len(c)): ema[i] = alpha*c[i] + (1-alpha)*ema[i-1]
    mom = (c/np.maximum(ema,1e-12) - 1.0)
    v_ma = pd.Series(v).rolling(win, min_periods=win//3).mean().bfill().values
    v_sd = pd.Series(v).rolling(win, min_periods=win//3).std().replace(0, np.nan).bfill().values
    vz  = (v - v_ma) / np.where(v_sd>0, v_sd, 1.0)

    out = pd.DataFrame({"timestamp": g["timestamp"].values, "ret": ret, "rv": rv, "mom": mom, "vz": vz})
    return out

def apply_scaler_np(X: np.ndarray, mu, sd):
    if mu is None or sd is None: return X
    mu = np.asarray(mu, dtype=np.float32); sd = np.asarray(sd, dtype=np.float32)
    sd = np.where(sd>0, sd, 1.0); return (X - mu) / sd

# =================== DIRECTION (Binary) ===================
@torch.no_grad()
def tcn_direction(symbol: str):
    df = get_recent_1m(symbol, minutes=SEQ_LEN+H+10)
    if df is None or len(df) < (SEQ_LEN+1): return None, 0.0
    feats = build_features(df)
    if len(feats) < SEQ_LEN: return None, 0.0

    X = feats[FEATS].tail(SEQ_LEN).to_numpy().astype(np.float32)
    X = apply_scaler_np(X, SCALER_MU, SCALER_SD)
    x_t = torch.from_numpy(X[None, ...]).to(DEVICE)  # [1,L,F]

    logit = TCN_MODEL(x_t)           # [1]
    p = float(torch.sigmoid(logit).item())  # P(up)

    if VERBOSE: print(f"[DEBUG][{symbol}] p_up={p:.4f} thr={SIG_THR:.3f} band={NO_TRADE_BAND:.3f}")

    # 대기 밴드
    if abs(p - 0.5) < NO_TRADE_BAND:
        base = None
    else:
        if   p >= SIG_THR: base = "Buy"
        elif p <= 1.0 - SIG_THR: base = "Sell"
        else: base = None

    # ENTRY_MODE
    side = base
    if ENTRY_MODE == "inverse":
        if   base == "Buy": side = "Sell"
        elif base == "Sell": side = "Buy"
        else: side = None
    elif ENTRY_MODE == "random":
        side = base or random.choice(["Buy","Sell"])
    return side, (p if side else 0.0)

def choose_entry(symbol: str): return tcn_direction(symbol)

# =================== BROKER + LOGIC (동일) ===================
def _round_down(x: float, step: float) -> float:
    if step<=0: return x
    return math.floor(x/step)*step

class PaperBroker:
    def __init__(self):
        self.pos: Dict[str,dict] = {}
        self.qty_step = 0.001
        self.min_qty = 0.001
    def try_open(self, symbol: str, side: str, mid: float, ts: int):
        eq = get_wallet_equity()
        qty = _round_down(eq * ENTRY_EQUITY_PCT * LEVERAGE / max(mid,1e-9), self.qty_step)
        if qty < self.min_qty: return False
        self.pos[symbol] = {"side": side, "entry": mid, "qty": qty, "opened": ts}
        return True
    def try_close(self, symbol: str, ts: int) -> Optional[float]:
        if symbol not in self.pos: return None
        _,_,mark = get_quote(symbol)
        p = self.pos[symbol]; entry=p["entry"]; qty=p["qty"]
        pnl = (mark-entry)*qty if p["side"]=="Buy" else (entry-mark)*qty
        del self.pos[symbol]; return pnl
    def positions(self): return self.pos

BROKER = PaperBroker()

def unrealized_pnl_usd(side: str, qty: float, entry: float, mark: float) -> float:
    return (mark - entry) * qty if side=="Buy" else (entry - mark) * qty

def _entry_allowed_by_filters(symbol:str, side:str, entry_ref:float)->bool:
    v_bps = recent_vol_bps(symbol, VOL_WIN)
    if v_bps < V_BPS_FLOOR:
        if VERBOSE: print(f"[SKIP] {symbol} vol {v_bps:.1f}bps < floor {V_BPS_FLOOR}bps"); return False
    tp1_px = tp_from_bps(entry_ref, TP1_BPS, side)
    edge_bps = _bps_between(entry_ref, tp1_px)
    need_bps = _entry_cost_bps() + _exit_cost_bps() + MIN_EXPECTED_ROI_BPS
    if edge_bps < need_bps:
        if VERBOSE: print(f"[SKIP] {symbol} ROI edge {edge_bps:.1f} < need {need_bps:.1f}bps"); return False
    return True

def try_enter(symbol: str):
    global SESSION_START_TS, SESSION_TRADES

    if time.time() - SESSION_START_TS >= SESSION_MAX_MINUTES * 60:
        SESSION_START_TS = time.time(); SESSION_TRADES = 0
    if SESSION_MAX_TRADES > 0 and SESSION_TRADES >= SESSION_MAX_TRADES:
        if VERBOSE:
            remain = int(SESSION_MAX_MINUTES * 60 - (time.time() - SESSION_START_TS))
            print(f"[SKIP] session trade cap hit (remain {max(remain, 0)}s)")
        return
    if (time.time() - SESSION_START_TS) >= (SESSION_MAX_MINUTES * 60):
        if VERBOSE: print("[SKIP] session time cap hit"); return
    if COOLDOWN_UNTIL.get(symbol, 0.0) > time.time(): return
    if symbol in BROKER.positions(): return

    side, conf = choose_entry(symbol)
    if side not in ("Buy", "Sell"): return

    dq = ENTRY_HISTORY[symbol]; dq.append(side)
    if len(dq) < N_CONSEC_SIGNALS or any(x != side for x in dq):
        if VERBOSE: print(f"[SKIP] consec fail {symbol} need={N_CONSEC_SIGNALS} got={list(dq)}")
        return

    bid, ask, mid = get_quote(symbol)
    if mid <= 0: return

    entry_ref = float(ask if side == "Buy" else bid)
    if not _entry_allowed_by_filters(symbol, side, entry_ref): return

    ts = int(time.time())
    with ENTRY_LOCK:
        if symbol in BROKER.positions(): return
        if not BROKER.try_open(symbol, side, mid, ts): return

        horizon_min = max(1, int(calc_timeout_minutes(symbol)))
        POSITION_DEADLINE[symbol] = ts + horizon_min * 60

        tp1 = tp_from_bps(mid, TP1_BPS, side)
        tp2 = tp_from_bps(mid, TP2_BPS, side)
        tp3 = tp_from_bps(mid, TP3_BPS, side)
        sl  = mid * (1.0 - SL_ROI_PCT) if side == "Buy" else mid * (1.0 + SL_ROI_PCT)

        pos_qty = BROKER.positions()[symbol]["qty"]
        STATE[symbol] = {"tp1": tp1, "tp2": tp2, "tp3": tp3, "sl": sl,
                         "tp1_filled": False, "tp2_filled": False, "be_moved": False,
                         "abs_usd": _dynamic_abs_tp_usd(symbol, mid, pos_qty)}

        _log_trade({"ts": ts, "event": "ENTRY", "symbol": symbol, "side": side,
                    "qty": f"{pos_qty:.8f}", "entry": f"{mid:.6f}", "exit": "",
                    "tp1": f"{tp1:.6f}", "tp2": f"{tp2:.6f}", "tp3": f"{tp3:.6f}",
                    "sl": f"{sl:.6f}", "reason": "open", "mode": "paper"})

        SESSION_TRADES += 1
        if VERBOSE:
            print(f"[OPEN] {symbol} {side} entry={mid:.4f} timeout={horizon_min}m "
                  f"abs≈{STATE[symbol]['abs_usd']:.2f}USD  [EQUITY]={get_wallet_equity():.2f}")

def close_and_log(symbol: str, reason: str):
    now = int(time.time())
    if symbol not in BROKER.positions(): return
    p = BROKER.positions()[symbol]; side = p["side"]; entry = p["entry"]; qty = p["qty"]
    _,_,mark = get_quote(symbol)
    BROKER.try_close(symbol, now)
    gross, fee, new_eq = apply_trade_pnl(side, entry, mark, qty, taker_fee=TAKER_FEE)
    _log_trade({"ts": now, "event":"EXIT", "symbol":symbol, "side":"Sell" if side=="Buy" else "Buy",
                "qty":f"{qty:.8f}","entry":f"{entry:.6f}","exit":f"{mark:.6f}",
                "tp1":"","tp2":"","tp3":"","sl":"","reason":reason,"mode":"paper"})
    if VERBOSE: print(f"[{reason}] {symbol} gross={gross:.6f} fee={fee:.6f} eq={new_eq:.2f}")
    POSITION_DEADLINE.pop(symbol, None); STATE.pop(symbol, None)
    COOLDOWN_UNTIL[symbol] = time.time() + COOLDOWN_SEC

def monitor_loop(poll_sec: float = 1.0, max_loops: int = 2):
    loops = 0
    while loops < max_loops:
        _log_equity(get_wallet_equity())
        now = int(time.time())
        for sym, p in list(BROKER.positions().items()):
            side=p["side"]; entry=p["entry"]; qty=p["qty"]
            _,_,mark = get_quote(sym)
            if mark<=0: continue

            st = STATE.get(sym, None)
            if st is None:
                tp1 = tp_from_bps(entry, TP1_BPS, side)
                tp2 = tp_from_bps(entry, TP2_BPS, side)
                tp3 = tp_from_bps(entry, TP3_BPS, side)
                sl  = entry*(1.0 - SL_ROI_PCT) if side=="Buy" else entry*(1.0 + SL_ROI_PCT)
                st = {"tp1":tp1,"tp2":tp2,"tp3":tp3,"sl":sl,
                      "tp1_filled":False,"tp2_filled":False,"be_moved":False,
                      "abs_usd": _dynamic_abs_tp_usd(sym, entry, qty)}
                STATE[sym]=st

            # TP1
            if (not st["tp1_filled"]) and ((mark>=st["tp1"] and side=="Buy") or (mark<=st["tp1"] and side=="Sell")):
                part = qty*TP1_RATIO
                apply_trade_pnl(side, entry, st["tp1"], part, taker_fee=TAKER_FEE)
                p["qty"] = qty = max(0.0, qty - part)
                st["tp1_filled"]=True
                _log_trade({"ts":int(time.time()),"event":"TP1_FILL","symbol":sym,"side":side,
                            "qty":f"{part:.8f}","entry":f"{entry:.6f}","exit":f"{st['tp1']:.6f}",
                            "tp1":"","tp2":"","tp3":"","sl":"","reason":"","mode":"paper"})

            # TP2
            if qty>0 and (not st["tp2_filled"]) and ((mark>=st["tp2"] and side=="Buy") or (mark<=st["tp2"] and side=="Sell")):
                remain_ratio = 1.0 - TP1_RATIO
                part = min(qty, remain_ratio*TP2_RATIO)
                apply_trade_pnl(side, entry, st["tp2"], part, taker_fee=TAKER_FEE)
                p["qty"] = qty = max(0.0, qty - part)
                st["tp2_filled"]=True
                _log_trade({"ts":int(time.time()),"event":"TP2_FILL","symbol":sym,"side":side,
                            "qty":f"{part:.8f}","entry":f"{entry:.6f}","exit":f"{st['tp2']:.6f}",
                            "tp1":"","tp2":"","tp3":"","sl":"","reason":"","mode":"paper"})

            # BE
            if st["tp2_filled"] and not st["be_moved"]:
                be_px = entry*(1.0 + BE_EPS_BPS/10000.0) if side=="Buy" else entry*(1.0 - BE_EPS_BPS/10000.0)
                st["sl"] = be_px; st["be_moved"]=True
                _log_trade({"ts":int(time.time()),"event":"MOVE_SL","symbol":sym,"side":side,
                            "qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"",
                            "sl":f"{be_px:.6f}","reason":"BE","mode":"paper"})

            # Trailing
            if (st["tp1_filled"] if TRAIL_AFTER_TIER==1 else st["tp2_filled"]):
                if side=="Buy":  st["sl"] = max(st["sl"], mark*(1.0 - TRAIL_BPS/10000.0))
                else:            st["sl"] = min(st["sl"], mark*(1.0 + TRAIL_BPS/10000.0))

            if qty > 0:
                pnl = unrealized_pnl_usd(side, qty, entry, mark)
                abs_hit = pnl >= st["abs_usd"]
                sl_hit  = (mark <= st["sl"] and side=="Buy") or (mark >= st["sl"] and side=="Sell")
                ddl = POSITION_DEADLINE.get(sym); to_hit = (ddl and now >= ddl)
                if abs_hit or sl_hit or to_hit:
                    reason = "ABS_TP" if abs_hit else ("SL" if sl_hit else "HORIZON_TIMEOUT")
                    close_and_log(sym, reason); continue

        loops += 1; time.sleep(poll_sec)

def main():
    print(f"[START] TCN-BINARY PAPER | MODE={ENTRY_MODE} TESTNET={TESTNET} SIG_THR={SIG_THR} BAND={NO_TRADE_BAND}")
    _log_equity(get_wallet_equity())
    while True:
        if SCAN_PARALLEL:
            with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
                ex.map(try_enter, SYMBOLS)
        else:
            for s in SYMBOLS: try_enter(s)
        monitor_loop(1.0, 2); time.sleep(0.5)

if __name__ == "__main__":
    main()
