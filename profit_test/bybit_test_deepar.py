# -*- coding: utf-8 -*-
# PAPER 전용 (확장판): DeepAR ckpt 로드 → 방향 결정
# 진입 필터(ROI/변동성/연속신호) + 타임아웃/동적 ABS_TP + TP1/TP2 분할, BE/트레일링, 쿨다운/세션 상한
import os, time, math, csv, random, threading, collections
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
import torch
from pybit.unified_trading import HTTP
from concurrent.futures import ThreadPoolExecutor
import warnings, logging
warnings.filterwarnings("ignore", module="lightning")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
# ===== ENV =====
CATEGORY = "linear"
INTERVAL = "1"  # 1m
SYMBOLS = os.getenv("SYMBOLS","ASTERUSDT,AIAUSDT,0GUSDT,STBLUSDT,WLFIUSDT,LINEAUSDT,BARDUSDT,SOMIUSDT,UBUSDT,OPENUSDT").split(",")

ENTRY_MODE = os.getenv("ENTRY_MODE", "model")  # model|inverse|random
START_EQUITY = float(os.getenv("START_EQUITY","1000.0"))
LEVERAGE = float(os.getenv("LEVERAGE","20"))
ENTRY_EQUITY_PCT = float(os.getenv("ENTRY_EQUITY_PCT","0.20"))
TAKER_FEE = float(os.getenv("TAKER_FEE","0.0006"))
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER","1.0"))
FEE_SAFETY = float(os.getenv("FEE_SAFETY","1.2"))

# --- TP/SL(시뮬) ---
TP1_BPS = float(os.getenv("TP1_BPS","50.0"))
TP2_BPS = float(os.getenv("TP2_BPS","100.0"))
TP3_BPS = float(os.getenv("TP3_BPS","200.0"))
SL_ROI_PCT = float(os.getenv("SL_ROI_PCT","0.01"))
TP1_RATIO = float(os.getenv("TP1_RATIO","0.40"))
TP2_RATIO = float(os.getenv("TP2_RATIO","0.35"))
BE_EPS_BPS = float(os.getenv("BE_EPS_BPS","2.0"))
TRAIL_BPS = float(os.getenv("TRAIL_BPS","50.0"))
TRAIL_AFTER_TIER = int(os.getenv("TRAIL_AFTER_TIER","2"))  # 1: TP1후, 2: TP2후

# --- 동적 ABS_TP(ATR 기반) ---
ABS_TP_USD = float(os.getenv("ABS_TP_USD","0"))  # 0이면 동적 모드 사용
ABS_K = float(os.getenv("ABS_K","1.0"))          # ABS_TP_USD = max(ABS_TP_USD_FLOOR, ABS_K * ATR_USD)
ABS_TP_USD_FLOOR = float(os.getenv("ABS_TP_USD_FLOOR","5.0"))

# --- 타임아웃(ATR) ---
TIMEOUT_MODE = os.getenv("TIMEOUT_MODE", "atr")  # fixed|atr
VOL_WIN = int(os.getenv("VOL_WIN", "60"))
TIMEOUT_K = float(os.getenv("TIMEOUT_K", "1.5"))
TIMEOUT_MIN = int(os.getenv("TIMEOUT_MIN", "3"))
TIMEOUT_MAX = int(os.getenv("TIMEOUT_MAX", "20"))
PRED_HORIZ_MIN = int(os.getenv("PRED_HORIZ_MIN","1"))
_v = os.getenv("TIMEOUT_TARGET_BPS")
TIMEOUT_TARGET_BPS = float(_v) if (_v is not None and _v.strip() != "") else None

# --- 진입 품질 필터/세션 제한 ---
MIN_EXPECTED_ROI_BPS = float(os.getenv("MIN_EXPECTED_ROI_BPS","30.0"))  # ROI 필터
V_BPS_FLOOR = float(os.getenv("V_BPS_FLOOR","10.0"))                    # 변동성 하한(bps)
N_CONSEC_SIGNALS = int(os.getenv("N_CONSEC_SIGNALS","2"))               # 연속 동일 방향 요구
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC","30"))
SESSION_MAX_TRADES = int(os.getenv("SESSION_MAX_TRADES","6"))
SESSION_MAX_MINUTES = int(os.getenv("SESSION_MAX_MINUTES","30"))

# --- 스캔 ---
SCAN_PARALLEL = str(os.getenv("SCAN_PARALLEL","1")).lower() in ("1","y","yes","true")
SCAN_WORKERS  = int(os.getenv("SCAN_WORKERS","8"))

# --- 로그 ---
TRADES_CSV = os.getenv("TRADES_CSV",f"paper_trades_{ENTRY_MODE}_{START_EQUITY}_{LEVERAGE}.csv")
EQUITY_CSV = os.getenv("EQUITY_CSV",f"paper_equity_{ENTRY_MODE}_{START_EQUITY}_{LEVERAGE}.csv")
TRADES_FIELDS = ["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","reason","mode"]
EQUITY_FIELDS = ["ts","equity"]
VERBOSE = str(os.getenv("VERBOSE","1")).lower() in ("1","y","yes","true")

# ===== STATE =====
_equity_cache = START_EQUITY
POSITION_DEADLINE: Dict[str,int] = {}
STATE: Dict[str,dict] = {}  # symbol -> tp/sl/flags/abs_usd 등
ENTRY_LOCK = threading.Lock()
ENTRY_HISTORY: Dict[str,collections.deque] = {s: collections.deque(maxlen=max(1,N_CONSEC_SIGNALS)) for s in SYMBOLS}
COOLDOWN_UNTIL: Dict[str,float] = {}
SESSION_START_TS = time.time()
SESSION_TRADES = 0

# ===== MARKET (public only) =====
TESTNET = str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")
mkt = HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

# ===== I/O =====
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
    if VERBOSE:
        print(f"[TRADE] {row}")

def _log_equity(eq: float):
    global _equity_cache
    _equity_cache = float(eq)
    _ensure_csv()
    with open(EQUITY_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([int(time.time()), f"{_equity_cache:.6f}"])
    if VERBOSE: print(f"[EQUITY] {_equity_cache:.2f}")

def get_wallet_equity() -> float:
    return float(_equity_cache)

def apply_trade_pnl(side: str, entry: float, exit: float, qty: float, taker_fee: float = TAKER_FEE):
    gross = (exit - entry) * qty if side=="Buy" else (entry - exit) * qty
    fee = (entry * qty + exit * qty) * taker_fee
    new_eq = get_wallet_equity() + gross - fee
    _log_equity(new_eq)
    return gross, fee, new_eq

# ===== Market helpers =====
def get_quote(symbol: str) -> Tuple[float,float,float]:
    r = mkt.get_tickers(category=CATEGORY, symbol=symbol)
    row = ((r.get("result") or {}).get("list") or [{}])[0]
    bid = float(row.get("bid1Price") or 0.0)
    ask = float(row.get("ask1Price") or 0.0)
    last = float(row.get("lastPrice") or 0.0)
    mid = (bid+ask)/2.0 if bid>0 and ask>0 else (last or bid or ask)
    return bid, ask, float(mid or 0.0)

def get_recent_1m(symbol: str, minutes: int = 200) -> Optional[pd.DataFrame]:
    r = mkt.get_kline(category=CATEGORY, symbol=symbol, interval=INTERVAL, limit=min(minutes,1000))
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

def recent_vol_bps(symbol: str, minutes: int = None) -> float:
    minutes = minutes or VOL_WIN
    df = get_recent_1m(symbol, minutes=minutes)
    if df is None or len(df) < 5:
        return 10.0
    v_bps = float(df["log_return"].abs().median() * 1e4)
    return max(v_bps, 1.0)

def fee_threshold(taker_fee=TAKER_FEE, slip_bps=SLIPPAGE_BPS_TAKER, safety=FEE_SAFETY):
    rt = 2.0*taker_fee + 2.0*(slip_bps/1e4)
    return math.log(1.0 + safety*rt)

def tp_from_bps(entry: float, bps: float, side: str) -> float:
    return entry*(1.0 + bps/10000.0) if side=="Buy" else entry*(1.0 - bps/10000.0)

def _bps_between(p1:float,p2:float)->float:
    return abs((p2 - p1) / max(p1, 1e-12)) * 10_000.0

def _entry_cost_bps()->float:
    return TAKER_FEE*1e4 + SLIPPAGE_BPS_TAKER

def _exit_cost_bps()->float:
    return TAKER_FEE*1e4 + SLIPPAGE_BPS_TAKER

def calc_timeout_minutes(symbol: str) -> int:
    if TIMEOUT_MODE == "fixed":
        return max(1, int(PRED_HORIZ_MIN))
    target_bps = TIMEOUT_TARGET_BPS if TIMEOUT_TARGET_BPS is not None else TP1_BPS
    v_bps = recent_vol_bps(symbol, VOL_WIN)
    need = math.ceil((target_bps / v_bps) * TIMEOUT_K)
    return int(min(max(need, TIMEOUT_MIN), TIMEOUT_MAX))

def _dynamic_abs_tp_usd(symbol:str, mid:float, qty:float)->float:
    if ABS_TP_USD > 0:  # 고정 ABS 설정 시 우선
        return ABS_TP_USD
    v_bps = recent_vol_bps(symbol, VOL_WIN)
    atr_usd = (v_bps/1e4) * mid * qty
    return max(ABS_TP_USD_FLOOR, ABS_K * atr_usd)

# ===== DeepAR Inference =====
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.data import GroupNormalizer
import torch.nn.functional as F
from pytorch_forecasting.metrics import NormalDistributionLoss

SEQ_LEN = int(os.getenv("SEQ_LEN","60"))
PRED_LEN = int(os.getenv("PRED_LEN","60"))
MODEL_CKPT = os.getenv("MODEL_CKPT","../multimodel/models/multi_deepar_best.ckpt")

class SignAwareNormalLoss(NormalDistributionLoss):
    def __init__(self, reduction="mean", lambda_sign=0.3, alpha_sign=7.0):
        super().__init__(reduction=reduction); self.lambda_sign=float(lambda_sign); self.alpha_sign=float(alpha_sign)
    def _unpack_target(self, target):
        if isinstance(target,(tuple,list)): y=target[0]; w=target[1] if len(target)>1 else None
        elif isinstance(target,dict): y=target.get("target",target.get("y",target)); w=target.get("weight",None)
        else: y=target; w=None
        return y,w
    def forward(self,y_pred,target):
        base=super().forward(y_pred,target); y,w=self._unpack_target(target); mu=self.to_prediction(y_pred)
        if mu.shape!=y.shape: y=y.view_as(mu)
        logits=self.alpha_sign*mu; y_pos=(y>0).float()
        bce=F.binary_cross_entropy_with_logits(logits,y_pos,reduction="none")
        if w is not None:
            if w.shape!=bce.shape: w=w.view_as(bce)
            denom=torch.clamp(w.sum(),min=1.0); bce=(bce*w).sum()/denom
        else: bce=bce.mean()
        return base + self.lambda_sign*bce

DEEPar = None
try:
    DEEPar = DeepAR.load_from_checkpoint(MODEL_CKPT, map_location="cpu").eval()
except Exception as e:
    if VERBOSE: print(f"[WARN] DeepAR load failed: {e}")

def ma_side(symbol: str, minutes: int = 180) -> Optional[str]:
    df = get_recent_1m(symbol, minutes=minutes)
    if df is None or len(df) < 60:
        return None
    c = df["close"].to_numpy()
    ma20 = np.mean(c[-20:])
    ma60 = np.mean(c[-60:])
    if ma20 > ma60: return "Buy"
    if ma20 < ma60: return "Sell"
    return None

def build_infer_dataset(df_sym: pd.DataFrame):
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
    base = None
    try:
        if DEEPar is not None:
            df = get_recent_1m(symbol, minutes=SEQ_LEN+PRED_LEN+10)
            if df is not None and len(df) >= (SEQ_LEN+1):
                tsd = build_infer_dataset(df)
                dl = tsd.to_dataloader(train=False, batch_size=1, num_workers=0)
                mu = float(DEEPar.predict(dl, mode="prediction")[0, 0])
                if VERBOSE: print(f"[DEBUG] {symbol} mu={mu:.6g} thr={fee_threshold():.6g}")
                if   mu >  fee_threshold(): base = "Buy"
                elif mu < -fee_threshold(): base = "Sell"
    except Exception as e:
        if VERBOSE: print(f"[PRED][ERR] {symbol} -> {e}")

    # 모델이 애매/실패면 MA 백업
    if base is None:
        base = ma_side(symbol)

    # 엔트리 모드 적용
    if ENTRY_MODE == "model":
        side = base
    elif ENTRY_MODE == "inverse":
        if   base == "Buy": side = "Sell"
        elif base == "Sell": side = "Buy"
        else: side = None
    elif ENTRY_MODE == "random":
        side = base or random.choice(["Buy","Sell"])   # 모델 None이면 랜덤 허용
    else:
        side = base

    conf = 1.0 if side is not None else 0.0
    return side, conf

def choose_entry(symbol: str):
    return deepar_direction(symbol)

# ===== Paper Broker =====
def _round_down(x: float, step: float) -> float:
    if step<=0: return x
    return math.floor(x/step)*step

class PaperBroker:
    def __init__(self):
        self.pos: Dict[str,dict] = {}  # symbol -> {side, entry, qty, opened}
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
        del self.pos[symbol]
        return pnl
    def positions(self): return self.pos

BROKER = PaperBroker()

# ===== Helpers =====
def unrealized_pnl_usd(side: str, qty: float, entry: float, mark: float) -> float:
    return (mark - entry) * qty if side=="Buy" else (entry - mark) * qty

# ===== 진입 시 필터 =====
def _entry_allowed_by_filters(symbol:str, side:str, entry_ref:float)->bool:
    v_bps = recent_vol_bps(symbol, VOL_WIN)
    if v_bps < V_BPS_FLOOR:
        if VERBOSE: print(f"[SKIP] {symbol} vol {v_bps:.1f}bps < floor {V_BPS_FLOOR}bps")
        return False
    tp1_px = tp_from_bps(entry_ref, TP1_BPS, side)
    edge_bps = _bps_between(entry_ref, tp1_px)
    need_bps = _entry_cost_bps() + _exit_cost_bps() + MIN_EXPECTED_ROI_BPS
    if edge_bps < need_bps:
        if VERBOSE: print(f"[SKIP] {symbol} ROI edge {edge_bps:.1f} < need {need_bps:.1f}bps")
        return False
    return True

# ===== Entry/Exit =====
def try_enter(symbol: str):
    global SESSION_TRADES
    # 세션 제한
    if SESSION_TRADES >= SESSION_MAX_TRADES: return
    if (time.time() - SESSION_START_TS) >= (SESSION_MAX_MINUTES*60): return
    if COOLDOWN_UNTIL.get(symbol,0.0) > time.time(): return

    if symbol in BROKER.positions():
        return
    side, conf = choose_entry(symbol)
    if side not in ("Buy", "Sell"):
        return

    # 연속신호 체크
    dq = ENTRY_HISTORY[symbol]
    dq.append(side)
    if len(dq) < N_CONSEC_SIGNALS or any(x != side for x in dq):
        return

    bid, ask, mid = get_quote(symbol)
    if mid <= 0:
        return

    # 품질 필터(ROI/변동성)
    entry_ref = float(ask if side=="Buy" else bid)
    if not _entry_allowed_by_filters(symbol, side, entry_ref):
        return

    ts = int(time.time())

    with ENTRY_LOCK:
        if symbol in BROKER.positions():
            return
        if not BROKER.try_open(symbol, side, mid, ts):
            return

        # 타임아웃
        horizon_min = max(1, int(calc_timeout_minutes(symbol)))
        POSITION_DEADLINE[symbol] = ts + horizon_min * 60

        # TP/SL & ABS_TP 동적
        tp1 = tp_from_bps(mid, TP1_BPS, side)
        tp2 = tp_from_bps(mid, TP2_BPS, side)
        tp3 = tp_from_bps(mid, TP3_BPS, side)
        sl  = mid*(1.0 - SL_ROI_PCT) if side=="Buy" else mid*(1.0 + SL_ROI_PCT)
        pos_qty = BROKER.positions()[symbol]["qty"]
        STATE[symbol] = {
            "tp1":tp1,"tp2":tp2,"tp3":tp3,"sl":sl,
            "tp1_filled":False,"tp2_filled":False,"be_moved":False,
            "abs_usd": _dynamic_abs_tp_usd(symbol, mid, pos_qty)
        }

        _log_trade({"ts":ts,"event":"ENTRY","symbol":symbol,"side":side,
                    "qty":f"{pos_qty:.8f}","entry":f"{mid:.6f}","exit":"",
                    "tp1":f"{tp1:.6f}","tp2":f"{tp2:.6f}","tp3":f"{tp3:.6f}",
                    "sl":f"{sl:.6f}","reason":"open","mode":"paper"})
        SESSION_TRADES += 1
        if VERBOSE:
            print(f"[OPEN] {symbol} {side} entry={mid:.4f} timeout={horizon_min}m "
                  f"abs≈{STATE[symbol]['abs_usd']:.2f}USD  [EQUITY]={get_wallet_equity():.2f}")

def close_and_log(symbol: str, reason: str):
    now = int(time.time())
    if symbol not in BROKER.positions(): return
    p = BROKER.positions()[symbol]
    side = p["side"]; entry = p["entry"]; qty = p["qty"]
    _,_,mark = get_quote(symbol)
    BROKER.try_close(symbol, now)
    gross, fee, new_eq = apply_trade_pnl(side, entry, mark, qty, taker_fee=TAKER_FEE)
    _log_trade({"ts": now, "event":"EXIT", "symbol":symbol, "side":"Sell" if side=="Buy" else "Buy",
                "qty":f"{qty:.8f}","entry":f"{entry:.6f}","exit":f"{mark:.6f}",
                "tp1":"","tp2":"","tp3":"","sl":"","reason":reason,"mode":"paper"})
    if VERBOSE:
        print(f"[{reason}] {symbol} gross={gross:.6f} fee={fee:.6f} eq={new_eq:.2f}")
    POSITION_DEADLINE.pop(symbol, None)
    STATE.pop(symbol, None)
    COOLDOWN_UNTIL[symbol] = time.time() + COOLDOWN_SEC

# ===== Monitor =====
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
                # 복구(이상 시)
                tp1 = tp_from_bps(entry, TP1_BPS, side)
                tp2 = tp_from_bps(entry, TP2_BPS, side)
                tp3 = tp_from_bps(entry, TP3_BPS, side)
                sl  = entry*(1.0 - SL_ROI_PCT) if side=="Buy" else entry*(1.0 + SL_ROI_PCT)
                st = {"tp1":tp1,"tp2":tp2,"tp3":tp3,"sl":sl,
                      "tp1_filled":False,"tp2_filled":False,"be_moved":False,
                      "abs_usd": _dynamic_abs_tp_usd(sym, entry, qty)}
                STATE[sym]=st

            # TP1 fill (분할청산 시뮬: 부분 PnL 반영 + 잔량 감소)
            if (not st["tp1_filled"]) and ((mark>=st["tp1"] and side=="Buy") or (mark<=st["tp1"] and side=="Sell")):
                part = qty*TP1_RATIO
                apply_trade_pnl(side, entry, st["tp1"], part, taker_fee=TAKER_FEE)
                p["qty"] = qty = max(0.0, qty - part)
                st["tp1_filled"]=True
                _log_trade({"ts":int(time.time()),"event":"TP1_FILL","symbol":sym,"side":side,
                            "qty":f"{part:.8f}","entry":f"{entry:.6f}","exit":f"{st['tp1']:.6f}",
                            "tp1":"","tp2":"","tp3":"","sl":"","reason":"","mode":"paper"})

            # TP2 fill
            if qty>0 and (not st["tp2_filled"]) and ((mark>=st["tp2"] and side=="Buy") or (mark<=st["tp2"] and side=="Sell")):
                # 남은 잔량 기준 TP2_RATIO 만큼
                remain_ratio = 1.0 - TP1_RATIO
                part = min(qty, remain_ratio*TP2_RATIO)
                apply_trade_pnl(side, entry, st["tp2"], part, taker_fee=TAKER_FEE)
                p["qty"] = qty = max(0.0, qty - part)
                st["tp2_filled"]=True
                _log_trade({"ts":int(time.time()),"event":"TP2_FILL","symbol":sym,"side":side,
                            "qty":f"{part:.8f}","entry":f"{entry:.6f}","exit":f"{st['tp2']:.6f}",
                            "tp1":"","tp2":"","tp3":"","sl":"","reason":"","mode":"paper"})

            # BE 이동
            if st["tp2_filled"] and not st["be_moved"]:
                be_px = entry*(1.0 + BE_EPS_BPS/10000.0) if side=="Buy" else entry*(1.0 - BE_EPS_BPS/10000.0)
                st["sl"] = be_px
                st["be_moved"]=True
                _log_trade({"ts":int(time.time()),"event":"MOVE_SL","symbol":sym,"side":side,
                            "qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"",
                            "sl":f"{be_px:.6f}","reason":"BE","mode":"paper"})

            # 트레일링
            if (st["tp1_filled"] if TRAIL_AFTER_TIER==1 else st["tp2_filled"]):
                if side=="Buy":
                    st["sl"] = max(st["sl"], mark*(1.0 - TRAIL_BPS/10000.0))
                else:
                    st["sl"] = min(st["sl"], mark*(1.0 + TRAIL_BPS/10000.0))

            # 종료 조건(ABS/SL/Timeout) — 잔량 전량 종료
            if qty > 0:
                pnl = unrealized_pnl_usd(side, qty, entry, mark)
                abs_hit = pnl >= st["abs_usd"]
                sl_hit  = (mark <= st["sl"] and side=="Buy") or (mark >= st["sl"] and side=="Sell")
                ddl = POSITION_DEADLINE.get(sym)
                to_hit = (ddl and now >= ddl)
                if abs_hit or sl_hit or to_hit:
                    reason = "ABS_TP" if abs_hit else ("SL" if sl_hit else "HORIZON_TIMEOUT")
                    close_and_log(sym, reason)
                    continue

        loops += 1
        time.sleep(poll_sec)

# ===== Main =====
def main():
    print(f"[START] PAPER EXTENDED | MODE={ENTRY_MODE} TESTNET={TESTNET}")
    _log_equity(get_wallet_equity())
    while True:
        # 스캔
        if SCAN_PARALLEL:
            with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
                ex.map(try_enter, SYMBOLS)
        else:
            for s in SYMBOLS:
                try_enter(s)
        # 모니터 (짧게 여러번)
        monitor_loop(1.0, 2)
        time.sleep(0.5)

if __name__ == "__main__":
    main()