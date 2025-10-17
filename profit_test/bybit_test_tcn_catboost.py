# -*- coding: utf-8 -*-
"""
bybit_test_tcn_catboost.py  (hedge-mode patched)
- PAPER trading loop using TCN(+optional CatBoost) with Bybit public API
- Hedge mode: long/short per symbol are independent ((symbol, side) keys)
- Honors PRESET env strictly (OFF/A/B). No hidden preset re-application.
- Patches kept:
  * OPEN gate: min TP(bps) >= fee+slip+spread+safety
  * EXIT label: ABS_TP -> TAKE_PROFIT if net>0 else ABS_EXIT
  * EQUITY_MTM logging suppression
"""
import os, time, math, csv, random, threading, collections, logging, warnings, json
from typing import Optional, Dict, Tuple, List
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import requests, certifi

# ===== TLS / logging noise
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
warnings.filterwarnings("ignore", module="lightning")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ===== env helpers
def env_bool(name: str, default: bool=False) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in ("1","y","yes","true","on")

def env_int(name: str, default: int=0) -> int:
    v = os.getenv(name)
    try:    return int(str(v).strip()) if v and str(v).strip()!="" else default
    except: return default

def env_float(name: str, default: float=0.0) -> float:
    v = os.getenv(name)
    try:    return float(str(v).strip()) if v and str(v).strip()!="" else default
    except: return default

def env_str(name: str, default: str="") -> str:
    v = os.getenv(name)
    return str(v) if v and str(v).strip()!="" else default

def env_list(name: str, default_csv: str) -> List[str]:
    raw = os.getenv(name, default_csv)
    return [s.strip() for s in raw.split(",") if s.strip()]

VERBOSE    = env_bool("VERBOSE", True)

# turn CatBoost off by default
USE_CATBOOST = env_bool("USE_CATBOOST", False)
if USE_CATBOOST:
    try:
        from catboost import CatBoostClassifier, Pool
        CATBOOST_CBM  = os.getenv("CATBOOST_CBM",  "../multimodel/models/catboost_meta.cbm")
        CATBOOST_META = os.getenv("CATBOOST_META", "../multimodel/models/catboost_meta_features.json")
        CB_MODEL = CatBoostClassifier()
        CB_MODEL.load_model(CATBOOST_CBM)
        with open(CATBOOST_META, "r", encoding="utf-8") as f:
            CB_META = json.load(f)
    except Exception as e:
        CB_MODEL, CB_META = None, {}
        print(f"[WARN] CatBoost load failed: {e}")
# ===== alias support
_INITIAL_ENV_KEYS = set(os.environ.keys())
_ALIAS_TO_CANON = {
    "SPREAD_MAX_BPS": "SPREAD_BPS_MAX",
    "SPREAD_MAX": "SPREAD_BPS_MAX",
    "SPREAD_BPS_CEIL": "SPREAD_BPS_MAX",
    "ROI_BASE_MIN": "ROI_BASE_MIN_EXPECTED_BPS",
    "ROI_BASE_MIN_BPS": "ROI_BASE_MIN_EXPECTED_BPS",
    "ROI_MIN_BPS": "MIN_EXPECTED_ROI_BPS",
    "V_FLOOR": "V_BPS_FLOOR",
    "V_FLOOR_BPS": "V_BPS_FLOOR",
}
_CANON_TO_ALIASES = {}
for a, c in _ALIAS_TO_CANON.items():
    _CANON_TO_ALIASES.setdefault(c, []).append(a)

def _resolve_env_aliases():
    for alias, canon in _ALIAS_TO_CANON.items():
        if alias in os.environ and canon not in os.environ:
            os.environ[canon] = os.environ[alias]

def _user_overrode(canon: str) -> bool:
    if canon in _INITIAL_ENV_KEYS: return True
    for a in _CANON_TO_ALIASES.get(canon, []):
        if a in _INITIAL_ENV_KEYS: return True
    return False

def _print_config_summary():
    print(
        "[CONFIG] "
        f"PRESET={PRESET} | ENTRY_MODE={ENTRY_MODE} | "
        f"SPREAD_BPS_MAX={SPREAD_BPS_MAX} | "
        f"ROI_BASE_MIN_EXPECTED_BPS={ROI_BASE_MIN_EXPECTED_BPS} | "
        f"MIN_EXPECTED_ROI_BPS={MIN_EXPECTED_ROI_BPS} | "
        f"V_BPS_FLOOR={V_BPS_FLOOR} | "
        f"CB_THR={CB_THR} LOGIT_PROB_THR={LOGIT_PROB_THR} | "
        f"FAST_BE_BPS={FAST_BE_BPS} DECAY_GRACE_SEC={DECAY_GRACE_SEC} DECAY_PCT={DECAY_PCT} | "
        f"SAFETY_BPS={SAFETY_BPS} MTM_EPS={MTM_EPS} MTM_MIN_INTERVAL_SEC={MTM_MIN_INTERVAL_SEC}"
    )

_resolve_env_aliases()

# ===== BASE CONFIG
CATEGORY   = env_str("CATEGORY", "linear")
INTERVAL   = env_str("INTERVAL", "1")
SYMBOLS    = env_list("SYMBOLS", "BTCUSDT,ETHUSDT")
DIAG       = env_bool("DIAG", False)
TRACE      = env_bool("TRACE", True)
VERBOSE    = env_bool("VERBOSE", True)

ENTRY_MODE     = env_str("ENTRY_MODE", env_str("MODEL_MODE", "model"))
BYBIT_TESTNET  = env_bool("BYBIT_TESTNET", False)
TESTNET        = BYBIT_TESTNET

START_EQUITY         = env_float("START_EQUITY", 10000.0)
LEVERAGE             = env_float("LEVERAGE", 25.0)
ENTRY_EQUITY_PCT     = env_float("ENTRY_EQUITY_PCT", 0.5)
TAKER_FEE            = env_float("TAKER_FEE", 0.0006)
SLIPPAGE_BPS_TAKER   = env_float("SLIPPAGE_BPS_TAKER", 0.5)
FEE_SAFETY           = env_float("FEE_SAFETY", 1.2)

SAFETY_BPS           = env_float("SAFETY_BPS", 0.0)

MTM_EPS              = env_float("MTM_EPS", 0.01)
MTM_MIN_INTERVAL_SEC = env_float("MTM_MIN_INTERVAL_SEC", 5.0)

TP1_BPS       = env_float("TP1_BPS", 50.0)
TP2_BPS       = env_float("TP2_BPS", 80.0)
TP3_BPS       = env_float("TP3_BPS", 150.0)
SL_ROI_PCT    = env_float("SL_ROI_PCT", 0.015)
TP1_RATIO     = env_float("TP1_RATIO", 0.2)
TP2_RATIO     = env_float("TP2_RATIO", 0.3)
BE_EPS_BPS    = env_float("BE_EPS_BPS", 2.0)
TRAIL_BPS     = env_float("TRAIL_BPS", 50.0)
TRAIL_AFTER_TIER = env_int("TRAIL_AFTER_TIER", 2)

ABS_TP_USD        = env_float("ABS_TP_USD", 0.0)
ABS_K             = env_float("ABS_K", 1.0)
ABS_TP_USD_FLOOR  = env_float("ABS_TP_USD_FLOOR", 1.0)

N_CONSEC_SIGNALS  = env_int("N_CONSEC_SIGNALS", 1)
COOLDOWN_SEC      = env_int("COOLDOWN_SEC", 0)
SESSION_MAX_TRADES= env_int("SESSION_MAX_TRADES", 1000)
SESSION_MAX_MINUTES=env_int("SESSION_MAX_MINUTES", 1)

SCAN_PARALLEL     = env_bool("SCAN_PARALLEL", True)
SCAN_WORKERS      = env_int("SCAN_WORKERS", 8)

TRADES_CSV      = env_str("TRADES_CSV", f"paper_trades_{ENTRY_MODE}_{START_EQUITY}_{LEVERAGE}.csv")
EQUITY_CSV      = env_str("EQUITY_CSV", f"paper_equity_{ENTRY_MODE}_{START_EQUITY}_{LEVERAGE}.csv")
EQUITY_MTM_CSV  = env_str("EQUITY_MTM_CSV", f"paper_equity_mtm_{ENTRY_MODE}_{START_EQUITY}_{LEVERAGE}.csv")
TRADES_FIELDS   = ["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","reason","mode"]
EQUITY_FIELDS   = ["ts","equity"]

# ===== META/DECAY/TIMEOUT
META_ON          = env_int("META_ON", 0)
META_WIN         = env_int("META_WIN", 30)
META_MIN_HIT     = env_float("META_MIN_HIT", 0.4)
META_MIN_EVR     = env_float("META_MIN_EVR", 0.0)
META_MAX_DD_BPS  = env_float("META_MAX_DD_BPS", -300.0)
META_COOLDOWN    = env_int("META_COOLDOWN", 0)

DECAY_GRACE_SEC  = env_int("DECAY_GRACE_SEC", 60)
DECAY_PCT        = env_float("DECAY_PCT", 0.005)
MIN_LOCK_SEC     = env_int("MIN_LOCK_SEC", 3)
FAST_BE_BPS      = env_float("FAST_BE_BPS", 10.0)

TIMEOUT_MODE      = env_str("TIMEOUT_MODE", "fixed")   # fixed|atr|model
VOL_WIN           = env_int("VOL_WIN", 60)
TIMEOUT_K         = env_float("TIMEOUT_K", 1.5)
TIMEOUT_MIN       = env_int("TIMEOUT_MIN", 1)
TIMEOUT_MAX       = env_int("TIMEOUT_MAX", 2)
PRED_HORIZ_MIN    = env_int("PRED_HORIZ_MIN", 1)
_v = os.getenv("TIMEOUT_TARGET_BPS")
TIMEOUT_TARGET_BPS = float(_v) if (_v is not None and str(_v).strip()!="") else None

# ===== MODEL THRESHOLDS / ROI FILTERS
CB_Q_WIN   = env_int("CB_Q_WIN", 200)
CB_Q_PERC  = env_float("CB_Q_PERC", 0.70)
MIN_CB_THR = env_float("MIN_CB_THR", 0.55)
CB_THR          = env_float("CB_THR", 0.48)
LOGIT_PROB_THR  = env_float("LOGIT_PROB_THR", 0.48)

MIN_EXPECTED_ROI_BPS      = env_float("MIN_EXPECTED_ROI_BPS", 2.0)
ROI_BASE_MIN_EXPECTED_BPS = env_float("ROI_BASE_MIN_EXPECTED_BPS", MIN_EXPECTED_ROI_BPS)
ROI_LOWVOL_BOOST_BPS      = env_float("ROI_LOWVOL_BOOST_BPS", 0.0)
ROI_POORMETA_BOOST_BPS    = env_float("ROI_POORMETA_BOOST_BPS", 0.0)
V_BPS_FLOOR               = env_float("V_BPS_FLOOR", 8.0)

MAX_DD_BPS_DAY    = env_float("MAX_DD_BPS_DAY", -800.0)
MAX_CONSEC_LOSS   = env_int("MAX_CONSEC_LOSS", 5)
RISK_COOLDOWN_SEC = env_int("RISK_COOLDOWN_SEC", 0)
HOLD_UNTIL_TIMEOUT = env_bool("HOLD_UNTIL_TIMEOUT", True)

SPREAD_BPS_MAX  = env_float("SPREAD_BPS_MAX", 0.0)

# ===== SESSION RESTART (kept)
SESSION_WARMUP_SEC      = env_int("SESSION_WARMUP_SEC", 10)
SESSION_TARGET_PNL_BPS  = env_float("SESSION_TARGET_PNL_BPS", +100000.0)
SESSION_MAX_DD_BPS      = env_float("SESSION_MAX_DD_BPS", -1500.0)
SESSION_HARD_MINUTES    = env_float("SESSION_HARD_MINUTES", 180.0)
SESSION_COOLDOWN_SEC    = env_int("SESSION_COOLDOWN_SEC", 0)
SESSION_BANK_PROFITS    = env_bool("SESSION_BANK_PROFITS", False)
SESSION_START_EQUITY    = env_float("SESSION_START_EQUITY", START_EQUITY)

# ===== RUNTIME STATE  (hedge mode uses (symbol, side) keys)
_equity_cache = START_EQUITY
POSITION_DEADLINE: Dict[Tuple[str,str], int] = {}
STATE: Dict[Tuple[str,str], dict] = {}
ENTRY_LOCK = threading.Lock()
ENTRY_HISTORY: Dict[str,collections.deque] = {s: collections.deque(maxlen=max(1,N_CONSEC_SIGNALS)) for s in SYMBOLS}
COOLDOWN_UNTIL: Dict[str,float] = {}
SESSION_START_TS = time.time()
SESSION_TRADES = 0

META_TRADES: Dict[str, deque] = {s: deque(maxlen=max(5, META_WIN)) for s in SYMBOLS}
META_BLOCK_UNTIL: Dict[str, float] = {}

CB_CONF_BUF: Dict[str, deque] = {s: deque(maxlen=max(50, CB_Q_WIN)) for s in SYMBOLS}
RISK_CONSEC_LOSS = 0
RISK_DAY_CUM_BPS = 0.0
RISK_LAST_RESET_DAY = time.strftime("%Y-%m-%d")

_mtm_last_val: Optional[float] = None
_mtm_last_ts: float = 0.0
CB_MODEL, CB_META = None, {}

# ===== PRESETS
def apply_scalping_preset(preset: str = "A"):
    global TP1_BPS, TP2_BPS, TP3_BPS, SL_ROI_PCT
    global V_BPS_FLOOR, N_CONSEC_SIGNALS
    global CB_Q_PERC, MIN_CB_THR, FAST_BE_BPS, TRAIL_BPS, TRAIL_AFTER_TIER
    global TIMEOUT_MODE, TIMEOUT_MIN, TIMEOUT_MAX, PRED_HORIZ_MIN
    global SPREAD_BPS_MAX, ENTRY_EQUITY_PCT
    global ROI_BASE_MIN_EXPECTED_BPS, ROI_LOWVOL_BOOST_BPS

    def set_if_user_not_overrode(name: str, value):
        if not _user_overrode(name):
            globals()[name] = value

    p = (preset or "").upper()
    if p == "OFF":
        print("[PRESET] OFF (env overrides only)")
        return

    if p == "A":
        set_if_user_not_overrode("TP1_BPS", 15.0); set_if_user_not_overrode("TP2_BPS", 22.0); set_if_user_not_overrode("TP3_BPS", 40.0)
        set_if_user_not_overrode("SL_ROI_PCT", 0.012)
        set_if_user_not_overrode("V_BPS_FLOOR", 10.0)
        set_if_user_not_overrode("ROI_BASE_MIN_EXPECTED_BPS", 5.0)
        set_if_user_not_overrode("ROI_LOWVOL_BOOST_BPS", 3.0)
        set_if_user_not_overrode("N_CONSEC_SIGNALS", 1)
        set_if_user_not_overrode("CB_Q_PERC", 0.75); set_if_user_not_overrode("MIN_CB_THR", 0.50)
        set_if_user_not_overrode("FAST_BE_BPS", 10.0); set_if_user_not_overrode("TRAIL_BPS", 20.0); set_if_user_not_overrode("TRAIL_AFTER_TIER", 1)
        set_if_user_not_overrode("TIMEOUT_MODE", "model"); set_if_user_not_overrode("TIMEOUT_MIN", 1); set_if_user_not_overrode("TIMEOUT_MAX", 2); set_if_user_not_overrode("PRED_HORIZ_MIN", 1)
        set_if_user_not_overrode("SPREAD_BPS_MAX", 6.0)
        if not _user_overrode("ENTRY_EQUITY_PCT"):
            globals()["ENTRY_EQUITY_PCT"] = min(ENTRY_EQUITY_PCT, 0.10)
    else:
        set_if_user_not_overrode("TP1_BPS", 20.0); set_if_user_not_overrode("TP2_BPS", 30.0); set_if_user_not_overrode("TP3_BPS", 60.0)
        set_if_user_not_overrode("SL_ROI_PCT", 0.015)
        set_if_user_not_overrode("V_BPS_FLOOR", 3.0)
        set_if_user_not_overrode("ROI_BASE_MIN_EXPECTED_BPS", 10.0)
        set_if_user_not_overrode("ROI_LOWVOL_BOOST_BPS", 2.0)
        set_if_user_not_overrode("N_CONSEC_SIGNALS", 1)
        set_if_user_not_overrode("CB_Q_PERC", 0.70); set_if_user_not_overrode("MIN_CB_THR", 0.50)
        set_if_user_not_overrode("FAST_BE_BPS", 12.0); set_if_user_not_overrode("TRAIL_BPS", 25.0); set_if_user_not_overrode("TRAIL_AFTER_TIER", 1)
        set_if_user_not_overrode("TIMEOUT_MODE", "model"); set_if_user_not_overrode("TIMEOUT_MIN", 1); set_if_user_not_overrode("TIMEOUT_MAX", 2); set_if_user_not_overrode("PRED_HORIZ_MIN", 1)
        set_if_user_not_overrode("SPREAD_BPS_MAX", 8.0)
        if not _user_overrode("ENTRY_EQUITY_PCT"):
            globals()["ENTRY_EQUITY_PCT"] = min(ENTRY_EQUITY_PCT, 0.15)

    print(
        f"[PRESET] scalping preset='{p}' "
        f"| TP1={TP1_BPS} TP2={TP2_BPS} TP3={TP3_BPS} "
        f"| V_FLOOR={V_BPS_FLOOR} ROI_BASE_MIN={ROI_BASE_MIN_EXPECTED_BPS} "
        f"| CB(q={CB_Q_PERC:.2f}, min={MIN_CB_THR:.2f}) "
        f"| TO(min={TIMEOUT_MIN}, max={TIMEOUT_MAX}) "
        f"| SPREAD≤{SPREAD_BPS_MAX}bps | N_CONSEC={N_CONSEC_SIGNALS} "
        f"| ENTRY_EQUITY_PCT={ENTRY_EQUITY_PCT}"
    )

# ===== read & apply PRESET =====
PRESET = env_str("PRESET", "A")
#apply_scalping_preset(PRESET)
_print_config_summary()

WHY = os.getenv("WHY","0").lower() in ("1","y","yes","true")

def why_log(symbol, side, p_tcn, cb_conf, vol_bps, spread_bps, edge_bps, need_bps,
            consec_ok, cooldown_ts, meta_ts):
    if not WHY: return
    cd  = max(0.0, cooldown_ts - time.time())
    mb  = max(0.0, meta_ts - time.time())
    print(
        f"[WHY][{symbol}] side={side} tcn={p_tcn:.3f} cb={cb_conf:.3f} "
        f"vol={vol_bps:.1f}/{V_BPS_FLOOR} spd={spread_bps:.1f}/{SPREAD_BPS_MAX} "
        f"roi_edge={edge_bps:.1f} need={need_bps:.1f} "
        f"consec={'OK' if consec_ok else 'WAIT'} cooldown={cd:.0f}s meta={mb:.0f}s"
    )

def reapply_env_overrides_after_preset():
    global TP1_BPS, TP2_BPS, TP3_BPS, SL_ROI_PCT
    global V_BPS_FLOOR, N_CONSEC_SIGNALS
    global CB_Q_PERC, MIN_CB_THR, FAST_BE_BPS, TRAIL_BPS, TRAIL_AFTER_TIER
    global TIMEOUT_MODE, TIMEOUT_MIN, TIMEOUT_MAX, PRED_HORIZ_MIN
    global SPREAD_BPS_MAX, ENTRY_EQUITY_PCT
    global ROI_BASE_MIN_EXPECTED_BPS, ROI_LOWVOL_BOOST_BPS

    SPREAD_BPS_MAX = env_float("SPREAD_BPS_CEIL", env_float("SPREAD_BPS_MAX", SPREAD_BPS_MAX))
    TP1_BPS   = env_float("TP1_BPS", TP1_BPS)
    TP2_BPS   = env_float("TP2_BPS", TP2_BPS)
    TP3_BPS   = env_float("TP3_BPS", TP3_BPS)
    SL_ROI_PCT= env_float("SL_ROI_PCT", SL_ROI_PCT)
    V_BPS_FLOOR  = env_float("V_BPS_FLOOR", V_BPS_FLOOR)
    ROI_BASE_MIN_EXPECTED_BPS = env_float("MIN_EXPECTED_ROI_BPS", ROI_BASE_MIN_EXPECTED_BPS)
    ROI_LOWVOL_BOOST_BPS      = env_float("ROI_LOWVOL_BOOST_BPS", ROI_LOWVOL_BOOST_BPS)
    N_CONSEC_SIGNALS = env_int("N_CONSEC_SIGNALS", N_CONSEC_SIGNALS)
    CB_Q_PERC   = env_float("CB_Q_PERC", CB_Q_PERC)
    MIN_CB_THR  = env_float("MIN_CB_THR", MIN_CB_THR)
    FAST_BE_BPS = env_float("FAST_BE_BPS", FAST_BE_BPS)
    TRAIL_BPS   = env_float("TRAIL_BPS", TRAIL_BPS)
    TRAIL_AFTER_TIER = env_int("TRAIL_AFTER_TIER", TRAIL_AFTER_TIER)
    TIMEOUT_MODE   = env_str("TIMEOUT_MODE", TIMEOUT_MODE)
    TIMEOUT_MIN    = env_int("TIMEOUT_MIN", TIMEOUT_MIN)
    TIMEOUT_MAX    = env_int("TIMEOUT_MAX", TIMEOUT_MAX)
    PRED_HORIZ_MIN = env_int("PRED_HORIZ_MIN", PRED_HORIZ_MIN)
    ENTRY_EQUITY_PCT = env_float("ENTRY_EQUITY_PCT", ENTRY_EQUITY_PCT)

_PRESET_APPLIED = False
def maybe_apply_preset():
    global _PRESET_APPLIED
    if _PRESET_APPLIED: return
    _p = os.getenv("PRESET", "OFF").upper()
    apply_scalping_preset(_p)
    reapply_env_overrides_after_preset()
    _PRESET_APPLIED = True

def print_runtime_header():
    print(f"[START] PAPER TCN+CatBoost | TESTNET={BYBIT_TESTNET} | PUBLIC_API=bybit v5 | HEDGE=ON")
    print(f"[EQUITY] {START_EQUITY:.2f}")
    print(
        f"[CONFIG] MODE={ENTRY_MODE} | LEV={LEVERAGE}x | ENTRY_PCT={ENTRY_EQUITY_PCT:.2f} "
        f"| TP1/2/3={TP1_BPS}/{TP2_BPS}/{TP3_BPS}bps | SL_ROI={SL_ROI_PCT*100:.2f}% "
        f"| SPREAD≤{SPREAD_BPS_MAX}bps | ROI_MIN={ROI_BASE_MIN_EXPECTED_BPS}bps "
        f"| V_FLOOR={V_BPS_FLOOR}bps | CB(q={CB_Q_PERC:.2f}, min={MIN_CB_THR:.2f}) "
        f"| TIMEOUT({TIMEOUT_MODE}:{TIMEOUT_MIN}-{TIMEOUT_MAX}, H={PRED_HORIZ_MIN}) "
        f"| SAFETY_BPS={SAFETY_BPS}"
    )

# ===== MARKET (Bybit public)
class BybitPublic:
    def __init__(self, testnet: bool = False, timeout: int = 10):
        self.base = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self.s = requests.Session()
        self.s.verify = certifi.where()
        self.timeout = timeout
        self.headers = {"User-Agent": "bybit-public-client/1.0", "Accept": "application/json"}

    def _get(self, path: str, params: dict, retries: int = 3, backoff: float = 0.4):
        url = self.base + path
        for i in range(retries):
            try:
                r = self.s.get(url, params=params, headers=self.headers, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(backoff * (i+1)); continue
                return r.json()
            except Exception:
                time.sleep(backoff * (i+1))
        return {}

    def get_tickers(self, category: str, symbol: str):
        return self._get("/v5/market/tickers", {"category": category, "symbol": symbol})

    def get_kline_primary(self, category: str, symbol: str, interval: str, limit: int):
        return self._get("/v5/market/kline", {"category": category, "symbol": symbol, "interval": interval, "limit": min(int(limit),1000)})

    def get_kline_mark(self, category: str, symbol: str, interval: str, limit: int):
        return self._get("/v5/market/mark-price-kline", {"category": category, "symbol": symbol, "interval": interval, "limit": min(int(limit),1000)})

mkt = BybitPublic(testnet=TESTNET, timeout=10)

BYBIT_MAIN = "https://api.bybit.com"
BYBIT_TEST = "https://api-testnet.bybit.com"
USE_TESTNET = TESTNET

def _bybit_kline(host, symbol, minutes):
    limit = min(int(minutes), 500)
    qs = {"category":"linear","symbol":symbol,"interval":"1","limit":str(limit)}
    r = requests.get(f"{host}/v5/market/kline", params=qs, timeout=5)
    data = r.json()
    lst = (((data.get("result") or {}).get("list")) or [])
    return lst, data

def get_recent_1m(symbol: str, minutes: int = 300):
    host = BYBIT_TEST if USE_TESTNET else BYBIT_MAIN
    lst, raw = _bybit_kline(host, symbol, minutes)
    if not lst and host != BYBIT_MAIN:
        lst, raw = _bybit_kline(BYBIT_MAIN, symbol, minutes)
    if not lst:
        print(f"[BYBIT KLINE EMPTY] sym={symbol} host={host} raw={raw}")
        return None
    rows = lst[::-1]
    df = pd.DataFrame([{
        "timestamp": pd.to_datetime(int(z[0]), unit="ms", utc=True),
        "open": float(z[1]), "high": float(z[2]), "low": float(z[3]),
        "close": float(z[4]), "volume": float(z[5]), "turnover": float(z[6]),
    } for z in rows])
    return df

def get_quote(symbol: str) -> Tuple[float,float,float]:
    r = mkt.get_tickers(category=CATEGORY, symbol=symbol)
    row = ((r.get("result") or {}).get("list") or [{}])[0]
    bid = float(row.get("bid1Price") or 0.0)
    ask = float(row.get("ask1Price") or 0.0)
    last = float(row.get("lastPrice") or 0.0)
    mid = (bid+ask)/2.0 if bid>0 and ask>0 else (last or bid or ask)
    return bid, ask, float(mid or 0.0)

# ===== utils
def tp_from_bps(entry: float, bps: float, side: str) -> float:
    return entry*(1.0 + bps/10000.0) if side=="Buy" else entry*(1.0 - bps/10000.0)

def _bps_between(p1:float,p2:float)->float:
    return abs((p2 - p1) / max(p1, 1e-12)) * 10_000.0

def _entry_cost_bps()->float:
    return TAKER_FEE*1e4 + SLIPPAGE_BPS_TAKER

def _exit_cost_bps()->float:
    return TAKER_FEE*1e4 + SLIPPAGE_BPS_TAKER

def _min_tp_required_bps(spread_bps: float) -> float:
    return 2.0*(_entry_cost_bps()) + float(spread_bps) + float(SAFETY_BPS)

def recent_vol_bps(symbol: str, minutes: int = None) -> float:
    minutes = minutes or VOL_WIN
    df = get_recent_1m(symbol, minutes=minutes)
    if df is None or len(df) < 5:
        return 10.0
    logret = np.log(df["close"]).diff().abs()
    v_bps = float(logret.median() * 1e4)
    return max(v_bps, 1.0)

def fee_threshold(taker_fee=TAKER_FEE, slip_bps=SLIPPAGE_BPS_TAKER, safety=FEE_SAFETY):
    rt = 2.0*taker_fee + 2.0*(slip_bps/1e4)
    return math.log(1.0 + safety*rt)

def _log_thr_from_bps(target_bps: float) -> float:
    return math.log1p(target_bps / 1e4)

# ===== TCN
# ===== TCN (load patch)
TCN_CKPT = os.getenv("TCN_CKPT", "../multimodel/models_daily/daily_limit_tp_ep035.ckpt")
TCN_FEATS = ["ret","rv","mom","vz"]
TCN_SEQ_LEN_FALLBACK = 240
TCN_HAS_MU = False
TCN_THR_PROB = float(os.getenv("TCN_THR_PROB","0.48"))
TCN_MODEL=None; TCN_CFG={}; TCN_MU=None; TCN_SD=None; TCN_SEQ_LEN=TCN_SEQ_LEN_FALLBACK; TCN_HEAD="reg"

def _strip_prefix(state, prefixes=("model.", "net.", "module.")):
    out={}
    for k,v in state.items():
        nk=k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk]=v
    return out

try:
    ck = torch.load(TCN_CKPT, map_location="cpu")
    # Lightning(.ckpt) 호환
    raw_state = ck.get("model", None)
    if raw_state is None:
        raw_state = ck.get("state_dict", None)
    if raw_state is None and isinstance(ck, dict) and all(isinstance(k, str) for k in ck.keys()):
        raw_state = ck  # 순수 state dict

    if raw_state is None or not isinstance(raw_state, dict) or len(raw_state)==0:
        raise RuntimeError(f"bad checkpoint format keys={list(ck.keys()) if isinstance(ck,dict) else type(ck)}")

    state = _strip_prefix(raw_state)

    # cfg/스케일러 추출
    TCN_CFG = ck.get("cfg", {})
    if "FEATS" in TCN_CFG: TCN_FEATS = TCN_CFG["FEATS"]
    TCN_SEQ_LEN = int(TCN_CFG.get("SEQ_LEN", TCN_SEQ_LEN_FALLBACK))
    TCN_MU = ck.get("scaler_mu", ck.get("mu", None))
    TCN_SD = ck.get("scaler_sd", ck.get("sd", None))

    # 교체
    # 교체
    cand = [k for k in state.keys()
            if k.endswith("tcn.0.conv1.weight_v") or k.endswith("tcn.0.conv1.weight")
            or k.endswith("tcn.0.c1.weight_v") or k.endswith("tcn.0.c1.weight")]
    if not cand:
        # 가장 일반화된 백업: tcn.0.* 에서 weight 포함
        cand = [k for k in state.keys() if k.startswith("tcn.0.") and ("weight" in k)]
    if not cand:
        raise RuntimeError(f"no TCN first conv weight key found; sample={list(state.keys())[:6]}")

    k0 = cand[0]
    w0 = state[k0]
    hidden = w0.shape[0];
    in_feat = w0.shape[1];
    k_size = w0.shape[2]
    levels = 1 + max(int(k.split('.')[1]) for k in state.keys()
                     if k.startswith("tcn.") and (".conv1." in k or ".c1." in k))

    head_w = state.get("head.weight")
    head_out = head_w.shape[0] if head_w is not None else 2

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
        def __init__(self, in_feat, hidden=128, levels=6, k=3, dropout=0.1, head_out=2):
            super().__init__()
            self.head_out = head_out
            layers=[]; ch_in=in_feat
            for i in range(levels):
                layers += [TemporalBlock(ch_in, hidden, k, 2**i, dropout)]
                ch_in = hidden
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Linear(hidden, head_out)
        def forward(self, x):
            x = x.transpose(1,2)
            h = self.tcn(x)[:, :, -1]
            o = self.head(h)
            if self.head_out == 1:   # cls
                return None, o.squeeze(-1)
            else:                    # reg/joint
                return o[:,0], o[:,1]

    TCN_MODEL = TCNBinary(in_feat=in_feat, hidden=hidden, levels=levels, k=k_size, dropout=0.1, head_out=head_out).eval()
    TCN_MODEL.load_state_dict(state, strict=False)  # 일부 키 차이 허용
    TCN_HEAD = "cls" if TCN_MODEL.head.out_features == 1 else "reg"

    print(f"[TCN] loaded: feats={len(TCN_FEATS)} seq={TCN_SEQ_LEN} head={TCN_HEAD} in={in_feat} hid={hidden} L={levels} k={k_size}")
    # in_feat와 피처 수 불일치 시 13채널 기본 세트로 강제
    if 'in_feat' in locals() and in_feat == 13:
        TCN_FEATS = ["ret",
                     "rv5", "rv15", "rv30", "rv60",
                     "mom5", "mom15", "mom30", "mom60",
                     "vz20", "vz60",
                     "atr14",
                     "bias1"]

except Exception as e:
    print(f"[WARN] TCN load failed: {e}")
    TCN_MODEL=None

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
        out = self.drop2(self.relu2(self.chomp2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNBinary(nn.Module):
    def __init__(self, in_feat, hidden=128, levels=6, k=3, dropout=0.1, head_out=2):
        super().__init__()
        self.head_out = head_out
        layers=[]; ch_in=in_feat
        for i in range(levels):
            layers += [TemporalBlock(ch_in, hidden, k, 2**i, dropout)]
            ch_in = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, head_out)

    def forward(self, x):
        x = x.transpose(1,2)
        h = self.tcn(x)[:, :, -1]
        o = self.head(h)
        if self.head_out == 1:   # cls
            return None, o.squeeze(-1)
        else:                    # joint/reg
            return o[:,0], o[:,1]

def _build_tcn_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.sort_values("timestamp").copy()
    c = g["close"].astype(float).values
    h = g["high"].astype(float).values
    l = g["low"].astype(float).values
    v = g["volume"].astype(float).values

    # ret
    logc = np.log(np.maximum(c, 1e-12))
    ret = np.zeros_like(logc); ret[1:] = logc[1:] - logc[:-1]

    # rv
    def roll_std(x, w):
        s = pd.Series(x).rolling(w, min_periods=max(2, w//3)).std()
        return s.fillna(method="bfill").values
    rv5, rv15, rv30, rv60 = (roll_std(ret, 5), roll_std(ret, 15), roll_std(ret, 30), roll_std(ret, 60))

    # mom (price / EMA - 1)
    def mom_w(x, w):
        ema = pd.Series(x).ewm(span=w, adjust=False).mean().values
        return (x/np.maximum(ema,1e-12) - 1.0)
    mom5, mom15, mom30, mom60 = (mom_w(c,5), mom_w(c,15), mom_w(c,30), mom_w(c,60))

    # vol z-score
    def volz(x, w):
        s = pd.Series(x)
        mu = s.rolling(w, min_periods=max(2,w//3)).mean()
        sd = s.rolling(w, min_periods=max(2,w//3)).std()
        sd = sd.replace(0, np.nan)
        z = (s - mu) / sd
        return z.fillna(method="bfill").fillna(0.0).values
    vz20, vz60 = (volz(v,20), volz(v,60))

    # ATR14
    prev_close = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[ (h-l), np.abs(h - prev_close), np.abs(l - prev_close) ], axis=1)
    atr14 = pd.Series(tr).rolling(14, min_periods=5).mean().fillna(method="bfill").values

    # bias
    bias1 = np.ones_like(ret)

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(g["timestamp"].values),
        "ret": ret,
        "rv5": rv5, "rv15": rv15, "rv30": rv30, "rv60": rv60,
        "mom5": mom5, "mom15": mom15, "mom30": mom30, "mom60": mom60,
        "vz20": vz20, "vz60": vz60,
        "atr14": atr14,
        "bias1": bias1,
    })
    return out

def _apply_tcn_scaler(X: np.ndarray, mu, sd):
    if mu is None or sd is None:
        return X
    mu = np.asarray(mu, dtype=np.float32).reshape(1, -1)
    sd = np.asarray(sd, dtype=np.float32).reshape(1, -1)
    sd = np.where(sd>0, sd, 1.0)
    # 스케일러 차원 안 맞으면 스킵
    if mu.shape[1] != X.shape[1]:
        return X
    return (X - mu) / sd


def _tcn_thr():
    mode = os.getenv("THR_MODE", "fixed").lower()
    if mode == "fee" and TCN_HAS_MU:
        return float(fee_threshold())
    return float(os.getenv("MODEL_THR_BPS", "5.0")) / 1e4

# (_tcn_thr 정의 바로 아래) 두 번째 로더 교체
if TCN_MODEL is None:  # 첫 로더가 실패했을 때만 재시도
    try:
        ck = torch.load(TCN_CKPT, map_location="cpu")
        state = ck.get("model", ck)
        TCN_CFG = ck.get("cfg", {})
        if "FEATS" in TCN_CFG: TCN_FEATS = TCN_CFG["FEATS"]
        TCN_SEQ_LEN = int(TCN_CFG.get("SEQ_LEN", TCN_SEQ_LEN_FALLBACK))

        cand = [k for k in state.keys()
                if k.endswith("tcn.0.conv1.weight_v") or k.endswith("tcn.0.conv1.weight")
                or k.endswith("tcn.0.c1.weight_v")   or k.endswith("tcn.0.c1.weight")]
        if not cand:
            cand = [k for k in state.keys() if k.startswith("tcn.0.") and ("weight" in k)]
        k0 = cand[0]
        w0 = state[k0]
        hidden = w0.shape[0]; in_feat = w0.shape[1]; k_size = w0.shape[2]
        levels = 1 + max(int(k.split('.')[1]) for k in state.keys()
                         if k.startswith("tcn.") and (".conv1." in k or ".c1." in k))
        head_w = state.get("head.weight")
        head_out = head_w.shape[0] if head_w is not None else 2

        TCN_MODEL = TCNBinary(in_feat=in_feat, hidden=hidden, levels=levels, k=k_size,
                              dropout=0.1, head_out=head_out).eval()
        TCN_MODEL.load_state_dict(state, strict=False)
        TCN_HEAD = "cls" if TCN_MODEL.head.out_features == 1 else "reg"
        TCN_MU = ck.get("scaler_mu", None); TCN_SD = ck.get("scaler_sd", None)
    except Exception as e:
        print(f"[WARN] TCN load failed: {e}")  # 여기서 TCN_MODEL=None으로 덮어쓰지 않음
PREDICTION_LAG = env_int("PREDICTION_LAG", 0)  # 0=현재, 1=1분 후, 2=2분 후


@torch.no_grad()
def tcn_direction(symbol: str):
    if TCN_MODEL is None:
        return None, 0.0, _tcn_thr()

    lag = max(0, PREDICTION_LAG)
    df = get_recent_1m(symbol, minutes=TCN_SEQ_LEN + 10 + lag)
    if df is None or len(df) < (TCN_SEQ_LEN + lag + 1):
        return None, 0.0, _tcn_thr()

    feats = _build_tcn_features(df)

    # lag만큼 과거 데이터 사용
    if lag > 0:
        X = feats[TCN_FEATS].iloc[-(TCN_SEQ_LEN + lag):-lag].to_numpy().astype(np.float32)
    else:
        X = feats[TCN_FEATS].tail(TCN_SEQ_LEN).to_numpy().astype(np.float32)

    X = _apply_tcn_scaler(X, TCN_MU, TCN_SD)
    x_t = torch.from_numpy(X[None, ...])
    y_hat, logit = TCN_MODEL(x_t)
    if y_hat is None:  # cls
        p = float(torch.sigmoid(logit).item())
        prob_thr = float(os.getenv("LOGIT_PROB_THR", f"{LOGIT_PROB_THR}"))
        side = "Buy" if p > prob_thr else ("Sell" if p < (1.0 - prob_thr) else None)
        score = p - 0.5
        thr = prob_thr - 0.5
        if VERBOSE: print(f"[DEBUG][TCN][{symbol}] p={p:.3f} thr={prob_thr:.2f} -> {side}")
        return side, score, thr

    mu = float(y_hat.item())
    thr = _tcn_thr()
    side = "Buy" if mu > thr else ("Sell" if mu < -thr else None)
    if VERBOSE: print(f"[DEBUG][TCN][{symbol}] mu={mu:.6g} thr={thr:.6g} -> {side}")
    return side, mu, thr

@torch.no_grad()
def tcn_time_to_target_minutes(symbol: str, side: str, target_bps: float):
    if TCN_HEAD == "cls":
        return None
    t_side, mu_t, _thr = tcn_direction(symbol)
    if mu_t is None: return None
    step = abs(float(mu_t))
    if step <= 1e-8: return None
    thr = _log_thr_from_bps(target_bps)
    need = int(math.ceil(thr / step))
    return max(1, need)

# ===== CatBoost (optional)
try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_CBM  = os.getenv("CATBOOST_CBM",  "../multimodel/models/catboost_meta.cbm")
    CATBOOST_META = os.getenv("CATBOOST_META", "../multimodel/models/catboost_meta_features.json")
    CB_MODEL: Optional[CatBoostClassifier] = CatBoostClassifier()
    CB_MODEL.load_model(CATBOOST_CBM)
    with open(CATBOOST_META, "r", encoding="utf-8") as f:
        CB_META = json.load(f)
except Exception as e:
    CB_MODEL = None
    CB_META = {}
    print(f"[WARN] CatBoost load failed: {e}")

def _latest_base_feats(df: pd.DataFrame) -> Optional[dict]:
    need = 120
    if df is None or len(df) < need: return None
    g = df.sort_values("timestamp").copy()
    c = g["close"].astype(float).values
    h = g["high"].astype(float).values
    l = g["low"].astype(float).values
    v = g["volume"].astype(float).values
    logc = np.log(np.maximum(c, 1e-12))
    ret1 = np.zeros_like(logc); ret1[1:] = logc[1:] - logc[:-1]
    def roll_std(arr, w):
        s = pd.Series(arr).rolling(w, min_periods=max(2, w//3)).std()
        return float(s.iloc[-1])
    def mom_w(arr, w):
        ema = pd.Series(arr).ewm(span=w, adjust=False).mean()
        return float(arr[-1]/max(ema.iloc[-1], 1e-12) - 1.0)
    def volz(arr, w):
        s = pd.Series(arr)
        mu = s.rolling(w, min_periods=max(2,w//3)).mean()
        sd = s.rolling(w, min_periods=max(2,w//3)).std().replace(0, np.nan)
        val = (arr[-1] - float(mu.iloc[-1])) / (float(sd.iloc[-1]) if pd.notna(sd.iloc[-1]) and sd.iloc[-1]>0 else 1.0)
        return float(val)
    prev_close = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[ (h-l), np.abs(h - prev_close), np.abs(l - prev_close) ], axis=1)
    atr14 = pd.Series(tr).rolling(14, min_periods=5).mean().iloc[-1]
    feats = {"ret1": float(ret1[-1]),
             "rv5":  roll_std(ret1, 5),  "rv15": roll_std(ret1, 15),
             "rv30": roll_std(ret1, 30), "rv60": roll_std(ret1, 60),
             "mom5":  mom_w(c, 5),   "mom15": mom_w(c, 15),
             "mom30": mom_w(c, 30),  "mom60": mom_w(c, 60),
             "vz20": volz(v, 20), "vz60": volz(v, 60),
             "atr14": float(atr14 if pd.notna(atr14) else 0.0)}
    if any([not np.isfinite(v) for v in feats.values()]): return None
    return feats

@torch.no_grad()
def catboost_direction(symbol: str) -> Tuple[Optional[str], float]:
    def _log(msg: str):
        if VERBOSE or TRACE:
            print(msg)
    AGREE_REQUIRED = int(os.getenv("AGREE_REQUIRED","0"))
    ALLOW_TCN_ONLY = int(os.getenv("ALLOW_TCN_ONLY","1"))

    minutes = max(300, int(TCN_SEQ_LEN + 80))
    df = get_recent_1m(symbol, minutes=minutes)
    if df is None or len(df) < (TCN_SEQ_LEN + 61):
        _log(f"[CB][{symbol}] SKIP: SHORT_DF len={0 if df is None else len(df)} need>={TCN_SEQ_LEN+61}")
        return (None, 0.0)

    try:
        t_side, t_score, t_thr = tcn_direction(symbol)
    except Exception as e:
        _log(f"[CB][{symbol}] SKIP: TCN_ERR {e}")
        return (None, 0.0)
    t_ok = (t_side in ("Buy","Sell")) and (abs(float(t_score)) >= float(t_thr))
    _log(f"[CB][{symbol}][TCN] side={t_side} score={t_score:.6g} thr={t_thr:.6g} ok={int(t_ok)}")
    if AGREE_REQUIRED and not t_ok:
        _log(f"[CB][{symbol}] -> SKIP: REQUIRE_TCN")
        return (None, 0.0)

    if (CB_MODEL is None) or (not CB_META):
        if ALLOW_TCN_ONLY and t_ok:
            side = t_side
            if ENTRY_MODE == "inverse":
                side = "Sell" if side == "Buy" else "Buy"
            _log(f"[CB][{symbol}] NO_CB -> PASS(TCN_ONLY) {side}")
            return side, float(abs(t_score))
        _log(f"[CB][{symbol}] SKIP: NO_CB & TCN_FAIL")
        return (None, 0.0)

    base = _latest_base_feats(df)
    if base is None:
        _log(f"[CB][{symbol}] SKIP: BASE_FEATS_NA")
        return (None, 0.0)

    feat_cols: List[str] = CB_META.get("feature_cols", [])
    cat_cols:  List[str] = CB_META.get("cat_cols", ["symbol"])
    base_thr:  float      = float(os.getenv("CB_THR", CB_META.get("threshold_prob", 0.55)))

    row = dict(base)
    if "tcn_mu" in feat_cols: row["tcn_mu"] = float(t_score)
    if "tcn_logit" in feat_cols and "tcn_logit" not in row: row["tcn_logit"] = 0.0
    for c in feat_cols:
        if c not in row:
            _log(f"[CB][{symbol}] SKIP: MISSING_FEAT {c}")
            return (None, 0.0)

    X = pd.DataFrame([[row[c] for c in feat_cols] + [symbol]*len(cat_cols)], columns=feat_cols + cat_cols)
    pool = Pool(X, cat_features=[len(feat_cols)+i for i in range(len(cat_cols))])

    try:
        proba = CB_MODEL.predict_proba(pool)[0]
    except Exception as e:
        _log(f"[CB][{symbol}] SKIP: CB_ERR {e}")
        return (None, 0.0)

    pred_idx = int(np.argmax(proba))
    inv_map = {0:-1, 1:0, 2:+1}
    y_hat = inv_map.get(pred_idx, 0)
    cb_conf = float(np.max(proba))
    cb_side = "Buy" if y_hat > 0 else ("Sell" if y_hat < 0 else None)

    CB_CONF_BUF.setdefault(symbol, deque(maxlen=max(50, CB_Q_WIN))).append(cb_conf)
    def _cb_dyn(symbol:str, fallback:float)->float:
        buf = CB_CONF_BUF.get(symbol)
        if not buf or len(buf) < max(20, int(CB_Q_WIN*0.4)): return float(fallback)
        arr = np.asarray(list(buf), dtype=float)
        q = float(np.quantile(arr, min(max(CB_Q_PERC, 0.5), 0.9)))
        return float(min(max(q, MIN_CB_THR), fallback + 0.05))

    dyn_thr = _cb_dyn(symbol, base_thr)
    thr_prob = max(float(MIN_CB_THR), float(dyn_thr))
    _log(f"[CB][{symbol}][CB] pred={cb_side or 'NoTrade'} conf={cb_conf:.3f} thr(base={base_thr:.2f}, dyn={dyn_thr:.2f}) -> use={thr_prob:.2f}")

    if cb_side is None or cb_conf < thr_prob:
        if ALLOW_TCN_ONLY and t_ok:
            side = t_side
            if ENTRY_MODE == "inverse":
                side = "Sell" if side == "Buy" else "Buy"
            _log(f"[CB][{symbol}] CB_WEAK -> PASS(TCN_ONLY) {side} (cb={cb_conf:.3f}<thr={thr_prob:.2f})")
            return side, float(max(cb_conf, abs(t_score)))
        _log(f"[CB][{symbol}] -> SKIP: { 'CB_PROB_LOW' if cb_side else 'CB_NOTRADE' }")
        return (None, cb_conf)

    if AGREE_REQUIRED and t_ok and cb_side != t_side:
        _log(f"[CB][{symbol}] -> SKIP: DISAGREE")
        return (None, cb_conf)

    final_side = cb_side
    if ENTRY_MODE == "inverse":
        final_side = "Sell" if cb_side == "Buy" else "Buy"
    _log(f"[CB][{symbol}] -> PASS {final_side} (conf={cb_conf:.3f})")
    return (final_side, cb_conf)

# ===== Paper Broker (hedge)
def _round_down(x: float, step: float) -> float:
    if step<=0: return x
    return math.floor(x/step)*step

class PaperBroker:
    def __init__(self):
        self.pos: Dict[Tuple[str,str],dict] = {}
        self.qty_step = 0.001
        self.min_qty = 0.001
    def key(self, symbol,str_side): return (symbol, str_side)
    def has_side(self, symbol, side): return self.key(symbol, side) in self.pos
    def try_open(self, symbol: str, side: str, ts: int):
        bid, ask, mid = get_quote(symbol)
        if bid<=0 or ask<=0 or mid<=0: return False
        px = _exec_taker_price(side, bid, ask, SLIPPAGE_BPS_TAKER)
        eq = get_wallet_equity()
        qty = _round_down(eq * ENTRY_EQUITY_PCT * LEVERAGE / max(px,1e-9), self.qty_step)
        if qty < self.min_qty: return False
        self.pos[self.key(symbol, side)] = {"side": side, "entry": px, "qty": qty, "opened": ts}
        return True
    def positions(self): return self.pos

BROKER = PaperBroker()

def unrealized_pnl_usd(side: str, qty: float, entry: float, mark: float) -> float:
    return (mark - entry) * qty if side=="Buy" else (entry - mark) * qty

# ===== I/O
def _ensure_csv():
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(TRADES_FIELDS)
    if not os.path.exists(EQUITY_CSV):
        with open(EQUITY_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(EQUITY_FIELDS)
    if not os.path.exists(EQUITY_MTM_CSV):
        with open(EQUITY_MTM_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(EQUITY_FIELDS)

def _log_equity_mtm(eq: float):
    global _mtm_last_val, _mtm_last_ts
    now_ts = time.time()
    if (_mtm_last_val is None) or (abs(float(eq) - float(_mtm_last_val)) > MTM_EPS) or ((now_ts - _mtm_last_ts) >= MTM_MIN_INTERVAL_SEC):
        _ensure_csv()
        with open(EQUITY_MTM_CSV,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([int(now_ts), f"{float(eq):.6f}"])
        if VERBOSE: print(f"[EQUITY_MTM] {float(eq):.2f}")
        _mtm_last_val = float(eq); _mtm_last_ts = now_ts

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

def get_equity_mtm() -> float:
    base = get_wallet_equity()
    upnl = 0.0
    for (sym, side), p in BROKER.positions().items():
        qty, entry = p["qty"], p["entry"]
        _,_,mark = get_quote(sym)
        if mark > 0:
            upnl += unrealized_pnl_usd(side, qty, entry, mark)
    return base + upnl

def get_wallet_equity() -> float:
    return float(_equity_cache)

def apply_trade_pnl(side: str, entry: float, exit: float, qty: float, taker_fee: float = TAKER_FEE):
    gross = (exit - entry) * qty if side=="Buy" else (entry - exit) * qty
    fee = (entry * qty + exit * qty) * taker_fee
    new_eq = get_wallet_equity() + gross - fee
    _log_equity(new_eq)
    return gross, fee, new_eq

# ===== META / SESSION
def _bps(pnl_usd: float, notional_usd: float) -> float:
    if notional_usd <= 0: return 0.0
    return (pnl_usd / notional_usd) * 10_000.0

def _meta_push(symbol: str, pnl_bps: float):
    META_TRADES.setdefault(symbol, deque(maxlen=max(5, META_WIN))).append(float(pnl_bps))

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
    return hit, mean, mdd

class SessionManager:
    def __init__(self):
        self.reset_counters()
        self.lockbox = 0.0
    def reset_counters(self):
        self.t0 = time.time()
        self.trades = 0
        self.pnl_bps = 0.0
        self.dd_min_bps = 0.0
        self.warmup_until = self.t0 + SESSION_WARMUP_SEC
    def in_warmup(self) -> bool:
        return time.time() < self.warmup_until
    def on_trade_close(self, pnl_bps: float):
        self.trades += 1
        self.pnl_bps += pnl_bps
        self.dd_min_bps = min(self.dd_min_bps, self.pnl_bps)
    def session_time_over(self) -> bool:
        return (time.time() - self.t0) >= SESSION_HARD_MINUTES
    def should_restart(self) -> Tuple[bool,str]:
        if self.pnl_bps >= SESSION_TARGET_PNL_BPS: return True, "HIT_TARGET"
        if self.pnl_bps <= SESSION_MAX_DD_BPS:     return True, "HIT_DRAWDOWN"
        if self.session_time_over():               return True, "HIT_TIME"
        if SESSION_MAX_TRADES>0 and self.trades >= SESSION_MAX_TRADES: return True, "HIT_TRADE_CAP"
        return False, ""

SESS = SessionManager()

def _clear_runtime_state():
    ENTRY_HISTORY.clear()
    for s in SYMBOLS:
        ENTRY_HISTORY[s] = collections.deque(maxlen=max(1, N_CONSEC_SIGNALS))
        COOLDOWN_UNTIL[s] = 0.0
    # prune STATE/POSITION_DEADLINE non-existing symbols
    for d in (POSITION_DEADLINE, STATE):
        for k in list(d.keys()):
            if k[0] not in SYMBOLS:
                d.pop(k, None)
    for s in SYMBOLS:
        CB_CONF_BUF[s].clear()
        META_TRADES[s].clear()
    global RISK_CONSEC_LOSS
    RISK_CONSEC_LOSS = 0

def _risk_guard_on_close(symbol: str, pnl_bps: float):
    global RISK_CONSEC_LOSS, RISK_DAY_CUM_BPS, RISK_LAST_RESET_DAY, COOLDOWN_UNTIL
    today = time.strftime("%Y-%m-%d")
    if today != RISK_LAST_RESET_DAY:
        RISK_LAST_RESET_DAY = today
        RISK_DAY_CUM_BPS = 0.0
        RISK_CONSEC_LOSS = 0
    RISK_DAY_CUM_BPS += float(pnl_bps)
    if pnl_bps <= 0: RISK_CONSEC_LOSS += 1
    else:            RISK_CONSEC_LOSS = 0
    breach = (RISK_DAY_CUM_BPS <= MAX_DD_BPS_DAY) or (RISK_CONSEC_LOSS >= MAX_CONSEC_LOSS)
    if breach:
        until = time.time() + RISK_COOLDOWN_SEC
        for s in SYMBOLS:
            COOLDOWN_UNTIL[s] = until
        if VERBOSE:
            print(f"[RISK] cooldown {RISK_COOLDOWN_SEC}s (day_dd={RISK_DAY_CUM_BPS:.1f}bps, consec={RISK_CONSEC_LOSS})")

# ===== ENTRY/FILTERS
def _dynamic_abs_tp_usd(symbol:str, mid:float, qty:float)->float:
    if ABS_TP_USD > 0: return ABS_TP_USD
    v_bps = recent_vol_bps(symbol, VOL_WIN)
    atr_usd = (v_bps/1e4) * mid * qty
    return max(ABS_TP_USD_FLOOR, ABS_K * atr_usd)

def _entry_allowed_by_filters(symbol: str, side: str, entry_ref: float) -> bool:
    v_bps = recent_vol_bps(symbol, VOL_WIN)
    if v_bps < V_BPS_FLOOR:
        if VERBOSE: print(f"[SKIP] {symbol} vol {v_bps:.1f}bps < floor {V_BPS_FLOOR}bps")
        return False

    bid, ask, _ = get_quote(symbol)
    spr_bps = None
    if bid > 0 and ask > 0:
        spr_bps = _bps_between(bid, ask)
        if spr_bps > SPREAD_BPS_MAX:
            if VERBOSE: print(f"[SKIP] {symbol} spread {spr_bps:.1f}bps > {SPREAD_BPS_MAX}bps")
            return False

    tp1_px   = tp_from_bps(entry_ref, TP1_BPS, side)
    edge_bps = _bps_between(entry_ref, tp1_px)

    if spr_bps is not None:
        req_min = _min_tp_required_bps(spr_bps)
        if edge_bps < req_min:
            if VERBOSE:
                print(f"[SKIP] {symbol} MIN_TP_BREACH tp={edge_bps:.1f}bps < req={req_min:.1f}bps")
            return False

    base_need = _entry_cost_bps() + _exit_cost_bps() + ROI_BASE_MIN_EXPECTED_BPS
    need_bps  = base_need
    if v_bps < (2.0 * V_BPS_FLOOR):
        need_bps += ROI_LOWVOL_BOOST_BPS
    if META_ON:
        hit, evr, mdd = _meta_stats(symbol)
        if (hit < META_MIN_HIT) or (evr < META_MIN_EVR) or (mdd <= META_MAX_DD_BPS):
            need_bps += ROI_POORMETA_BOOST_BPS
    if edge_bps < need_bps:
        if VERBOSE: print(f"[SKIP] {symbol} ROI edge {edge_bps:.1f} < need {need_bps:.1f}bps (v={v_bps:.1f})")
        return False

    return True

def _meta_allows(symbol: str) -> bool:
    if not META_ON:
        return True
    if META_BLOCK_UNTIL.get(symbol, 0.0) > time.time():
        if VERBOSE: print(f"[META][BLOCK] {symbol} cooldown")
        return False
    hit, mean_bps, mdd = _meta_stats(symbol)
    ok = (hit >= META_MIN_HIT) and (mean_bps >= META_MIN_EVR) and (mdd > META_MAX_DD_BPS)
    if not ok:
        META_BLOCK_UNTIL[symbol] = time.time() + META_COOLDOWN
        if VERBOSE: print(f"[META][PAUSE] {symbol} hit={hit:.2f} evr={mean_bps:.1f} mdd={mdd:.1f} -> rest {META_COOLDOWN}s")
        return False
    return True

def choose_entry(symbol: str):
    # TCN-only path
    side, score, thr = tcn_direction(symbol)
    if side in ("Buy", "Sell") and abs(float(score)) >= float(thr):
        # “confidence” 자리에 절대값 점수 사용
        return side, float(abs(score))
    return None, 0.0

def _scan_tick(symbol: str, side_probe: Optional[str]=None):
    now = time.time()
    fields = []; reasons=[]
    has_long = BROKER.has_side(symbol, "Buy")
    has_short= BROKER.has_side(symbol, "Sell")
    fields.append(f"posL={int(has_long)} posS={int(has_short)}")
    cd_left = max(0, int(COOLDOWN_UNTIL.get(symbol,0.0) - now))
    fields.append(f"cool={cd_left}s")
    if cd_left>0: reasons.append("COOLDOWN")
    if META_ON:
        hit, evr, mdd = _meta_stats(symbol)
        meta_ok = (hit >= META_MIN_HIT) and (evr >= META_MIN_EVR) and (mdd > META_MAX_DD_BPS)
        meta_block = META_BLOCK_UNTIL.get(symbol,0.0) > now
        fields.append(f"meta_ok={int(meta_ok)} hit={hit:.2f} evr={evr:.1f} mdd={mdd:.1f} block={int(meta_block)}")
        if (not meta_ok) or meta_block: reasons.append("META")
    try:
        t_side, mu, thr = tcn_direction(symbol)
        fields.append(f"tcn_side={t_side} mu={mu:.6g} thr={thr:.6g}")
        if t_side is None: reasons.append("NO_TCN_SIG")
    except Exception as e:
        fields.append(f"tcn_err={e}"); reasons.append("TCN_ERR"); t_side=None; thr=0
    if USE_CATBOOST and CB_MODEL is not None and CB_META:
        cb_side, cb_conf = catboost_direction(symbol)
        fields.append(f"cb_side={cb_side} cb_conf={cb_conf:.2f}")
        # 필요 시 여기서만 CatBoost 기준 추가 필터링
    else:
        cb_side, cb_conf = None, 0.0
        # CatBoost 미사용 시 CB_BLOCK 이유는 추가하지 않음

    v_bps = recent_vol_bps(symbol, VOL_WIN)
    vol_ok = v_bps >= V_BPS_FLOOR
    fields.append(f"vol={v_bps:.1f}bps floor={V_BPS_FLOOR:.1f} {'OK' if vol_ok else 'LOW'}")
    if not vol_ok: reasons.append("LOW_VOL")
    probe = side_probe or t_side
    if probe in ("Buy","Sell"):
        bid, ask, mid = get_quote(symbol)
        if (probe=="Buy" and ask>0) or (probe=="Sell" and bid>0):
            entry_ref = float(ask if probe=="Buy" else bid)
            tp1 = tp_from_bps(entry_ref, TP1_BPS, probe)
            edge_bps = _bps_between(entry_ref, tp1)
            need_bps = _entry_cost_bps() + _exit_cost_bps() + ROI_BASE_MIN_EXPECTED_BPS
            fields.append(f"roi_edge={edge_bps:.1f} need={need_bps:.1f} {'OK' if edge_bps>=need_bps else 'LOW'}")
            if bid>0 and ask>0:
                sp = _bps_between(bid, ask); req = _min_tp_required_bps(sp)
                fields.append(f"min_tp_req={req:.1f}bps {'OK' if edge_bps>=req else 'LOW'}")
                if edge_bps < req: reasons.append("MIN_TP")
    dq = ENTRY_HISTORY[symbol]; need = max(1, N_CONSEC_SIGNALS)
    fields.append(f"consec={len(dq)}/{need} buf={list(dq)}")
    if need>1 and (len(dq)<need or any(x != (probe or "") for x in dq)):
        reasons.append("CONSEC")
    verdict = "PASS" if not reasons else ("SKIP:" + ",".join(reasons))
    print(f"[SCAN] {symbol} | " + " | ".join(fields) + f" -> {verdict}")

def try_enter(symbol: str):
    # cooldown
    if COOLDOWN_UNTIL.get(symbol, 0.0) > time.time():
        return

    # model signal
    side, conf = choose_entry(symbol)
    if side not in ("Buy", "Sell"):
        return

    # hedge: same-side exists -> skip
    if BROKER.has_side(symbol, side):
        return

    # meta filter
    if not _meta_allows(symbol):
        return

    # consecutive signals gate
    dq = ENTRY_HISTORY[symbol]
    need = max(1, N_CONSEC_SIGNALS)
    dq.append(side)
    if need > 1 and (len(dq) < need or any(x != side for x in dq)):
        if VERBOSE: print(f"[SKIP] consec fail {symbol} need={need} got={list(dq)}")
        if TRACE: _scan_tick(symbol, side)
        return

    # quotes
    bid, ask, mid = get_quote(symbol)
    if bid <= 0 or ask <= 0 or mid <= 0:
        return

    # pre-entry filters use executable reference price
    entry_ref = float(ask if side == "Buy" else bid)
    if not _entry_allowed_by_filters(symbol, side, entry_ref):
        if TRACE: _scan_tick(symbol, side)
        return

    if TRACE: _scan_tick(symbol, side)

    ts = int(time.time())
    with ENTRY_LOCK:
        # re-check state after lock
        if BROKER.has_side(symbol, side):
            return
        # execute entry at taker with slippage inside broker
        if not BROKER.try_open(symbol, side, ts):
            return

        # use actual filled entry price
        entry_px = float(BROKER.positions()[(symbol, side)]["entry"])
        pos_qty  = float(BROKER.positions()[(symbol, side)]["qty"])

        # timeout horizon
        horizon_min = max(1, int(calc_timeout_minutes(symbol, side)))
        POSITION_DEADLINE[(symbol, side)] = ts + horizon_min * 60

        # targets and SL from filled price
        tp1 = tp_from_bps(entry_px, TP1_BPS, side)
        tp2 = tp_from_bps(entry_px, TP2_BPS, side)
        tp3 = tp_from_bps(entry_px, TP3_BPS, side)
        sl  = entry_px * (1.0 - SL_ROI_PCT) if side == "Buy" else entry_px * (1.0 + SL_ROI_PCT)

        STATE[(symbol, side)] = {
            "tp1": tp1, "tp2": tp2, "tp3": tp3, "sl": sl,
            "tp1_filled": False, "tp2_filled": False, "be_moved": False,
            "abs_usd": _dynamic_abs_tp_usd(symbol, entry_px, pos_qty),
            "mfe_px": entry_px, "opened_ts": ts
        }

        _log_trade({
            "ts": ts, "event": "ENTRY", "symbol": symbol, "side": side,
            "qty": f"{pos_qty:.8f}", "entry": f"{entry_px:.6f}", "exit": "",
            "tp1": f"{tp1:.6f}", "tp2": f"{tp2:.6f}", "tp3": f"{tp3:.6f}",
            "sl": f"{sl:.6f}", "reason": "open", "mode": "paper"
        })
        if VERBOSE:
            print(f"[OPEN] {symbol} {side} entry={entry_px:.6f} timeout={horizon_min}m  [EQUITY]={get_wallet_equity():.2f}")

def calc_timeout_minutes(symbol: str, side: Optional[str] = None) -> int:
    if TIMEOUT_MODE == "fixed":
        return max(1, int(PRED_HORIZ_MIN))
    if TIMEOUT_MODE == "model":
        target_bps = TIMEOUT_TARGET_BPS if TIMEOUT_TARGET_BPS is not None else TP1_BPS
        m2 = tcn_time_to_target_minutes(symbol, side or "Buy", target_bps)
        if m2 is not None:
            chosen = int(min(max(m2, TIMEOUT_MIN), TIMEOUT_MAX))
            if VERBOSE: print(f"[TIMEOUT][{symbol}] tcn={m2}m -> {chosen}m (target={target_bps}bps)")
            return chosen
        if VERBOSE: print(f"[TIMEOUT][{symbol}] model-time unavailable -> fallback to ATR")
    target_bps = TIMEOUT_TARGET_BPS if TIMEOUT_TARGET_BPS is not None else TP1_BPS
    v_bps = recent_vol_bps(symbol, VOL_WIN); v_bps = max(v_bps, 1e-6)
    need = math.ceil((target_bps / v_bps) * TIMEOUT_K)
    return int(min(max(need, TIMEOUT_MIN), TIMEOUT_MAX))

def close_and_log(symbol: str, side: str, reason: str):
    now = int(time.time())
    if not BROKER.has_side(symbol, side): return
    p = BROKER.positions()[(symbol, side)]
    entry = float(p["entry"]); qty = float(p["qty"])
    bid, ask, _ = get_quote(symbol)
    # 반대 사이드로 시장가 청산
    exit_side = "Sell" if side=="Buy" else "Buy"
    exec_px = _exec_taker_price(exit_side, bid, ask, SLIPPAGE_BPS_TAKER)

    # 포지션 닫기
    del BROKER.positions()[(symbol, side)]
    gross, fee, new_eq = apply_trade_pnl(side, entry, exec_px, qty, taker_fee=TAKER_FEE)
    notional = max(entry * qty, 1e-9)
    pnl_bps = _bps(gross - fee, notional)

    final_tag = "TAKE_PROFIT" if (reason=="ABS_TP" and (gross - fee) > 0) else ("ABS_EXIT" if reason=="ABS_TP" else reason)
    _meta_push(symbol, pnl_bps)
    _log_trade({"ts": now,"event":"EXIT","symbol":symbol,"side":exit_side,
                "qty":f"{qty:.8f}","entry":f"{entry:.6f}","exit":f"{exec_px:.6f}",
                "tp1":"","tp2":"","tp3":"","sl":"","reason":final_tag,"mode":"paper"})
    if VERBOSE:
        print(f"[{final_tag}] {symbol}/{side} pnl_bps={pnl_bps:.1f} gross={gross:.6f} fee={fee:.6f} eq={new_eq:.2f}")

    POSITION_DEADLINE.pop((symbol, side), None)
    STATE.pop((symbol, side), None)
    COOLDOWN_UNTIL[symbol] = time.time() + COOLDOWN_SEC
    _risk_guard_on_close(symbol, pnl_bps)

def monitor_loop(poll_sec: float = 1.0, max_loops: int = 2):
    loops = 0
    while loops < max_loops:
        # mark-to-market 로그
        _log_equity_mtm(get_equity_mtm())
        now = int(time.time())

        for (sym, side), p in list(BROKER.positions().items()):
            entry = float(p["entry"])
            qty   = float(p["qty"])
            bid, ask, mark = get_quote(sym)
            if mark <= 0:
                continue

            # 상태 초기화 보정
            st = STATE.get((sym, side))
            if st is None:
                tp1 = tp_from_bps(entry, TP1_BPS, side)
                tp2 = tp_from_bps(entry, TP2_BPS, side)
                tp3 = tp_from_bps(entry, TP3_BPS, side)
                sl  = entry * (1.0 - SL_ROI_PCT) if side == "Buy" else entry * (1.0 + SL_ROI_PCT)
                st = {
                    "tp1": tp1, "tp2": tp2, "tp3": tp3, "sl": sl,
                    "tp1_filled": False, "tp2_filled": False, "be_moved": False,
                    "abs_usd": _dynamic_abs_tp_usd(sym, entry, qty),
                    "mfe_px": entry, "opened_ts": int(p.get("opened", now))
                }
                STATE[(sym, side)] = st

            # 타임아웃 판정
            ddl = POSITION_DEADLINE.get((sym, side))
            to_hit = bool(ddl and now >= int(ddl))

            # 타임아웃 모드 처리
            if HOLD_UNTIL_TIMEOUT:
                if to_hit:
                    close_and_log(sym, side, "HORIZON_TIMEOUT")
                    continue
            else:
                # 분할청산 로직
                if (not st["tp1_filled"]) and ((side == "Buy" and mark >= st["tp1"]) or (side == "Sell" and mark <= st["tp1"])):
                    part = qty * TP1_RATIO
                    apply_trade_pnl(side, entry, st["tp1"], part, taker_fee=TAKER_FEE)
                    BROKER.positions()[(sym, side)]["qty"] = qty = max(0.0, qty - part)
                    st["tp1_filled"] = True
                    _log_trade({"ts": now, "event": "TP1_FILL", "symbol": sym, "side": side,
                                "qty": f"{part:.8f}", "entry": f"{entry:.6f}", "exit": f"{st['tp1']:.6f}",
                                "tp1": "", "tp2": "", "tp3": "", "sl": "", "reason": "", "mode": "paper"})
                if qty > 0 and (not st["tp2_filled"]) and ((side == "Buy" and mark >= st["tp2"]) or (side == "Sell" and mark <= st["tp2"])):
                    remain_ratio = 1.0 - TP1_RATIO
                    part = min(qty, remain_ratio * TP2_RATIO)
                    apply_trade_pnl(side, entry, st["tp2"], part, taker_fee=TAKER_FEE)
                    BROKER.positions()[(sym, side)]["qty"] = qty = max(0.0, qty - part)
                    st["tp2_filled"] = True
                    _log_trade({"ts": now, "event": "TP2_FILL", "symbol": sym, "side": side,
                                "qty": f"{part:.8f}", "entry": f"{entry:.6f}", "exit": f"{st['tp2']:.6f}",
                                "tp1": "", "tp2": "", "tp3": "", "sl": "", "reason": "", "mode": "paper"})

                # BE 이동
                if qty > 0:
                    if st["tp2_filled"] and not st["be_moved"]:
                        be_px = entry * (1.0 + BE_EPS_BPS / 10000.0) if side == "Buy" else entry * (1.0 - BE_EPS_BPS / 10000.0)
                        st["sl"] = be_px; st["be_moved"] = True
                        _log_trade({"ts": now, "event": "MOVE_SL", "symbol": sym, "side": side,
                                    "qty": "", "entry": "", "exit": "", "tp1": "", "tp2": "", "tp3": "",
                                    "sl": f"{be_px:.6f}", "reason": "BE", "mode": "paper"})
                    # 빠른 BE
                    if not st["be_moved"]:
                        gain_bps = _bps_between(entry, mark)
                        if (side == "Buy" and mark < entry) or (side == "Sell" and mark > entry):
                            gain_bps = 0.0
                        if gain_bps >= FAST_BE_BPS:
                            be_px = entry * (1.0 + BE_EPS_BPS / 10000.0) if side == "Buy" else entry * (1.0 - BE_EPS_BPS / 10000.0)
                            st["sl"] = be_px; st["be_moved"] = True
                            _log_trade({"ts": now, "event": "MOVE_SL", "symbol": sym, "side": side,
                                        "qty": "", "entry": "", "exit": "", "tp1": "", "tp2": "", "tp3": "",
                                        "sl": f"{be_px:.6f}", "reason": "FAST_BE", "mode": "paper"})

                    # 트레일링
                    if (st["tp1_filled"] if TRAIL_AFTER_TIER == 1 else st["tp2_filled"]):
                        if side == "Buy":
                            st["sl"] = max(st["sl"], mark * (1.0 - TRAIL_BPS / 10000.0))
                        else:
                            st["sl"] = min(st["sl"], mark * (1.0 + TRAIL_BPS / 10000.0))

            # 공통 종료 조건(ABS/SL/타임아웃/감소)
            if qty > 0:
                age = now - st["opened_ts"]
                if side == "Buy":
                    st["mfe_px"] = max(st["mfe_px"], mark)
                    draw = (st["mfe_px"] - mark) / max(st["mfe_px"], 1e-12)
                else:
                    st["mfe_px"] = min(st["mfe_px"], mark)
                    draw = (mark - st["mfe_px"]) / max(st["mfe_px"], 1e-12)

                decay_hit = (age >= DECAY_GRACE_SEC) and (draw >= DECAY_PCT)
                sl_hit    = ((mark <= st["sl"]) if side == "Buy" else (mark >= st["sl"]))
                pnl_usd   = unrealized_pnl_usd(side, qty, entry, mark)
                abs_hit   = pnl_usd >= st["abs_usd"]

                if (age >= MIN_LOCK_SEC) and (abs_hit or sl_hit or to_hit or decay_hit):
                    reason = "ABS_TP" if abs_hit else ("SL" if sl_hit else ("HORIZON_TIMEOUT" if to_hit else "PROFIT_DECAY"))
                    close_and_log(sym, side, reason)
                    continue

        loops += 1
        time.sleep(poll_sec)

def _whynot(symbol: str):
    msgs = []
    now = time.time()
    if COOLDOWN_UNTIL.get(symbol,0.0) > now: msgs.append(f"cooldown={int(COOLDOWN_UNTIL[symbol]-now)}s")
    msgs.append(f"posL={int(BROKER.has_side(symbol,'Buy'))} posS={int(BROKER.has_side(symbol,'Sell'))}")
    if META_ON:
        hit, mean_bps, mdd = _meta_stats(symbol)
        meta_block = META_BLOCK_UNTIL.get(symbol,0.0) > now
        ok = (hit >= META_MIN_HIT) and (mean_bps >= META_MIN_EVR) and (mdd > META_MAX_DD_BPS)
        msgs.append(f"meta_ok={int(ok)} hit={hit:.2f} evr={mean_bps:.1f} mdd={mdd:.1f} block={int(meta_block)}")
    try:
        t_side, mu, thr = tcn_direction(symbol)
        if TCN_MODEL is None: msgs.append("tcn=OFF")
        else: msgs.append(f"tcn_side={t_side} mu/score={mu:.3f} thr={thr:.3f}")
    except Exception as e:
        msgs.append(f"tcn_err={e}")
    bid, ask, mid = get_quote(symbol)
    v_bps = recent_vol_bps(symbol, VOL_WIN)
    msgs.append(f"vol={v_bps:.1f}bps floor={V_BPS_FLOOR:.1f} {'OK' if v_bps>=V_BPS_FLOOR else 'LOW'}")
    if 't_side' in locals() and t_side in ("Buy","Sell"):
        entry_ref = float(ask if t_side=='Buy' else bid)
        if entry_ref>0:
            tp1 = tp_from_bps(entry_ref, TP1_BPS, t_side)
            edge_bps = _bps_between(entry_ref, tp1)
            need_bps = _entry_cost_bps() + _exit_cost_bps() + ROI_BASE_MIN_EXPECTED_BPS
            msgs.append(f"roi_edge={edge_bps:.1f} need={need_bps:.1f} {'OK' if edge_bps>=need_bps else 'LOW'}")
            if bid>0 and ask>0:
                sp = _bps_between(bid, ask); req = _min_tp_required_bps(sp)
                msgs.append(f"min_tp_req={req:.1f}bps {'OK' if edge_bps>=req else 'LOW'}")
    dq = ENTRY_HISTORY[symbol]; need = max(1, N_CONSEC_SIGNALS)
    msgs.append(f"consec_need={need} buf={list(dq)}")
    print("[WHYNOT]", symbol, " | ".join(msgs))

# ===== MAIN
maybe_apply_preset()

# ==== HOT SYMBOL PICKER (Bybit v5) ====
def find_hot_symbols(n: int = 5, min_turnover_usdt: float = 30_000_000.0) -> list[str]:
    """
    24h 변동률(|price24hPcnt|) 큰 선물(USDT Perp)만 상위 n개 선택.
    - min_turnover_usdt: 24h 거래대금(USDT) 하한
    반환: ["INUSDT","KGENUSDT", ...]
    """
    try:
        r = mkt.get_tickers(category="linear", symbol="")  # 모든 선물 심볼
        rows = ((r.get("result") or {}).get("list") or [])
    except Exception:
        rows = []

    picks = []
    for it in rows:
        sym = str(it.get("symbol") or "")
        # USDT 선물만, (현물/USDC/옵션 제외)
        if not sym.endswith("USDT"):
            continue
        # 24h 변동률(%), 24h 거래대금(USDT)
        pc = float(it.get("price24hPcnt") or 0.0) * 100.0   # e.g. 0.53 -> 53%
        turn = float(it.get("turnover24h") or 0.0)          # already USDT
        if turn < float(min_turnover_usdt):
            continue
        picks.append((abs(pc), turn, sym))

    # 변동률↓, 대금↓ 정렬 후 상위 n개 심볼만
    picks.sort(key=lambda x: (-x[0], -x[1]))
    hot = [sym for _, _, sym in picks[:max(1, n)]]

    if hot:
        os.environ["SYMBOLS"] = ",".join(hot)   # 스크립트 전역에서 바로 사용
        print(f"[HOT] selected {len(hot)} symbols: {hot} (min_turnover={min_turnover_usdt:.0f} USDT)")
    else:
        print("[HOT] no symbols passed the filters.")
    return hot
# --- replace this whole function ---
def _clear_runtime_state():
    """Reset runtime dicts safely even if SYMBOLS changed at runtime."""
    # prune old keys that are no longer in SYMBOLS
    keep = set(SYMBOLS)
    for d in (ENTRY_HISTORY, COOLDOWN_UNTIL, POSITION_DEADLINE, STATE, CB_CONF_BUF, META_TRADES):
        for k in list(d.keys()):
            if isinstance(k, tuple):  # STATE may use (sym,side) in live code; paper uses sym
                if k[0] not in keep:
                    d.pop(k, None)
            else:
                if k not in keep:
                    d.pop(k, None)

    # ensure all current symbols exist
    for s in SYMBOLS:
        ENTRY_HISTORY[s] = collections.deque(maxlen=max(1, N_CONSEC_SIGNALS))
        COOLDOWN_UNTIL[s] = 0.0
        POSITION_DEADLINE.pop(s, None)
        STATE.pop(s, None)
        CB_CONF_BUF.setdefault(s, deque(maxlen=max(50, CB_Q_WIN))).clear()
        META_TRADES.setdefault(s, deque(maxlen=max(5, META_WIN))).clear()

    global RISK_CONSEC_LOSS
    RISK_CONSEC_LOSS = 0

def sync_symbols_runtime():
    _clear_runtime_state()


def _exec_taker_price(side: str, bid: float, ask: float, slip_bps: float = SLIPPAGE_BPS_TAKER) -> float:
    s = 1.0 + slip_bps/1e4
    if side == "Buy":
        return ask * s
    else:
        return bid / s  # Sell은 bid에서 미끄러짐
'''
SYMBOLS = find_hot_symbols(n=5, min_turnover_usdt=100_000_000.0)  # 원하는 하한으로 조정
# hot picker 실행 직후
hot = find_hot_symbols(n=5, min_turnover_usdt=1e8)
if hot:
    SYMBOLS[:] = hot  # 또는 SYMBOLS = hot
    sync_symbols_runtime()
'''

def main():
    print_runtime_header()
    _log_equity(get_wallet_equity())
    _log_equity_mtm(get_equity_mtm())

    if DIAG:
        for s in SYMBOLS:
            _whynot(s)
        return

    while True:
        if SCAN_PARALLEL:
            with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
                ex.map(try_enter, SYMBOLS)
        else:
            for s in SYMBOLS:
                try_enter(s)
        monitor_loop(1.0, 2)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
