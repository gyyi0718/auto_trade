# -*- coding: utf-8 -*-
"""
Bybit(USDT-Perp) 시세 기반 동시틱(동일 시각) 페이퍼 트레이딩 + 콘솔 로그
- 시세: Bybit v5 public REST (category=linear)
- 동기화: TickBus가 1초마다 스냅샷 생성 → 모든 시나리오가 같은 tick_id 처리
- 전략: three_model_consensus(symbol) 있으면 사용. 없으면 1m MA(20/60) 교차
- 체결 가정: 테이커 수수료·슬리피지 적용
- 리스크: TP1/TP2/TP3, TP2 후 BE, 이후 트레일 SL, 보유시간 제한
- 결과: 시나리오별 CSV ./logs_sync/, 콘솔 로그 실시간 출력
"""
import os, time, math, csv, threading, requests, signal, torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
os.environ["LIGHTNING_LOG_LEVEL"] = "ERROR"  # Lightning 정보로그 숨김
from lightning.pytorch import Trainer

# =========================
# CONFIG
# =========================
SYMBOLS = [
    "ETHUSDT","BTCUSDT","SOLUSDT"
]

# 시나리오: 레버리지 × 시작 소지금
LEVERAGE_LIST   = [10, 25, 100]
START_CASH_LIST = [100.0, 500.0, 1000.0]

ENTRY_MODE = os.getenv("ENTRY_MODE", "model")  # model | inverse | random
MAX_OPEN   = 6         # 시나리오별 동시 보유 심볼 수 상한
RISK_PCT_OF_EQUITY = 0.30
ENTRY_PORTION      = 0.40
MIN_NOTIONAL_USDT  = 10.0
MAX_NOTIONAL_ABS   = 2000.0
MAX_NOTIONAL_PCT_OF_EQUITY = 20.0  # equity * 20%

# 체결 가정
TAKER_FEE    = 0.0006
SLIP_BPS_IN  = 1.0     # 진입 슬리피지
SLIP_BPS_OUT = 1.0     # 청산 슬리피지

# TP/SL
TP1_BPS, TP2_BPS, TP3_BPS = 50.0, 100.0, 200.0  # 0.50%, 1.00%, 2.00%
TP1_RATIO, TP2_RATIO = 0.40, 0.35
BE_AFTER_TIER = 2
BE_EPS_BPS    = 2.0
TRAIL_BPS     = 50.0     # 0.50%
TRAIL_AFTER_TIER = 2
SL_ROI_PCT    = 0.01     # 1% 손절
MAX_HOLD_SEC  = 3600
COOLDOWN_SEC  = 10

# 마켓데이터
TICK_INTERVAL_SEC   = 1.0   # 1초 주기
USE_MARK_PRICE_FOR_RISK = True
MAX_MARK_FAIR_DIFF  = 0.01  # mid vs mark 괴리 1% 넘으면 진입 회피

# 로그 출력 옵션
VERBOSE              = True
HEARTBEAT_SEC        = 5.0   # 마켓데이터 하트비트 간격
WORKER_STATUS_EVERY  = 5     # n틱마다 워커 상태 출력

# 파일
OUT_DIR = "./logs_sync"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Bybit v5 Public REST (quotes only)
# =========================
BYBIT_API = "https://api.bybit.com"

USE_GPU = torch.cuda.is_available()
PL_TRAINER = Trainer(
    accelerator="gpu" if USE_GPU else "cpu",
    devices=1,
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=False,
    num_sanity_val_steps=0,
)



def _get(path: str, params: dict=None, retries: int=3):
    url = BYBIT_API + path
    params = params or {}
    last_err = None
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, timeout=5)
            if r.status_code == 200:
                j = r.json()
                # Bybit v5: retCode == 0 이 성공
                if isinstance(j, dict) and j.get("retCode") == 0:
                    return j.get("result")
                last_err = f"retCode={j.get('retCode')} retMsg={j.get('retMsg')}"
            else:
                last_err = f"HTTP {r.status_code} {r.text}"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.1)
    raise RuntimeError(f"HTTP GET failed: {path} {params} -> {last_err}")

def _iv_map_1m(iv: str) -> str:
    # 입력 "1m" → Bybit interval "1"
    iv = str(iv).lower().strip()
    return {"1m":"1","3m":"3","5m":"5","15m":"15","30m":"30",
            "1h":"60","2h":"120","4h":"240","1d":"D"}.get(iv, "1")

def get_all_book_tickers() -> Dict[str, Tuple[float,float,float,float]]:
    """
    {symbol: (bid, ask, mid, mark)}. 우리 유니버스만 필터.
    /v5/market/tickers?category=linear
    """
    res = _get("/v5/market/tickers", {"category":"linear"})
    items = (res or {}).get("list", []) if isinstance(res, dict) else []
    out = {}
    u = set(SYMBOLS)
    for it in items:
        sym = it.get("symbol")
        if sym not in u:
            continue
        # bid/ask는 문자열로 옴
        bid = float(it.get("bid1Price") or 0)
        ask = float(it.get("ask1Price") or 0)
        last = float(it.get("lastPrice") or 0)
        mark = float(it.get("markPrice") or 0)
        if bid>0 and ask>0:
            mid = (bid+ask)/2.0
        elif last>0:
            mid = last
            if bid==0: bid = last
            if ask==0: ask = last
        else:
            continue
        out[sym] = (bid, ask, mid, mark)
    return out

def get_mark_price(symbol: str) -> float:
    res = _get("/v5/market/tickers", {"category":"linear", "symbol":symbol})
    items = (res or {}).get("list", [])
    if not items:
        return 0.0
    return float(items[0].get("markPrice") or 0.0)

def get_klines(symbol: str, interval: str="1m", limit: int=120) -> pd.DataFrame:
    iv = _iv_map_1m(interval)
    res = _get("/v5/market/kline", {"category":"linear","symbol":symbol,"interval":iv,"limit":int(min(1500,limit))})
    rows = (res or {}).get("list", [])
    if not rows:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    # Bybit v5 kline row: [start, open, high, low, close, volume, turnover]
    df = pd.DataFrame(rows, columns=["start","open","high","low","close","volume","turnover"])
    df["timestamp"] = pd.to_datetime(df["start"].astype(np.int64), unit="ms")
    for c in ("open","high","low","close","volume"):
        df[c] = df[c].astype(float)
    return df[["timestamp","open","high","low","close","volume"]].sort_values("timestamp").reset_index(drop=True)

# =========================
# Tick bus (synchronized)
# =========================
class TickBus:
    def __init__(self):
        self._lock = threading.Lock()
        self._evt  = threading.Event()
        self.snapshot: Dict[str, Tuple[float,float,float,float]] = {}
        self.tick_id = 0
        self.ts = 0.0
        self._stop = False
        self._last_hb = 0.0

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        # 연결 점검 1회
        try:
            snap = get_all_book_tickers()
            print(f"[CHECK] connectivity ok. symbols={len(snap)}")
        except Exception as e:
            print(f"[CHECK][ERR] {e}")

    def stop(self):
        self._stop = True

    def wait_tick(self, last_seen: int, timeout: float=5.0) -> Optional[int]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.tick_id > last_seen:
                return self.tick_id
            self._evt.wait(0.05)
        return None

    def _run(self):
        while not self._stop:
            try:
                snap = get_all_book_tickers()
                if snap:
                    with self._lock:
                        self.snapshot = snap
                        self.ts = time.time()
                        self.tick_id += 1
                    self._evt.set()
                    self._evt.clear()
                    if VERBOSE and (self.ts - self._last_hb) >= HEARTBEAT_SEC:
                        self._last_hb = self.ts
                        ex = {k:v for k,v in snap.items() if k in ("BTCUSDT","ETHUSDT")}
                        ex_str = ", ".join(f"{k} mid={v[2]:.2f} mark={v[3]:.2f}" for k,v in ex.items())
                        print(f"[MD] tick={self.tick_id} n={len(snap)} {ex_str}")
                else:
                    if VERBOSE:
                        print("[MD][WARN] empty snapshot")
            except Exception as e:
                print("[MD][ERR]", e)
            time.sleep(TICK_INTERVAL_SEC)

    def get(self) -> Tuple[int, float, Dict[str, Tuple[float,float,float,float]]]:
        with self._lock:
            return self.tick_id, self.ts, dict(self.snapshot)

# =========================
# Utility
# =========================
def _opp(side: str) -> str:
    return "Sell" if side=="Buy" else "Buy"

def _roi_pct(side: str, entry: float, px: float) -> float:
    return (px/entry - 1.0) if side=="Buy" else (entry/px - 1.0)

def _round_down_qty(q: float, step: float) -> float:
    if step<=0: return q
    return math.floor(q/step)*step

# 심볼 규격 대략치(페이퍼용). 필요시 수정.
SYMBOL_RULES = {
    "BTCUSDT": {"tick":0.5,   "step":0.001, "min":0.001},
    "ETHUSDT": {"tick":0.05,  "step":0.001, "min":0.001},
    "SOLUSDT": {"tick":0.001, "step":0.01,  "min":0.01},
    "XRPUSDT": {"tick":0.0001,"step":1.0,   "min":1.0},
    "DOGEUSDT":{"tick":0.00001,"step":1.0,  "min":1.0},
    "LINKUSDT":{"tick":0.0005,"step":0.01,  "min":0.01},
    "ADAUSDT": {"tick":0.0001,"step":1.0,   "min":1.0},
    "SUIUSDT": {"tick":0.0001,"step":1.0,   "min":1.0},
    "1000PEPEUSDT":{"tick":0.00001,"step":1.0,"min":1.0},
    "MNTUSDT": {"tick":0.0001,"step":1.0,   "min":1.0},
}
DEFAULT_RULE = {"tick":0.0001,"step":0.001,"min":0.001}
def rule_of(sym:str): return SYMBOL_RULES.get(sym, DEFAULT_RULE)

# =========================
# Strategy
# =========================
# =========================
# Strategy (캐시 적용)
# =========================
def _ma_side(symbol: str) -> Tuple[Optional[str], float]:
    """
    1분 MA(20/60) 교차. 심볼별 결과를 TTL 동안 캐시해 API 남발 방지.
    """
    # 함수 내부 정적 캐시
    if not hasattr(_ma_side, "_cache"):
        _ma_side._cache = {}           # type: ignore[attr-defined]
        _ma_side._lock = threading.Lock()  # type: ignore[attr-defined]
        _ma_side._ttl  = 15.0          # type: ignore[attr-defined]  # 초

    now = time.time()
    with _ma_side._lock:  # type: ignore[attr-defined]
        rec = _ma_side._cache.get(symbol)  # type: ignore[attr-defined]
        if rec and (now - rec["ts"] < _ma_side._ttl):  # type: ignore[attr-defined]
            return rec["val"]

    try:
        df = get_klines(symbol, "1m", 120)
        if len(df) < 60:
            side_conf = (None, 0.0)
        else:
            closes = df["close"].values
            ma20 = float(np.mean(closes[-20:]))
            ma60 = float(np.mean(closes[-60:]))
            if ma20 > ma60:
                side_conf = ("Buy", 0.60)
            elif ma20 < ma60:
                side_conf = ("Sell", 0.60)
            else:
                side_conf = (None, 0.0)
    except Exception:
        side_conf = (None, 0.0)

    with _ma_side._lock:  # type: ignore[attr-defined]
        _ma_side._cache[symbol] = {"ts": now, "val": side_conf}  # type: ignore[attr-defined]
    return side_conf


def choose_entry(symbol: str) -> Tuple[Optional[str], float]:
    """
    1순위: three_model_consensus(symbol[, trainer=PL_TRAINER]) 사용.
    2순위: MA 교차.
    """
    fn = globals().get("three_model_consensus")
    if callable(fn):
        try:
            # three_model_consensus가 trainer 인자를 받는 경우
            side_cons, confs, _, _ = fn(symbol, trainer=PL_TRAINER)  # type: ignore
        except TypeError:
            # trainer 인자를 안 받는 경우
            side_cons, confs, _, _ = fn(symbol)  # type: ignore
        except Exception:
            side_cons, confs = None, None

        if side_cons in ("Buy", "Sell"):
            conf_list = [c for c in (confs or []) if isinstance(c, (int, float))]
            conf = float(max(conf_list or [0.6]))
            return side_cons, conf

    return _ma_side(symbol)

def choose_entry(symbol: str) -> Tuple[Optional[str], float]:
    try:
        side_cons, confs, _, _ = globals()["three_model_consensus"](symbol)  # type: ignore
        if side_cons in ("Buy","Sell"):
            conf = float(max([c for c in (confs or []) if isinstance(c,(int,float))] or [0.6]))
            return side_cons, conf
    except Exception:
        pass
    return _ma_side(symbol)

def apply_mode(side: Optional[str]) -> Optional[str]:
    if side not in ("Buy","Sell"): return None
    m = (ENTRY_MODE or "model").lower()
    if m=="model": return side
    if m=="inverse": return _opp(side)
    if m=="random": return "Buy" if (time.time()*1000)%2<1 else "Sell"
    return side

# =========================
# Broker (paper)
# =========================
class PaperBroker:
    def __init__(self, starting_cash: float, leverage: int, tag: str):
        self.cash = float(starting_cash)
        self.lev  = int(leverage)
        self.tag  = tag
        self.pos: Dict[str, dict] = {}
        self.cool: Dict[str, float] = defaultdict(float)
        self.last_eq_ts = 0.0
        self.trades_csv = os.path.join(OUT_DIR, f"trades_{tag}_binance.csv")
        self.equity_csv = os.path.join(OUT_DIR, f"equity_{tag}_binance.csv")
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.trades_csv):
            with open(self.trades_csv,"w",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow(["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","pnl","roi","cash","eq","hold"])
        if not os.path.exists(self.equity_csv):
            with open(self.equity_csv,"w",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow(["ts","equity","cash","upnl"])

    def _wtrade(self, row: dict):
        with open(self.trades_csv,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(k,"") for k in ["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","pnl","roi","cash","eq","hold"]])
        if VERBOSE:
            ev=row.get("event",""); sym=row.get("symbol","")
            print(f"[{self.tag}][{ev}] {sym} side={row.get('side','')} qty={row.get('qty','')} entry={row.get('entry','')} exit={row.get('exit','')} pnl={row.get('pnl','')}")

    def _weq(self, ts: float, eq: float, upnl: float):
        if time.time() - self.last_eq_ts < 3.0: return
        with open(self.equity_csv,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([int(ts), f"{eq:.6f}", f"{self.cash:.6f}", f"{upnl:+.6f}"])
        self.last_eq_ts = time.time()

    def can_open(self) -> bool:
        return len(self.pos) < MAX_OPEN

    def equity(self, mids: Dict[str,float]) -> Tuple[float, float]:
        up = 0.0
        for sym, p in self.pos.items():
            mid = mids.get(sym)
            if mid is None:
                continue
            if p["side"]=="Buy":
                up += p["qty"]*(mid - p["entry"])
            else:
                up += p["qty"]*(p["entry"] - mid)
        return self.cash + up, up

    def try_open(self, sym: str, side: str, mid: float, ts: float):
        if sym in self.pos: return
        if not self.can_open(): return

        # 마지막 미드 사용해 equity 정확히 산출
        mids_ref = getattr(self, "_last_mids", {})
        eq, _ = self.equity(mids_ref)

        free_cash = self.cash
        use_cap = eq * (RISK_PCT_OF_EQUITY / 1.0)
        use_cash = min(free_cash * ENTRY_PORTION, use_cap)
        if use_cash <= 0 or mid <= 0: return

        notional_raw = use_cash * self.lev
        notional_cap = min(MAX_NOTIONAL_ABS, eq * (MAX_NOTIONAL_PCT_OF_EQUITY / 100.0))
        notional = max(MIN_NOTIONAL_USDT, min(notional_raw, notional_cap))

        rule = rule_of(sym)
        slip = SLIP_BPS_IN / 10000.0
        entry = mid * (1.0 + slip if side == "Buy" else 1.0 - slip)
        qty = notional / max(entry, 1e-12)
        qty = _round_down_qty(qty, rule["step"])
        if qty < rule["min"]:
            qty = rule["min"]

        tp1 = entry * (1.0 + TP1_BPS / 10000.0) if side == "Buy" else entry * (1.0 - TP1_BPS / 10000.0)
        tp2 = entry * (1.0 + TP2_BPS / 10000.0) if side == "Buy" else entry * (1.0 - TP2_BPS / 10000.0)
        tp3 = entry * (1.0 + TP3_BPS / 10000.0) if side == "Buy" else entry * (1.0 - TP3_BPS / 10000.0)
        sl = entry * (1.0 - SL_ROI_PCT) if side == "Buy" else entry * (1.0 + SL_ROI_PCT)

        self.cash -= notional * TAKER_FEE  # 진입 수수료
        self.pos[sym] = {
            "side": side, "qty": qty, "entry": entry, "tp": [tp1, tp2, tp3],
            "tp_done": [False, False, False], "tp_ratio": [TP1_RATIO, TP2_RATIO, 1.0],
            "sl": sl, "be": False, "entry_ts": ts
        }
        eq_now, _ = self.equity({sym: mid})
        self._wtrade(
            {"ts": int(ts), "event": "ENTRY", "symbol": sym, "side": side, "qty": f"{qty:.8f}", "entry": f"{entry:.6f}",
             "exit": "",
             "tp1": f"{tp1:.6f}", "tp2": f"{tp2:.6f}", "tp3": f"{tp3:.6f}", "sl": f"{sl:.6f}",
             "pnl": "", "roi": "", "cash": f"{self.cash:.6f}", "eq": f"{eq_now:.6f}", "hold": ""})

    def _partial_exit(self, sym: str, px: float, ratio: float, reason: str, ts: float):
        p = self.pos[sym]
        side, entry, qty0 = p["side"], p["entry"], p["qty"]
        close_qty = max(0.0, min(qty0, qty0*ratio))
        if close_qty <= 0: return
        slip = SLIP_BPS_OUT/10000.0
        exit_px = px*(1.0 - slip) if side=="Buy" else px*(1.0 + slip)

        pnl = close_qty*((exit_px-entry) if side=="Buy" else (entry-exit_px))
        fee = close_qty*exit_px*TAKER_FEE
        pnl_after = pnl - fee
        self.cash += pnl_after
        p["qty"] = float(qty0 - close_qty)

        hold = max(0.0, ts - p["entry_ts"])
        eq_now,_ = self.equity({sym:px})
        self._wtrade({"ts":int(ts),"event":"EXIT_PARTIAL","symbol":sym,"side":side,"qty":f"{close_qty:.8f}",
                      "entry":f"{entry:.6f}","exit":f"{exit_px:.6f}",
                      "tp1":"","tp2":"","tp3":"","sl":f"{p.get('sl',0):.6f}",
                      "pnl":f"{pnl_after:.6f}","roi":f"{_roi_pct(side,entry,exit_px):.6f}",
                      "cash":f"{self.cash:.6f}","eq":f"{eq_now:.6f}","hold":f"{hold:.2f}"})

        if p["qty"] <= 0:
            del self.pos[sym]

    def close_all(self, sym: str, px: float, reason: str, ts: float):
        if sym not in self.pos: return
        self._partial_exit(sym, px, 1.0, reason, ts)

    # =========================
    # PaperBroker (정확한 equity 산출을 위해 on_tick/try_open 수정)
    # =========================
    def on_tick(self, tick_ts: float, mids: Dict[str, float], bids: Dict[str, float], asks: Dict[str, float]):
        # 마지막 미드 저장 → try_open에서 정확한 equity 사용
        self._last_mids = mids  # 동적 속성
        eq, up = self.equity(mids)
        self._weq(tick_ts, eq, up)

        to_close = []
        for sym, p in list(self.pos.items()):
            if sym not in mids:
                continue
            mid = mids[sym];
            bid = bids[sym];
            ask = asks[sym]
            side = p["side"];
            qty = p["qty"];
            entry = p["entry"]

            # TP hit
            for i, tp_px in enumerate(p["tp"]):
                if p["tp_done"][i] or qty <= 0: continue
                hit = (mid >= tp_px) if side == "Buy" else (mid <= tp_px)
                if hit:
                    px_exec = bid if side == "Buy" else ask
                    self._partial_exit(sym, px_exec, p["tp_ratio"][i], f"TP{i + 1}", tick_ts)
                    p = self.pos.get(sym)
                    if not p: break
                    p["tp_done"][i] = True
                    qty = p["qty"]
                    if (i + 1) >= BE_AFTER_TIER and not p["be"]:
                        be_eps = BE_EPS_BPS / 10000.0
                        p["sl"] = entry * (1.0 + be_eps) if side == "Buy" else entry * (1.0 - be_eps)
                        p["be"] = True
        if hasattr(self, "_last_mids"):
            mids_ref = self._last_mids  # type: ignore[attr-defined]
        else:
            mids_ref = {}

        # trailing / SL / time stop
        for sym, p in list(self.pos.items()):
            if sym not in mids:
                continue
            mid = mids[sym];
            bid = bids[sym];
            ask = asks[sym]
            side = p["side"];
            entry = p["entry"]

            tiers_done = sum(1 for x in p["tp_done"] if x)
            if tiers_done >= TRAIL_AFTER_TIER:
                if side == "Buy":
                    trail = mid * (1.0 - TRAIL_BPS / 10000.0)
                    if p["be"]:
                        trail = max(trail, entry * (1.0 + BE_EPS_BPS / 10000.0))
                    if trail > p["sl"]: p["sl"] = trail
                else:
                    trail = mid * (1.0 + TRAIL_BPS / 10000.0)
                    if p["be"]:
                        trail = min(trail, entry * (1.0 - BE_EPS_BPS / 10000.0))
                    if p["sl"] == 0 or trail < p["sl"]: p["sl"] = trail

            sl_now = p.get("sl", 0.0) or 0.0
            if sl_now > 0:
                if (side == "Buy" and bid <= sl_now) or (side == "Sell" and ask >= sl_now):
                    px_exec = bid if side == "Buy" else ask
                    to_close.append((sym, px_exec, "SL"))

            if MAX_HOLD_SEC and (tick_ts - p["entry_ts"]) >= MAX_HOLD_SEC:
                to_close.append((sym, mid, "TIMEOUT"))

        for sym, px, reason in to_close:
            self.close_all(sym, px, reason, tick_ts)
            self.cool[sym] = time.time() + COOLDOWN_SEC


# =========================
# Worker (scenario)
# =========================
class ScenarioWorker(threading.Thread):
    def __init__(self, bus: TickBus, leverage: int, start_cash: float):
        tag = f"lev{leverage}_cash{int(start_cash)}"
        super().__init__(daemon=True, name=f"Worker-{tag}")
        self.bus = bus
        self.tag = tag
        self.broker = PaperBroker(start_cash, leverage, tag)
        self._stop = False
        self._last_tick = 0

    def stop(self): self._stop = True

    def run(self):
        printed_zero = False
        while not self._stop:
            tid = self.bus.wait_tick(self._last_tick, timeout=5.0)
            if tid is None:
                if VERBOSE and not printed_zero:
                    print(f"[{self.tag}] 대기 중… tick 없음")
                    printed_zero = True
                continue
            printed_zero = False
            self._last_tick = tid
            tid2, ts, snap = self.bus.get()
            if tid2 != tid:
                continue

            bids={}; asks={}; mids={}; marks={}
            for s,(b,a,m,mark) in snap.items():
                bids[s]=b; asks[s]=a; mids[s]=m; marks[s]=mark

            # 1) 동일 틱에서 관리
            self.broker.on_tick(ts, mids, bids, asks)

            # 2) 동일 틱에서 진입 판단
            opened = 0
            for sym in SYMBOLS:
                if sym not in mids: continue
                if self.broker.cool.get(sym,0.0) > time.time(): continue
                if sym in self.broker.pos: continue
                if not self.broker.can_open(): break

                side, conf = choose_entry(sym)
                side = apply_mode(side)
                if side not in ("Buy","Sell") or conf < 0.55:
                    continue

                if USE_MARK_PRICE_FOR_RISK:
                    try:
                        mark = marks.get(sym) or get_mark_price(sym)
                        if mark>0 and abs(mark - mids[sym])/max(mids[sym],1e-9) > MAX_MARK_FAIR_DIFF:
                            continue
                    except Exception as e:
                        if VERBOSE:
                            print(f"[{self.tag}][WARN] mark fetch {sym}: {e}")

                self.broker.try_open(sym, side, mids[sym], ts)
                opened += 1

            if VERBOSE and (tid % WORKER_STATUS_EVERY == 0):
                pos_cnt = len(self.broker.pos)
                eq,_ = self.broker.equity(mids)
                print(f"[{self.tag}] tick={tid} pos={pos_cnt} opened={opened} eq={eq:.2f}")

# =========================
# Main
# =========================
def main():
    print(f"[START] {len(LEVERAGE_LIST)*len(START_CASH_LIST)} scenarios | symbols={len(SYMBOLS)} | interval={TICK_INTERVAL_SEC}s")
    print(f"ENTRY_MODE={ENTRY_MODE}  TP={TP1_BPS}/{TP2_BPS}/{TP3_BPS}bps  SL={SL_ROI_PCT*100:.2f}%")

    bus = TickBus(); bus.start()

    workers: List[ScenarioWorker] = []
    for lev in LEVERAGE_LIST:
        for cash in START_CASH_LIST:
            workers.append(ScenarioWorker(bus, lev, cash))

    for w in workers: w.start()

    stop_flag = threading.Event()
    def _sigint(_sig,_frm): stop_flag.set()
    signal.signal(signal.SIGINT, _sigint)

    try:
        while not stop_flag.is_set():
            time.sleep(0.5)
    finally:
        for w in workers: w.stop()
        bus.stop()
        print("[STOP] done")

if __name__ == "__main__":
    main()
