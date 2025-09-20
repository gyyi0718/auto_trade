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
import os, time, math, csv, threading, requests, signal
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
SYMBOLS = [
    "ETHUSDT","BTCUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
    "BNBUSDT","ADAUSDT","LINKUSDT","UNIUSDT","TRXUSDT",
    "LTCUSDT","MNTUSDT","SUIUSDT","1000PEPEUSDT",
    "XLMUSDT","ARBUSDT","APTUSDT","OPUSDT","AVAXUSDT"
]

# 시나리오: 레버리지 × 시작 소지금
LEVERAGE_LIST   = [10, 25, 100]
START_CASH_LIST = [100.0,1000.0,10000.0]

ENTRY_MODE = os.getenv("ENTRY_MODE", "inverse")  # model | inverse | random
EXEC_MODE = "taker"  # "ideal" | "taker"
MAKER_FEE  = float(os.getenv("MAKER_FEE", "0.0"))

MAX_OPEN   = 10         # 시나리오별 동시 보유 심볼 수 상한
RISK_PCT_OF_EQUITY = 0.30
ENTRY_PORTION      = 1.0
MIN_NOTIONAL_USDT  = 10.0
MAX_NOTIONAL_ABS   = 2000.0
MAX_NOTIONAL_PCT_OF_EQUITY = 20.0  # equity * 20%
MAX_SPREAD_BPS = 4.0         # 스프레드가 0.04% 초과면 진입 금지

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

SLIP_BPS_BASE_IN  = 2.0      # 진입 기본 0.02%
SLIP_BPS_BASE_OUT = 2.0      # 청산 기본 0.02%


PS_BASE_IN  = 2.0      # 진입 기본 0.02%
SLIP_BPS_BASE_OUT = 2.0      # 청산 기본 0.02%
# 로그 출력 옵션
VERBOSE              = True
HEARTBEAT_SEC        = 5.0   # 마켓데이터 하트비트 간격
WORKER_STATUS_EVERY  = 5     # n틱마다 워커 상태 출력

# === CONFIG ===
ENTRY_EQUITY_PCT   = 0.20   # 매 진입시 자본의 20%를 노셔널로 사용
REENTRY_ON_DIP     = True   # 재진입 사용
REENTRY_PCT        = 0.20   # 마지막 청산가 대비 20% 유리하게 움직이면 재진입
REENTRY_SIZE_PCT   = 0.20   # 재진입도 자본의 20% 노셔널

# 파일
OUT_DIR = "./logs_sync"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Bybit v5 Public REST (quotes only)
# =========================
BYBIT_API = "https://api.bybit.com"

import numpy as np

def ev_long_short_from_samples(p0: float,
                               samples: np.ndarray,
                               tp_bps=(50.0,100.0,200.0),
                               sl_frac=0.01,
                               fee_taker=0.0006,
                               slip_bps=10.0):
    """
    p0: 진입가
    samples: (M,H) 예측가격 경로 (DeepAR 샘플)
    tp_bps: (tp1,tp2,tp3) 중 최소 TP만 사용해 EV 보수적 산정(필요시 바꿔도 됨)
    sl_frac: SL 비율(예: 0.01=1%)
    fee_taker: 주문 왕복 총 수수료의 절반(편익/비용 양쪽 적용 위해 왕복은 2배 반영)
    slip_bps: 체결 슬리피지 bps(왕복 기준의 절반으로 여기서 입력. 최종 2배 곱해짐)
    return: (EV_long, EV_short, p_long_win, p_short_win)
    """
    if samples is None or len(samples)==0:
        return None, None, 0.0, 0.0

    M, H = samples.shape
    tp = min(tp_bps)/1e4  # 예: 50bps -> 0.005
    sl = float(sl_frac)

    # 비용(bps→수익률) : 수수료+슬립 각각 왕복 적용
    cost = 2*fee_taker + 2*(slip_bps/1e4)

    LW = LL = SW = SL = 0
    p_up = p0*(1+tp); p_dn = p0*(1-sl)
    p_up_s = p0*(1+sl); p_dn_s = p0*(1-tp)

    for path in samples:
        # Long: TP 먼저? SL 먼저?
        hit_tp = next((i for i,p in enumerate(path) if p>=p_up), None)
        hit_sl = next((i for i,p in enumerate(path) if p<=p_dn), None)
        if hit_tp is not None and (hit_sl is None or hit_tp < hit_sl): LW += 1
        elif hit_sl is not None and (hit_tp is None or hit_sl < hit_tp): LL += 1

        # Short: TP=하락 tp, SL=상승 sl
        hit_tp_s = next((i for i,p in enumerate(path) if p<=p_dn_s), None)
        hit_sl_s = next((i for i,p in enumerate(path) if p>=p_up_s), None)
        if hit_tp_s is not None and (hit_sl_s is None or hit_tp_s < hit_sl_s): SW += 1
        elif hit_sl_s is not None and (hit_tp_s is None or hit_sl_s < hit_tp_s): SL += 1

    pLW = LW/max(M,1); pLL = LL/max(M,1)
    pSW = SW/max(M,1); pSL = SL/max(M,1)

    EV_long  = pLW*tp - pLL*sl - cost
    EV_short = pSW*tp - pSL*sl - cost
    return EV_long, EV_short, pLW, pSW

def get_deepar_samples(symbol: str, H: int = 60, M: int = 200):
    """
    (M,H) 가격 경로 반환. 네 모델 추론 코드로 대체.
    없으면 None 리턴해서 EV 필터 패스.
    """
    try:
        # 예시) 최근 상태로부터 분포 샘플 → 가격 누적
        # mu, sigma = ...  # 너의 모델에서 얻기
        # eps = np.random.randn(M,H)
        # r = mu + sigma*eps                # Δlog-price
        # y0 = np.log(p0); y = y0 + r.cumsum(axis=1)
        # return np.exp(y)
        return None
    except:
        return None

def _dyn_slip_bps(notional_usdt: float, base_bps: float) -> float:
    # notional↑ -> slip↑, 과격하지 않게 sqrt 스케일
    return base_bps * (1.0 + 0.5 * math.sqrt(max(notional_usdt, 1.0) / 1000.0))

def _bps_spread(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0: return 1e9
    return 1e4 * (ask / bid - 1.0)
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
            #print(f"[CHECK] connectivity ok. symbols={len(snap)}")
        except Exception as e:
            #print(f"[CHECK][ERR] {e}")
            print("\t")
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
                        #print(f"[MD] tick={self.tick_id} n={len(snap)} {ex_str}")
                else:
                    if VERBOSE:
                        #print("[MD][WARN] empty snapshot")
                        print("\t")
            except Exception as e:
                #print("[MD][ERR]", e)
                print("\t")
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
def _ma_side(symbol: str) -> Tuple[Optional[str], float]:
    try:
        df = get_klines(symbol, "1m", 120)
        if len(df) < 60: return None, 0.0
        closes = df["close"].values
        ma20 = float(np.mean(closes[-20:]))
        ma60 = float(np.mean(closes[-60:]))
        if ma20>ma60: return "Buy", 0.60
        if ma20<ma60: return "Sell", 0.60
        return None, 0.0
    except Exception:
        return None, 0.0

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
        self.lev = int(leverage)
        self.tag = tag
        self.pos: Dict[str, dict] = {}
        self.cool: Dict[str, float] = defaultdict(float)
        self.last_eq_ts = 0.0
        self.last_exit: Dict[str, dict] = {}  # {sym: {"side": "Buy"/"Sell", "px": float, "ts": float}}

        self.trades_csv = os.path.join(OUT_DIR, f"trades_{tag}_bybit.csv")
        self.equity_csv = os.path.join(OUT_DIR, f"equity_{tag}_bybit.csv")
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
            #print(f"[{self.tag}][{ev}] {sym} side={row.get('side','')} qty={row.get('qty','')} entry={row.get('entry','')} exit={row.get('exit','')} pnl={row.get('pnl','')}")

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

    def try_open(self, sym: str, side: str, mid: float, ts: float, notional_override: float = None):
        if sym in self.pos or not self.can_open():
            return
        if mid <= 0:
            return

        eq, _ = self.equity({})
        free_cash = self.cash  # 미사용이지만 유지

        # 목표 노셔널: override가 없으면 자본×비율×레버리지
        if notional_override is not None:
            target_notional = float(notional_override)
        else:
            target_notional = eq * ENTRY_EQUITY_PCT * self.lev

        # 상·하한 캡에도 레버리지 반영
        cap_abs = MAX_NOTIONAL_ABS * self.lev
        cap_pct = eq * self.lev * (MAX_NOTIONAL_PCT_OF_EQUITY / 100.0)
        notional_cap = min(cap_abs, cap_pct)
        target_notional = max(MIN_NOTIONAL_USDT, min(target_notional, notional_cap))
        if target_notional <= 0:
            return

        rule = rule_of(sym)

        # 진입가 및 수수료
        if EXEC_MODE == "ideal":
            entry = mid
            fee_entry = 0.0
        else:
            slip = SLIP_BPS_IN / 10000.0
            entry = mid * (1.0 + slip) if side == "Buy" else mid * (1.0 - slip)
            fee_entry = target_notional * TAKER_FEE

        # 수량
        qty_raw = target_notional / max(entry, 1e-12)
        qty = _round_down_qty(qty_raw, rule["step"])
        if qty < rule["min"]:
            qty = rule["min"]

        # 캡 재확인
        max_qty_cap = notional_cap / max(entry, 1e-12)
        max_qty_cap = math.floor(max_qty_cap / rule["step"]) * rule["step"]
        if max_qty_cap > 0:
            qty = min(qty, max_qty_cap)
        if qty <= 0:
            return

        # TP/SL
        tp1 = entry * (1.0 + TP1_BPS / 10000.0) if side == "Buy" else entry * (1.0 - TP1_BPS / 10000.0)
        tp2 = entry * (1.0 + TP2_BPS / 10000.0) if side == "Buy" else entry * (1.0 - TP2_BPS / 10000.0)
        tp3 = entry * (1.0 + TP3_BPS / 10000.0) if side == "Buy" else entry * (1.0 - TP3_BPS / 10000.0)
        sl = entry * (1.0 - SL_ROI_PCT) if side == "Buy" else entry * (1.0 + SL_ROI_PCT)

        # 현금 반영
        self.cash -= fee_entry

        self.pos[sym] = {
            "side": side, "qty": qty, "entry": entry,
            "tp": [tp1, tp2, tp3], "tp_done": [False, False, False],
            "tp_ratio": [TP1_RATIO, TP2_RATIO, 1.0],
            "sl": sl, "be": False, "entry_ts": ts
        }
        eq_now, _ = self.equity({sym: mid})
        self._wtrade({
            "ts": int(ts), "event": "ENTRY", "symbol": sym, "side": side, "qty": f"{qty:.8f}",
            "entry": f"{entry:.6f}", "exit": "", "tp1": f"{tp1:.6f}", "tp2": f"{tp2:.6f}",
            "tp3": f"{tp3:.6f}", "sl": f"{sl:.6f}", "pnl": "", "roi": "",
            "cash": f"{self.cash:.6f}", "eq": f"{eq_now:.6f}", "hold": ""
        })

    def _partial_exit(self, sym: str, px_mid_bid_ask: Tuple[float, float, float], ratio: float, reason: str, ts: float):
        p = self.pos[sym]
        side, entry, qty0 = p["side"], p["entry"], p["qty"]
        mid, bid, ask = px_mid_bid_ask
        close_qty = max(0.0, min(qty0, qty0 * ratio))
        if close_qty <= 0:
            return

        if EXEC_MODE == "ideal":
            exit_px = mid
            fee = 0.0
        else:
            slip_out_bps = _dyn_slip_bps(close_qty * mid, SLIP_BPS_BASE_OUT)
            if side == "Buy":
                exit_px = bid * (1.0 - slip_out_bps / 1e4)
            else:
                exit_px = ask * (1.0 + slip_out_bps / 1e4)
            fee = close_qty * exit_px * TAKER_FEE

        pnl = close_qty * ((exit_px - entry) if side == "Buy" else (entry - exit_px))
        pnl_after = pnl - fee
        self.cash += pnl_after
        p["qty"] = float(qty0 - close_qty)

        hold = max(0.0, ts - p["entry_ts"])
        eq_now, _ = self.equity({sym: mid})
        self._wtrade({
            "ts": int(ts), "event": "EXIT_PARTIAL", "symbol": sym, "side": side,
            "qty": f"{close_qty:.8f}", "entry": f"{entry:.6f}", "exit": f"{exit_px:.6f}",
            "tp1": "", "tp2": "", "tp3": "", "sl": f"{p.get('sl', 0):.6f}",
            "pnl": f"{pnl_after:.6f}", "roi": f"{_roi_pct(side, entry, exit_px):.6f}",
            "cash": f"{self.cash:.6f}", "eq": f"{eq_now:.6f}", "hold": f"{hold:.2f}"
        })

        if p["qty"] <= 0:
            del self.pos[sym]
            # 재진입 트리거용 마지막 청산가 저장(옵션)
            self.last_exit[sym] = {"side": side, "px": exit_px, "ts": ts}

    def close_all(self, sym: str, px: float, reason: str, ts: float):
        if sym not in self.pos: return
        self._partial_exit(sym, px, 1.0, reason, ts)

    def on_tick(self, tick_ts: float, mids: Dict[str, float], bids: Dict[str, float], asks: Dict[str, float]):
        eq, upnl = self.equity(mids)
        self._weq(tick_ts, eq, upnl)
        to_close = []

        for sym, p in list(self.pos.items()):
            if sym not in mids:
                continue
            mid, bid, ask = mids[sym], bids[sym], asks[sym]
            side, entry = p["side"], p["entry"]
            qty = p["qty"]

            # TP 체커
            for i, tp_px in enumerate(p["tp"]):
                if p["tp_done"][i] or qty <= 0:
                    continue
                hit = (mid >= tp_px) if side == "Buy" else (mid <= tp_px)
                if hit:
                    self._partial_exit(sym, (mid, bid, ask), p["tp_ratio"][i], f"TP{i + 1}", tick_ts)
                    if sym not in self.pos:
                        break
                    p = self.pos[sym];
                    p["tp_done"][i] = True;
                    qty = p["qty"]
                    # BE 승격
                    if (i + 1) >= BE_AFTER_TIER and not p["be"]:
                        be_eps = BE_EPS_BPS / 1e4
                        p["sl"] = entry * (1.0 + be_eps) if side == "Buy" else entry * (1.0 - be_eps)
                        p["be"] = True

            if sym not in self.pos:
                continue

            # 트레일링 SL
            tiers_done = sum(1 for x in p["tp_done"] if x)
            if tiers_done >= TRAIL_AFTER_TIER:
                if side == "Buy":
                    trail = mid * (1.0 - TRAIL_BPS / 1e4)
                    if p["be"]: trail = max(trail, entry * (1.0 + BE_EPS_BPS / 1e4))
                    if trail > p["sl"]: p["sl"] = trail
                else:
                    trail = mid * (1.0 + TRAIL_BPS / 1e4)
                    if p["be"]: trail = min(trail, entry * (1.0 - BE_EPS_BPS / 1e4))
                    if p["sl"] == 0 or trail < p["sl"]: p["sl"] = trail

            # SL 트리거
            sl_now = float(p.get("sl") or 0.0)
            if sl_now > 0:
                if (side == "Buy" and (bid <= sl_now)) or (side == "Sell" and (ask >= sl_now)):
                    to_close.append((sym, (mid, bid, ask), "SL"))

            # 보유 시간 제한
            if MAX_HOLD_SEC and (tick_ts - p["entry_ts"]) >= MAX_HOLD_SEC:
                to_close.append((sym, (mid, bid, ask), "TIMEOUT"))

        for sym, px_pack, reason in to_close:
            # 전량 청산
            if sym in self.pos:
                self._partial_exit(sym, px_pack, 1.0, reason, tick_ts)
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
                    printed_zero = True
                continue
            printed_zero = False
            self._last_tick = tid

            tid2, ts, snap = self.bus.get()
            if tid2 != tid:
                continue

            bids = {};
            asks = {};
            mids = {};
            marks = {}
            for s, (b, a, m, mark) in snap.items():
                bids[s] = b;
                asks[s] = a;
                mids[s] = m;
                marks[s] = mark

            # 포지션 관리
            self.broker.on_tick(ts, mids, bids, asks)

            opened = 0
            for sym in SYMBOLS:
                if sym not in mids:
                    continue
                if self.broker.cool.get(sym, 0.0) > time.time():
                    continue
                if sym in self.broker.pos:
                    continue
                if not self.broker.can_open():
                    break

                # 재진입: 마지막 청산가 대비 20% 유리하면 같은 방향 재진입
                did_reenter = False
                if REENTRY_ON_DIP and sym in self.broker.last_exit:
                    re = self.broker.last_exit[sym]
                    last_side, last_px = re["side"], re["px"]
                    mid = mids[sym]
                    cond = (mid <= last_px * (1.0 - REENTRY_PCT)) if last_side == "Buy" else (
                                mid >= last_px * (1.0 + REENTRY_PCT))
                    if cond:
                        eq, _ = self.broker.equity(mids)
                        # 레버리지 반영한 재진입 노셔널과 캡
                        lev = self.broker.lev
                        cap_abs = MAX_NOTIONAL_ABS * lev
                        cap_pct = eq * lev * (MAX_NOTIONAL_PCT_OF_EQUITY / 100.0)
                        notional = max(
                            MIN_NOTIONAL_USDT,
                            min(eq * REENTRY_SIZE_PCT * lev, min(cap_abs, cap_pct))
                        )
                        self.broker.try_open(sym, last_side, mids[sym], ts, notional_override=notional)
                        opened += 1
                        did_reenter = True

                if did_reenter:
                    continue

                # 일반 엔트리
                # --- 교체본: EV 필터 추가 ---
                # 일반 엔트리
                side, conf = choose_entry(sym)
                side = apply_mode(side)
                if side not in ("Buy", "Sell") or conf < 0.55:
                    continue

                mid = mids[sym]
                # DeepAR 샘플에서 EV 계산(미구현이면 get_deepar_samples가 None을 리턴 → 원래 로직 그대로)
                try:
                    samples = get_deepar_samples(sym, H=60, M=200)
                except Exception:
                    samples = None

                EVL, EVS, pLW, pSW = ev_long_short_from_samples(
                    p0=mid,
                    samples=samples,
                    tp_bps=(float(TP1_BPS), float(TP2_BPS), float(TP3_BPS)),
                    sl_frac=float(SL_ROI_PCT),
                    fee_taker=float(TAKER_FEE),
                    slip_bps=float(SLIP_BPS_IN)
                )

                if EVL is not None:
                    # 기대값이 음수면 진입 건너뜀
                    if max(EVL, EVS) <= 0:
                        if VERBOSE:
                            print(f"[EV][SKIP] {sym} EVL={EVL:.6f} EVS={EVS:.6f} pLW={pLW:.2f} pSW={pSW:.2f}")
                        continue
                    # EV 우위 방향으로 강제
                    side = "Buy" if EVL >= EVS else "Sell"
                    # 보수 필터(선택): 승률 컷
                    if side == "Buy" and pLW < 0.55:
                        if VERBOSE:
                            print(f"[EV][CUT] {sym} long pLW={pLW:.2f} < 0.55")
                        continue
                    if side == "Sell" and pSW < 0.55:
                        if VERBOSE:
                            print(f"[EV][CUT] {sym} short pSW={pSW:.2f} < 0.55")
                        continue
                    if VERBOSE:
                        print(f"[EV][ENTER] {sym} side={side} EVL={EVL:.6f} EVS={EVS:.6f} pLW={pLW:.2f} pSW={pSW:.2f}")

                self.broker.try_open(sym, side, mid, ts)
                opened += 1

            if VERBOSE and (tid % WORKER_STATUS_EVERY == 0):
                pos_cnt = len(self.broker.pos)
                eq, _ = self.broker.equity(mids)
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
