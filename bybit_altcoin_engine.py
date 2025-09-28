# -*- coding: utf-8 -*-
"""
bybit_altcoin_engine.py
- settings.config 읽어서 동작
- trade_mode: off | paper | live
- 실험용 로직은 strategy_hooks.py 로 분리
"""

import os, time, csv, math, pathlib, warnings
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from configparser import ConfigParser

warnings.filterwarnings("ignore")

# ===== 로깅/CSV =====
LOG_DIR = pathlib.Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRADES_FIELDS = [
    "ts","event","symbol","side","qty",
    "entry_price","tp_price","sl_price",
    "order_id","tp_order_id","sl_order_id",
    "reason","pnl_usdt","roi","extra","mode"
]

def trade_csv_path(leverage: int, csv_save_mode: str, symbol: str="")->pathlib.Path:
    if csv_save_mode == "split" and symbol:
        p = LOG_DIR / f"trades_{symbol}.csv"
    else:
        p = LOG_DIR / f"live_trades_leverage{leverage}.csv"
    if not p.exists():
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(TRADES_FIELDS)
    return p

def equity_csv_path(leverage: int)->pathlib.Path:
    p = LOG_DIR / f"live_equity_leverage{leverage}.csv"
    if not p.exists():
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts","cash","upnl","equity"])
    return p

def write_trade_row(row: dict, leverage: int, csv_mode: str):
    path = trade_csv_path(leverage, csv_mode, row.get("symbol",""))
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            row.get("ts",""), row.get("event",""), row.get("symbol",""),
            row.get("side",""), row.get("qty",""),
            row.get("entry_price",""), row.get("tp_price",""), row.get("sl_price",""),
            row.get("order_id",""), row.get("tp_order_id",""), row.get("sl_order_id",""),
            row.get("reason",""), row.get("pnl_usdt",""), row.get("roi",""),
            row.get("extra",""), row.get("mode",""),
        ])

_last_eq_log_ts = 0.0
def log_equity_snapshot(get_equity_breakdown, leverage: int):
    global _last_eq_log_ts
    now = time.time()
    if now - _last_eq_log_ts < 3.0:
        return
    p = equity_csv_path(leverage)
    eq, cash, upnl, _ = get_equity_breakdown()
    with open(p, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([int(now), f"{cash:.6f}", f"{upnl:+.6f}", f"{eq:.6f}"])
    _last_eq_log_ts = now


# ===== 설정 로더 =====
class Settings:
    def __init__(self, path="settings.config"):
        self.cfg = ConfigParser()
        if os.path.exists(path):
            self.cfg.read(path, encoding="utf-8")
        else:
            raise FileNotFoundError(f"config not found: {path}")

        # ENV override helper
        def envs(name, default):
            v = os.getenv(name)
            return v if v not in (None, "") else self.cfg_get_str(*default)

        def envi(name, default):
            v = os.getenv(name)
            return int(v) if v not in (None, "") else self.cfg_get_int(*default)

        def envf(name, default):
            v = os.getenv(name)
            return float(v) if v not in (None, "") else self.cfg_get_float(*default)

        # bybit
        self.API_KEY    = envs("BYBIT_API_KEY", ("bybit","api_key"))
        self.API_SECRET = envs("BYBIT_API_SECRET", ("bybit","api_secret"))
        self.TESTNET    = envs("BYBIT_TESTNET", ("bybit","testnet")).lower() == "true"
        self.CATEGORY   = self.cfg_get_str("bybit","category","linear")
        self.LEVERAGE   = envi("LEVERAGE", ("bybit","leverage"))
        self.TAKER_FEE  = envf("TAKER_FEE", ("bybit","taker_fee"))

        # modes
        self.TRADE_MODE   = envs("TRADE_MODE", ("mode","trade_mode")) # off|paper|live
        self.MODEL_MODE   = envs("MODE", ("mode","model_mode"))       # model|inverse|random
        self.CSV_SAVE_MODE= self.cfg_get_str("mode","csv_save_mode","single")

        # strategy
        self.USE_DEEPAR = envs("USE_DEEPAR", ("strategy","use_deepar")) == "1"
        self.CKPT_PATH  = self.cfg_get_str("strategy","ckpt_path","models/multi_deepar_model.ckpt")
        self.SEQ_LEN    = self.cfg_get_int("strategy","seq_len",60)
        self.PRED_LEN   = self.cfg_get_int("strategy","pred_len",10)

        # risk
        self.RISK_PCT_OF_EQUITY = self.cfg_get_float("risk","risk_pct_of_equity",0.30)
        self.ENTRY_PORTION      = self.cfg_get_float("risk","entry_portion",0.40)
        self.MAX_OPEN           = self.cfg_get_int("risk","max_open",4)

        # guards
        self.SPREAD_REJECT_BPS  = self.cfg_get_float("quality_guards","spread_reject_bps",35.0)
        self.GAP_REJECT_BPS     = self.cfg_get_float("quality_guards","gap_reject_bps",400.0)

        # tp/sl
        self.TP1_BPS  = self.cfg_get_float("tp_sl","tp1_bps",15.0)
        self.TP2_BPS  = self.cfg_get_float("tp_sl","tp2_bps",50.0)
        self.TP3_BPS  = self.cfg_get_float("tp_sl","tp3_bps",100.0)
        self.TP1_RATIO= self.cfg_get_float("tp_sl","tp1_ratio",0.25)
        self.TP2_RATIO= self.cfg_get_float("tp_sl","tp2_ratio",0.35)
        self.BE_EPS_BPS     = self.cfg_get_float("tp_sl","be_eps_bps",2.0)
        self.BE_AFTER_TIER  = self.cfg_get_int("tp_sl","be_after_tier",2)
        self.TRAIL_BPS      = self.cfg_get_float("tp_sl","trail_bps",40.0)
        self.TRAIL_AFTER_TIER = self.cfg_get_int("tp_sl","trail_after_tier",1)
        self.STOP_LOSS_PCT  = self.cfg_get_float("tp_sl","stop_loss_pct",0.03)
        self.MAX_HOLD_SEC   = self.cfg_get_int("tp_sl","max_hold_sec",3600)
        self.COOLDOWN_SEC   = self.cfg_get_int("tp_sl","cooldown_sec",60)
        self.TAKE_PROFIT_USD = self.cfg_get_float("tp_sl","take_profit_usd",0.0)
        self.TAKE_PROFIT_REL_PCT = self.cfg_get_float("tp_sl","take_profit_rel_pct",0.005)

        # scoring
        self.CONF_MIN        = self.cfg_get_float("scoring","conf_min",0.75)
        self.PNL_MIN_USD     = self.cfg_get_float("scoring","pnl_min_usd",0.01)
        self.SHARPE_MIN      = self.cfg_get_float("scoring","sharpe_min",0.15)
        self.KELLY_CAP       = self.cfg_get_float("scoring","kelly_cap",1.0)
        self.KELLY_MAX_ON_CASH = self.cfg_get_float("scoring","kelly_max_on_cash",0.35)

        # symbols
        self.SYMBOLS = [s.strip() for s in self.cfg_get_str("symbols","list","").split(",") if s.strip()]
        self.EXCLUDE_SYMBOLS = [s.strip() for s in self.cfg_get_str("symbols","exclude","").split(",") if s.strip()]

        # windows (예: "09-12,13-16")
        self.ALLOW_WINDOWS = []
        w = self.cfg_get_str("windows","allow_local","").strip()
        if w:
            for rng in w.split(","):
                try:
                    a,b = rng.split("-")
                    self.ALLOW_WINDOWS.append((int(a),int(b)))
                except:
                    pass

    def cfg_get_str(self, section, key, default=None):
        try: return self.cfg.get(section, key)
        except: return default

    def cfg_get_int(self, section, key, default=None):
        try: return self.cfg.getint(section, key)
        except: return default

    def cfg_get_float(self, section, key, default=None):
        try: return self.cfg.getfloat(section, key)
        except: return default


# ===== 공통 유틸 =====
def D(x)->Decimal: return Decimal(str(x))

def round_step(x: Decimal, step: Decimal)->Decimal:
    if step <= 0: step = Decimal("0.001")
    return (x / step).to_integral_value(rounding=ROUND_DOWN) * step

def fnum(x, default=0.0)->float:
    try:
        if x is None: return float(default)
        s = str(x).strip().lower()
        if s in {"","nan","none","null","inf","-inf"}: return float(default)
        return float(s)
    except Exception:
        return float(default)


# ===== 브로커 인터페이스 =====
class Broker:
    def get_quote(self, symbol:str)->Tuple[float,float,float]:
        raise NotImplementedError
    def get_kline(self, symbol:str, limit:int=120)->pd.DataFrame:
        raise NotImplementedError
    def ensure_leverage(self, symbol:str, leverage:int):
        pass
    def place_market(self, symbol:str, side:str, use_usdt:float, leverage:int)->Tuple[Optional[str], Optional[Decimal], Optional[Decimal]]:
        raise NotImplementedError
    def reduce_market(self, symbol:str, exit_side:str, qty:Decimal)->Optional[str]:
        raise NotImplementedError
    def cancel_all(self, symbol:str):
        pass


# ===== 실거래 Bybit 브로커 =====
class RealBybitBroker(Broker):
    def __init__(self, api_key:str, api_secret:str, testnet:bool, category:str):
        from pybit.unified_trading import HTTP
        self.client = HTTP(api_key=api_key, api_secret=api_secret,
                           recv_window=300000, timeout=10, testnet=testnet)
        self.category = category

    def get_quote(self, symbol:str)->Tuple[float,float,float]:
        res = self.client.get_tickers(category=self.category, symbol=symbol)
        row = (res.get("result", {}) or {}).get("list", [{}])[0]
        bid = fnum(row.get("bid1Price"), 0.0)
        ask = fnum(row.get("ask1Price"), 0.0)
        last = fnum(row.get("lastPrice"), 0.0)
        if bid > 0 and ask > 0:
            return bid, ask, (bid+ask)/2.0
        p = last if last > 0 else self.get_last_price_fallback(symbol)
        return p, p, p

    def get_last_price_fallback(self, symbol:str)->float:
        res = self.client.get_tickers(category=self.category, symbol=symbol)
        row = (res.get("result", {}) or {}).get("list", [{}])[0]
        for k in ("lastPrice","markPrice","bid1Price"):
            p = fnum(row.get(k), 0.0)
            if p > 0: return p
        raise RuntimeError(f"{symbol} price query failed")

    def get_kline(self, symbol:str, limit:int=120)->pd.DataFrame:
        res = self.client.get_kline(category=self.category, symbol=symbol, interval="1", limit=limit)
        rows = (res.get("result", {}) or {}).get("list", [])
        df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
        return df[["timestamp","open","high","low","close","volume"]].copy()

    def ensure_leverage(self, symbol:str, leverage:int):
        try:
            self.client.set_leverage(category=self.category, symbol=symbol,
                                     buyLeverage=str(leverage), sellLeverage=str(leverage))
        except Exception as e:
            print("[WARN] set_leverage:", e)

    def _symbol_filters(self, symbol:str)->Tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
        info = self.client.get_instruments_info(category=self.category, symbol=symbol)
        row = (info.get("result",{}) or {}).get("list",[{}])[0]
        lotSz = D(row.get("lotSizeFilter",{}).get("qtyStep","0.001"))
        minQty = D(row.get("lotSizeFilter",{}).get("minOrderQty","0.0"))
        maxQty = D(row.get("lotSizeFilter",{}).get("maxOrderQty","999999999"))
        minNotional = D(row.get("lotSizeFilter",{}).get("minNotionalValue","0.0"))
        tickSz = D(row.get("priceFilter",{}).get("tickSize","0.01"))
        return lotSz, minQty, maxQty, minNotional, tickSz

    def place_market(self, symbol:str, side:str, use_usdt:float, leverage:int)->Tuple[Optional[str], Optional[Decimal], Optional[Decimal]]:
        # 수량 산출
        bid, ask, mid = self.get_quote(symbol)
        ref_price = D(ask if side=="Buy" else bid)
        qty_step, min_qty, max_qty, min_notional, tick = self._symbol_filters(symbol)
        use = D(use_usdt)
        if use <= 0: return None, None, None
        notional = use * leverage
        qty = round_step(notional / max(ref_price, D("1e-9")), qty_step)
        if qty < min_qty: qty = min_qty
        if min_notional > 0 and qty * ref_price < min_notional:
            qty = round_step((min_notional / ref_price) + qty_step, qty_step)
        if qty > max_qty: qty = max_qty
        if qty <= 0: return None, None, None

        try:
            r = self.client.place_order(category=self.category, symbol=symbol, side=side,
                                        orderType="Market", qty=str(qty),
                                        timeInForce="IOC", reduceOnly=False)
            order_id = (r.get("result",{}) or {}).get("orderId")
            return order_id, ref_price, qty
        except Exception as e:
            print("[ERROR] place_market:", e)
            return None, None, None

    def reduce_market(self, symbol:str, exit_side:str, qty:Decimal)->Optional[str]:
        try:
            r = self.client.place_order(category=self.category, symbol=symbol,
                                        side=exit_side, orderType="Market",
                                        qty=str(qty), timeInForce="IOC",
                                        reduceOnly=True, positionIdx=0)
            return (r.get("result",{}) or {}).get("orderId")
        except Exception as e:
            print("[WARN] reduce_market:", e)
            return None

    def cancel_all(self, symbol:str):
        try:
            self.client.cancel_all_orders(category=self.category, symbol=symbol)
        except Exception as e:
            print("[WARN] cancel_all:", e)


# ===== 페이퍼 브로커 =====
class PaperBroker(Broker):
    """
    시세는 Bybit에서 조회하지만, 주문은 내부 시뮬레이션 (orderId는 가짜)
    """
    def __init__(self, category:str):
        from pybit.unified_trading import HTTP
        self.client = HTTP(api_key="", api_secret="", recv_window=300000, timeout=10, testnet=True)
        self.category = category
        self.counter = 0

    def _oid(self)->str:
        self.counter += 1
        return f"paper-{self.counter:08d}"

    def get_quote(self, symbol:str)->Tuple[float,float,float]:
        res = self.client.get_tickers(category=self.category, symbol=symbol)
        row = (res.get("result", {}) or {}).get("list", [{}])[0]
        bid = fnum(row.get("bid1Price"), 0.0)
        ask = fnum(row.get("ask1Price"), 0.0)
        last = fnum(row.get("lastPrice"), 0.0)
        if bid > 0 and ask > 0:
            return bid, ask, (bid+ask)/2.0
        p = last if last > 0 else self.get_last_price_fallback(symbol)
        return p, p, p

    def get_last_price_fallback(self, symbol:str)->float:
        res = self.client.get_tickers(category=self.category, symbol=symbol)
        row = (res.get("result", {}) or {}).get("list", [{}])[0]
        for k in ("lastPrice","markPrice","bid1Price"):
            p = fnum(row.get(k), 0.0)
            if p > 0: return p
        return 0.0

    def get_kline(self, symbol:str, limit:int=120)->pd.DataFrame:
        res = self.client.get_kline(category=self.category, symbol=symbol, interval="1", limit=limit)
        rows = (res.get("result", {}) or {}).get("list", [])
        df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
        return df[["timestamp","open","high","low","close","volume"]].copy()

    def ensure_leverage(self, symbol:str, leverage:int): pass

    def place_market(self, symbol:str, side:str, use_usdt:float, leverage:int)->Tuple[Optional[str], Optional[Decimal], Optional[Decimal]]:
        bid, ask, mid = self.get_quote(symbol)
        ref_price = D(ask if side=="Buy" else bid)
        if ref_price <= 0: return None, None, None
        notional = D(use_usdt) * leverage
        qty = notional / ref_price
        oid = self._oid()
        return oid, ref_price, qty

    def reduce_market(self, symbol:str, exit_side:str, qty:Decimal)->Optional[str]:
        return self._oid()

    def cancel_all(self, symbol:str): pass


# ===== 엔진 =====
import strategy_hooks as SH

class Engine:
    def __init__(self, S: Settings, broker: Broker):
        self.S = S
        self.broker = broker
        self.open_pos: Dict[str, dict] = {}
        self.cooldown_until: Dict[str, float] = {}

    # ----- 시간창 허용 -----
    def _local_hour_kst(self, ts: Optional[float]=None)->int:
        t = ts if ts is not None else time.time()
        return int((time.gmtime(t).tm_hour + 9) % 24)

    def trading_allowed_now(self)->bool:
        if not self.S.ALLOW_WINDOWS:
            return True
        h = self._local_hour_kst()
        for start, end in self.S.ALLOW_WINDOWS:
            if start == end:
                return True
            if start < end:
                if start <= h < end: return True
            else:
                if h >= start or h < end: return True
        return False

    # ----- PnL / Equity -----
    def get_equity_breakdown(self):
        cash = 0.0
        upnl_total = 0.0
        upnl_each = {}
        for sym, p in self.open_pos.items():
            try:
                _, _, mid = self.broker.get_quote(sym)
                qty = float(p.get("qty", 0.0) or 0.0)
                entry = float(p["entry"])
                side = p["side"]
                pnl = qty * ((mid - entry) if side=="Buy" else (entry - mid))
                upnl_each[sym] = pnl
                upnl_total += pnl
            except Exception:
                pass
        equity = cash + upnl_total
        return equity, cash, upnl_total, upnl_each

    # ----- 진입 -----
    def try_enter(self, symbol: str) -> bool:
        if symbol in self.open_pos: return False
        if symbol in self.S.EXCLUDE_SYMBOLS: return False
        now = time.time()
        if self.cooldown_until.get(symbol, 0) > now: return False
        if len(self.open_pos) >= self.S.MAX_OPEN: return False
        if not self.trading_allowed_now(): return False

        df = self.broker.get_kline(symbol, 120)
        bid, ask, mid = self.broker.get_quote(symbol)
        last_close = float(df["close"].iloc[-1]) if len(df) else mid
        if not SH.pass_quality_filters(bid, ask, mid, last_close,
                                       self.S.SPREAD_REJECT_BPS, self.S.GAP_REJECT_BPS):
            return False

        # 예측
        tp_pred, horizon_sec, conf = SH.predict_future_price(
            symbol, df, self.S.TP1_BPS, self.S.USE_DEEPAR,
            self.S.CKPT_PATH, self.S.SEQ_LEN, self.S.PRED_LEN
        )
        base_side = "Buy" if tp_pred > mid else "Sell"
        side = SH.pick_side_with_mode(base_side, self.S.MODEL_MODE)

        # 리스크/현금 가정
        equity, cash, upnl, _ = self.get_equity_breakdown()
        equity = max(1.0, equity)
        use_cash_cap = equity * self.S.RISK_PCT_OF_EQUITY
        use_cash_base = min(use_cash_cap, equity * self.S.ENTRY_PORTION)
        if use_cash_base <= 0: return False

        # 샤프/켈리 스코어
        sigma_per_min = self._sigma_pct_from_df(df, 120)
        target_for_score = mid * (1.0 + (self.S.TP1_BPS/10000.0 if side=="Buy" else -self.S.TP1_BPS/10000.0))
        denom = mid * (1.0 / self.S.LEVERAGE + self.S.TAKER_FEE)
        est_qty_float = (use_cash_base / denom) if denom > 0 else 0.0

        qm = SH.score_signal_with_kelly_sharpe(
            mid, target_for_score, side, int(horizon_sec),
            est_qty_float, sigma_per_min, conf,
            self.S.CONF_MIN, self.S.PNL_MIN_USD,
            self.S.SHARPE_MIN, self.S.KELLY_CAP
        )
        if not qm["ok"]:
            # off 모드라면 진입은 안하지만 로그/평가 용도로만
            if self.S.TRADE_MODE == "off":
                return False
            return False

        kf = min(qm["kelly_frac"], self.S.KELLY_MAX_ON_CASH)
        use_final = use_cash_base * max(0.08, kf)

        # 모드별 실행
        if self.S.TRADE_MODE == "off":
            # 아무 주문도 실행하지 않음(테스트/로그 전용)
            return False

        # 실 거래/페이퍼 공통: 주문
        self.broker.ensure_leverage(symbol, self.S.LEVERAGE)
        oid, ref_px, qty = self.broker.place_market(symbol, side, float(use_final), self.S.LEVERAGE)
        if oid is None or ref_px is None or qty is None:
            return False

        entry = float(ref_px); qty = float(qty)

        # 초기 SL
        sl_price = entry * (1.0 - self.S.STOP_LOSS_PCT) if side=="Buy" else entry * (1.0 + self.S.STOP_LOSS_PCT)

        # TP 티어 구성
        tp_prices, tp_ratios = SH.build_tp_tiers(
            entry, side, self.S.TP1_BPS, self.S.TP2_BPS, self.S.TP3_BPS,
            self.S.TP1_RATIO, self.S.TP2_RATIO
        )

        self.open_pos[symbol] = {
            "side": side,
            "qty": qty,
            "entry": entry,
            "entry_ts": time.time(),
            "sl_price": float(sl_price),
            "tp_prices": tp_prices,
            "tp_done": [False, False, False],
            "tp_ratios": tp_ratios,
            "be_moved": False,
            "peak": entry, "trough": entry,
            "mode": self.S.MODEL_MODE,
        }

        write_trade_row({
            "ts": int(time.time()), "event": "ENTRY", "symbol": symbol, "side": side,
            "qty": f"{qty:.10f}", "entry_price": f"{entry:.6f}",
            "tp_price": "", "sl_price": f"{float(sl_price):.6f}",
            "order_id": str(oid), "tp_order_id": "", "sl_order_id": "",
            "reason": "open", "pnl_usdt": "", "roi": "", "extra": f"tp={','.join([f'{x:.6f}' for x in tp_prices])}",
            "mode": self.S.MODEL_MODE
        }, self.S.LEVERAGE, self.S.CSV_SAVE_MODE)
        return True

    # ----- 관리 -----
    def manage_positions(self):
        log_equity_snapshot(self.get_equity_breakdown, self.S.LEVERAGE)
        now = time.time()

        for symbol, p in list(self.open_pos.items()):
            side = p["side"]; entry = float(p["entry"])
            qty_all = float(p.get("qty", 0.0) or 0.0)
            bid, ask, mid = self.broker.get_quote(symbol)

            # 시간 제한
            if now - p["entry_ts"] >= self.S.MAX_HOLD_SEC:
                self._close_all(symbol, "EXIT_TIMESTOP")
                continue

            # 즉시익절 (절대/상대)
            notional = entry * qty_all
            pnl_usd = qty_all * ((mid - entry) if side=="Buy" else (entry - mid))
            if self.S.TAKE_PROFIT_USD > 0.0 and pnl_usd >= self.S.TAKE_PROFIT_USD:
                self._close_all(symbol, "EXIT_TP_USD"); continue
            if self.S.TAKE_PROFIT_REL_PCT > 0.0 and pnl_usd >= notional * self.S.TAKE_PROFIT_REL_PCT:
                self._close_all(symbol, "EXIT_TP_REL"); continue

            # 피크/트로프
            if side == "Buy": p["peak"] = max(p["peak"], mid)
            else: p["trough"] = min(p["trough"], mid)

            # 부분익절
            for i, tp_px in enumerate(p["tp_prices"]):
                if p["tp_done"][i]: continue
                hit = (mid >= tp_px) if side == "Buy" else (mid <= tp_px)
                if hit:
                    remaining_qty = float(p.get("qty", 0.0) or 0.0)
                    if remaining_qty <= 0: break
                    close_qty = remaining_qty if i==2 else max(0.0, min(remaining_qty, remaining_qty * p["tp_ratios"][i]))
                    oid, px = self._close_reduce(symbol, side, close_qty, f"TP{i+1}")
                    p["qty"] = max(0.0, remaining_qty - close_qty)
                    p["tp_done"][i] = True

                    write_trade_row({
                        "ts": int(time.time()), "event": f"EXIT_TP{i+1}", "symbol": symbol, "side": side,
                        "qty": f"{close_qty:.10f}", "entry_price": f"{entry:.6f}",
                        "tp_price": f"{tp_px:.6f}", "sl_price": f"{p['sl_price']:.6f}",
                        "order_id": "", "tp_order_id": oid or "", "sl_order_id": "",
                        "reason": f"TP{i+1}", "pnl_usdt": f"{close_qty*((px-entry) if side=='Buy' else (entry-px)):.6f}",
                        "roi": f"{SH.roi(side,entry,px):.6f}", "extra": "", "mode": p.get("mode", self.S.MODEL_MODE),
                    }, self.S.LEVERAGE, self.S.CSV_SAVE_MODE)

                    # BE 이동
                    if not p["be_moved"] and (i+1) >= self.S.BE_AFTER_TIER:
                        eps = self.S.BE_EPS_BPS / 10_000.0
                        p["sl_price"] = entry * (1.0 + eps) if side=="Buy" else entry * (1.0 - eps)
                        p["be_moved"] = True
                        write_trade_row({
                            "ts": int(time.time()), "event": "MOVE_BE", "symbol": symbol, "side": side,
                            "qty": "", "entry_price": f"{entry:.6f}", "tp_price": "", "sl_price": f"{p['sl_price']:.6f}",
                            "order_id": "", "tp_order_id": "", "sl_order_id": "",
                            "reason": f"BE_AFTER_TIER_{self.S.BE_AFTER_TIER}", "pnl_usdt": "", "roi": "",
                            "extra": f"eps_bps={self.S.BE_EPS_BPS}", "mode": p.get("mode", self.S.MODEL_MODE),
                        }, self.S.LEVERAGE, self.S.CSV_SAVE_MODE)

            # 트레일링 SL
            tiers_done = sum(1 for x in p["tp_done"] if x)
            if tiers_done >= self.S.TRAIL_AFTER_TIER:
                if side == "Buy":
                    trail_px = p["peak"] * (1.0 - self.S.TRAIL_BPS / 10_000.0)
                    p["sl_price"] = max(p["sl_price"], trail_px)
                else:
                    trail_px = p["trough"] * (1.0 + self.S.TRAIL_BPS / 10_000.0)
                    p["sl_price"] = min(p["sl_price"], trail_px)

            # SL 히트
            if (side == "Buy" and mid <= p["sl_price"]) or (side == "Sell" and mid >= p["sl_price"]):
                self._close_all(symbol, "EXIT_SL"); continue

            # 전량 청산 시 정리
            if float(p.get("qty", 0.0) or 0.0) <= 0.0:
                self.broker.cancel_all(symbol)
                self.cooldown_until[symbol] = time.time() + self.S.COOLDOWN_SEC
                self.open_pos.pop(symbol, None)

    # ----- 내부 헬퍼 -----
    def _sigma_pct_from_df(self, df: pd.DataFrame, lookback=120)->float:
        s = df.tail(lookback)["close"]
        lr = np.log(s).diff().dropna().values
        if lr.size == 0: return 0.003
        return float(np.std(lr))

    def _close_reduce(self, symbol: str, side: str, qty: float, reason: str):
        exit_side = "Sell" if side=="Buy" else "Buy"
        oid = self.broker.reduce_market(symbol, exit_side, D(qty))
        _, _, mid = self.broker.get_quote(symbol)
        return oid, mid

    def _close_all(self, symbol: str, reason: str):
        p = self.open_pos.get(symbol)
        if not p: return
        side = p["side"]
        qty = float(p.get("qty", 0.0) or 0.0) or 1e9  # fail-safe
        oid, px = self._close_reduce(symbol, side, qty, reason)
        pnl = qty * ((px - p["entry"]) if side=="Buy" else (p["entry"] - px))

        write_trade_row({
            "ts": int(time.time()), "event": "EXIT_ALL", "symbol": symbol, "side": side,
            "qty": f"{qty:.10f}", "entry_price": f"{p['entry']:.6f}",
            "tp_price": "", "sl_price": f"{p['sl_price']:.6f}",
            "order_id": "", "tp_order_id": "", "sl_order_id": "",
            "reason": reason, "pnl_usdt": f"{pnl:.6f}",
            "roi": f"{SH.roi(side,p['entry'],px):.6f}", "extra": "", "mode": p.get("mode", self.S.MODEL_MODE),
        }, self.S.LEVERAGE, self.S.CSV_SAVE_MODE)

        self.broker.cancel_all(symbol)
        self.cooldown_until[symbol] = time.time() + self.S.COOLDOWN_SEC
        self.open_pos.pop(symbol, None)

    # ----- 루프 -----
    def run(self, symbols: List[str]):
        print(f"[RUN] TRADE_MODE={self.S.TRADE_MODE} TESTNET={self.S.TESTNET} LEV={self.S.LEVERAGE} USE_DEEPAR={self.S.USE_DEEPAR}")
        while True:
            try:
                for sym in symbols:
                    self.try_enter(sym)
                self.manage_positions()
            except KeyboardInterrupt:
                print("Interrupted.")
                break
            except Exception as e:
                print("[ERR] main:", e)
            time.sleep(1.0)


# ===== 엔트리포인트 =====
def make_broker(S: Settings) -> Broker:
    if S.TRADE_MODE == "live":
        return RealBybitBroker(S.API_KEY, S.API_SECRET, S.TESTNET, S.CATEGORY)
    elif S.TRADE_MODE == "paper":
        return PaperBroker(S.CATEGORY)
    else:  # off
        # off 모드에서도 시세는 필요하니 paper 브로커 사용(주문은 안함)
        return PaperBroker(S.CATEGORY)

if __name__ == "__main__":
    S = Settings("settings.config")
    broker = make_broker(S)
    engine = Engine(S, broker)
    engine.run(S.SYMBOLS)
