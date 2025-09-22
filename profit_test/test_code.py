# -*- coding: utf-8 -*-
"""
partial_tp_live_backtest.py
- 가정: 매 분봉 시작가에 진입, 방향은 항상 '정답'이라고 가정
- 데이터: Bybit public 1m Kline (최근 M분)
- 전략 비교:
  A) 분할익절(TP1/TP2/TP3, BE 미적용, 1분 내 충족 안되면 종가 청산)
  B) 단일익절(TP3만 전량, 미충족 시 종가 청산)
  C) 종가청산(익절 없음)
- 수수료: 테이커 왕복(진입+청산) 적용
- 포지션 크기: qty = equity * ENTRY_EQUITY_PCT * LEVERAGE / entry
"""
import os, math, time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

# ====== ENV ======
CATEGORY = "linear"
INTERVAL = "1"  # 1m
TESTNET = str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")

SYMBOLS = os.getenv("SYMBOLS","ASTERUSDT,AIAUSDT,0GUSDT,STBLUSDT,WLFIUSDT,LINEAUSDT,AVMTUSDT,BARDUSDT,SOMIUSDT,UBUSDT,OPENUSDT").split(",")

MINS = int(os.getenv("MINS","10"))             # 최근 분봉 개수
START_EQUITY = float(os.getenv("START_EQUITY","10000"))
LEVERAGE = float(os.getenv("LEVERAGE","20"))
ENTRY_EQUITY_PCT = float(os.getenv("ENTRY_EQUITY_PCT","0.20"))
TAKER_FEE = float(os.getenv("TAKER_FEE","0.0006"))

TP1_BPS = float(os.getenv("TP1_BPS","50"))
TP2_BPS = float(os.getenv("TP2_BPS","100"))
TP3_BPS = float(os.getenv("TP3_BPS","200"))
TP1_RATIO = float(os.getenv("TP1_RATIO","0.40"))
TP2_RATIO = float(os.getenv("TP2_RATIO","0.35"))
# 나머지 0.25는 TP3

VERBOSE = str(os.getenv("VERBOSE","1")).lower() in ("1","true","y","yes")

# ====== API ======
mkt = HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

def get_1m_df(symbol: str, limit: int) -> pd.DataFrame:
    r = mkt.get_kline(category=CATEGORY, symbol=symbol, interval=INTERVAL, limit=min(limit,1000))
    rows = (r.get("result") or {}).get("list") or []
    rows = rows[::-1]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "ts": int(z[0]),
        "open": float(z[1]), "high": float(z[2]), "low": float(z[3]), "close": float(z[4]),
        "vol": float(z[5])
    } for z in rows])
    return df

# ====== math ======
def px_from_bps(entry: float, bps: float, side: str) -> float:
    return entry*(1+bps/1e4) if side=="Buy" else entry*(1-bps/1e4)

def fees(entry: float, exits: List[Tuple[float,float]], qty_total: float) -> float:
    """
    exits: [(exit_price, exit_qty), ...]
    fee = taker*(entry*qty_total + sum(exit_i*qty_i))
    """
    notional_in  = entry * qty_total
    notional_out = sum(p*q for p,q in exits)
    return TAKER_FEE * (notional_in + notional_out)

def trade_pnl(entry: float, legs: List[Tuple[float,float,str]]) -> float:
    """
    legs: [(exit_price, qty, side)], side=Buy/Sell of original position
    PnL = sum((exit-entry)*q) for Buy, sum((entry-exit)*q) for Sell
    """
    pnl = 0.0
    for ex, q, side in legs:
        if side=="Buy":
            pnl += (ex - entry)*q
        else:
            pnl += (entry - ex)*q
    return pnl

# ====== simulate one bar given perfect direction ======
def simulate_one_bar(entry: float, high: float, low: float, close: float, eq: float) -> Dict[str, float]:
    """
    방향은 항상 정답: 양봉이면 Long, 음봉이면 Short로 가정.
    한 개 분봉 내에서 TP 충족 여부만 본다. 미충족 수량은 종가 청산.
    리턴: {'A': pnl_after_fee, 'B': ..., 'C': ...}, 그리고 새 equity는 바깥에서 합산.
    """
    # 방향 결정: close-open 기준. 동일하면 0수익 가정으로 종가청산만 수행.
    if close > entry:
        side = "Buy"; mfe = high - entry
    elif close < entry:
        side = "Sell"; mfe = entry - low
    else:
        side = "Flat"; mfe = 0.0

    if side == "Flat":
        return {"A": 0.0, "B": 0.0, "C": 0.0}

    # 포지션 사이즈
    qty = (eq * ENTRY_EQUITY_PCT * LEVERAGE) / max(entry,1e-12)
    if qty <= 0:
        return {"A": 0.0, "B": 0.0, "C": 0.0}

    # 목표가
    tp1 = px_from_bps(entry, TP1_BPS, side)
    tp2 = px_from_bps(entry, TP2_BPS, side)
    tp3 = px_from_bps(entry, TP3_BPS, side)

    # ===== A) 분할익절 =====
    legs_A: List[Tuple[float,float,str]] = []  # (exit_px, qty, side)
    exits_for_fee_A: List[Tuple[float,float]] = []

    filled1 = False; filled2 = False; filled3 = False
    # MFE로 충족 판정
    hit1 = (high >= tp1) if side=="Buy" else (low <= tp1)
    hit2 = (high >= tp2) if side=="Buy" else (low <= tp2)
    hit3 = (high >= tp3) if side=="Buy" else (low <= tp3)

    remain = qty
    if hit1:
        q1 = qty*TP1_RATIO
        legs_A.append((tp1, q1, side)); exits_for_fee_A.append((tp1, q1))
        remain -= q1; filled1=True
    if hit2 and remain>0:
        q2 = min(remain, qty*(1.0-TP1_RATIO)*TP2_RATIO)
        legs_A.append((tp2, q2, side)); exits_for_fee_A.append((tp2, q2))
        remain -= q2; filled2=True
    if hit3 and remain>0:
        q3 = remain
        legs_A.append((tp3, q3, side)); exits_for_fee_A.append((tp3, q3))
        remain = 0.0; filled3=True
    # 남은 수량 종가 청산
    if remain > 0:
        legs_A.append((close, remain, side)); exits_for_fee_A.append((close, remain))

    gross_A = trade_pnl(entry, legs_A)
    fee_A   = fees(entry, exits_for_fee_A, qty)
    pnl_A   = gross_A - fee_A

    # ===== B) 단일익절(TP3만) =====
    legs_B: List[Tuple[float,float,str]] = []
    exits_for_fee_B: List[Tuple[float,float]] = []
    if hit3:
        legs_B.append((tp3, qty, side)); exits_for_fee_B.append((tp3, qty))
    else:
        legs_B.append((close, qty, side)); exits_for_fee_B.append((close, qty))
    gross_B = trade_pnl(entry, legs_B)
    fee_B   = fees(entry, exits_for_fee_B, qty)
    pnl_B   = gross_B - fee_B

    # ===== C) 종가 청산 =====
    legs_C = [(close, qty, side)]
    gross_C = trade_pnl(entry, legs_C)
    fee_C   = fees(entry, [(close, qty)], qty)
    pnl_C   = gross_C - fee_C

    return {"A": pnl_A, "B": pnl_B, "C": pnl_C}

def backtest_symbol(symbol: str, mins: int, start_eq: float) -> Dict[str, float]:
    df = get_1m_df(symbol, mins)
    if df.empty or len(df) < 5:
        return {"A":0.0,"B":0.0,"C":0.0,"trades":0}

    eqA = start_eq; eqB = start_eq; eqC = start_eq
    pnlA = pnlB = pnlC = 0.0
    trades = 0

    # 각 분봉에서 '해당 분봉 시작가=entry'로 1회 거래
    for i in range(len(df)):
        o = df["open"].iat[i]; h = df["high"].iat[i]; l = df["low"].iat[i]; c = df["close"].iat[i]
        # A
        rA = simulate_one_bar(o,h,l,c, eqA)
        eqA += rA["A"]; pnlA += rA["A"]
        # B
        rB = simulate_one_bar(o,h,l,c, eqB)
        eqB += rB["B"]; pnlB += rB["B"]
        # C
        rC = simulate_one_bar(o,h,l,c, eqC)
        eqC += rC["C"]; pnlC += rC["C"]

        trades += 1

    if VERBOSE:
        print(f"[{symbol}] trades={trades}  PnL_A={pnlA:.2f}  PnL_B={pnlB:.2f}  PnL_C={pnlC:.2f}")
    return {"A":pnlA, "B":pnlB, "C":pnlC, "trades":trades}

def main():
    t0 = time.time()
    rows = []
    sumA=sumB=sumC=0.0
    for s in SYMBOLS:
        try:
            r = backtest_symbol(s.strip(), MINS, START_EQUITY)
        except Exception as e:
            if VERBOSE: print(f"[ERR] {s}: {e}")
            r = {"A":0.0,"B":0.0,"C":0.0,"trades":0}
        rows.append({"symbol":s, **r})
        sumA += r["A"]; sumB += r["B"]; sumC += r["C"]

    df = pd.DataFrame(rows).sort_values("A", ascending=False)
    print("\n=== Per-symbol PnL (USD) — 최근 {}분, 방향 정답 가정 ===".format(MINS))
    print(df.to_string(index=False, formatters={"A":"{:,.2f}".format,"B":"{:,.2f}".format,"C":"{:,.2f}".format}))

    print("\n=== Totals (USD) ===")
    print(f"A) Partial TP     : {sumA:,.2f}")
    print(f"B) Single TP at T3: {sumB:,.2f}")
    print(f"C) Close at bar end: {sumC:,.2f}")

    dur = time.time()-t0
    print(f"\nDone in {dur:.2f}s  | SYMS={len(SYMBOLS)}  | MINS={MINS}")

if __name__ == "__main__":
    main()
