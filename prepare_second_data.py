#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
필요 패키지:
    pip install websocket-client
설정:
    SYMBOL, MARKET, OUT_CSV만 바꿔서 사용하세요.
동작:
    - Binance aggTrade 스트림 수신
    - 1초 버킷으로 OHLCV 집계
    - 초가 바뀔 때마다 CSV에 한 줄씩 즉시 기록(append)
"""

import os, csv, json, time, signal, sys
from decimal import Decimal, getcontext
from datetime import datetime, timezone, timedelta
import websocket

# ===== 사용자 설정 =====
SYMBOL  = "SOLUSDT"               # 예: ETHUSDT, BTCUSDT
MARKET  = "futures"               # "futures"(USDT-M) 또는 "spot"
OUT_CSV = f"{SYMBOL}_1s.csv"      # 저장할 CSV 경로
FILL_GAPS = True                  # 중간에 거래가 없던 초를 직전 종가로 채울지 여부

# ===== 내부 설정 =====
getcontext().prec = 28
CSV_HEADER = [
    "symbol","epoch_sec","time_utc","time_kst",
    "open","high","low","close",
    "volume","quote_volume","trades",
    "taker_buy_base","taker_sell_base"
]

def ws_url(symbol: str, market: str) -> str:
    s = symbol.lower()
    if market.lower() == "futures":
        return f"wss://fstream.binance.com/stream?streams={s}@aggTrade"
    return f"wss://stream.binance.com:9443/stream?streams={s}@aggTrade"

def iso_utc(sec: int) -> str:
    return datetime.fromtimestamp(sec, tz=timezone.utc).isoformat(timespec="seconds")

def iso_kst(sec: int) -> str:
    kst = timezone(timedelta(hours=9))
    return datetime.fromtimestamp(sec, tz=timezone.utc).astimezone(kst).isoformat(timespec="seconds")

class CsvWriter:
    def __init__(self, path: str):
        self.path = path
        self._ensure_header()

    def _ensure_header(self):
        need_header = (not os.path.exists(self.path)) or os.path.getsize(self.path) == 0
        if need_header:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_HEADER)

    def write_row(self, row: list):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

class OneSecondOHLC:
    def __init__(self, symbol: str, csv_writer: CsvWriter, fill_gaps: bool = True):
        self.symbol = symbol.upper()
        self.w = csv_writer
        self.fill_gaps = fill_gaps

        self.cur_sec = None
        self.o = self.h = self.l = self.c = None
        self.v = Decimal(0)
        self.qv = Decimal(0)
        self.trades = 0
        self.taker_buy = Decimal(0)
        self.taker_sell = Decimal(0)
        self.last_close = None

    def _reset_bucket(self):
        self.o = self.h = self.l = self.c = None
        self.v = Decimal(0)
        self.qv = Decimal(0)
        self.trades = 0
        self.taker_buy = Decimal(0)
        self.taker_sell = Decimal(0)

    def _finalize_write(self, sec: int):
        if self.o is None:
            if not self.fill_gaps or self.last_close is None:
                return
            o = h = l = c = self.last_close
            v = Decimal(0); qv = Decimal(0); n = 0
            tb = Decimal(0); ts = Decimal(0)
        else:
            o, h, l, c = self.o, self.h, self.l, self.c
            v, qv, n = self.v, self.qv, self.trades
            tb, ts = self.taker_buy, self.taker_sell

        self.w.write_row([
            self.symbol, sec, iso_utc(sec), iso_kst(sec),
            str(o), str(h), str(l), str(c),
            str(v), str(qv), n, str(tb), str(ts)
        ])
        self.last_close = c

    def _roll_to(self, new_sec: int):
        if self.cur_sec is not None and new_sec > self.cur_sec + 1 and self.fill_gaps:
            # 중간 빈 초 채우기
            for s in range(self.cur_sec + 1, new_sec):
                self._finalize_write(s)

        if self.cur_sec is not None:
            self._finalize_write(self.cur_sec)

        self.cur_sec = new_sec
        self._reset_bucket()

    def add_trade(self, price: Decimal, qty: Decimal, is_buyer_maker: bool, trade_ms: int):
        sec = trade_ms // 1000
        if self.cur_sec is None:
            self.cur_sec = sec

        if sec != self.cur_sec:
            self._roll_to(sec)

        if self.o is None:
            self.o = self.h = self.l = price
        else:
            if price > self.h: self.h = price
            if price < self.l: self.l = price
        self.c = price
        self.v += qty
        self.qv += qty * price
        self.trades += 1

        # m=True → buyer is maker(매도 테이커 체결), m=False → buyer is taker(매수 테이커 체결)
        if is_buyer_maker:
            self.taker_sell += qty
        else:
            self.taker_buy += qty

    def flush(self):
        if self.cur_sec is not None:
            self._finalize_write(self.cur_sec)
            self.cur_sec = None
        self._reset_bucket()

class Collector:
    def __init__(self, symbol: str, market: str, out_csv: str, fill_gaps=True):
        self.url = ws_url(symbol, market)
        self.ohlc = OneSecondOHLC(symbol, CsvWriter(out_csv), fill_gaps)
        self._stop = False
        self.ws = None

    def on_open(self, ws):
        print(f"[OPEN] {self.url}", flush=True)

    def on_message(self, ws, message):
        try:
            msg = json.loads(message)
            data = msg.get("data", msg)  # combined stream or single
            if data.get("e") != "aggTrade":
                return
            # aggTrade fields: p(price), q(quantity), T(trade time ms), m(is buyer maker)
            p = Decimal(data["p"])
            q = Decimal(data["q"])
            t_ms = int(data["T"])
            m = bool(data["m"])
            self.ohlc.add_trade(p, q, m, t_ms)
        except Exception as e:
            print(f"[on_message][ERR] {e}", flush=True)

    def on_error(self, ws, error):
        print(f"[WS][ERR] {error}", flush=True)

    def on_close(self, ws, code, msg):
        print(f"[CLOSE] code={code} msg={msg}", flush=True)

    def run(self):
        backoff = 1.0
        while not self._stop:
            try:
                self.ws = websocket.WebSocketApp(
                    self.url,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                )
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[RUN][ERR] {e}", flush=True)

            if self._stop: break
            time.sleep(min(60.0, backoff))
            backoff = min(60.0, backoff * 2)

        self.ohlc.flush()

    def stop(self):
        self._stop = True
        try:
            if self.ws:
                self.ws.close()
        except:
            pass
        self.ohlc.flush()

def main():
    col = Collector(SYMBOL, MARKET, OUT_CSV, fill_gaps=FILL_GAPS)

    def _sig(*_):
        print("[SIG] stop", flush=True)
        col.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    col.run()

if __name__ == "__main__":
    main()
