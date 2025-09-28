# ✅ 실시간 트레이딩 전체 코드 (시장가 진입 + 지정가 청산 + 익절 기준 0.1% net)

import os, time, warnings
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from collections import deque
import requests
from binance.client import Client
from decimal import Decimal, ROUND_DOWN

warnings.filterwarnings("ignore")

# ───── Binance API 설정 ─────
API_KEY    = "kynaVBENI8TIS8AcZICK2Sn52mJfIqHpCHfKv2IaSM7TQtKaazn44KGcsJRERjbE"
API_SECRET = "z2ZtL0S8eIE34C8bL1XpAV0jEhcc8JEm8VBOEZbo5Rbe4d9HkN2F4V2AGukhtXyT"
client = Client(API_KEY, API_SECRET)


# ───── 트레이딩 설정 ─────
symbol = "BTCUSDT"
seq_len = 60
pred_len = 10
interval = "1m"
SLEEP_SEC = 5  # 60
slippage = 0.0005
fee_rate = 0.0004
thresh_roi = 0.0003  # 더 높은 진입 ROI 기준
ENTRY_WAIT = 5
EXIT_WAIT = 3
LEVERAGE = 20
MIN_PROFIT_USD = 0.01
ENTRY_PORTION = 0.2
MAX_POSITION_VALUE = 1000
fee_rate_ratio = 1.2

# ✅ 손익 기준
def compute_required_roi(capital, now_price):
    position_value = capital * LEVERAGE
    required_roi = (MIN_PROFIT_USD / position_value) + (fee_rate * fee_rate_ratio)
    return required_roi


# ───── 로그 설정 ─────
LOG_FILE = f"{symbol}_trade_log.csv"
PREDICTION_LOG_FILE = f"{symbol}_predict_log.csv"

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "event", "direction", "price", "roi", "value", "capital", "note"]).to_csv(
        LOG_FILE, index=False)
if not os.path.exists(PREDICTION_LOG_FILE):
    pd.DataFrame(
        columns=["timestamp", "now_price", "target_price", "pred_pct_roi", "target_min", "target_roi", "real_roi",
                 "entered"]).to_csv(PREDICTION_LOG_FILE, index=False)


def log_event(event, direction, price, roi, value, capital, note=""):
    pd.DataFrame([{"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "event": event, "direction": direction, "price": price,
                   "roi": roi, "value": value, "capital": capital, "note": note}]).to_csv(LOG_FILE, mode="a",
                                                                                          header=False, index=False)


def log_prediction(now_price, target_price, target_min, target_roi, real_roi, entered):
    pred_pct_roi = ((target_price / now_price) - 1) * 100
    pd.DataFrame([{"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "now_price": now_price,
                   "target_price": target_price,
                   "pred_pct_roi": pred_pct_roi,
                   "target_min": target_min,
                   "target_roi": target_roi,
                   "real_roi": real_roi,
                   "entered": entered}]).to_csv(PREDICTION_LOG_FILE, mode="a", header=False, index=False)


# ───── 자산 조회 ─────
def get_futures_balance(asset="USDT"):
    try:
        balances = client.futures_account_balance(recvWindow=10000)
        for b in balances:
            if b["asset"] == asset:
                return float(b["availableBalance"])
    except:
        return 0.0


# ───── 주문 함수 ─────
def place_market_order(symbol, side, qty):
    try:
        return client.futures_create_order(symbol=symbol, side=side, type="MARKET", quantity=qty, recvWindow=10000)
    except:
        return None


def place_limit_order(symbol, side, qty, price):
    try:
        return client.futures_create_order(symbol=symbol, side=side, type="LIMIT", timeInForce="GTC", quantity=qty,
                                           price=round(price, 6), recvWindow=10000)
    except:
        return None


def wait_until_filled(order_id, symbol, timeout):
    start = time.time()
    while time.time() - start < timeout:
        try:
            status = client.futures_get_order(symbol=symbol, orderId=order_id)
            if status["status"] == "FILLED": return True
        except:
            pass
        time.sleep(1)
    return False


# ───── 모델 로드 ─────
df_all = pd.read_csv(f"{symbol}_deepar_input.csv").dropna()
dataset = TimeSeriesDataSet(df_all,
                            time_idx="time_idx", target="log_return", group_ids=["series_id"],
                            max_encoder_length=seq_len, max_prediction_length=pred_len,
                            time_varying_known_reals=["time_idx", "volume"],
                            time_varying_unknown_reals=["log_return"],
                            add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True)

model = DeepAR.load_from_checkpoint(f"{symbol}_deepar_model.ckpt")
model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


# ───── 데이터 수집 ─────
def fetch_ohlcv(symbol, limit=100):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    res = requests.get(url, params=params).json()
    df = pd.DataFrame(res, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "q", "n", "tb",
                                    "tq", "ignore"])
    df = df.astype({"close": float, "volume": float})
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df.dropna()


def predict(df, now_price):
    df = df.copy().reset_index(drop=True)
    df["time_idx"] = np.arange(len(df))
    df["series_id"] = symbol
    ds = TimeSeriesDataSet.from_dataset(dataset, df, predict=True, stop_randomization=True)
    dl = ds.to_dataloader(train=False, batch_size=1)
    raw = model.predict(dl, mode="raw")[0].cpu().numpy()[0, :, :pred_len]
    pct_returns = np.exp(raw) - 1
    avg_roi = np.mean(pct_returns, axis=0)
    cum = np.cumsum(np.log1p(pct_returns), axis=1)
    pred_prices = now_price * np.exp(cum)
    return np.mean(pred_prices, axis=0), avg_roi


# ───── 실시간 루프 ─────
print("[INFO] 실시간 트레이딩 시작")
data_queue = deque(fetch_ohlcv(symbol, seq_len + pred_len).to_dict("records"), maxlen=seq_len + pred_len)
is_holding = False
entry_price = None
entry_time = None
direction = None


# 안전 수량 계산 함수
def calculate_safe_quantity(now_price, capital, ENTRY_PORTION, LEVERAGE, slippage, fee_rate, min_notional=5.0):
    now_price = Decimal(str(now_price))
    capital = Decimal(str(capital))
    ENTRY_PORTION = Decimal(str(ENTRY_PORTION))
    LEVERAGE = Decimal(str(LEVERAGE))
    slippage = Decimal(str(slippage))
    fee_rate = Decimal(str(fee_rate))

    effective_price = now_price * (1 + slippage + fee_rate)
    position_value = capital * ENTRY_PORTION * LEVERAGE
    # 한 번만 수행
    exchange_info = client.futures_exchange_info()
    symbol_info = next(s for s in exchange_info['symbols'] if s['symbol'] == symbol)
    qty_precision = symbol_info['quantityPrecision']

    # 이후
    quantity = round(position_value / effective_price, qty_precision)
    qty_value = quantity * now_price
    required_margin = (quantity * effective_price) / LEVERAGE
    max_margin = capital * ENTRY_PORTION

    return {
        "quantity": float(quantity),
        "qty_value": float(qty_value),
        "required_margin": float(required_margin),
        "max_margin": float(max_margin),
        "valid": qty_value >= min_notional and required_margin <= max_margin
    }

# ================== 루프 시작 ==================
while True:
    # 1) 데이터 수집
    df_new = fetch_ohlcv(symbol, limit=2)
    if df_new.empty:
        time.sleep(SLEEP_SEC)
        continue

    data_queue.append(df_new.iloc[-1])
    df_seq = pd.DataFrame(list(data_queue))
    if len(df_seq) < seq_len + pred_len:
        time.sleep(SLEEP_SEC)
        continue

    # 2) 예측
    now_price = df_seq["close"].iloc[-pred_len - 1]
    avg_prices, avg_roi_seq = predict(df_seq, now_price)
    target_idx = np.argmax(np.abs(avg_roi_seq))
    target_roi = avg_roi_seq[target_idx]
    target_price = avg_prices[target_idx]
    target_min = target_idx + 1
    real_roi = abs(target_roi) - fee_rate * fee_rate_ratio
    entered = False

    print("=" * 80)
    print(f"[{datetime.now()}] 🔮 ROI: {[f'{x * 100:.4f}%' for x in avg_roi_seq]}")
    print(f"🎯 Target ROI = {target_roi * 100:.4f}% @ {target_min}분 후 → 예상가: {target_price:.6f} | 현재가: {now_price:.6f}")
    print("=" * 80)

    # 3) 진입 조건 검사
    if not is_holding and abs(target_roi) > thresh_roi:
        capital = get_futures_balance("USDT")
        expected_profit = capital * abs(target_roi)
        real_roi = abs(target_roi) - fee_rate * fee_rate_ratio

        print(f"[DEBUG] 진입 조건 확인: target_roi={target_roi:.5f}, real_roi={real_roi:.5f}, expected_profit=${expected_profit:.3f}")
        if real_roi <= fee_rate * fee_rate_ratio or expected_profit < MIN_PROFIT_USD:
            print("❌ 수수료 감안 수익 부족 → 생략")
            continue

        direction = "LONG" if target_roi > 0 else "SHORT"
        side = "BUY" if direction == "LONG" else "SELL"
        qinfo = calculate_safe_quantity(now_price, capital, ENTRY_PORTION, LEVERAGE, slippage, fee_rate)
        if not qinfo["valid"]:
            print("❌ 주문 생략 → 최소 $5 또는 증거금 초과")
            continue

        quantity = qinfo["quantity"]
        try:
            # — 진입 주문만 감싸기 —
            order = client.futures_create_order(
                symbol=symbol, side=side, type="MARKET",
                quantity=quantity, recvWindow=10000
            )
        except Exception as e:
            print(f"❌ 시장가 진입 실패: {e}")
            log_event("ENTRY_FAILED", direction, now_price, 0, 0, capital, note=str(e))
        else:
            # 주문 성공 시 모니터링 시작
            entry_price = float(order.get("avgFillPrice", now_price))
            entry_time = datetime.now()
            is_holding = True
            max_roi = 0
            target_hold_seconds = target_min * 60

            print(f"🚀 [시장가 진입] [{direction}] @ {entry_price:.6f} for {target_min}분")
            log_event("ENTRY_MARKET", direction, entry_price, 0, capital, capital, f"target_min={target_min}")
            print(f"▶▶▶ 모니터링 루프 진입 (목표 보유: {target_hold_seconds}s)")

            # — 모니터링 루프 —
            while is_holding:
                try:
                    df_live = fetch_ohlcv(symbol, limit=3)
                    if df_live.shape[0] < 2:
                        print("⚠️ 실시간 데이터 부족 → skip")
                        time.sleep(2)
                        continue

                    latest_price = df_live["close"].iloc[-1]
                    held_seconds = (datetime.now() - entry_time).seconds
                    roi = ((latest_price - entry_price) / entry_price
                           if direction=="LONG" else (entry_price - latest_price)/entry_price)
                    value = capital * (1 + roi - fee_rate)
                    expected_profit = capital * roi

                    max_roi = max(max_roi, roi)
                    trailing_trigger = max_roi - 0.002

                    print(f"[{datetime.now()}] 📉 HOLD ROI={roi*100:+.4f}% Held={held_seconds}s MaxROI={max_roi*100:.4f}%")

                    # 청산 조건
                    should_exit = False
                    if roi <= trailing_trigger and max_roi > 0.002:
                        should_exit, exit_reason = True, "TRAILING_STOP"
                    elif held_seconds >= target_hold_seconds:
                        if roi > 0 and expected_profit >= MIN_PROFIT_USD:
                            should_exit, exit_reason = True, "TIME_LIMIT_PROFIT"
                        else:
                            should_exit, exit_reason = True, "TIME_LIMIT"

                    if should_exit:
                        print(f"💰 EXIT {exit_reason} ROI={roi*100:.4f}% Held={held_seconds}s")
                        side_exit = "SELL" if direction=="LONG" else "BUY"
                        try:
                            order_exit = place_limit_order(symbol, side_exit, quantity, latest_price)
                            if order_exit and wait_until_filled(order_exit['orderId'], symbol, EXIT_WAIT):
                                exec_price = latest_price
                                log_event(f"EXIT_{exit_reason}", direction, exec_price, roi, value, capital)
                            else:
                                print("⚠️ 지정가 미체결 → 시장가 청산")
                                mkt = place_market_order(symbol, side_exit, quantity)
                                exec_price = float(mkt.get("avgFillPrice", latest_price)) if mkt else latest_price
                                roi = ((exec_price-entry_price)/entry_price if direction=="LONG"
                                       else (entry_price-exec_price)/entry_price)
                                value = capital * (1 + roi - fee_rate)
                                log_event(f"EXIT_{exit_reason}_MARKET", direction, exec_price, roi, value, capital)
                        except Exception as e:
                            print(f"❌ 청산 오류: {e}")
                            log_event("EXIT_FAILED", direction, latest_price, roi, value, capital, note=str(e))
                        finally:
                            is_holding = False
                            capital = value
                            print("◀◀◀ 모니터링 루프 종료")
                            break

                except Exception as e:
                    print(f"⚠️ 모니터링 중 오류: {e}")
                    time.sleep(2)
                    continue

    # 4) 예측 로그 기록 & 대기
    log_prediction(now_price, target_price, target_min, float(target_roi), real_roi, entered)
    time.sleep(SLEEP_SEC)
