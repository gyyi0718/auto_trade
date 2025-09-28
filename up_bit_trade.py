# ===================== 업비트 현물 자동매매 (DeepAR 예측 기반) =====================
# 붙여넣기 후 .env 또는 런치 설정에 UPBIT_ACCESS_KEY/UPBIT_SECRET_KEY 넣으세요.
# (직접 상수로 넣고 싶다면 아래 os.getenv 부분을 문자열로 교체)
import sys
import os, time, warnings, uuid, hashlib, base64, json
import requests
import pandas as pd
import numpy as np
import torch
from datetime import datetime,timedelta, timezone
from collections import deque
from decimal import Decimal
from urllib.parse import urlencode
import jwt
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

ACCESS_KEY = "PGbto8x6q9OBB7K9iewTf7KcRUrxgzNCZ1xdOsS0"
SECRET_KEY = "I1v7h50FZny2cWHxnDKx2rUkYoMnfaIxN2N4v6lI"
SYMBOLS    = ["XNYUSDT","INUSDT","YALAUSDT","CARVUSDT","AIOUSDT","USELESSUSDT"]

symbol = "KRW-ETH"
LOG_FILE = f"{symbol}_trade_log.csv"
PREDICTION_LOG_FILE = f"{symbol}_predict_log.csv"
interval_minutes = 2
seq_len            = 60
pred_len           = 10
interval           = "1m"
SLEEP_SEC          = 5
slippage           = 0.0005      # 편도 슬리피지율
fee_rate           = 0.0004      # 편도 수수료율 (0.04%)
thresh_roi         = 0.0005      # 진입 예측 ROI 절대값 기준
LEVERAGE           = 1           # 레버리지 없이 운용
MIN_PROFIT_USD     = 0.001        # 최소 실현 이익($)
ENTRY_PORTION      = 0.4         # 진입에 사용할 자본 비율
LOG_FILE            = f"{symbol}_trade_log.csv"
PREDICTION_LOG_FILE = f"{symbol}_predict_log.csv"
CKPT_FILE = f"{symbol}_deepar_model.ckpt"
MIN_PROFIT_KRW = 5

KST = timezone(timedelta(hours=9))
interval_minutes = 1  # 기존 전역값 사용
# 가장 가까운 .env 로드

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

def DBG(msg):
    print(msg, flush=True)
def load_env_with_debug():
    candidates = [find_dotenv(), Path(__file__).with_name(".env"), Path.cwd() / ".env"]
    loaded = []
    for c in candidates:
        if c and Path(c).exists():
            load_dotenv(c, override=True)
            loaded.append(str(c))
    print("[.env loaded]", loaded or "NONE")
    print("[cwd]", Path.cwd())
    print("[__file__]", Path(__file__).resolve())

load_env_with_debug()



def mask(s): return s[:6]+"..."+s[-4:] if s else "<EMPTY>"
print("[UPBIT] ACCESS_KEY =", mask(ACCESS_KEY), "len=", len(ACCESS_KEY))
print("[UPBIT] SECRET_KEY =", "***", "len=", len(SECRET_KEY))

if not ACCESS_KEY or not SECRET_KEY:
    raise RuntimeError("ACCESS_KEY/SECRET_KEY 환경변수 비어있음 (.env or 환경변수 확인)")


# ───── 유틸: 업비트 인증 헤더 ─────
def upbit_headers(query: dict | None = None):
    payload = {"access_key": ACCESS_KEY, "nonce": str(uuid.uuid4())}

    # ✅ 쿼리 있는 요청은 query_hash 필수
    if query:
        # Upbit는 쿼리를 키=값&... 형태로 정렬 없이 그대로 해싱 권장 (urlencode 기본 사용)
        qs = urlencode(query, doseq=True)
        h = hashlib.sha512()
        h.update(qs.encode("utf-8"))
        payload["query_hash"] = h.hexdigest()
        payload["query_hash_alg"] = "SHA512"

    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    if isinstance(token, bytes):  # PyJWT v1 호환
        token = token.decode("utf-8")

    # (디버그) 토큰 페이로드 확인
    try:
        p = token.split(".")[1]
        # Base64 URL-safe padding
        pad = "=" * (-len(p) % 4)
        payload_decoded = json.loads(base64.urlsafe_b64decode(p + pad))
        assert payload_decoded.get("access_key") == ACCESS_KEY, "JWT payload access_key mismatch"
    except Exception as e:
        print("WARN: JWT payload decode 실패(디버그용):", e)

    return {"Authorization": f"Bearer {token}"}

# ───── 자격 확인 ─────
def verify_upbit_credentials():
    url = "https://api.upbit.com/v1/accounts"
    r = requests.get(url, headers=upbit_headers(), timeout=10)
    try:
        data = r.json()
    except ValueError:
        return False, f"Non-JSON response {r.status_code}: {r.text[:200]}"

    if isinstance(data, list):
        return True, "OK"

    if isinstance(data, dict) and "error" in data:
        e = data["error"]
        return False, f"{e.get('name','?')} - {e.get('message','')}"

    return False, f"Unexpected schema: {data}"

ok, reason = verify_upbit_credentials()
print("[UPBIT VERIFY]", "OK" if ok else "FAIL", reason)
if not ok:
    raise RuntimeError(f"Upbit credential check failed: {reason}")

# ───── 잔고 조회 ─────
def get_balance(currency="KRW") -> float:
    url = "https://api.upbit.com/v1/accounts"
    r = requests.get(url, headers=upbit_headers(), timeout=10)

    try:
        data = r.json()
    except ValueError:
        raise RuntimeError(f"Non-JSON response: {r.status_code} {r.text[:200]}")

    if isinstance(data, list):
        for acc in data:
            if isinstance(acc, dict) and acc.get("currency") == currency:
                return float(acc.get("balance", "0"))
        return 0.0

    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        name = err.get("name", "UnknownError")
        msg = err.get("message", "")
        raise RuntimeError(f"Upbit API error: {name} - {msg}")

    raise RuntimeError(f"Unexpected response schema: {json.dumps(data)[:300]}")

# ───── 1) 로깅 ─────
def setup_logging():
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=[
            "timestamp","event","direction","price","roi","value","capital","note"
        ]).to_csv(LOG_FILE, index=False)
    if not os.path.exists(PREDICTION_LOG_FILE):
        pd.DataFrame(columns=[
            "timestamp","now_price","target_price","pred_pct_roi",
            "target_min","target_roi","real_roi","entered"
        ]).to_csv(PREDICTION_LOG_FILE, index=False)

def log_event(event, direction, price, roi, value, capital, note=""):
    pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event, "direction": direction, "price": price,
        "roi": roi, "value": value, "capital": capital, "note": note
    }]).to_csv(LOG_FILE, mode="a", header=False, index=False)

def log_prediction(now_price, target_price, target_min, target_roi, real_roi, entered):
    pred_pct_roi = (target_price/now_price - 1) * 100
    pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "now_price": now_price,
        "target_price": target_price,
        "pred_pct_roi": pred_pct_roi,
        "target_min": target_min,
        "target_roi": target_roi,
        "real_roi": real_roi,
        "entered": entered
    }]).to_csv(PREDICTION_LOG_FILE, mode="a", header=False, index=False)

# ───── 2) 모델 로드 ─────
def load_deepar_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    hparams.pop("dataset_parameters", None)
    tmp = f"{symbol}_tmp.ckpt"
    torch.save(ckpt, tmp)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepAR.load_from_checkpoint(tmp, map_location=device)
    return model.to(device).eval()



def get_ticker_price(market):
    url = "https://api.upbit.com/v1/ticker"
    headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0 (UpbitBot/1.0)"}
    for attempt in range(3):
        try:
            r = requests.get(url, params={"markets": market}, headers=headers, timeout=10)
            if r.status_code == 429:
                time.sleep(0.5 * (attempt + 1)); continue
            if r.status_code != 200:
                print(f"[WARN] candles HTTP {r.status_code} url={r.url} body={r.text[:200]}")
                return pd.DataFrame()
            try:
                j = r.json()
                if isinstance(j, list) and j:
                    return float(j[0]["trade_price"])
                print(f"[WARN] Unexpected ticker schema: {j}")
                return np.nan
            except Exception:
                print(f"[WARN] Non-JSON ticker: {r.text[:200]}")
                time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            print(f"[ERROR] get_ticker_price exception: {e}")
            time.sleep(0.5 * (attempt + 1))
    return np.nan

def ensure_valid_market(market: str):
    url = "https://api.upbit.com/v1/market/all"
    try:
        r = requests.get(url, params={"isDetails": "false"}, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"/v1/market/all 조회 실패: {e}")

    markets = {d.get("market") for d in data if isinstance(d, dict)}
    if market not in markets:
        # 가까운 후보 몇 개 추천
        krw_markets = sorted([m for m in markets if m.startswith("KRW-")])
        ex = ", ".join(krw_markets[:10])
        raise RuntimeError(
            f"유효하지 않은 마켓 코드: {market}\n"
            f"예: {ex} ..."
        )

def order_chance(market):
    url = "https://api.upbit.com/v1/orders/chance"
    params = {"market": market}
    res = requests.get(url, headers=upbit_headers(params), params=params, timeout=10).json()
    # KRW 마켓의 경우 bid["min_total"]이 통상 5000원
    bid_fee = float(res["bid_fee"]); ask_fee = float(res["ask_fee"])
    min_total = float(res["market"]["bid"].get("min_total", "5000"))
    return bid_fee, ask_fee, min_total

def post_order(params):
    url = "https://api.upbit.com/v1/orders"
    headers = upbit_headers(params)
    # 업비트는 POST에 querystring 전송을 허용합니다(params=)
    res = requests.post(url, headers=headers, params=params, timeout=10).json()
    if isinstance(res, dict) and res.get("error"):
        raise RuntimeError(res["error"])
    return res

def get_order(uuid_):
    url = "https://api.upbit.com/v1/order"
    params = {"uuid": uuid_}
    res = requests.get(url, headers=upbit_headers(params), params=params, timeout=10).json()
    if isinstance(res, dict) and res.get("error"):
        raise RuntimeError(res["error"])
    return res

def avg_filled_price(order_detail):
    trades = order_detail.get("trades", [])
    if not trades:
        # 시장가 매수의 경우 price는 0일 수 있음 → 체결정보로만 계산
        return float(order_detail.get("price") or 0) or 0.0
    tot = 0.0; vol = 0.0
    for t in trades:
        p = float(t["price"]); v = float(t["volume"])
        tot += p * v; vol += v
    return (tot/vol) if vol > 0 else 0.0

# ───── 4) 수량/금액 계산 ─────
def net_cost():  # 왕복 수수료+슬리피지
    return 2*fee_rate + 2*slippage

def expected_net_profit_krw(krw_to_spend, roi):
    return krw_to_spend * roi - krw_to_spend * net_cost()

def calculate_buy_krw_amount(now_price, krw_balance, min_total):
    krw_to_spend = krw_balance * ENTRY_PORTION
    # 최소 주문금액 이상 보장
    if krw_to_spend < min_total:
        return 0.0
    return float(Decimal(str(krw_to_spend)).quantize(Decimal("1")))  # 정수 KRW로

# ───── 5) 데이터 & 예측 ─────
# --- 견고한 캔들 조회 (429/비JSON/오류 스키마 처리 + 재시도) ---
def fetch_ohlcv(market, count=2):
    url = f"https://api.upbit.com/v1/candles/minutes/{interval_minutes}"
    headers = {"Accept":"application/json","User-Agent":"UpbitBot/1.0"}
    try:
        r = requests.get(url, params={"market": market, "count": count}, headers=headers, timeout=10)
        DBG(f"[HTTP] {r.status_code} {r.url}")
        if r.status_code != 200:
            DBG(f"[WARN] candles body[:200]={r.text[:200]}")
            return pd.DataFrame()
        j = r.json()
        if not isinstance(j, list) or not j:
            DBG(f"[WARN] candles unexpected schema: {type(j)} {str(j)[:120]}")
            return pd.DataFrame()
        df = pd.DataFrame(j)[["trade_price","candle_acc_trade_volume"]].rename(
            columns={"trade_price":"close","candle_acc_trade_volume":"volume"}
        )
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        DBG(f"[ERROR] fetch_ohlcv: {e}")
        return pd.DataFrame()


def predict(df, now_price):
    df = df.copy().reset_index(drop=True)

    # ✅ 누락/NaN 대비: log_return 재계산
    if "log_return" not in df.columns or df["log_return"].isna().any():
        df["close"] = df["close"].astype(float)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df = df.dropna(subset=["log_return"]).reset_index(drop=True)

    df["time_idx"]  = np.arange(len(df))
    df["series_id"] = symbol
    ds = TimeSeriesDataSet.from_dataset(dataset, df, predict=True, stop_randomization=True)
    dl = ds.to_dataloader(train=False, batch_size=1)
    raw = model.predict(dl, mode="raw")[0].cpu().numpy()[0, :, :pred_len]
    pct_returns = np.exp(raw) - 1
    avg_roi = np.mean(pct_returns, axis=0)
    cum = np.cumsum(np.log1p(pct_returns), axis=1)
    pred_prices = now_price * np.exp(cum)
    return np.mean(pred_prices, axis=0), avg_roi

# ───── 6) 이익 시청 모니터 (현물, 타임아웃/손절 최소화 버전) ─────
def monitor_until_profit(entry_price, direction, volume):
    assert direction == "LONG", "Upbit 현물은 SHORT 미지원"
    need = net_cost() + 0.0001  # 수수료/슬리피지 + 작은 버퍼
    hard_sl = 0.0030            # -0.30% 고정 손절 (원하면 조정)
    start = datetime.now()
    timeout = pred_len * 60 + 30

    while True:
        df_live = fetch_ohlcv(symbol, count=2)
        if df_live.empty:
            time.sleep(1); continue
        last = df_live["close"].iloc[-1]
        roi  = (last - entry_price)/entry_price
        print(f"[HOLD] ROI={roi*100:+.4f}% Price={last}")

        if roi >= need:
            # 시장가 매도
            od = post_order({"market": symbol, "side": "ask", "ord_type": "market",
                             "volume": f"{volume:.8f}"})
            time.sleep(0.4)
            detail = get_order(od["uuid"])
            exit_price = avg_filled_price(detail) or last
            final_roi  = (exit_price - entry_price)/entry_price
            return ("TP", final_roi, exit_price)

        if roi <= -hard_sl:
            od = post_order({"market": symbol, "side": "ask", "ord_type": "market",
                             "volume": f"{volume:.8f}"})
            time.sleep(0.4)
            detail = get_order(od["uuid"])
            exit_price = avg_filled_price(detail) or last
            final_roi  = (exit_price - entry_price)/entry_price
            return ("SL", final_roi, exit_price)

        if (datetime.now() - start).total_seconds() >= timeout:
            od = post_order({"market": symbol, "side": "ask", "ord_type": "market",
                             "volume": f"{volume:.8f}"})
            time.sleep(0.4)
            detail = get_order(od["uuid"])
            exit_price = avg_filled_price(detail) or last
            final_roi  = (exit_price - entry_price)/entry_price
            return ("TIMEOUT", final_roi, exit_price)

        time.sleep(1)

# ───── 7) 메인 ─────
if __name__ == "__main__":
    setup_logging()

    # 데이터셋 & 모델 준비(업비트용 CSV 필요: f"{symbol}_deepar_input.csv")
    df_all = pd.read_csv(f"{symbol}_deepar_input.csv").dropna()
    dataset = TimeSeriesDataSet(
        df_all, time_idx="time_idx", target="log_return", group_ids=["series_id"],
        max_encoder_length=seq_len, max_prediction_length=pred_len,
        time_varying_known_reals=["time_idx","volume"],
        time_varying_unknown_reals=["log_return"],
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True
    )
    model = load_deepar_model(CKPT_FILE)

    # 주문 한도/수수료 확인
    try:
        bid_fee, ask_fee, min_total = order_chance(symbol)
        # 계정 수수료가 다르면 fee_rate를 자동 반영
        fee_rate = max(fee_rate, max(bid_fee, ask_fee))
        print(f"[INFO] Upbit fee(bid/ask): {bid_fee}/{ask_fee}, min_total={min_total}")
    except Exception as e:
        print(f"[WARN] orders/chance 실패: {e}")
        min_total = 5000.0  # KRW 기본값 추정치

    ensure_valid_market(symbol)

    print("[INFO] 실시간 트레이딩 시작(Upbit Spot)")
    seed_df = fetch_ohlcv(symbol, count=seq_len + pred_len)
    if seed_df.empty:
        raise RuntimeError("초기 캔들 로딩 실패")
    # ─── 메인 루프 ───
    # 큐는 shift(1)로 빠지는 1개를 고려해서 +1
    data_q = deque(seed_df.to_dict("records"), maxlen=seq_len + pred_len + 1)
    print(f"[BOOT] queue maxlen={data_q.maxlen}")

    heartbeat = 0
    while True:
        heartbeat += 1
        print(f"[LOOP] tick {datetime.now().strftime('%H:%M:%S')} ({heartbeat})", flush=True)

        # ❶ (옵션) 고아 포지션 정리 훅 — 함수가 있으면 쓰고, 없으면 지워도 됨
        # if cleanup_orphaned_position():
        #     time.sleep(SLEEP_SEC); continue

        # ❷ 데이터
        df_new = fetch_ohlcv(symbol, count=2)  # ← Upbit는 count 파라미터!
        if df_new.empty:
            print("[LOOP] empty candles → sleep", flush=True)
            time.sleep(SLEEP_SEC)
            continue

        last_close = float(df_new["close"].iloc[-1])
        data_q.append(df_new.iloc[-1])
        df_seq = pd.DataFrame(list(data_q))

        # log_return 계산(첫 행 NaN → drop)
        df_seq["close"] = df_seq["close"].astype(float)
        df_seq["log_return"] = np.log(df_seq["close"] / df_seq["close"].shift(1))
        df_seq = df_seq.dropna(subset=["log_return"]).reset_index(drop=True)

        need_len = seq_len + pred_len
        cur_len = len(df_seq)
        print(f"[SEQ] len={cur_len} need>={need_len} last_close={last_close}", flush=True)
        if cur_len < need_len:
            time.sleep(SLEEP_SEC)
            continue

        # 필요 구간만 유지
        df_seq = df_seq.tail(need_len).reset_index(drop=True)

        # ❸ 예측
        now_price = float(df_seq["close"].iloc[-pred_len - 1])
        t0 = time.time()
        prices, rois = predict(df_seq, now_price)
        dt = time.time() - t0

        idx = int(np.argmax(np.abs(rois)))
        target_roi = float(rois[idx])
        target_pr = float(prices[idx])
        target_min = idx + 1
        net_cost_pct = net_cost()
        net_roi = target_roi - net_cost_pct  # (현물; 수수료/슬리피지 반영 추정치)

        # 보기좋은 로그 (원래 스타일)
        print("=" * 80, flush=True)
        print(f"[{datetime.now()}] 🔮 preds: {[f'{x * 100:.4f}%' for x in rois]}", flush=True)
        print(
            f"🎯 Target {target_roi * 100:.4f}% @{target_min}m | now={now_price:.6f} → target≈{target_pr:.6f}",
            flush=True,
        )
        print(
            f"🧮 netCost≈{net_cost_pct * 100:.3f}% | netROI≈{net_roi * 100:.4f}% | pred_time={dt:.3f}s",
            flush=True,
        )
        print("=" * 80, flush=True)

        # ❸-보너스: 예상 순이익(원) 미리 보여주기
        try:
            krw_balance_preview = get_balance("KRW")
            krw_to_spend_preview = calculate_buy_krw_amount(now_price, krw_balance_preview, min_total)
            if krw_to_spend_preview > 0:
                exp_net_profit = expected_net_profit_krw(krw_to_spend_preview, target_roi)
                print(
                    f"💵 plan: spend≈₩{krw_to_spend_preview:,.0f} → exp.netProfit≈₩{exp_net_profit:,.0f}",
                    flush=True,
                )
        except Exception as e:
            print(f"[WARN] preview profit calc failed: {e}", flush=True)

        # ❹ 진입 (현물은 롱만)
        entered = False
        if target_roi > thresh_roi:
            # 최소 주문금액/최소 순이익 체크
            krw_balance = get_balance("KRW")
            krw_to_spend = calculate_buy_krw_amount(now_price, krw_balance, min_total)
            if krw_to_spend <= 0:
                print("❌ 최소 주문금액 부족 → 스킵", flush=True)
                log_prediction(now_price, target_pr, target_min, target_roi, net_roi, False)
                time.sleep(SLEEP_SEC)
                continue

            if expected_net_profit_krw(krw_to_spend, target_roi) < MIN_PROFIT_KRW:
                print("❌ 예상 순이익(KRW) 부족 → 스킵", flush=True)
                log_prediction(now_price, target_pr, target_min, target_roi, net_roi, False)
                time.sleep(SLEEP_SEC)
                continue

            # 시장가 매수
            try:
                od = post_order({"market": symbol, "side": "bid", "ord_type": "price", "price": str(int(krw_to_spend))})
                time.sleep(0.4)
                detail = get_order(od["uuid"])
                entry_price = avg_filled_price(detail) or get_ticker_price(symbol)
                position_volume = float(detail.get("executed_volume") or 0.0)
                if position_volume <= 0:
                    raise RuntimeError("매수 체결 수량 0")
            except Exception as e:
                print(f"❌ 매수 실패: {e}", flush=True)
                log_event("ENTRY_FAILED", "LONG", now_price, 0, krw_balance, krw_balance, note=str(e))
                log_prediction(now_price, target_pr, target_min, target_roi, net_roi, False)
                time.sleep(SLEEP_SEC)
                continue

            is_holding = True
            entered = True
            print(f"🚀 ENTRY [LONG] @ {entry_price:.6f}  vol={position_volume:.8f}", flush=True)
            log_event("ENTRY_MARKET", "LONG", entry_price, 0, krw_balance, krw_balance, f"min={target_min}")

            # ❺ 모니터링/청산
            reason, final_roi, exit_price = monitor_until_profit(entry_price, "LONG", position_volume)
            is_holding = False
            krw_after = get_balance("KRW")
            log_event(f"EXIT_{reason}", "LONG", exit_price, final_roi, krw_after, krw_after,
                      note=f"vol={position_volume:.8f}")

        # ❻ 기록 & 대기
        log_prediction(now_price, target_pr, target_min, target_roi, net_roi, entered)
        time.sleep(SLEEP_SEC)
