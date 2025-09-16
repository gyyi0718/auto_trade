# -*- coding: utf-8 -*-
"""
Bybit Futures Bot + DeepAR TP 예측 + 예측시간 내 미달성 시 강제 청산
- 시장가 진입 → 포지션/체결 확인 → 예측가 기반 TP(limit, reduceOnly)
- 예측 시간 초과 시 강제 시장가 청산
- pybit는 시세/심볼 필터만 사용, 지갑/주문/레버리지는 서명 REST로 직접 호출 (10002 완화)
"""

import os, time, math, logging, warnings, random, json, hmac, hashlib, requests
from typing import Dict, Tuple, Optional, List
from decimal import Decimal, ROUND_DOWN, InvalidOperation
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import torch
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ================== 설정 ==================
LEVERAGE = 100
ENTRY_PORTION = 0.40               # 가용잔고 중 사용 비율
CATEGORY = "linear"
CKPT_PATH = "models/multi_deepar_model.ckpt"
API_KEY    = ""
API_SECRET = ""

SEQ_LEN = 60
PRED_LEN = 60
MAX_OPEN = 4
SYMBOLS = ["ETHUSDT","BTCUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","LINKUSDT","ADAUSDT","MNTUSDT","SUIUSDT","CRVUSDT"]
REPREDICT_EVERY_SEC = 60
TICKER_BATCH = 50   # 한번에 조회할 심볼 수(과도한 대량 요청 방지)
LAST_SIGNALS = {}         # {sym: {"side": "Buy/Sell", "target": float, "eta_sec": int, "conf": float, "ts": float}}

# ================== 클라이언트 / 상수 ==================
client = HTTP(api_key=API_KEY, api_secret=API_SECRET, recv_window=300000, timeout=10, testnet=False)

BYBIT_HOST = "https://api.bybit.com"
RECV_WINDOW_MS = 300000  # 5분

positions: Dict[str, dict] = {}
_deepar_model = None
# 파일 상단 근처에 추가
LAST_SIGNALS = {}  # {symbol: {"side": "Buy/Sell", "target": float, "eta_sec": int, "now": float, "ts": time.time()}}
# === 진입 필터 파라미터 ===
TAKER_FEE = 0.0006          # 테이커 수수료(한 번)
ROUND_TRIP_FEE = TAKER_FEE * 2.0   # 진입+청산 왕복
MIN_ABS_PNL_USD = 1
MIN_CONFIDENCE = 0.9


import math

# ------- 설정(원하면 조정) -------
CONF_MIN      = 0.70     # 최소 confidence
PNL_MIN_USD   = 0.50     # 최소 예상 절대 PnL(달러)
SHARPE_MIN    = 0.15     # 최소 샤프 프록시
KELLY_CAP     = 0.25     # Kelly 상한(과도한 베팅 방지, 25%)
STOP_RATIO    = 0.5      # 손절폭 = TP거리의 50%로 가정 (R=TP/SL=2)

def _phi(z: float) -> float:
    # 표준정규 CDF (간단 근사)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def compute_sigma_pct_from_df(df: pd.DataFrame, lookback: int = 120) -> float:
    """
    분봉 log-return의 표준편차(퍼센트)를 추정 -> 분당 시그마(%).
    """
    s = df.tail(lookback)["close"]
    lr = np.log(s).diff().dropna().values
    if lr.size == 0:
        return 0.003  # 0.3% 기본값
    return float(np.std(lr))  # 이미 로그수익률이므로 '비율'

def score_signal_with_kelly_sharpe(
    now_price: float,
    target_price: float,
    side: str,                  # "Buy" or "Sell"
    horizon_sec: int,           # 예측 도달시간(초)
    est_qty: float,             # 예상 수량(진입 전 산정)
    sigma_per_min: float,       # 분당 변동성(로그수익률 표준편차)
    confidence: float,          # 0~1
) -> dict:
    """
    반환 dict:
      ok(bool), reason(str), exp_usd(float), sharpe(float),
      kelly_frac(float in [0,KELLY_CAP]), score(float 0~1)
    """
    sign = 1.0 if side == "Buy" else -1.0
    dist   = (target_price - now_price) * sign          # 유리한 방향 거리(가격)
    exp_usd = max(0.0, dist) * float(est_qty)          # 예상 절대 달러 PnL (음수면 0)
    if dist <= 0:
        return {"ok": False, "reason": "target not favorable", "exp_usd": 0.0,
                "sharpe": 0.0, "kelly_frac": 0.0, "score": 0.0}

    # 시간 스케일
    horizon_min = max(1.0, horizon_sec / 60.0)

    # 기대수익률(로그근사 아님, 단순 % 변화)
    exp_ret_pct = dist / max(now_price, 1e-9)  # ~ 비율
    # 변동성 스케일링 (√t): 분당 시그마를 horizon에 맞게 확대
    sigma_h = max(1e-9, sigma_per_min * math.sqrt(horizon_min))

    # 샤프 프록시 (드리프트=exp_ret_pct, 분산~sigma_h)
    sharpe = exp_ret_pct / sigma_h

    # 승률 추정: z=exp_ret/sigma → Phi(z)
    p_win = float(np.clip(_phi(sharpe), 0.0, 1.0))
    # 손익비(R): TP/SL, SL = STOP_RATIO * TP거리
    R = 1.0 / max(1e-9, STOP_RATIO)  # STOP_RATIO=0.5 → R=2
    q = 1.0 - p_win

    # Kelly fraction: f* = p - q/R
    kelly_raw = p_win - (q / R)
    kelly_frac = float(np.clip(kelly_raw, 0.0, KELLY_CAP))

    # 품질 점수(0~1): Sharpe, confidence, 달러 PnL를 간단히 결합
    # - Sharpe를 0~1로 눌러주는 스케일
    sharpe_norm = float(np.clip(sharpe / 1.0, 0.0, 1.0))  # 샤프 1.0 이상이면 1로 캡
    # - PnL도 완충: PNL_MIN_USD에서 1.0 근처로
    pnl_norm = float(np.clip(exp_usd / max(1e-9, 5.0 * PNL_MIN_USD), 0.0, 1.0))
    score = 0.4 * sharpe_norm + 0.4 * float(np.clip(confidence, 0.0, 1.0)) + 0.2 * pnl_norm

    # 필터링
    if confidence < CONF_MIN:
        return {"ok": False, "reason": f"conf<{CONF_MIN}", "exp_usd": exp_usd,
                "sharpe": sharpe, "kelly_frac": kelly_frac, "score": score}
    if exp_usd < PNL_MIN_USD:
        return {"ok": False, "reason": f"pnl<{PNL_MIN_USD}$", "exp_usd": exp_usd,
                "sharpe": sharpe, "kelly_frac": kelly_frac, "score": score}
    if sharpe < SHARPE_MIN:
        return {"ok": False, "reason": f"sharpe<{SHARPE_MIN}", "exp_usd": exp_usd,
                "sharpe": sharpe, "kelly_frac": kelly_frac, "score": score}

    return {"ok": True, "reason": "ok", "exp_usd": exp_usd,
            "sharpe": sharpe, "kelly_frac": kelly_frac, "score": score}


def _fmt_secs(s: int) -> str:
    s = max(int(s), 0)
    m, r = divmod(s, 60)
    return f"{m}m {r}s" if m else f"{r}s"

# ================== 공통 유틸 ==================
def _d(x) -> Decimal:
    return Decimal(str(x))

def _round_step(x: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        step = Decimal("0.001")
    return (x / step).to_integral_value(rounding=ROUND_DOWN) * step

def fnum(x, default=0.0) -> float:
    try:
        if x is None: return float(default)
        s = str(x).strip()
        if s.lower() in {"", "nan", "none", "null", "inf", "-inf"}: return float(default)
        return float(s)
    except Exception:
        return float(default)

# ================== REST 서명/요청 ==================
def _server_time_ms() -> int:
    r = requests.get(f"{BYBIT_HOST}/v5/market/time", timeout=3)
    r.raise_for_status()
    res = r.json().get("result", {}) or {}
    if "timeNano" in res:
        return int(int(res["timeNano"]) / 1_000_000)
    if "timeSecond" in res:
        return int(res["timeSecond"]) * 1000
    return int(time.time() * 1000) - 100

def _sign_v5(ts_ms: int, api_key: str, recv_window: int, payload: str, secret: str) -> str:
    to_sign = str(ts_ms) + api_key + str(recv_window) + payload
    return hmac.new(secret.encode(), to_sign.encode(), hashlib.sha256).hexdigest()

def _rest_request(method: str, path: str, query: dict | None = None, body: dict | None = None):
    url = f"{BYBIT_HOST}{path}"
    query = query or {}
    body = body or {}
    ts = _server_time_ms()
    if method.upper() == "GET":
        payload = requests.models.RequestEncodingMixin._encode_params(query)
    else:
        payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False)

    sign = _sign_v5(ts, API_KEY, RECV_WINDOW_MS, payload, API_SECRET)
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": str(ts),
        "X-BAPI-RECV-WINDOW": str(RECV_WINDOW_MS),
        "X-BAPI-SIGN": sign,
        "Content-Type": "application/json",
    }
    if method.upper() == "GET":
        resp = requests.get(url, params=query, headers=headers, timeout=10)
    else:
        resp = requests.post(url, params=query, headers=headers, data=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()

# ---- REST 래퍼들 ----
def rest_get_wallet_balance_unified() -> tuple[Decimal, Decimal]:
    data = _rest_request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})
    lst = data.get("result", {}).get("list", [])
    if not lst:
        return Decimal("0"), Decimal("0")
    top = lst[0]
    # USDT coin 우선
    avail = Decimal("0")
    for c in top.get("coin", []):
        if c.get("coin") == "USDT":
            for k in ("availableToWithdraw","availableBalance","transferBalance","equity"):
                v = c.get(k)
                if v not in (None, "", "null"):
                    avail = _d(v); break
            break
    if avail <= 0 and top.get("totalAvailableBalance") not in (None, "", "null"):
        avail = _d(top["totalAvailableBalance"])
    total = _d(top.get("totalEquity", top.get("totalWalletBalance", top.get("totalAvailableBalance", "0"))))
    return avail, total

def rest_set_leverage(symbol: str, lev: int):
    body = {"category": CATEGORY, "symbol": symbol, "buyLeverage": str(lev), "sellLeverage": str(lev)}
    return _rest_request("POST", "/v5/position/set-leverage", None, body)

def rest_place_order(category: str, symbol: str, side: str, orderType: str,
                     qty: str, price: str | None = None,
                     reduceOnly: bool | None = None, timeInForce: str | None = None):
    body = {"category": category, "symbol": symbol, "side": side, "orderType": orderType, "qty": qty}
    if price is not None:      body["price"] = price
    if reduceOnly is not None: body["reduceOnly"] = reduceOnly
    if timeInForce is not None:body["timeInForce"] = timeInForce

    res = _rest_request("POST", "/v5/order/create", None, body)

    # ★ 핵심: retCode 검사 (HTTP 200이라도 실패 가능)
    rc = res.get("retCode", res.get("ret_code"))
    if rc not in (0, "0", None):  # 일부 성공 응답은 None일 수 있음(구 버전)
        msg = res.get("retMsg", res.get("ret_msg", "unknown error"))
        raise Exception(f"BYBIT_ERR {rc}: {msg} | resp={res}")

    return res

def rest_get_open_orders(symbol: str, order_filter: Optional[str] = None, limit: int = 50):
    # orderFilter 예: "tpslOrder" 등. (필요 없으면 None)
    q = {"category": CATEGORY, "symbol": symbol, "limit": limit}
    if order_filter:
        q["orderFilter"] = order_filter
    return _rest_request("GET", "/v5/order/realtime", q)


# ================== 심볼 필터/반올림 ==================
_filters_cache: Dict[str, Tuple[Decimal, Decimal, Decimal, Decimal, Decimal]] = {}

def get_symbol_filters(symbol: str) -> Tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
    """
    반환: (qty_step, min_qty, max_qty, min_notional, tick)
    """
    if symbol in _filters_cache:
        return _filters_cache[symbol]
    r = client.get_instruments_info(category=CATEGORY, symbol=symbol)
    info = r.get("result", {}).get("list", [])
    if not info:
        res = (Decimal("0.001"), Decimal("0.001"), Decimal("999999999"), Decimal("0"), Decimal("0.0001"))
        _filters_cache[symbol] = res
        return res
    info = info[0]
    lot = info.get("lotSizeFilter", {})
    pricef = info.get("priceFilter", {})
    def sdec(v, d="0"):
        try: return _d(v)
        except: return _d(d)
    qty_step     = sdec(lot.get("qtyStep"),     "0.001")
    min_qty      = sdec(lot.get("minOrderQty"), "0.001")
    max_qty      = sdec(lot.get("maxOrderQty"), "999999999")
    min_notional = sdec(lot.get("minNotional"), "0")
    tick         = sdec(pricef.get("tickSize"), "0.0001")
    if qty_step <= 0: qty_step = Decimal("0.001")
    if min_qty  <= 0: min_qty  = qty_step
    if tick     <= 0: tick     = Decimal("0.0001")
    res = (qty_step, min_qty, max_qty, min_notional, tick)
    _filters_cache[symbol] = res
    return res

def quantize_qty(symbol: str, qty_float: float) -> Decimal:
    qty_step, min_qty, *_ = get_symbol_filters(symbol)
    q = _round_step(_d(qty_float), qty_step)
    if q < min_qty: q = min_qty
    return q

def quantize_price(symbol: str, price_float: float) -> Decimal:
    *_, tick = get_symbol_filters(symbol)
    return _round_step(_d(price_float), tick)

# ================== 시세/잔고 ==================
class BalanceCache:
    def __init__(self, ttl_sec=20):
        self.ttl = ttl_sec
        self.last_ts = 0.0
        self.avail = Decimal("0")
        self.total = Decimal("0")
BAL_CACHE = BalanceCache(ttl_sec=20)

def get_available_balance_cached() -> float:
    now = time.time()
    if now - BAL_CACHE.last_ts <= BAL_CACHE.ttl and BAL_CACHE.avail > 0:
        return float(BAL_CACHE.avail)
    try:
        avail, total = rest_get_wallet_balance_unified()
        BAL_CACHE.avail, BAL_CACHE.total, BAL_CACHE.last_ts = avail, total, now
        return float(avail)
    except Exception as e:
        print(f"[WARN] balance refresh failed: {e}")
        return float(BAL_CACHE.avail) if BAL_CACHE.avail > 0 else 0.0

def fetch_recent_data(symbol: str, limit: int = 120) -> pd.DataFrame:
    res = client.get_kline(category=CATEGORY, symbol=symbol, interval="1", limit=limit)
    rows = res["result"]["list"]
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df = df[["timestamp","open","high","low","close","volume"]].copy()
    df["series_id"] = symbol
    return df

def get_last_price(symbol: str) -> float:
    res = client.get_tickers(category=CATEGORY, symbol=symbol)
    row = res.get("result", {}).get("list", [{}])[0]
    for k in ("lastPrice","markPrice","bid1Price"):
        p = fnum(row.get(k), 0.0)
        if p > 0: return p
    raise ValueError(f"{symbol} 현재가 조회 실패: {row}")

# ================== 모델/예측 ==================
def load_deepar_model():
    global _deepar_model
    if _deepar_model is None:
        _deepar_model = DeepAR.load_from_checkpoint(CKPT_PATH)



def _extract_quantile_array(pred_out):
    """
    pred_out을 안전하게 numpy (B, T, Q)로 변환
    허용 입력: torch.Tensor, np.ndarray, list, dict({'prediction': ...})
    """
    # dict 형태 처리
    if isinstance(pred_out, dict):
        for k in ("prediction", "predictions", "output"):
            if k in pred_out:
                pred_out = pred_out[k]
                break

    # list -> array
    if isinstance(pred_out, list):
        pred_out = np.array(pred_out)

    # torch.Tensor -> numpy
    if isinstance(pred_out, torch.Tensor):
        arr = pred_out.detach().cpu().numpy()
    elif isinstance(pred_out, np.ndarray):
        arr = pred_out
    else:
        # 예외: 모르는 형식이면 한 번 더 np.array 시도
        arr = np.array(pred_out)

    # 차원 보정
    # 가능한 경우: (T, Q) / (B, T, Q) / (Q, T) 등
    if arr.ndim == 2:
        # (T, Q) 또는 (Q, T) 가정 → (B=1, T, Q)로 만듦
        # Q가 3개(0.1,0.5,0.9)일 확률이 높으니 마지막 축이 Q가 아니면 전치
        if arr.shape[1] in (3, 5):  # 3 or 5 quantiles typical
            arr = arr[None, :, :]
        elif arr.shape[0] in (3, 5):
            arr = arr.transpose(1, 0)[None, :, :]
        else:
            # 모르면 마지막 축을 Q로 보고 B 차원만 추가
            arr = arr[None, :, :]
    elif arr.ndim == 3:
        # (B, T, Q) 또는 (B, Q, T)
        B, A, C = arr.shape
        # Q가 3/5면 그 축을 Q로 맞추도록 전치
        if C in (3, 5):
            pass  # (B, T, Q)로 보정 완료
        elif A in (3, 5):
            arr = arr.transpose(0, 2, 1)  # (B, T, Q)
        else:
            # 불명확하면 그냥 (B, T, Q)로 간주
            pass
    else:
        # 다른 특이 케이스 → 1D/4D 등은 에러
        raise RuntimeError(f"Unexpected prediction shape: {arr.shape}")

    # 최종 (B, T, Q) 보장
    return arr


def predict_future_price(symbol: str, df: pd.DataFrame) -> Tuple[float, int, float]:
    """
    p50 경로에서 'TP 보정 이후' |TP - now|가 가장 큰 지점 선택.
    return: (best_target_price, best_target_sec, confidence_at_best)
    """
    load_deepar_model()
    df = df.sort_values("timestamp").copy()
    df["time_idx"] = np.arange(len(df))
    df["series_id"] = symbol
    df["log_return"] = np.log(df["close"]).diff().fillna(0.0)

    enc = df.iloc[-SEQ_LEN:].copy()
    enc["time_idx"] -= enc["time_idx"].min()

    fut = pd.DataFrame({
        "time_idx": np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN),
        "series_id": symbol,
        "log_return": [0.0] * PRED_LEN,
    })
    combined = pd.concat([enc, fut], ignore_index=True)

    dataset = TimeSeriesDataSet(
        combined,
        time_idx="time_idx",
        target="log_return",
        group_ids=["series_id"],
        max_encoder_length=SEQ_LEN,
        max_prediction_length=PRED_LEN,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["log_return"],
        static_categoricals=["series_id"],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    dl = dataset.to_dataloader(train=False, batch_size=1)

    now_price = get_last_price(symbol)

    # p50, p10, p90 경로
    try:
        # 1차 시도: mode_kwargs로 전달 (버전 호환)
        q_raw = _deepar_model.predict(
            dl,
            mode="quantiles",
            mode_kwargs={"quantiles": [0.1, 0.5, 0.9]},
            return_x=False
        )
    except TypeError:
        # 일부 버전은 quantiles를 최상위로 받음
        q_raw = _deepar_model.predict(
            dl,
            mode="quantiles",
            quantiles=[0.1, 0.5, 0.9],
            return_x=False
        )

    def to_price_path(logret):
        lr_cum = np.cumsum(logret)
        return now_price * np.exp(lr_cum)

    q_arr = _extract_quantile_array(q_raw)  # (B, T, Q)
    # 배치 1 가정
    q10 = q_arr[0, :, 0]
    q50 = q_arr[0, :, 1]
    q90 = q_arr[0, :, 2]

    p10 = to_price_path(q10)
    p50 = to_price_path(q50)
    p90 = to_price_path(q90)

    # 각 시점별: 방향 결정(+보정) → 절대 기대이익(= |TP_adj - now|) 최대 인덱스 선택
    best_idx = 0
    best_gain = -1.0
    for i in range(len(p50)):
        raw_tp = float(p50[i])
        # 방향은 "예측값 vs 현재가"로 결정
        side = "Buy" if raw_tp > now_price else "Sell"

        # 엔트리 전에 쓰던 TP 보정 규칙 그대로 적용
        if side == "Buy":
            tp_adj = max(raw_tp, round(now_price * 1.01, 3))
        else:
            tp_adj = min(raw_tp, round(now_price * 0.99, 3))

        gain = abs(tp_adj - now_price)  # 수량은 결정시 거의 상수 → Δ가격만 최적화

        if gain > best_gain:
            best_gain = gain
            best_idx = i
            best_tp_adj = tp_adj

    # 선택된 시점의 신뢰도(밴드 폭 기반)
    band = float(p90[best_idx] - p10[best_idx])
    rel_unc = band / max(now_price, 1e-9)
    cap = 0.01
    confidence = float(np.clip(1.0 - (rel_unc / cap), 0.0, 1.0))

    target_price = float(best_tp_adj)
    target_sec = (best_idx + 1) * 60  # 몇 분 뒤인지
    return target_price, target_sec, confidence

# ================== 포지션/주문 ==================
def ensure_leverage(symbol: str, lev: int = LEVERAGE):
    try:
        rest_set_leverage(symbol, lev)
    except Exception as e:
        print(f"[WARN] set_leverage({symbol}) → {e}")

def get_position_info(symbol: str) -> Optional[dict]:
    """
    현재 심볼의 포지션(사이즈>0) 1개만 반환
    """
    try:
        r = rest_get_positions(symbol)
        for pos in r.get("result", {}).get("list", []):
            sz = pos.get("size")
            ep = pos.get("entryPrice")
            side = pos.get("side")
            if not sz or not ep or side not in ("Buy", "Sell"):
                continue
            size = float(sz)
            if size > 0:
                return {"entry_price": float(ep), "side": side, "size": size}
        return None
    except Exception as e:
        print(f"[ERROR] get_position_info 실패(REST): {e}")
        return None
# === (추가) 오픈 오더 조회/취소 ===
def rest_get_open_orders(symbol: str):
    # v5 realtime: 현재 열린 주문
    return _rest_request("GET", "/v5/order/realtime",
                         {"category": CATEGORY, "symbol": symbol})

def rest_cancel_order(symbol: str, order_id: str):
    body = {"category": CATEGORY, "symbol": symbol, "orderId": order_id}
    return _rest_request("POST", "/v5/order/cancel", None, body)

def wait_for_position(symbol: str, expect_side: str, timeout_s: float = 8.0, order_id: Optional[str] = None) -> Optional[dict]:
    """
    시장가 주문 직후 포지션 반영될 때까지 REST로 폴링.
    실패 시 체결내역으로 평균가/수량 폴백.
    """
    t0 = time.time()
    # 1) 포지션 폴링
    while time.time() - t0 < timeout_s:
        pos = get_position_info(symbol)
        if pos and pos.get("side") in ("Buy","Sell") and float(pos.get("size", 0)) > 0:
            return pos
        time.sleep(0.35 + random.uniform(0, 0.2))

    # 2) 체결 폴백 (order_id 가 있으면 정확히, 없으면 최근 체결)
    try:
        ex = rest_get_executions(symbol, order_id=order_id, limit=50)
        fills = ex.get("result", {}).get("list", [])
        total_qty = 0.0
        total_quote = 0.0
        last_side = None
        for f in fills:
            q = fnum(f.get("execQty"), 0.0)
            p = fnum(f.get("execPrice"), 0.0)
            s = f.get("side")
            if q > 0 and p > 0:
                total_qty += q
                total_quote += q * p
                last_side = s
        if total_qty > 0:
            return {"entry_price": total_quote / total_qty, "side": last_side or expect_side, "size": total_qty}
    except Exception as e:
        print(f"[WARN] executions 폴백 실패: {e}")

    print("[ERR] wait_for_position timeout")
    return None
# (교체) TP 지정가 → orderId 반환


def rest_amend_order(category: str, symbol: str, order_id: str,
                     price: str | None = None, qty: str | None = None):
    body = {"category": category, "symbol": symbol, "orderId": order_id}
    if price is not None: body["price"] = price
    if qty   is not None: body["qty"]   = qty
    return _rest_request("POST", "/v5/order/amend", None, body)

def rest_cancel_order(category: str, symbol: str, order_id: str):
    body = {"category": category, "symbol": symbol, "orderId": order_id}
    return _rest_request("POST", "/v5/order/cancel", None, body)

# (추가) TP 업서트: target 변동/유실 시 강제 적용
def upsert_tp_order(symbol: str, side: str, qty_float: float, new_target_price: float,
                    prev_order_id: Optional[str] = None, diff_pct_threshold: float = 0.001):
    """
    - side: 진입 방향(포지션 방향). TP는 반대 방향으로 reduceOnly.
    - prev_order_id가 있으면 amend를 먼저 시도, 실패하면 cancel→새로 생성.
    - diff_pct_threshold: 기존 target과의 차이가 이 비율(=0.1%)보다 크면 가격 갱신.
    return: (tp_order_id, action)  action in {"placed","amended","replaced","skipped"}
    """
    opp = "Sell" if side == "Buy" else "Buy"
    qty_q = quantize_qty(symbol, qty_float)
    px_q  = quantize_price(symbol, new_target_price)

    # positions에 저장된 이전 target과 비교해 너무 자잘한 변동은 스킵(틱 2칸 + 0.1% 기준)
    prev = positions.get(symbol)
    if prev:
        old_px = float(prev.get("target_price", 0.0))
        # 틱 2칸 또는 비율 중 큰 것
        *_ , tick = get_symbol_filters(symbol)
        tick_guard = float(tick * 2)
        diff = abs(float(px_q) - old_px)
        rel  = diff / max(old_px, 1e-9)
        if diff < max(tick_guard, old_px * diff_pct_threshold):
            return prev.get("tp_order_id"), "skipped"

    # 1) 이전 주문이 있으면 amend 먼저
    if prev_order_id:
        try:
            rest_amend_order(CATEGORY, symbol, prev_order_id, price=str(px_q), qty=str(qty_q))
            print(f"[TP-AMEND] {symbol} → {opp} {qty_q} @ {px_q} (orderId={prev_order_id})")
            return prev_order_id, "amended"
        except Exception as e:
            print(f"[WARN] TP amend 실패: {e} → cancel & replace 시도")

            # 실패시 cancel
            try:
                rest_cancel_order(CATEGORY, symbol, prev_order_id)
                time.sleep(0.15 + random.uniform(0, 0.1))
            except Exception as ce:
                print(f"[WARN] TP cancel 실패(무시 가능): {ce}")

    # 2) 새로 생성
    try:
        resp = rest_place_order(CATEGORY, symbol, opp, "Limit", str(qty_q),
                                price=str(px_q), reduceOnly=True, timeInForce="GTC")
        new_oid = resp.get("result", {}).get("orderId")
        print(f"[TP-SET] {symbol} {opp} {qty_q} @ {px_q} (orderId={new_oid})")
        return new_oid, ("replaced" if prev_order_id else "placed")
    except Exception as e:
        print(f"[ERROR] {symbol} TP 주문 실패: {e}")
        return prev_order_id, "error"
# === add: cancel-all (symbol 전체 오더 취소) ===
def rest_cancel_all(symbol: str):
    try:
        return _rest_request("POST", "/v5/order/cancel-all", None, {"category": CATEGORY, "symbol": symbol})
    except Exception as e:
        print(f"[WARN] cancel-all 실패({symbol}): {e}")
        return None

def place_tp_limit(symbol: str, side: str, qty_float: float, price_float: float):
    opp = "Sell" if side == "Buy" else "Buy"
    qty_q = quantize_qty(symbol, qty_float)
    px_q  = quantize_price(symbol, price_float)
    try:
        resp = rest_place_order(
            CATEGORY, symbol, opp, "Limit",
            str(qty_q), price=str(px_q),
            reduceOnly=True, timeInForce="GTC"
        )
        oid = resp.get("result", {}).get("orderId")
        print(f"[TP-SET] {symbol} {opp} {qty_q} @ {px_q} (orderId={oid})")
        return oid
    except Exception as e:
        print(f"[ERROR] {symbol} TP 주문 실패: {e}")
        return None


# 진입: 가용잔고(avail_usdt) 인자로 받아서 심볼별 지갑 재조회 안함
ENTRY_TAKER_FEE = Decimal("0.0006")   # 대략 추정
SAFETY_BUFFER   = Decimal("0.015")    # 1.5% 버퍼

def _max_qty_from_avail(avail_usdt: Decimal, price: Decimal, lev: int,
                        qty_step: Decimal, min_qty: Decimal) -> Decimal:
    denom = price * (Decimal(1)/Decimal(lev) + ENTRY_TAKER_FEE) * (Decimal(1) + SAFETY_BUFFER)
    if denom <= 0: return Decimal("0")
    raw = avail_usdt / denom
    q = _round_step(raw, qty_step)
    if q < min_qty: q = min_qty
    return q

def place_market_safely(symbol: str, side: str, use_usdt: float, leverage: int = LEVERAGE) -> Tuple[Optional[str], Optional[str], Optional[Decimal]]:
    ensure_leverage(symbol, leverage)
    price = _d(get_last_price(symbol))
    qty_step, min_qty, max_qty, min_notional, tick = get_symbol_filters(symbol)

    use = _d(use_usdt)
    if use <= 0:
        return None, "[ABORT] 주문 금액 0", None

    qty = _max_qty_from_avail(use, price, leverage, qty_step, min_qty)

    if min_notional > 0 and qty * price < min_notional:
        qty = _round_step((min_notional / price) + qty_step, qty_step)

    if qty <= 0:
        return None, "[ABORT] 최종 수량 0", None

    def _try(q: Decimal):
        try:
            resp = rest_place_order(CATEGORY, symbol, side, "Market", str(q))
            oid = resp.get("result", {}).get("orderId")
            print(f"[ENTRY] {symbol} {side} qty={q} @ ~{price}")
            return oid, None
        except Exception as e:
            s = str(e)
            if "110007" in s or "not enough" in s.lower():
                return None, "110007"
            if "10001" in s and "Qty invalid" in s:
                return None, "10001"
            return None, f"REST_ERR:{s}"

    oid, err = _try(qty)
    if not err:
        return oid, None, qty

    if err == "110007":
        q = qty
        for i in range(4):
            q = _round_step(q * Decimal("0.7"), qty_step)
            if q < min_qty: break
            oid2, err2 = _try(q)
            if not err2:
                print(f"[ENTRY] {symbol} {side} qty={q} (백오프 {i+1})")
                return oid2, None, q
        return None, f"[FAIL] 110007 지속", None

    if err == "10001":
        return None, "[FAIL] 10001 규격 오류", None

    # 기타 에러면 포기
    return None, err, None

import random

def sync_server_time(max_tries: int = 3) -> bool:
    """
    Bybit 서버시간을 미리 한 번 호출해서 로컬-서버 타임슬립 여유를 만든다.
    pybit 구버전에서 10002를 줄이는 데 도움.
    """
    for _ in range(max_tries):
        try:
            r = requests.get("https://api.bybit.com/v5/market/time", timeout=3)
            if r.ok:
                # 100~250ms 랜덤 대기 → 다음 요청 타임스탬프가 서버보다 살짝 뒤로 가도록
                time.sleep(0.1 + random.random() * 0.15)
                return True
        except Exception:
            pass
    return False

def call_with_timesync(fn, *args, **kwargs):
    """
    API 호출 래퍼: 10002/400 같은 타임스탬프 예외가 보이면
    서버시간 프리싱크 후 지수백오프로 재시도.
    """
    last_err = None
    for i in range(4):  # 최대 4회
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            s = str(e)
            last_err = s
            if "ErrCode: 10002" in s or "retries exceeded maximum" in s or "Bad request" in s:
                sync_server_time()
                time.sleep(0.5 + 0.3 * i)  # 0.5s, 0.8s, 1.1s, 1.4s
                continue
            raise
    raise RuntimeError(f"[call_with_timesync] retries exhausted: {last_err}")
def close_position_market(symbol: str, side: str, qty_float: float):
    """
    포지션 조회 결과가 None이어도 reduceOnly 시장가로 '그냥' 닫기 시도.
    - 먼저 cancel-all로 잔여 TP/대기 주문 제거
    - reduceOnly Market 주문 시도 (REST → pybit 폴백)
    - 에러가 '포지션 없음/수량 부족' 류면 이미 닫힌 것으로 간주
    - 마지막에 포지션/체결 확인 후 로그
    """
    try:
        # 0) 잔여 주문 제거(충돌 방지)
        rest_cancel_all(symbol)

        # 1) 남은 수량 기준으로 그냥 reduceOnly 시장가 클로즈 시도
        qty = quantize_qty(symbol, qty_float)
        opp = "Sell" if side == "Buy" else "Buy"

        def _try_reduce_only():
            try:
                return rest_place_order(
                    category=CATEGORY, symbol=symbol, side=opp,
                    orderType="Market", qty=str(qty),
                    reduceOnly=True  # 핵심: reduceOnly!
                )
            except Exception as e:
                # 대표적인 메시지: position size not enough / order qty exceeds position qty
                s = str(e).lower()
                if any(x in s for x in ["position", "not enough", "exceed", "does not exist"]):
                    return "NO_POSITION"
                # pybit 폴백
                try:
                    return call_with_timesync(
                        client.place_order,
                        category=CATEGORY, symbol=symbol, side=opp,
                        orderType="Market", qty=str(qty),
                        reduceOnly=True
                    )
                except Exception as ee:
                    s2 = str(ee).lower()
                    if any(x in s2 for x in ["position", "not enough", "exceed", "does not exist"]):
                        return "NO_POSITION"
                    raise

        r = _try_reduce_only()
        if r == "NO_POSITION":
            print(f"[EXIT] {symbol} 이미 포지션 없음으로 간주(side={side})")
            return

        print(f"[EXIT] {symbol} 시장가 청산 side={side} qty={qty}")
    except Exception as e:
        print(f"[ERROR] {symbol} 시장가 청산 실패: {e}")

def rest_get_positions(symbol: str):
    return _rest_request("GET", "/v5/position/list", {"category": CATEGORY, "symbol": symbol})

def rest_get_executions(symbol: str, order_id: Optional[str] = None, limit: int = 50):
    q = {"category": CATEGORY, "symbol": symbol, "limit": limit}
    if order_id:
        q["orderId"] = order_id
    return _rest_request("GET", "/v5/execution/list", q)

def show_targets_heartbeat():
    if not LAST_SIGNALS:
        return
    lines = []
    lines.append("┌─ TARGET BOARD ───────────────────────────────────────────┐")
    lines.append("│  SYMBOL   SIDE   NOW         TARGET      ETA             │")
    lines.append("├─────────────────────────────────────────────────────────┤")
    for sym, s in LAST_SIGNALS.items():
        side  = s["side"][:4]
        now   = s["now"]
        tgt   = s["target"]
        eta   = _fmt_secs(int(s["eta_sec"]))
        lines.append(f"│  {sym:<7}  {side:<4}  {now:>10.3f}  →  {tgt:>10.3f}  in {eta:<8}  │")
    lines.append("└─────────────────────────────────────────────────────────┘")
    print("\n".join(lines))

# ================== 엔트리/모니터 ==================
def try_enter(symbol: str, avail_usdt: float, min_abs_pnl_usd: float = 0.10, min_conf: float = 0.70):
    """
    - TP는 최초 1회만 설정 (이미 오픈/TP 있으면 재설정 안 함)
    - 예측된 분(target_min)과 함께 [TRACK]에 기록
    """
    try:
        # 이미 트래킹 중이면 스킵
        if symbol in positions:
            return

        # 1) 예측
        df = fetch_recent_data(symbol, limit=120)
        target_price, target_sec, confidence = predict_future_price(symbol, df)

        if confidence < MIN_CONFIDENCE:
            print(f"[SKIP] {symbol} confidence={confidence:.2f} < {MIN_CONFIDENCE}")
            return
        else:
            print(f"[OK] {symbol} confidence={confidence:.2f} < {MIN_CONFIDENCE}")
        # 2) 현재가/방향
        now_price = get_last_price(symbol)
        side = "Buy" if target_price > now_price else "Sell"

        # 신뢰도 필터
        if confidence < min_conf:
            print(f"[SKIP] {symbol} confidence={confidence:.2f} < {min_conf:.2f}")
            return

        # 3) 사용할 예산
        use_usdt = float(avail_usdt) * float(ENTRY_PORTION)
        if use_usdt <= 0:
            print(f"[SKIP] {symbol} 진입 실패: 가용 자산 부족")
            return

        # 4) 예상 수량과 기대수익 체크(달러)
        price_dec = Decimal(str(now_price))
        qty_step, min_qty, max_qty, min_notional, tick = get_symbol_filters(symbol)
        exp_qty_dec = _max_qty_from_avail(Decimal(str(use_usdt)), price_dec, LEVERAGE, qty_step, min_qty)
        if min_notional > 0 and exp_qty_dec * price_dec < min_notional:
            exp_qty_dec = _round_step((min_notional / price_dec) + qty_step, qty_step)
        if exp_qty_dec <= 0:
            print(f"[SKIP] {symbol} 예상 수량 0 (budget={use_usdt:.4f}$)")
            return
        if max_qty > 0 and exp_qty_dec > max_qty:
            exp_qty_dec = _round_step(max_qty, qty_step)

        exp_qty = float(exp_qty_dec)

        # TP 보정 미리 적용(체크용)
        if side == "Buy":
            tp_check = max(target_price, round(now_price * 1.01, 3))
        else:
            tp_check = min(target_price, round(now_price * 0.99, 3))

        exp_pnl_usd = abs(tp_check - now_price) * exp_qty
        if exp_pnl_usd < min_abs_pnl_usd:
            print(f"[SKIP] {symbol} 기대수익 부족: ≈{exp_pnl_usd:.2f}$ < {min_abs_pnl_usd:.2f}$ "
                  f"(qty≈{exp_qty}, entry≈{now_price}, target≈{tp_check})")
            return
        else:
            print(f"[OK] {symbol} 진입조건 충족: 예상수익≈{exp_pnl_usd:.2f}$ "
                  f"(qty≈{exp_qty:.4g}, entry≈{now_price}, target≈{tp_check})")

        # 5) 실제 진입
        # === 품질 평가: Sharpe + Kelly + confidence + 예상PnL ===
        sigma_per_min = compute_sigma_pct_from_df(df, lookback=120)

        # “보정된 TP” 미리보기(너 이미 쓰던 함수)
        tp_preview = adjust_tp_preview(side, now_price, target_price)

        # 예상 수량(너 기존 로직)
        price_dec = _d(now_price)
        qty_step, min_qty, max_qty, min_notional, tick = get_symbol_filters(symbol)
        est_qty_dec = _max_qty_from_avail(_d(use_usdt), price_dec, LEVERAGE, qty_step, min_qty)
        if min_notional > 0 and est_qty_dec * price_dec < min_notional:
            est_qty_dec = _round_step((min_notional / price_dec) + qty_step, qty_step)
        if est_qty_dec <= 0:
            print(f"[SKIP] {symbol} 예상 수량 0"); return
        if max_qty > 0 and est_qty_dec > max_qty:
            est_qty_dec = _round_step(max_qty, qty_step)
        est_qty = float(est_qty_dec)

        qm = score_signal_with_kelly_sharpe(
            now_price=now_price,
            target_price=tp_preview,
            side=side,
            horizon_sec=int(target_sec),
            est_qty=est_qty,
            sigma_per_min=sigma_per_min,
            confidence=confidence,
        )

        print(f"[QUALITY] {symbol} conf={confidence:.2f} sharpe={qm['sharpe']:.3f} "
              f"kelly={qm['kelly_frac']:.3f} expPnL≈{qm['exp_usd']:.2f}$ score={qm['score']:.2f} ({qm['reason']})")

        if not qm["ok"]:
            print(f"[SKIP] {symbol} 품질필터 미통과 → {qm['reason']}")
            return

        # Kelly로 진입 금액 스케일 (ENTRY_PORTION × Kelly)
        scaled_usdt = float(ENTRY_PORTION) * float(qm["kelly_frac"]) * float(avail_usdt)
        # 최소 보정: 너무 작으면 원래 use_usdt의 일부라도 쓰자
        use_usdt_final = max(scaled_usdt, 0.25 * use_usdt)

        # === 실제 진입 ===
        oid, err, qty_dec = place_market_safely(symbol, side, use_usdt_final, LEVERAGE)
        if err:
            print(f"[ENTRY-FAIL] {symbol} {err}"); return


        # 6) 포지션 확인
        pos = wait_for_position(symbol, side, 8.0, order_id=oid)
        if not pos:
            print(f"[ERROR] {symbol} 진입 실패: 포지션 정보 없음")
            return
        entry_price = float(pos["entry_price"])
        pos_qty     = float(pos["size"])

        # 7) TP 최종 보정(최초 1회만)
        if side == "Buy":
            target_price = max(target_price, round(entry_price * 1.01, 3))
        else:
            target_price = min(target_price, round(entry_price * 0.99, 3))

        tp_order_id = place_tp_limit(symbol, side, pos_qty, target_price)

        # 8) 포지션 트래킹 저장
        # 포지션 트래킹 (추가 필드 저장)
        positions[symbol] = {
            "entry_price": entry_price,
            "target_price": target_price,
            "target_time": time.time() + target_sec,
            "qty": pos_qty,
            "side": side,
            "predicted_sec": target_sec,
            "conf": float(confidence),
            "kelly": float(qm["kelly_frac"]),
            "sharpe": float(qm["sharpe"]),
        }
        print(f"[TRACK] {symbol} side={side} entry={entry_price} "
              f"target={target_price} in {target_sec}s (conf={confidence:.2f}, kelly={qm['kelly_frac']:.2f})")


    except Exception as e:
        print(f"[ERROR] {symbol} 진입 실패: {e}")

# (추가) 재예측 간격/업데이트 임계치
RE_PREDICT_SEC   = 45          # 45초마다 재예측
TARGET_UPDATE_BP = 0.0008      # 8bp(0.08%) 이상 변하면 TP 갱신

# ================== 모니터링 ==================
def _secs_left(ts):
    return int(max(0, ts - time.time()))
def rest_cancel_all(symbol: str):
    try:
        return _rest_request("POST", "/v5/order/cancel-all", None, {"category": CATEGORY, "symbol": symbol})
    except Exception as e:
        print(f"[WARN] cancel-all 실패({symbol}): {e}")
        return None

def monitor_positions():
    to_remove = []
    for sym, pos in list(positions.items()):
        try:
            price = get_last_price(sym)
            side  = pos["side"]
            qty   = float(pos["qty"])
            tp    = float(pos["target_price"])

            # 보기용 로그 (남겨둬도 무방)
            print(f"[TARGET] {sym} side={side} target={tp:.3f} price={price:.3f}")

            # A) 방향 뒤집힘이면 즉시 시장가 청산
            flipped = maybe_close_on_direction_flip(sym, pos)
            if flipped:
                to_remove.append(sym)
                continue

            # B) TP 선도달이면 시장가 백업 청산 (원하면 이 블록 유지)
            if (side == "Buy" and price >= tp) or (side == "Sell" and price <= tp):
                print(f"[TP-HIT] {sym} → 시장가 정리")
                # 열려있는 TP/대기오더가 있으면 먼저 정리(선택)
                try:
                    rest_cancel_all(sym)
                except Exception:
                    pass
                close_position_market(sym, side, qty)
                to_remove.append(sym)
                continue

            # ※ 타임아웃(시간초과) 청산은 완전히 제거

        except Exception as e:
            print(f"[MONITOR-ERR] {sym}: {e}")

    for sym in to_remove:
        positions.pop(sym, None)

# ================== 보조 ==================
def get_top_symbols_by_volume(symbols: list, top_n: int = 5) -> List[str]:
    url = f"{BYBIT_HOST}/v5/market/tickers"
    data = requests.get(url, params={"category": CATEGORY}, timeout=10).json().get("result", {}).get("list", [])
    filtered = [s for s in data if s["symbol"] in symbols]
    sorted_by_vol = sorted(filtered, key=lambda x: float(x["turnover24h"]), reverse=True)
    return [s["symbol"] for s in sorted_by_vol[:top_n]]


def adjust_tp_preview(side: str, now_price: float, predicted_target: float) -> float:
    """
    엔트리 전 미리 TP를 실제와 동일 규칙으로 보정해 '체크용' TP를 만든다.
    - Buy  : min  +1% (now*1.01) 이상
    - Sell : max  -1% (now*0.99) 이하
    """
    if side == "Buy":
        min_tp = round(now_price * 1.01, 3)
        return max(predicted_target, min_tp)
    else:
        max_tp = round(now_price * 0.99, 3)
        return min(predicted_target, max_tp)

def repredict_signal(symbol: str) -> dict:
    """
    심볼 재예측 → {'side','target','eta_sec','conf','now_price','ts'}
    """
    df = fetch_recent_data(symbol, limit=120)
    target_price, target_sec, confidence = predict_future_price(symbol, df)
    now_price = get_last_price(symbol)
    side = "Buy" if target_price > now_price else "Sell"
    sig = {
        "side": side,
        "target": float(target_price),
        "eta_sec": int(target_sec),
        "conf": float(confidence),
        "now": float(now_price),
        "ts": time.time(),
    }
    LAST_SIGNALS[symbol] = sig
    return sig

def maybe_close_on_direction_flip(symbol: str, pos: dict):
    """
    일정 주기마다 재예측. 예측 방향이 포지션 방향과 반대면 즉시 시장가 청산.
    """
    # 주기 체크
    last = LAST_SIGNALS.get(symbol, {})
    if last and time.time() - last.get("ts", 0) < REPREDICT_EVERY_SEC:
        return False  # 아직 재예측 주기 미도달

    sig = repredict_signal(symbol)
    old_side = pos["side"]
    new_side = sig["side"]

    if (old_side == "Buy" and new_side == "Sell") or (old_side == "Sell" and new_side == "Buy"):
        # 방향 뒤집힘 → 즉시 청산
        qty = float(pos["qty"])
        print(f"[FLIP-EXIT] {symbol} {old_side}→{new_side} (conf={sig['conf']:.2f}) → 시장가 청산")
        rest_cancel_all(symbol)         # 남아있을 TP/대기 주문 정리(있다면)
        close_position_market(symbol, old_side, qty)
        return True

    # 방향 유지 → 아무 것도 하지 않음 (TP/타임아웃 무시 전략이라면 그대로 홀딩)
    return False

# ================== 메인 ==================
if __name__ == "__main__":
    while True:
        try:
            avail_usdt = get_available_balance_cached()
            top_symbols = get_top_symbols_by_volume(SYMBOLS, top_n=MAX_OPEN)
            print(f"[INFO] 거래량 상위 {MAX_OPEN}개 심볼: {top_symbols} | avail≈{avail_usdt:.4f} USDT")

            # (★추가) 현황 보드 출력
            show_targets_heartbeat()

            for sym in top_symbols:
                if sym not in positions:
                    try_enter(sym, avail_usdt)

            monitor_positions()
            time.sleep(5)
        except Exception as e:
            print(f"[LOOP ERROR] {e}")
            time.sleep(10)

