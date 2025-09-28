
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import os, math, time, requests, numpy as np, pandas as pd, torch, torch.nn as nn
from collections import deque
from decimal import Decimal, ROUND_DOWN
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.deepar import DeepAR
from binance.client import Client

# ================== USER CONFIG ==================
symbol     = "ETHUSDT"
CKPT_FILE  = f"{symbol}_deepar_model.ckpt"     # DeepAR 체크포인트(필수)
INPUT_CSV  = f"{symbol}_deepar_input.csv"      # 학습 스키마/통계 재사용(필수)
PROFIT_CLS_PATH = f"{symbol}_profit_cls.pt"    # Profit 게이트(옵션: 없으면 비활성)
interval   = "1m"

# 모델/전략 파라미터(실거래와 동일)
seq_len    = 60
pred_len   = 60

# ===== 실시간 엑싯 모드 =====
REALTIME_EXIT = True     # 보유 중에 실시간으로 TP/SL/Trail/BE 판정
POLL_MS       = 10       # 실시간 가격 폴링 주기(ms)

# 수수료/슬리피지 (소수)
TAKER_FEE_ONEWAY = 0.0004
SLIPPAGE_ONEWAY  = 0.0005
FEE_ROUNDTRIP    = TAKER_FEE_ONEWAY * 2
SLIP_ROUNDTRIP   = SLIPPAGE_ONEWAY  * 2

LEVERAGE       = 20
ENTRY_PORTION  = 0.8       # 목표 진입 비율(이론치) — 실제는 lot 강화로 eff_portion으로 반영
MIN_PROFIT_USD = 0.01
thresh_roi     = 0.0001    # px ROI 문턱(0.01%=0.0001)

# Risk controls (px 기준)
TP_PX         = 0.0030     # 0.30%
SL_PX         = 0.0010     # 0.10%
TRAIL_PX      = 0.0015     # 0.15%
LIQ_BUFFER    = 0.02       # (실거래 보호용, 여기선 미사용)
GAP_GUARD_PX  = 0.0025     # (참고) 진입 직후 과도한 갭 방어

# Profit 게이트 컷오프(확률) — 옵션
P_PROFIT_CUTOFF = 0.60

# 초기자본
INITIAL_CAPITAL = 10

# 저장 파일
EQUITY_CSV = f"{symbol}_livebt_equity.csv"
TRADES_CSV = f"{symbol}_livebt_trades.csv"

# ===== TESTNET SWITCH & CLIENT =====
USE_TESTNET = True
API_KEY= "vdRj1Ta31Ntn1XOSWwZgaiVp1TfZDe1onjdlnwMZpy5epXppunKQGeJTOeqZqVmz"
API_SECRET =  "zMoSP3VQcwG5nCysDCw6Xw0PV2EJX2lf86Zga2Udato6Aqyu8QL5lO1Hyn5hQPdM"

client = Client(API_KEY, API_SECRET, testnet=USE_TESTNET)
FAPI_BASE = "https://testnet.binancefuture.com" if USE_TESTNET else "https://fapi.binance.com"

# ================== HELPERS ==================
def pct(x, d=4, signed=False):
    if x is None or (isinstance(x, float) and math.isnan(x)): return "nan"
    v = float(x) * 100
    return f"{v:+.{d}f}%" if signed else f"{v:.{d}f}%"

def price_move_threshold_for_fees(leverage: int) -> float:
    return (FEE_ROUNDTRIP + SLIP_ROUNDTRIP) / leverage

def binance_server_now():
    r = requests.get(f"{FAPI_BASE}/fapi/v1/time", timeout=10)
    r.raise_for_status()
    ms = r.json()["serverTime"]
    return pd.Timestamp(ms, unit="ms", tz="UTC")

def fetch_klines(symbol, start_ms=None, end_ms=None, interval="1m", limit=1000):
    """
    연속 klines. start_ms/end_ms는 ms(UTC). 없으면 최신 limit개.
    """
    url = f"{FAPI_BASE}/fapi/v1/klines"
    params = dict(symbol=symbol, interval=interval, limit=limit)
    if start_ms is not None: params["startTime"] = int(start_ms)
    if end_ms   is not None: params["endTime"]   = int(end_ms)

    out = []
    while True:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data: break
        out.extend(data)
        if len(data) < limit:
            break
        last_open = data[-1][0]
        params["startTime"] = last_open + 1

    df = pd.DataFrame(out, columns=[
        "open_time","open","high","low","close","volume","close_time","q","n","tb","tq","ignore"
    ])
    if df.empty: return df
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df.dropna(inplace=True)
    return df.reset_index(drop=True)

def fetch_latest_klines(symbol, limit=2):
    """가장 최근 봉 포함 limit개(보통 2개) 반환."""
    return fetch_klines(symbol, interval=interval, limit=limit)

# ====== Exchange filters (LOT_SIZE) to simulate real quantity rounding ======
_EXINFO_CACHE = None
def _get_exchange_info():
    global _EXINFO_CACHE
    if _EXINFO_CACHE is None:
        r = requests.get(f"{FAPI_BASE}/fapi/v1/exchangeInfo", timeout=10)
        r.raise_for_status()
        _EXINFO_CACHE = r.json()
    return _EXINFO_CACHE

def get_symbol_filters(symbol):
    info = _get_exchange_info()
    sym = next(s for s in info["symbols"] if s["symbol"] == symbol)
    lot = next(f for f in sym["filters"] if f["filterType"] == "LOT_SIZE")
    step = float(lot["stepSize"])
    minq = float(lot["minQty"])
    prec = int(sym.get("quantityPrecision", 3))
    return {"stepSize": step, "minQty": minq, "quantityPrecision": prec}

def _quantize_down(qty, step):
    dq   = Decimal(str(qty))
    dstep= Decimal(str(step))
    q    = (dq / dstep).to_integral_value(rounding=ROUND_DOWN) * dstep
    return float(q)

def calc_safe_qty_and_eff_portion(now_price, capital, entry_portion, leverage):
    """
    실거래와 동일한 LOT_SIZE 제약을 적용하여
    - 체결 가능한 수량(qty)
    - eff_portion = (qty * now_price) / (capital * leverage)
    를 계산.
    """
    filters = get_symbol_filters(symbol)
    step, min_qty, prec = filters["stepSize"], filters["minQty"], filters["quantityPrecision"]
    # 목표 수량(이론치)
    raw_qty = (capital * entry_portion * leverage) / now_price
    qty     = _quantize_down(raw_qty, step)
    qty     = float(f"{qty:.{prec}f}")
    valid   = qty >= min_qty
    if not valid:
        return {"quantity": 0.0, "eff_portion": 0.0, "valid": False}

    entry_notional = qty * now_price
    eff_portion    = (entry_notional / (capital * leverage)) if capital > 0 else 0.0
    return {"quantity": qty, "eff_portion": float(eff_portion), "valid": True}

# ================== MODEL LOADING ==================
def load_deepar_model(ckpt_path, symbol):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt.get("hyper_parameters", {}).pop("dataset_parameters", None)
    tmp = f"{symbol}_tmp.ckpt"
    torch.save(ckpt, tmp)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepAR.load_from_checkpoint(tmp, map_location=device)
    return model.to(device).eval(), device

class ProfitClassifier(nn.Module):
    def __init__(self, in_ch=3, hidden=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(in_ch, hidden, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers>1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,1)
        )
    def forward(self, x):
        out,_ = self.gru(x); h = out[:, -1, :]
        return self.head(h)

def load_profit_classifier(path):
    if not os.path.exists(path):
        print(f"[INFO] Profit classifier not found: {path} (gate disabled)")
        return None, None, False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ProfitClassifier().to(device).eval()
    sd = torch.load(path, map_location=device)
    state = sd.get("model", sd)
    model.load_state_dict(state)
    print(f"[INFO] Profit classifier loaded: {path}")
    return model, device, True

def _pick_pred_from_dict(d):
    for k in ["prediction","pred","y_hat","y","output","outs"]:
        if k in d: return d[k]
    return next(iter(d.values()))

def to_tensor(pred):
    if isinstance(pred, torch.Tensor): return pred
    if isinstance(pred, np.ndarray):   return torch.from_numpy(pred)
    if isinstance(pred, dict):         return to_tensor(_pick_pred_from_dict(pred))
    if isinstance(pred, (list, tuple)):
        flats, stack = [], list(pred)
        while stack:
            x = stack.pop(0)
            if isinstance(x, torch.Tensor): flats.append(x)
            elif isinstance(x, np.ndarray): flats.append(torch.from_numpy(x))
            elif isinstance(x, dict):      stack.insert(0, _pick_pred_from_dict(x))
            elif isinstance(x, (list, tuple)): stack = list(x) + stack
            else: flats.append(torch.as_tensor(x))
        if len(flats) == 1: return flats[0]
        try:   return torch.cat(flats, dim=0)
        except Exception:
            flats = [f.unsqueeze(0) for f in flats]
            return torch.cat(flats, dim=0)
    return torch.as_tensor(pred)

# ================== PREDICT ==================
def predict_prices_and_pxroi(model, dataset, df_seq, now_price, pred_len, n_samples=64):
    tmp = df_seq.copy().reset_index(drop=True)
    tmp["time_idx"]  = np.arange(len(tmp))
    tmp["series_id"] = symbol
    ds = TimeSeriesDataSet.from_dataset(dataset, tmp, predict=True, stop_randomization=True)
    dl = ds.to_dataloader(train=False, batch_size=1)
    with torch.no_grad():
        raw = model.predict(dl, mode="prediction", n_samples=n_samples)
        y   = to_tensor(raw).detach().cpu()
    if y.ndim == 3:   samples = y[:, 0, :].numpy()
    elif y.ndim == 2: samples = y.numpy()[None, 0, :]
    elif y.ndim == 1: samples = y.numpy()[None, :]
    else: raise RuntimeError(f"Unexpected pred shape: {tuple(y.shape)}")

    pct_steps = np.expm1(samples)      # (S, T)
    avg       = pct_steps.mean(axis=0) # (T,)
    cum       = np.cumsum(np.log1p(pct_steps), axis=1)
    prices    = now_price * np.exp(cum)         # (S, T)
    prices_mean = prices.mean(axis=0)           # (T,)
    return prices_mean, avg

# ================== PROFIT GATE ==================
def profit_gate_probability(df_seq, vol_mu, vol_sd, profit_model, profit_device):
    if profit_model is None: return None
    tail = df_seq.tail(seq_len).copy()
    lr   = tail["log_return"].astype(np.float32).values
    vol  = tail["volume"].astype(np.float32).values
    vz   = (vol - vol_mu) / (vol_sd + 1e-8)
    rv   = pd.Series(df_seq["log_return"].astype(float)).rolling(seq_len).std(ddof=0).fillna(0.0).values[-seq_len:].astype(np.float32)
    feats = np.stack([lr, vz, rv], axis=1).astype(np.float32)  # (T,3)
    x = torch.from_numpy(feats).unsqueeze(0).to(profit_device)
    with torch.no_grad():
        p = torch.sigmoid(profit_model(x)).item()
    return p

# ================== EXIT SIM (OHLC TOUCH) ==================
def step_exit_with_ohlc(entry_price, direction, bar_ohlc, state):
    """
    새 분봉 하나가 들어올 때 OHLC 경로로 TP/SL/Trail/BE 판정.
    """
    EPS = 1e-9
    best_px = state.get("best_px", 0.0)
    tp_hit  = state.get("tp_hit", False)
    be_armed= state.get("be_armed", False)

    o, h, l, c = bar_ohlc
    path = (o, h, l, c) if c >= o else (o, l, h, c)

    fee_th_px = (FEE_ROUNDTRIP + SLIP_ROUNDTRIP) / LEVERAGE
    BREAKEVEN_ARM_PX = max(0.0006, fee_th_px * 1.2)
    POST_TP_TRAIL = min(TRAIL_PX, 0.0002)

    def px_roi_of(price):
        return (price - entry_price) / entry_price if direction == "LONG" else (entry_price - price) / entry_price

    for px in path:
        px_roi = px_roi_of(px)
        acct_roi_unreal = (px_roi * LEVERAGE) - (TAKER_FEE_ONEWAY + SLIPPAGE_ONEWAY)

        if px_roi > best_px: best_px = px_roi
        if best_px + EPS >= TP_PX: tp_hit = True
        if not be_armed and best_px >= BREAKEVEN_ARM_PX - EPS: be_armed = True

        # 0) 브레이크이븐 스톱
        if be_armed and px_roi <= 0.0 + EPS:
            realized_px_roi = px_roi
            acct_roi = (realized_px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
            return True, acct_roi, "BREAKEVEN_STOP", realized_px_roi, {"best_px":best_px,"tp_hit":tp_hit,"be_armed":be_armed}

        # 1) 손절
        if px_roi <= -SL_PX - EPS or acct_roi_unreal <= -(SL_PX*LEVERAGE - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)) - EPS:
            realized_px_roi = px_roi
            acct_roi = (realized_px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
            return True, acct_roi, "STOP_LOSS", realized_px_roi, {"best_px":best_px,"tp_hit":tp_hit,"be_armed":be_armed}

        # 2) 즉시 익절
        tp_acct_equiv = TP_PX * LEVERAGE - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
        if px_roi >= TP_PX - EPS or acct_roi_unreal >= tp_acct_equiv - EPS:
            realized_px_roi = px_roi
            acct_roi = (realized_px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
            return True, acct_roi, "TAKE_PROFIT", realized_px_roi, {"best_px":best_px,"tp_hit":tp_hit,"be_armed":be_armed}

        # 3) TP 이후 초미세 트레일 잠금
        if tp_hit and (best_px - px_roi) >= min(TRAIL_PX, 0.0002) - EPS:
            realized_px_roi = px_roi
            acct_roi = (realized_px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
            return True, acct_roi, "TRAILING_TP_LOCK", realized_px_roi, {"best_px":best_px,"tp_hit":tp_hit,"be_armed":be_armed}

        # 4) 일반 트레일링
        if best_px > 0 and (best_px - px_roi) >= TRAIL_PX - EPS:
            realized_px_roi = px_roi
            acct_roi = (realized_px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
            return True, acct_roi, "TRAILING_EXIT", realized_px_roi, {"best_px":best_px,"tp_hit":tp_hit,"be_armed":be_armed}

    return False, 0.0, None, None, {"best_px":best_px,"tp_hit":tp_hit,"be_armed":be_armed}

# ================== REALTIME EXIT (틱 단위, 시그널만 판단)
def monitor_until_exit_live(entry_price, direction, capital, poll_ms=10):
    """
    보유 중 실시간(mid→mark) 가격으로 TP/SL/Trail/BE 즉시 판정.
    반환: realized_acct_roi(미사용), reason, realized_px_roi(미사용), updated_state
    """
    EPS = 1e-9
    best_px = 0.0
    tp_hit  = False
    be_armed= False

    fee_th_px = (FEE_ROUNDTRIP + SLIP_ROUNDTRIP) / LEVERAGE
    BREAKEVEN_ARM_PX = max(0.0006, fee_th_px * 1.2)
    POST_TP_TRAIL    = min(TRAIL_PX, 0.0002)

    url_book = f"{FAPI_BASE}/fapi/v1/ticker/bookTicker"
    url_mark = f"{FAPI_BASE}/fapi/v1/premiumIndex"

    while True:
        latest = None
        try:
            r = requests.get(url_book, params={"symbol":symbol}, timeout=2)
            d = r.json()
            bid = float(d["bidPrice"]); ask = float(d["askPrice"])
            latest = (bid + ask) / 2.0
        except Exception:
            try:
                r = requests.get(url_mark, params={"symbol":symbol}, timeout=2)
                latest = float(r.json()["markPrice"])
            except Exception as e:
                print(f"[WARN] live price fetch: {e}")
                time.sleep(max(0.1, poll_ms/1000.0)); continue

        px_roi = (latest - entry_price) / entry_price if direction == "LONG" else (entry_price - latest) / entry_price
        acct_roi_unreal = (px_roi * LEVERAGE) - (TAKER_FEE_ONEWAY + SLIPPAGE_ONEWAY)

        if px_roi > best_px: best_px = px_roi
        if best_px + EPS >= TP_PX: tp_hit = True
        if not be_armed and best_px >= BREAKEVEN_ARM_PX - EPS: be_armed = True

        reason = None
        if reason is None and be_armed and px_roi <= 0.0 + EPS: reason = "BREAKEVEN_STOP"
        if reason is None and (px_roi <= -SL_PX - EPS or acct_roi_unreal <= -(SL_PX*LEVERAGE - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)) - EPS): reason = "STOP_LOSS"
        tp_acct_equiv = TP_PX * LEVERAGE - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
        if reason is None and (px_roi >= TP_PX - EPS or acct_roi_unreal >= tp_acct_equiv - EPS): reason = "TAKE_PROFIT"
        if reason is None and tp_hit and (best_px - px_roi) >= POST_TP_TRAIL - EPS: reason = "TRAILING_TP_LOCK"
        if reason is None and best_px > 0 and (best_px - px_roi) >= TRAIL_PX - EPS: reason = "TRAILING_EXIT"

        if reason:
            return 0.0, reason, 0.0, {"best_px":best_px, "tp_hit":tp_hit, "be_armed":be_armed}

        time.sleep(max(0.1, poll_ms/1000.0))

# ================== TESTNET 주문 헬퍼 ==================
def place_market_and_wait_fill(side, qty, reduce_only=False, timeout=5.0):
    """
    테스트넷 선물에 시장가 주문을 넣고 FILLED 될 때까지 짧게 폴링.
    반환: (avg_price, executed_qty, order_id)
    """
    params = dict(symbol=symbol, side=side, type="MARKET", quantity=qty, newOrderRespType="RESULT")
    if reduce_only:
        params["reduceOnly"] = "true"

    resp = client.futures_create_order(**params)
    oid = resp["orderId"]

    avg_price = None
    executed = 0.0
    t0 = time.time()
    while time.time() - t0 < timeout:
        od = client.futures_get_order(symbol=symbol, orderId=oid)
        executed = float(od.get("executedQty", 0.0))
        status = od.get("status", "")
        if status == "FILLED":
            try:
                avg_price = float(od.get("avgPrice", 0.0)) or None
            except Exception:
                avg_price = None
            break
        time.sleep(0.1)

    if avg_price is None:
        # 백업: 체결 내역으로 평균가 계산
        trades = client.futures_account_trades(symbol=symbol, orderId=oid)
        if trades:
            qty_sum = sum(float(t["qty"]) for t in trades)
            cost    = sum(float(t["quoteQty"]) for t in trades)
            if qty_sum > 0:
                avg_price = cost / qty_sum
                executed  = qty_sum

    return avg_price, executed, oid

# ================== MAIN (REALTIME-LIKE LOOP with TESTNET ORDERS) ==================
if __name__ == "__main__":
    # ===== FUTURES TESTNET 계정 설정 =====
    try:
        # 원웨이 모드(헤지 끔)
        client.futures_change_position_mode(dualSidePosition="false")  # "false"=one-way
        # 마진 타입 (초기 1회만 성공하고 이후엔 에러가 날 수 있음 → 경고만 출력)
        try:
            client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
        except Exception as e:
            print("[WARN] margin type:", e)
        # 레버리지
        client.futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
    except Exception as e:
        print("[WARN] futures account setup:", e)

    # 통계/데이터셋 스키마 재사용
    df_all = pd.read_csv(INPUT_CSV).dropna()
    VOL_MU = float(df_all["volume"].mean()) if "volume" in df_all.columns else 0.0
    VOL_SD = float(df_all["volume"].std())  if "volume" in df_all.columns else 1.0

    dataset = TimeSeriesDataSet(
        df_all,
        time_idx="time_idx",
        target="log_return",
        group_ids=["series_id"],
        max_encoder_length=seq_len,
        max_prediction_length=pred_len,
        time_varying_known_reals=["time_idx", "volume"],
        time_varying_unknown_reals=["log_return"],
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True
    )

    model, device = load_deepar_model(CKPT_FILE, symbol)
    profit_model, profit_device, gate_enabled = load_profit_classifier(PROFIT_CLS_PATH)

    # 부트스트랩: 직전 확정봉 기준으로 seq_len+pred_len 확보
    now_utc = binance_server_now()
    start_boot = int((now_utc - pd.Timedelta(minutes=seq_len + pred_len + 5)).timestamp() * 1000)
    df_hist = fetch_klines(symbol, start_ms=start_boot, end_ms=None, interval=interval, limit=1000)
    if df_hist.empty or len(df_hist) < (seq_len + pred_len + 1):
        raise RuntimeError("초기 히스토리 부족")

    window_len = seq_len + pred_len
    data_q = deque(df_hist.tail(window_len).to_dict("records"), maxlen=window_len)

    tz = "Asia/Seoul"
    capital = INITIAL_CAPITAL
    equity_curve = []
    trade_logs = []

    is_holding = False
    direction = None
    entry_price = None
    hold_state = {"best_px":0.0, "tp_hit":False, "be_armed":False}
    open_eff_portion = 0.0  # 이번 트레이드의 실효 진입 비율 저장
    filled_qty_open  = 0.0  # 실제 체결된 진입 수량 저장
    trades = 0

    # 마지막으로 본 close_time으로 새 봉 감지
    last_close_time = int(data_q[-1]["close_time"])

    print(f"[INFO] Live-like TESTNET started (fees+slip+lot via real fills). Equity=${capital:.2f}, L=x{LEVERAGE}, TargetPortion={ENTRY_PORTION*100:.0f}%")
    while True:
        try:
            # 최신 2봉 가져와서 새 확정봉 감지
            df2 = fetch_latest_klines(symbol, limit=2)
            if df2.empty:
                time.sleep(1); continue

            ct = int(df2.iloc[-1]["close_time"])   # 마지막 행의 close_time (막 닫힌 봉)
            if ct == last_close_time:
                time.sleep(1); continue  # 아직 새 봉 안 닫힘

            # 새 확정봉 하나 들어옴
            last_close_time = ct
            new_bar = df2.iloc[-1].to_dict()

            # 큐에 추가
            data_q.append(new_bar)
            df_seq = pd.DataFrame(list(data_q))

            # 타임스탬프(한국)
            ts = pd.to_datetime(new_bar["close_time"], unit="ms", utc=True).tz_convert(tz)

            if not is_holding:
                # 예측 기준 now_price = "막 닫힌 이전 봉의 종가"
                now_price = df_seq["close"].iloc[-pred_len - 1]

                # 예측
                _, rois = predict_prices_and_pxroi(model, dataset, df_seq, now_price, pred_len, n_samples=64)
                idx = int(np.argmax(np.abs(rois)))
                target_roi = float(rois[idx])
                target_min = idx + 1
                direction  = "LONG" if target_roi > 0 else "SHORT"

                # Profit 게이트
                p_profit = profit_gate_probability(df_seq, VOL_MU, VOL_SD, profit_model, profit_device) if gate_enabled else None
                gate_pass = True if not gate_enabled else (p_profit is not None and p_profit >= P_PROFIT_CUTOFF)

                # 기본 문턱
                fee_threshold   = price_move_threshold_for_fees(LEVERAGE)
                entry_threshold = max(abs(thresh_roi), fee_threshold)

                # LOT_SIZE 반영한 '실효 진입비율'로 예상 순이익 판단
                qinfo = calc_safe_qty_and_eff_portion(now_price, capital, ENTRY_PORTION, LEVERAGE)
                eff_p = qinfo["eff_portion"]
                notional_eff   = capital * LEVERAGE * eff_p
                expected_net_px  = max(0.0, abs(target_roi) - fee_threshold)
                expected_net_usd = notional_eff * expected_net_px

                if (abs(target_roi) >= entry_threshold) and gate_pass and qinfo["valid"] and expected_net_usd >= MIN_PROFIT_USD:
                    # ===== 실제 테스트넷 시장가 진입 =====
                    side = "BUY" if direction == "LONG" else "SELL"
                    qty_target = qinfo["quantity"]

                    entry_px, filled_qty_open, _ = place_market_and_wait_fill(side, qty_target, reduce_only=False, timeout=5.0)

                    if not entry_px or filled_qty_open < get_symbol_filters(symbol)["minQty"]:
                        print(f"[{ts:%Y-%m-%d %H:%M}] ❌ NO-FILL  want={qty_target:.6f} filled={filled_qty_open:.6f}")
                        is_holding = False
                        filled_qty_open = 0.0
                    else:
                        # 체결된 명목으로 eff_portion 재계산
                        entry_notional = entry_px * filled_qty_open
                        open_eff_portion = min(1.0, entry_notional / (capital * LEVERAGE))

                        entry_price = entry_px
                        is_holding  = True
                        hold_state  = {"best_px":0.0, "tp_hit":False, "be_armed":False}

                        print(f"[{ts:%Y-%m-%d %H:%M}] 🚀 ENTRY(TESTNET) {direction}  px={entry_price:.6f}  "
                              f"filled={filled_qty_open:.6f}  eff={open_eff_portion:.4f}  Eq=${capital:.2f}")
                else:
                    print(f"[{ts:%Y-%m-%d %H:%M}] ⏳ NO-ENTRY  pred={pct(target_roi, signed=True)}@{target_min}m  "
                          f"eff_portion={eff_p:.4f}  need≥${MIN_PROFIT_USD:.2f}  Eq=${capital:.2f}")

            else:
                # ===== EXIT =====
                if REALTIME_EXIT:
                    # 실시간(틱) 모드: 가격을 계속 보면서 즉시 엑싯 시그널만 판단
                    _, reason, _, hold_state = monitor_until_exit_live(
                        entry_price, direction, capital, poll_ms=POLL_MS
                    )

                    # 실제 테스트넷 반대 주문(감소전용)으로 청산
                    exit_side = "SELL" if direction == "LONG" else "BUY"
                    exit_px, exit_filled, _ = place_market_and_wait_fill(exit_side, filled_qty_open, reduce_only=True, timeout=5.0)
                    ts_live = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(tz)

                    if not exit_px or exit_filled <= 0:
                        print(f"[{ts_live:%Y-%m-%d %H:%M}] ❌ EXIT NO-FILL — 재시도 권장")
                        # 실패 시 다음 루프에서 다시 청산 시도
                        continue

                    realized_px_roi = ((exit_px - entry_price) / entry_price) if direction == "LONG" \
                                      else ((entry_price - exit_px) / entry_price)
                    realized_acct_roi = (realized_px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)

                    capital_before = capital
                    capital = capital * (1.0 + open_eff_portion * realized_acct_roi)
                    trades += 1

                    print(f"[{ts_live:%Y-%m-%d %H:%M}] 💥 EXIT(TESTNET) {reason}  "
                          f"entry={entry_price:.6f} exit={exit_px:.6f}  "
                          f"pxROI*={pct(realized_px_roi, signed=True)}  "
                          f"acctROI={pct(realized_acct_roi, signed=True)}  "
                          f"eff={open_eff_portion:.4f}  Eq ${capital_before:.2f}→${capital:.2f}")

                    trade_logs.append({
                        "timestamp": ts_live, "direction": direction, "reason": reason,
                        "realized_pxROI": realized_px_roi, "acctROI": realized_acct_roi,
                        "eff_portion": open_eff_portion,
                        "capital_before": capital_before, "capital_after": capital,
                        "entry_px": entry_price, "exit_px": exit_px,
                        "filled_qty": float(exit_filled)
                    })

                    # 리셋
                    is_holding  = False
                    direction   = None
                    entry_price = None
                    hold_state  = {"best_px":0.0, "tp_hit":False, "be_armed":False}
                    open_eff_portion = 0.0
                    filled_qty_open  = 0.0

                else:
                    # (기존) 새 확정봉의 OHLC 경로로 엑싯 시그널 판단 → 실제 주문으로 청산
                    bar_ohlc = [float(new_bar["open"]), float(new_bar["high"]), float(new_bar["low"]), float(new_bar["close"])]
                    exited, _, reason, _, hold_state = step_exit_with_ohlc(
                        entry_price, direction, bar_ohlc, hold_state
                    )

                    if exited:
                        exit_side = "SELL" if direction == "LONG" else "BUY"
                        exit_px, exit_filled, _ = place_market_and_wait_fill(exit_side, filled_qty_open, reduce_only=True, timeout=5.0)
                        ts_live = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(tz)

                        if not exit_px or exit_filled <= 0:
                            print(f"[{ts_live:%Y-%m-%d %H:%M}] ❌ EXIT NO-FILL — 재시도 권장")
                            continue

                        realized_px_roi = ((exit_px - entry_price) / entry_price) if direction == "LONG" \
                                          else ((entry_price - exit_px) / entry_price)
                        realized_acct_roi = (realized_px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)

                        capital_before = capital
                        capital = capital * (1.0 + open_eff_portion * realized_acct_roi)
                        trades += 1

                        print(f"[{ts_live:%Y-%m-%d %H:%M}] 💥 EXIT(TESTNET) {reason}  "
                              f"entry={entry_price:.6f} exit={exit_px:.6f}  "
                              f"pxROI*={pct(realized_px_roi, signed=True)}  "
                              f"acctROI={pct(realized_acct_roi, signed=True)}  "
                              f"eff={open_eff_portion:.4f}  Eq ${capital_before:.2f}→${capital:.2f}")

                        trade_logs.append({
                            "timestamp": ts_live, "direction": direction, "reason": reason,
                            "realized_pxROI": realized_px_roi, "acctROI": realized_acct_roi,
                            "eff_portion": open_eff_portion,
                            "capital_before": capital_before, "capital_after": capital,
                            "entry_px": entry_price, "exit_px": exit_px,
                            "filled_qty": float(exit_filled)
                        })

                        is_holding = False
                        direction = None
                        entry_price = None
                        hold_state = {"best_px":0.0, "tp_hit":False, "be_armed":False}
                        open_eff_portion = 0.0
                        filled_qty_open  = 0.0

            # 에쿼티 곡선 저장(매 분)
            equity_curve.append((ts, capital))

            # (선택) 주기적으로 CSV 저장
            if len(equity_curve) % 60 == 0:  # 한 시간마다
                pd.DataFrame(equity_curve, columns=["timestamp","equity"]).to_csv(EQUITY_CSV, index=False)
                pd.DataFrame(trade_logs).to_csv(TRADES_CSV, index=False)

        except KeyboardInterrupt:
            print("\n[STOP] Interrupted by user.")
            break
        except Exception as e:
            print(f"[WARN] loop error: {e}")
            time.sleep(1)
            continue

    # 종료 시 저장
    pd.DataFrame(equity_curve, columns=["timestamp","equity"]).to_csv(EQUITY_CSV, index=False)
    pd.DataFrame(trade_logs).to_csv(TRADES_CSV, index=False)
    print(f"[SAVED] Equity → {EQUITY_CSV}")
    print(f"[SAVED] Trades → {TRADES_CSV}")
