#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ⚠️ 실제 메인넷 실거래 버전 (테스트 후 사용하세요)

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

# 모델/전략 파라미터
seq_len    = 60
pred_len   = 60

# ===== 실시간 엑싯 모드 =====
REALTIME_EXIT = True      # 보유 중 실시간으로 TP/SL/Trail/BE/TimeStop 판정
POLL_MS       = 10        # 실시간 가격 폴링 주기(ms)

# 수수료/슬리피지 (초기값; 실거래에서는 아래에서 거래소 값으로 갱신함)
TAKER_FEE_ONEWAY = 0.0004
SLIPPAGE_ONEWAY  = 0.0005
FEE_ROUNDTRIP    = TAKER_FEE_ONEWAY * 2
SLIP_ROUNDTRIP   = SLIPPAGE_ONEWAY  * 2

LEVERAGE       = 20
ENTRY_PORTION  = 0.9       # 목표 진입 비율(이론치) — LOT_SIZE 반영 시 eff_portion으로 환산
MIN_PROFIT_USD = 0.01
thresh_roi     = 0.0001    # px ROI 문턱(0.01%=0.0001)

# ===== 동적 손절·익절 레시피 =====
# 'A' = 스캘핑(빠른 결단, TP=2*ATR, SL=1*ATR, Trail=1*ATR, BE=0.6*ATR)
# 'B' = 트렌드 팔로우(고정 TP 없음, SL=1.5*ATR, Trail=2.5*ATR, BE=1*ATR)
# 'C' = 평균회귀(밴드 복귀, SL=1.2*ATR, TP=1.5*ATR, TimeStop=1.5*target_min)
STRATEGY       = "A"       # 'A' | 'B' | 'C'
ATR_N          = 14        # ATR 기간
PARTIAL_TP_RATIO = 0.5     # 첫 익절 시 부분청산 비율(0~1)

# 초기자본(내부 시뮬레이션용; 실계정 잔고와 별개)
INITIAL_CAPITAL = 100

# ===== 리스크 가드 =====
MAX_DRAWDOWN_PCT      = 0.05   # 하루 시작 대비 -5%면 신규 진입 중단
MAX_CONSEC_LOSSES     = 3      # 연속 3회 손실 시 쿨다운
DAILY_TRADE_LIMIT     = 50     # 하루 최대 트레이드 수
MAX_POSITION_NOTIONAL = 200.0  # USDT 기준 1회 최대 포지션 명목가 상한
COOLDOWN_MINUTES      = 10     # 과열/연속손실 시 쿨다운

# 저장 파일
EQUITY_CSV = f"{symbol}_live_equity.csv"
TRADES_CSV = f"{symbol}_live_trades.csv"

# ===== 메인넷 연결 =====
USE_TESTNET = False  # ✅ 메인넷
API_KEY    = "kynaVBENI8TIS8AcZICK2Sn52mJfIqHpCHfKv2IaSM7TQtKaazn44KGcsJRERjbE"
API_SECRET = "z2ZtL0S8eIE34C8bL1XpAV0jEhcc8JEm8VBOEZbo5Rbe4d9HkN2F4V2AGukhtXyT"
client     = Client(API_KEY, API_SECRET)

FAPI_BASE = "https://fapi.binance.com"  # ✅ 메인넷 REST 베이스

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

# ====== Exchange filters (LOT_SIZE) ======
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
    LOT_SIZE 적용하여
    - 체결 가능한 수량(qty)
    - eff_portion = (qty * now_price) / (capital * leverage)
    계산
    """
    filters = get_symbol_filters(symbol)
    step, min_qty, prec = filters["stepSize"], filters["minQty"], filters["quantityPrecision"]
    raw_qty = (capital * entry_portion * leverage) / now_price
    qty     = _quantize_down(raw_qty, step)
    qty     = float(f"{qty:.{prec}f}")
    valid   = qty >= min_qty
    if not valid: return {"quantity": 0.0, "eff_portion": 0.0, "valid": False}

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

# ================== ATR & THRESHOLDS ==================
def compute_atr_px(df, n=14):
    """ ATR(n)를 가격대비 비율(px)로 반환 """
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    prev_c  = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(n).mean().iloc[-1]
    now_price = df["close"].iloc[-1]
    return float(atr / now_price)

def build_thresholds(atr_px, fee_th, strategy, target_min):
    tp_px = None; sl_px = None; trail_px = None; be_arm_px = None; time_stop_s = None
    if strategy == "A":  # 스캘핑
        sl_px     = max(1.0 * atr_px, fee_th * 1.2)
        tp_px     = 2.0 * atr_px
        trail_px  = 1.0 * atr_px
        be_arm_px = max(0.6 * atr_px, fee_th * 1.5)
        time_stop_s = int(max(60, target_min * 60 * 1.1))
    elif strategy == "B":  # 트렌드 팔로우
        sl_px     = max(1.5 * atr_px, fee_th * 1.2)
        tp_px     = None
        trail_px  = 2.5 * atr_px
        be_arm_px = max(1.0 * atr_px, fee_th * 1.5)
        time_stop_s = int(max(120, target_min * 60 * 2.0))
    else:  # "C" 평균회귀
        sl_px     = max(1.2 * atr_px, fee_th * 1.2)
        tp_px     = 1.5 * atr_px
        trail_px  = 1.0 * atr_px
        be_arm_px = max(0.6 * atr_px, fee_th * 1.5)
        time_stop_s = int(max(90, target_min * 60 * 1.5))
    return dict(tp_px=tp_px, sl_px=sl_px, trail_px=trail_px, be_arm_px=be_arm_px, time_stop_s=time_stop_s)

# ================== REALTIME EXIT (틱 단위) ==================
def monitor_until_exit_live(entry_price, direction, thr, poll_ms=10, entry_ts_utc=None):
    """
    실시간 가격으로 TP/SL/Trail/BE/TimeStop 판정.
    thr: dict(tp_px, sl_px, trail_px, be_arm_px, time_stop_s)
    반환: (reason, latest_price_at_decision)
    """
    EPS = 1e-9
    best_px = 0.0
    tp_hit  = False
    be_armed= False
    url_book = f"{FAPI_BASE}/fapi/v1/ticker/bookTicker"
    url_mark = f"{FAPI_BASE}/fapi/v1/premiumIndex"
    t0 = pd.Timestamp.utcnow() if entry_ts_utc is None else entry_ts_utc

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
        if (thr["tp_px"] is not None) and (best_px >= thr["tp_px"] - EPS): tp_hit = True
        if not be_armed and best_px >= max(thr["be_arm_px"], (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)/LEVERAGE*1.2) - EPS:
            be_armed = True

        # 0) 타임 스톱
        if thr.get("time_stop_s"):
            elapsed_s = (pd.Timestamp.utcnow() - t0).total_seconds()
            if elapsed_s >= thr["time_stop_s"]:
                return "TIME_STOP", latest

        # 1) 브레이크이븐 스톱
        if be_armed and px_roi <= 0.0 + EPS:
            return "BREAKEVEN_STOP", latest

        # 2) 손절
        if px_roi <= -thr["sl_px"] - EPS or acct_roi_unreal <= -(thr["sl_px"]*LEVERAGE - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)) - EPS:
            return "STOP_LOSS", latest

        # 3) 즉시 익절(고정 TP가 있을 때)
        if (thr["tp_px"] is not None):
            tp_acct_equiv = thr["tp_px"] * LEVERAGE - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
            if (px_roi >= thr["tp_px"] - EPS) or (acct_roi_unreal >= tp_acct_equiv - EPS):
                return "TAKE_PROFIT", latest

        # 4) TP 이후 잠금(더 촘촘)
        if tp_hit and thr["trail_px"] is not None and (best_px - px_roi) >= min(thr["trail_px"], 0.0002) - EPS:
            return "TRAILING_TP_LOCK", latest

        # 5) 일반 트레일링
        if thr["trail_px"] is not None and best_px > 0 and (best_px - px_roi) >= thr["trail_px"] - EPS:
            return "TRAILING_EXIT", latest

        time.sleep(max(0.1, poll_ms/1000.0))

# ================== 실거래용: 수수료 동기화 ==================
def refresh_commission_rates(symbol):
    """
    선물 메이커/테이커 수수료를 API에서 읽어와 반영.
    반환: (maker, taker)  # 소수(예: 0.0002 = 0.02%)
    """
    try:
        rate = client.futures_commission_rate(symbol=symbol)
        maker = float(rate["makerCommissionRate"])
        taker = float(rate["takerCommissionRate"])
        return maker, taker
    except Exception as e:
        print("[WARN] commission rate fetch:", e)
        return None, None

# ================== 주문 헬퍼(실거래) ==================
def place_market_and_wait_fill(side, qty, reduce_only=False, timeout=5.0):
    """
    선물에 시장가 주문을 넣고 FILLED 될 때까지 짧게 폴링.
    반환: (avg_price, executed_qty, order_id)
    """
    params = dict(symbol=symbol, side=side, type="MARKET", quantity=qty, newOrderRespType="RESULT")
    if reduce_only: params["reduceOnly"] = "true"
    resp = client.futures_create_order(**params, recvWindow=5000)  # ⏱ recvWindow 여유
    oid = resp["orderId"]

    avg_price = None
    executed = 0.0
    t0 = time.time()
    while time.time() - t0 < timeout:
        od = client.futures_get_order(symbol=symbol, orderId=oid, recvWindow=5000)
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
        trades = client.futures_account_trades(symbol=symbol, orderId=oid, recvWindow=5000)
        if trades:
            qty_sum = sum(float(t["qty"]) for t in trades)
            cost    = sum(float(t["quoteQty"]) for t in trades)
            if qty_sum > 0:
                avg_price = cost / qty_sum
                executed  = qty_sum

    return avg_price, executed, oid

# ================== MAIN ==================
if __name__ == "__main__":
    # ===== 계정/레버리지 설정 =====
    try:
        client.futures_change_position_mode(dualSidePosition="false")  # "false"=one-way
        try:
            client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
        except Exception as e:
            print("[WARN] margin type:", e)
        client.futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
    except Exception as e:
        print("[WARN] futures account setup:", e)

    # 실거래 수수료 동기화
    mk, tk = refresh_commission_rates(symbol)
    if tk is not None:
        TAKER_FEE_ONEWAY = tk
        FEE_ROUNDTRIP    = TAKER_FEE_ONEWAY * 2
        print(f"[INFO] Taker fee(one-way) set to {TAKER_FEE_ONEWAY} (from exchange)")

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

    # 히스토리 부트스트랩
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

    # 포지션 상태
    is_holding = False
    direction = None
    entry_price = None
    hold_thr = None
    entry_ts_utc = None
    open_eff_portion_total = 0.0
    qty_open_total  = 0.0     # 최초 진입 체결 수량
    qty_open_remain = 0.0     # 남은 보유 수량
    tp1_done = False
    trades = 0

    # 데일리 리스크 상태
    day_key = pd.Timestamp.utcnow().date()
    day_start_equity = capital
    trades_today = 0
    consec_losses = 0
    cooldown_until = None

    # 새 봉 감지
    last_close_time = int(data_q[-1]["close_time"])
    print(f"[INFO] LIVE (MAINNET) start — L={LEVERAGE}x, Strategy={STRATEGY}, InitEq=${capital:.2f}")

    while True:
        try:
            # 최신 2봉 가져와서 새 확정봉 감지
            df2 = fetch_latest_klines(symbol, limit=2)
            if df2.empty:
                time.sleep(1); continue

            ct = int(df2.iloc[-1]["close_time"])
            if ct == last_close_time:
                time.sleep(1); continue

            # 새 확정봉 하나 들어옴
            last_close_time = ct
            new_bar = df2.iloc[-1].to_dict()

            # 큐에 추가
            data_q.append(new_bar)
            df_seq = pd.DataFrame(list(data_q))

            # 타임스탬프(한국)
            ts = pd.to_datetime(new_bar["close_time"], unit="ms", utc=True).tz_convert(tz)

            # ===== 데일리 리셋/가드 =====
            now_date = pd.Timestamp.utcnow().date()
            if now_date != day_key:
                day_key = now_date
                day_start_equity = capital
                trades_today = 0
                consec_losses = 0
                cooldown_until = None
                print(f"[INFO] New trading day — reset counters. DayStartEq=${day_start_equity:.2f}")

            if not is_holding:
                # 쿨다운 중이면 스킵
                if cooldown_until and pd.Timestamp.utcnow() < cooldown_until:
                    print(f"[RISK] cooldown active until {cooldown_until} → skip entry")
                    equity_curve.append((ts, capital))
                    continue

                # DD 가드
                if capital <= day_start_equity * (1.0 - MAX_DRAWDOWN_PCT):
                    print(f"[RISK] daily drawdown reached → skip entries today")
                    equity_curve.append((ts, capital))
                    continue

                # 데일리 트레이드 제한
                if trades_today >= DAILY_TRADE_LIMIT:
                    print(f"[RISK] daily trade limit reached ({DAILY_TRADE_LIMIT}) → skip")
                    equity_curve.append((ts, capital))
                    continue

                # 예측 기준 now_price = "막 닫힌 이전 봉의 종가"
                now_price = df_seq["close"].iloc[-pred_len - 1]

                # 예측
                _, rois = predict_prices_and_pxroi(model, dataset, df_seq, now_price, pred_len, n_samples=64)
                idx = int(np.argmax(np.abs(rois)))
                target_roi = float(rois[idx])
                target_min = idx + 1
                direction  = "LONG" if target_roi > 0 else "SHORT"

                # Profit 게이트
                p_profit = profit_gate_probability(df_seq, float(VOL_MU), float(VOL_SD), profit_model, profit_device) if gate_enabled else None
                gate_pass = True if not gate_enabled else (p_profit is not None and p_profit >= 0.60)

                # 기본 문턱
                fee_threshold   = price_move_threshold_for_fees(LEVERAGE)
                entry_threshold = max(abs(thresh_roi), fee_threshold)

                # ATR 기반 동적 손절/익절/트레일/BE/타임스톱 계산
                atr_px = compute_atr_px(df_seq.tail(100), n=ATR_N)
                hold_thr = build_thresholds(atr_px, fee_threshold, STRATEGY, target_min)

                # 포지션 사이즈 산정(상한 적용)
                # 허용 entry_portion = min(설정, 상한/최대명목가)
                max_portion_by_notional = min(1.0, MAX_POSITION_NOTIONAL / max(1e-9, (capital * LEVERAGE)))
                entry_portion_allowed = min(ENTRY_PORTION, max_portion_by_notional)

                qinfo = calc_safe_qty_and_eff_portion(now_price, capital, entry_portion_allowed, LEVERAGE)
                eff_p = qinfo["eff_portion"]
                notional_eff   = capital * LEVERAGE * eff_p
                expected_net_px  = max(0.0, abs(target_roi) - fee_threshold)
                expected_net_usd = notional_eff * expected_net_px

                if (abs(target_roi) >= entry_threshold) and gate_pass and qinfo["valid"] and expected_net_usd >= MIN_PROFIT_USD:
                    # ===== 메인넷 시장가 진입 =====
                    side = "BUY" if direction == "LONG" else "SELL"
                    qty_target = qinfo["quantity"]

                    entry_px, filled_qty, _ = place_market_and_wait_fill(side, qty_target, reduce_only=False, timeout=5.0)

                    if not entry_px or filled_qty < get_symbol_filters(symbol)["minQty"]:
                        print(f"[{ts:%Y-%m-%d %H:%M}] ❌ NO-FILL  want={qty_target:.6f} filled={filled_qty:.6f}")
                        is_holding = False
                        qty_open_total = qty_open_remain = 0.0
                        open_eff_portion_total = 0.0
                        tp1_done = False
                    else:
                        entry_price = entry_px
                        # 체결된 명목으로 eff_portion 재계산
                        entry_notional = entry_px * filled_qty
                        open_eff_portion_total = min(1.0, entry_notional / (capital * LEVERAGE))

                        qty_open_total  = filled_qty
                        qty_open_remain = filled_qty
                        tp1_done = False
                        entry_ts_utc = pd.Timestamp.utcnow()

                        is_holding  = True
                        print(f"[{ts:%Y-%m-%d %H:%M}] 🚀 ENTRY(LIVE) {direction}  px={entry_price:.6f}  "
                              f"qty={filled_qty:.6f}  eff={open_eff_portion_total:.4f}  "
                              f"ATRpx={atr_px:.6f}  thr={hold_thr}  Eq=${capital:.2f}")
                else:
                    print(f"[{ts:%Y-%m-%d %H:%M}] ⏳ NO-ENTRY  pred={pct(target_roi, signed=True)}@{target_min}m  "
                          f"eff={eff_p:.4f} need≥${MIN_PROFIT_USD:.2f} Eq=${capital:.2f}")

            else:
                # ===== EXIT ===== (실시간 시그널 → 실제 주문)
                reason, latest_px = monitor_until_exit_live(
                    entry_price, direction, hold_thr, poll_ms=POLL_MS, entry_ts_utc=entry_ts_utc
                )
                exit_side = "SELL" if direction == "LONG" else "BUY"

                # --- 부분익절(첫 TAKE_PROFIT 시)
                if (reason == "TAKE_PROFIT") and (not tp1_done) and (PARTIAL_TP_RATIO > 0) and (qty_open_remain > 0):
                    qty_tp1 = max(get_symbol_filters(symbol)["minQty"], round(qty_open_total * PARTIAL_TP_RATIO, 6))
                    qty_tp1 = min(qty_tp1, qty_open_remain)
                    exit_px, exit_filled, _ = place_market_and_wait_fill(exit_side, qty_tp1, reduce_only=True, timeout=5.0)
                    ts_live = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(tz)

                    if exit_px and exit_filled > 0:
                        realized_px_roi = ((exit_px - entry_price) / entry_price) if direction == "LONG" \
                                          else ((entry_price - exit_px) / entry_price)
                        realized_acct_roi = (realized_px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)

                        # 부분 비중만큼 자본 업데이트
                        portion = float(exit_filled / qty_open_total)
                        eff_part = open_eff_portion_total * portion
                        capital_before = capital
                        capital = capital * (1.0 + eff_part * realized_acct_roi)

                        qty_open_remain -= exit_filled
                        open_eff_portion_total -= eff_part
                        tp1_done = True

                        print(f"[{ts_live:%Y-%m-%d %H:%M}] 💰 PARTIAL TP1 "
                              f"qty={exit_filled:.6f}/{qty_open_total:.6f}  "
                              f"entry={entry_price:.6f} exit={exit_px:.6f}  "
                              f"pxROI*={pct(realized_px_roi, signed=True)}  acctROI={pct(realized_acct_roi, signed=True)}  "
                              f"Eq ${capital_before:.2f}→${capital:.2f}  remain_qty={qty_open_remain:.6f}")

                        trade_logs.append({
                            "timestamp": ts_live, "direction": direction, "reason": "TP1_PARTIAL",
                            "realized_pxROI": realized_px_roi, "acctROI": realized_acct_roi,
                            "portion": portion, "capital_before": capital_before, "capital_after": capital,
                            "entry_px": entry_price, "exit_px": exit_px, "filled_qty": float(exit_filled)
                        })
                        # 보유 지속
                        equity_curve.append((ts_live, capital))
                        continue
                    else:
                        print("[WARN] Partial TP no-fill → 전체 청산 로직으로 진행")

                # --- 전체 청산
                qty_to_close = max(get_symbol_filters(symbol)["minQty"], round(qty_open_remain, 6))
                exit_px, exit_filled, _ = place_market_and_wait_fill(exit_side, qty_to_close, reduce_only=True, timeout=5.0)
                ts_live = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(tz)

                if not exit_px or exit_filled <= 0:
                    print(f"[{ts_live:%Y-%m-%d %H:%M}] ❌ EXIT NO-FILL — 재시도")
                    equity_curve.append((ts_live, capital))
                    continue

                realized_px_roi = ((exit_px - entry_price) / entry_price) if direction == "LONG" \
                                  else ((entry_price - exit_px) / entry_price)
                realized_acct_roi = (realized_px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)

                capital_before = capital
                capital = capital * (1.0 + open_eff_portion_total * realized_acct_roi)
                trades += 1
                trades_today += 1

                # 연속 손실/쿨다운 갱신
                if realized_acct_roi < 0:
                    consec_losses += 1
                    if consec_losses >= MAX_CONSEC_LOSSES:
                        cooldown_until = pd.Timestamp.utcnow() + pd.Timedelta(minutes=COOLDOWN_MINUTES)
                        print(f"[RISK] {consec_losses} losses in a row → cooldown until {cooldown_until}")
                else:
                    consec_losses = 0

                print(f"[{ts_live:%Y-%m-%d %H:%M}] 💥 EXIT(LIVE) {reason}  "
                      f"entry={entry_price:.6f} exit={exit_px:.6f}  "
                      f"pxROI*={pct(realized_px_roi, signed=True)}  "
                      f"acctROI={pct(realized_acct_roi, signed=True)}  "
                      f"eff={open_eff_portion_total:.4f}  Eq ${capital_before:.2f}→${capital:.2f}  "
                      f"trades_today={trades_today} DD={pct(1 - capital/day_start_equity, d=2)}")

                trade_logs.append({
                    "timestamp": ts_live, "direction": direction, "reason": reason,
                    "realized_pxROI": realized_px_roi, "acctROI": realized_acct_roi,
                    "eff_portion": open_eff_portion_total,
                    "capital_before": capital_before, "capital_after": capital,
                    "entry_px": entry_price, "exit_px": exit_px,
                    "filled_qty": float(exit_filled),
                    "tp1_done": tp1_done
                })

                # 리셋
                is_holding  = False
                direction   = None
                entry_price = None
                hold_thr    = None
                entry_ts_utc= None
                open_eff_portion_total = 0.0
                qty_open_total  = 0.0
                qty_open_remain = 0.0
                tp1_done        = False

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
