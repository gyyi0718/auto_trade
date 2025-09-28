#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import os, math, time, requests, numpy as np, pandas as pd, torch, torch.nn as nn
from collections import deque
from decimal import Decimal
from datetime import datetime
from binance.client import Client
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.deepar import DeepAR

# ========= API =========
API_KEY    = "kynaVBENI8TIS8AcZICK2Sn52mJfIqHpCHfKv2IaSM7TQtKaazn44KGcsJRERjbE"
API_SECRET = "z2ZtL0S8eIE34C8bL1XpAV0jEhcc8JEm8VBOEZbo5Rbe4d9HkN2F4V2AGukhtXyT"
client     = Client(API_KEY, API_SECRET)

# ========= Trading Config =========
symbol     = "BTCUSDT"
CKPT_FILE  = f"{symbol}_deepar_model.ckpt"   # DeepAR 체크포인트
PROFIT_CLS_PATH = f"{symbol}_profit_cls.pt"  # Profit 분류기 (없으면 게이트 비활성)

seq_len    = 60
pred_len   = 60
interval   = "1m"
SLEEP_SEC  = 3

# fees & slippage as DECIMALS (0.0004 = 0.04%)
TAKER_FEE_ONEWAY = 0.0004
SLIPPAGE_ONEWAY  = 0.0005

LEVERAGE       = 20
MIN_PROFIT_USD = 0.01
ENTRY_PORTION  = 0.3

# user pxROI threshold (decimal). ex) 0.001 = 0.1%
thresh_roi = 0.001

# Profit 게이트 컷오프(확률)
P_PROFIT_CUTOFF = 0.60   # 0.60~0.70 추천

LOG_FILE            = f"{symbol}_trade_log.csv"
PREDICTION_LOG_FILE = f"{symbol}_predict_log.csv"

# ===== Risk controls (DECIMALS) =====
TP_PX         = 0.0030   # 0.30% (소수)
SL_PX         = 0.0010   # 0.10%
TRAIL_PX      = 0.0015   # 0.15%
MAX_LOSS_ACCT = 0.0182   # 1.82% (px SL과 등가치)
MAX_HOLD_SEC  = 60 * pred_len
LIQ_BUFFER    = 0.02     # 청산가 대비 2% 버퍼

# ===== Derived roundtrip costs =====
FEE_ROUNDTRIP  = TAKER_FEE_ONEWAY * 2
SLIP_ROUNDTRIP = SLIPPAGE_ONEWAY  * 2

# ===== Entry/exit guards =====
# 진입 직후 체결가와 실시간가 괴리 방지 (슬리피지/급변시)
GAP_GUARD_PX = 0.0025    # 0.25% 이내 아니면 10초 내 대기

# ===== Helpers =====
def pct(x: float, digits: int = 4, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    val = float(x) * 100.0
    return (f"{val:+.{digits}f}%" if signed else f"{val:.{digits}f}%")

def fmt_seq_percent(seq, digits: int = 4) -> str:
    try:
        return ", ".join(pct(float(v), digits=digits, signed=True) for v in seq)
    except Exception:
        return str(seq)

def price_move_threshold_for_fees(leverage: int) -> float:
    return (FEE_ROUNDTRIP + SLIP_ROUNDTRIP) / leverage

def debug_thresholds(leverage: int, fee_threshold: float, entry_threshold: float):
    roundtrip_cost = (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
    chosen = "thresh_roi" if abs(thresh_roi) >= fee_threshold else "fee_threshold"
    print(f"[THRESH] fee_per_side={pct(TAKER_FEE_ONEWAY)}  "
          f"slip_per_side={pct(SLIPPAGE_ONEWAY)}  "
          f"roundtrip={pct(roundtrip_cost)}  leverage=x{leverage}")
    print(f"[THRESH] fee_threshold(px) = {fee_threshold:.6f} ({pct(fee_threshold)})")
    print(f"[THRESH] user_threshold(px)= {abs(thresh_roi):.6f} ({pct(abs(thresh_roi))})")
    print(f"[THRESH] entry_threshold(chosen={chosen}) = "
          f"{entry_threshold:.6f} ({pct(entry_threshold)})")

def print_prediction_log(rois, target_roi, target_min, capital, p_profit=None):
    fee_th   = price_move_threshold_for_fees(LEVERAGE)
    entry_th = max(abs(thresh_roi), fee_th)
    now_ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 80)
    print(f"[{now_ts}] 🔮 pxROI_seq%: {fmt_seq_percent(rois)}")
    print(f"🎯 Target pxROI={pct(target_roi, signed=True)} @ {int(target_min)}m")
    debug_thresholds(LEVERAGE, fee_th, entry_th)
    if capital > 0:
        extra   = MIN_PROFIT_USD / (capital * ENTRY_PORTION * LEVERAGE)
        need_px = max(abs(thresh_roi), fee_th + extra)
        print(f"[NEED] min |pxROI| = {pct(need_px)}  "
              f"(fee_th={pct(fee_th)}, extra={pct(extra)}; L=x{LEVERAGE}, cap={capital:.2f})")
    else:
        print("[NEED] capital=0 -> cannot compute min |pxROI|")
    if p_profit is not None:
        print(f"[GATE] P(profit)={p_profit*100:.2f}%  cutoff={P_PROFIT_CUTOFF*100:.0f}%  "
              f"=> {'PASS' if p_profit>=P_PROFIT_CUTOFF else 'REJECT'}")
    print("=" * 80)

# ===== Logging =====
def setup_logging():
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=[
            "timestamp","event","direction","price","roi","value","capital","note"
        ]).to_csv(LOG_FILE, index=False)
    if not os.path.exists(PREDICTION_LOG_FILE):
        pd.DataFrame(columns=[
            "timestamp","now_price","target_price","pred_pct_roi",
            "target_min","pred_roi","entered"
        ]).to_csv(PREDICTION_LOG_FILE, index=False)

def log_event(event, direction, price, roi, value, capital, note=""):
    pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event, "direction": direction, "price": price,
        "roi": roi, "value": value, "capital": capital, "note": note
    }]).to_csv(LOG_FILE, mode="a", header=False, index=False)

def log_prediction(now_price, target_price, target_min, pred_roi, entered):
    pred_pct_roi = (target_price / now_price - 1) * 100.0
    pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "now_price": now_price,
        "target_price": target_price,
        "pred_pct_roi": pred_pct_roi,
        "target_min": target_min,
        "pred_roi": pred_roi,
        "entered": entered
    }]).to_csv(PREDICTION_LOG_FILE, mode="a", header=False, index=False)

# ===== Model load (DeepAR) =====
def load_deepar_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt.get("hyper_parameters", {}).pop("dataset_parameters", None)
    tmp = f"{symbol}_tmp.ckpt"
    torch.save(ckpt, tmp)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepAR.load_from_checkpoint(tmp, map_location=device)
    return model.to(device).eval()

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

# ===== Profit Classifier (게이트) =====
class ProfitClassifier(nn.Module):
    def __init__(self, in_ch=3, hidden=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=in_ch, hidden_size=hidden, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :]
        logit = self.head(h)
        return logit

def load_profit_classifier(path):
    if not os.path.exists(path):
        print(f"[INFO] Profit classifier not found: {path} (gate disabled)")
        return None, None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ProfitClassifier().to(device).eval()
    sd = torch.load(path, map_location=device)
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    model.load_state_dict(state)
    print(f"[INFO] Profit classifier loaded: {path}")
    return model, device, True

def profit_gate_probability(df_seq, vol_mu, vol_sd, profit_model, profit_device):
    """
    df_seq: 최소 seq_len+1 rows 포함해야 함 (log_return/volume 보유)
    features: [log_return, volume_z, rolling_vol] (seq_len, 3)
    """
    if profit_model is None:
        return None
    tail = df_seq.tail(seq_len).copy()
    lr = tail["log_return"].astype(np.float32).values
    vol = tail["volume"].astype(np.float32).values
    vz  = (vol - vol_mu) / (vol_sd + 1e-8)
    rv_series = pd.Series(df_seq["log_return"].astype(float)).rolling(seq_len).std(ddof=0).fillna(0.0)
    rv = rv_series.values[-seq_len:].astype(np.float32)

    feats = np.stack([lr, vz, rv], axis=1).astype(np.float32)   # (seq_len, 3)
    x = torch.from_numpy(feats).unsqueeze(0).to(profit_device)  # (1, T, 3)
    with torch.no_grad():
        logit = profit_model(x)
        p = torch.sigmoid(logit).item()
    return p

# ===== Exchange helpers =====
def ensure_margin_and_leverage(symbol, leverage=1, isolated=True):
    try:
        client.futures_change_margin_type(symbol=symbol,
                                          marginType="ISOLATED" if isolated else "CROSSED")
    except Exception as e:
        if "-4046" not in str(e):
            print(f"[WARN] change_margin_type: {e}")
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
    except Exception as e:
        print(f"[WARN] change_leverage: {e}")
    try:
        pos_info = client.futures_position_information(symbol=symbol)
        exch_lev = int(float(pos_info[0]["leverage"]))
        mt = pos_info[0].get("marginType", "unknown")
        print(f"[INFO] Exchange leverage set: x{exch_lev} ({mt})")
    except Exception as e:
        print(f"[WARN] read leverage: {e}")

def get_futures_balance(asset="USDT"):
    for b in client.futures_account_balance(recvWindow=10000):
        if b["asset"] == asset:
            return float(b["availableBalance"])
    return 0.0

def place_market_order(symbol, side, qty):
    return client.futures_create_order(symbol=symbol, side=side, type="MARKET",
                                       quantity=qty, recvWindow=10000)

def calculate_safe_quantity(now_price, capital):
    pos_val = Decimal(str(capital)) * Decimal(str(ENTRY_PORTION)) * Decimal(str(LEVERAGE))
    qty = (pos_val / Decimal(str(now_price)))
    info  = client.futures_exchange_info()
    sym   = next(s for s in info["symbols"] if s["symbol"] == symbol)
    prec  = sym["quantityPrecision"]
    min_qty = float(next(f for f in sym["filters"] if f["filterType"]=="LOT_SIZE")["minQty"])
    qty = round(float(qty), prec)
    return {"quantity": qty, "valid": qty >= min_qty}

def cleanup_orphaned_position():
    capital   = get_futures_balance()
    positions = client.futures_position_information(symbol=symbol)
    for p in positions:
        amt = float(p["positionAmt"])
        if amt == 0: continue
        side  = "SELL" if amt > 0 else "BUY"
        mk    = place_market_order(symbol, side, abs(amt))
        price = float(mk.get("avgFillPrice", 0))
        log_event("FORCE_EXIT", "LONG" if amt>0 else "SHORT",
                  price, 0, capital, capital, note="orphan")
        return True
    return False

# ===== Data (for prediction only) =====
def fetch_ohlcv(symbol, limit=100):
    url = "https://fapi.binance.com/fapi/v1/klines"
    try:
        r = requests.get(url, params={"symbol":symbol, "interval":interval, "limit":limit}, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[WARN] fetch_ohlcv failed: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume","close_time","q","n","tb","tq","ignore"
    ])
    try:
        df["close"]  = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
    except Exception:
        df["close"]  = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df.dropna()

# ===== Live price (for position management) =====
def get_live_prices():
    """실시간 가격 소스 묶음: 우선순위 mid → mark → last"""
    last = mark = bid = ask = mid = None
    try:
        t = client.futures_symbol_ticker(symbol=symbol)  # last
        last = float(t["price"])
    except Exception:
        pass
    try:
        p = client.futures_mark_price(symbol=symbol)     # mark
        mark = float(p["markPrice"])
    except Exception:
        pass
    try:
        bt = client.futures_orderbook_ticker(symbol=symbol)  # best bid/ask
        bid = float(bt["bidPrice"]); ask = float(bt["askPrice"])
        mid = (bid + ask) / 2.0
    except Exception:
        pass
    live = mid if mid is not None else (mark if mark is not None else last)
    return {"live": live, "last": last, "mark": mark, "bid": bid, "ask": ask, "mid": mid}

# ===== Predict =====
def predict(df_seq, now_price, n_samples=64):
    """
    returns:
      prices_mean: (pred_len,)
      avg_roi:     (pred_len,)  per-horizon expected px ROI (decimal)
    """
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
    else: raise RuntimeError(f"Unexpected prediction shape: {tuple(y.shape)}")

    pct_steps = np.expm1(samples)               # (S, T)
    avg       = pct_steps.mean(axis=0)          # (T,)
    cum       = np.cumsum(np.log1p(pct_steps), axis=1)
    prices    = now_price * np.exp(cum)         # (S, T)
    prices_mean = prices.mean(axis=0)
    return prices_mean, avg

# ===== Risk-managed exit (LIVE PRICE) =====
def monitor_until_exit(entry_price, direction, quantity, capital):
    """
    Profit-first exit loop (live price 기반)
    - 미실현 acctROI: 레버리지 반영 - 편도(one-way) 비용만 차감
    - 실현 acctROI: 왕복 비용 차감
    - best_px가 TP를 '터치'하면: 즉시 익절 또는 초미세 트레일링으로 잠금
    - 브레이크이븐 스톱: 일정 수익 찍히면 0 밑으로 재하락 시 즉시 청산
    - 일반 트레일링/타임아웃/청산가 보호 포함
    """

    MIN_HOLD_SEC = 5
    EPS          = 1e-9

    side_exit = "SELL" if direction == "LONG" else "BUY"
    start_ts  = time.time()
    best_px   = 0.0
    tp_hit    = False
    be_armed  = False

    # px SL과 등가인 계정손절(왕복비용 고려)
    acct_sl_equiv = SL_PX * LEVERAGE - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
    acct_sl_equiv = max(acct_sl_equiv, 0.01)   # 안전 하한 1%

    # 브레이크이븐 무장 임계
    fee_th_px = (FEE_ROUNDTRIP + SLIP_ROUNDTRIP) / LEVERAGE
    BREAKEVEN_ARM_PX = max(0.0006, fee_th_px * 1.2)  # ≥0.06% or 비용*1.2

    POST_TP_TRAIL = min(TRAIL_PX, 0.0002)  # 0.02%

    print("⏳ Managing position with SL/TP/Trail/BE/Time/Liq-guard...")

    while True:
        # --- 실시간 가격 읽기 ---
        px = get_live_prices()
        latest = px["live"]
        if latest is None:
            time.sleep(0.5)
            continue

        now_sec = time.time() - start_ts

        # 진입 직후 과도한 갭 방어(10초)
        if now_sec < 10:
            gap = abs((latest - entry_price) / entry_price)
            if gap >= GAP_GUARD_PX:
                time.sleep(0.5)
                continue

        # px ROI (소수)
        if direction == "LONG":
            px_roi = (latest - entry_price) / entry_price
        else:
            px_roi = (entry_price - latest) / entry_price

        # 미실현 계정 ROI: 편도 비용만 반영
        acct_roi_unreal = (px_roi * LEVERAGE) - (TAKER_FEE_ONEWAY + SLIPPAGE_ONEWAY)

        # 최고치 갱신
        if px_roi > best_px:
            best_px = px_roi

        # TP 터치 여부
        if best_px + EPS >= TP_PX:
            tp_hit = True

        # 브레이크이븐 스톱 무장
        if not be_armed and best_px >= BREAKEVEN_ARM_PX - EPS:
            be_armed = True

        # 청산가 보호(참고: 일부 거래소는 지연 가능)
        try:
            pos_info  = client.futures_position_information(symbol=symbol)
            liq_price = float(pos_info[0].get("liquidationPrice", 0) or 0)
        except Exception:
            liq_price = 0.0

        liq_risk = False
        if liq_price > 0:
            if direction == "LONG"  and latest <= liq_price * (1.0 + LIQ_BUFFER): liq_risk = True
            if direction == "SHORT" and latest >= liq_price * (1.0 - LIQ_BUFFER): liq_risk = True

        print(f"[HOLD] pxROI={pct(px_roi, signed=True)}  "
              f"acctROI={pct(acct_roi_unreal, signed=True)}  "
              f"best_px={pct(best_px)}  "
              f"(decimals: px={px_roi:.6f}, acct={acct_roi_unreal:.6f})")

        # 최소 보유 시간
        if now_sec < MIN_HOLD_SEC:
            time.sleep(0.5)
            continue

        # --- 청산 사유 판정 ---
        reason = None

        # 0) 브레이크이븐 스톱: 무장되면 손익 0 밑으로 내려가면 즉시 청산
        if reason is None and be_armed and px_roi <= 0.0 + EPS:
            reason = "BREAKEVEN_STOP"

        # 1) 손절: px 또는 계정 손절
        if reason is None and (px_roi <= -SL_PX - EPS or acct_roi_unreal <= -acct_sl_equiv - EPS):
            reason = "STOP_LOSS"

        # 2) 즉시 익절: 현재 px가 TP 이상이거나 계정 ROI 등가 이상
        tp_acct_equiv = TP_PX * LEVERAGE - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
        if reason is None and (px_roi >= TP_PX - EPS or acct_roi_unreal >= tp_acct_equiv - EPS):
            reason = "TAKE_PROFIT"

        # 3) TP 터치 후 초미세 트레일 잠금
        if reason is None and tp_hit and (best_px - px_roi) >= POST_TP_TRAIL - EPS:
            reason = "TRAILING_TP_LOCK"

        # 4) 일반 트레일링
        if reason is None and best_px > 0 and (best_px - px_roi) >= TRAIL_PX - EPS:
            reason = "TRAILING_EXIT"

        # 5) 최대 보유 시간
        if reason is None and now_sec >= MAX_HOLD_SEC:
            reason = "TIME_EXIT"

        # 6) 청산가 근접 보호
        if reason is None and liq_risk:
            reason = "LIQUIDATION_GUARD"

        # --- 청산 실행 ---
        if reason:
            side_exit = "SELL" if direction == "LONG" else "BUY"
            try:
                mk = place_market_order(symbol, side_exit, quantity)
                price = float(mk.get("avgFillPrice", latest)) if mk else latest
            except Exception:
                price = latest

            # 실현 px/계정 ROI (왕복 비용 반영)
            if direction == "LONG":
                realized_px_roi = (price - entry_price) / entry_price
            else:
                realized_px_roi = (entry_price - price) / entry_price

            realized_acct_roi = (realized_px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)

            log_event(
                reason, direction, price, realized_acct_roi,
                capital * (1 + realized_acct_roi), capital,
                note=f"tp={TP_PX}, sl={SL_PX}, trail={TRAIL_PX}, be_arm={BREAKEVEN_ARM_PX:.6f}"
            )
            print(f"💥 EXIT[{reason}] @ {price:.6f}  "
                  f"pxROI={pct(realized_px_roi, signed=True)}  "
                  f"acctROI={pct(realized_acct_roi, signed=True)}  "
                  f"(decimals: {realized_acct_roi:.6f})")
            return realized_acct_roi

        time.sleep(0.5)

def monitor_until_profit(entry_price, direction, quantity, capital):
    return monitor_until_exit(entry_price, direction, quantity, capital)

# ===== Main =====
if __name__ == "__main__":
    ensure_margin_and_leverage(symbol, leverage=LEVERAGE, isolated=True)
    setup_logging()

    # 학습 입력 스키마와 동일한 CSV (정규화/피처용 통계에 사용)
    df_all = pd.read_csv(f"{symbol}_deepar_input.csv").dropna()
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
    model = load_deepar_model(CKPT_FILE)

    # Profit classifier (optional)
    profit_model, profit_device, gate_enabled = load_profit_classifier(PROFIT_CLS_PATH)

    print("[INFO] Live trading started")
    seed_df = fetch_ohlcv(symbol, seq_len + pred_len)
    data_q  = deque(seed_df.to_dict("records"), maxlen=seq_len + pred_len)

    is_holding = False
    capital    = get_futures_balance()

    while True:
        if cleanup_orphaned_position():
            time.sleep(SLEEP_SEC); continue

        # 예측용 최신 데이터(봉)
        df_new = fetch_ohlcv(symbol, limit=2)
        if df_new.empty:
            time.sleep(SLEEP_SEC); continue

        data_q.append(df_new.iloc[-1])
        df_seq = pd.DataFrame(list(data_q))
        if len(df_seq) < seq_len + pred_len:
            time.sleep(SLEEP_SEC); continue

        now_price   = df_seq["close"].iloc[-pred_len - 1]
        prices, rois= predict(df_seq, now_price)
        idx         = int(np.argmax(np.abs(rois)))
        target_roi  = float(rois[idx])           # decimal
        target_pr   = float(prices[idx])
        target_min  = idx + 1

        # Profit gate probability
        p_profit = profit_gate_probability(df_seq, VOL_MU, VOL_SD, profit_model, profit_device) \
                   if gate_enabled else None

        # 로그
        print_prediction_log(rois, target_roi, target_min, capital, p_profit=p_profit)

        entered = False

        # 기본 문턱(수수료+슬리피지/레버리지 보정 vs 사용자 지정)
        fee_threshold   = price_move_threshold_for_fees(LEVERAGE)
        entry_threshold = max(abs(thresh_roi), fee_threshold)

        # 게이트 통과 여부
        gate_pass = True
        if gate_enabled:
            gate_pass = (p_profit is not None) and (p_profit >= P_PROFIT_CUTOFF)

        # 진입 조건 (경계 포함)
        if (not is_holding) and (abs(target_roi) >= entry_threshold) and gate_pass:
            capital = get_futures_balance()

            notional         = capital * ENTRY_PORTION * LEVERAGE
            expected_net_px  = max(0.0, abs(target_roi) - fee_threshold)
            expected_net_usd = notional * expected_net_px

            if expected_net_usd < (MIN_PROFIT_USD - 1e-12):
                print("❌ 예상 순이익 부족 → 스킵")
                log_prediction(now_price, target_pr, target_min, target_roi, False)
                time.sleep(SLEEP_SEC); continue

            direction = "LONG" if target_roi > 0 else "SHORT"
            side      = "BUY"  if direction == "LONG" else "SELL"
            # 실시간가 기반 수량 산출을 위해 live 가격으로 업데이트(선택)
            px_live = get_live_prices()["live"] or now_price
            qinfo    = calculate_safe_quantity(px_live, capital)

            if not qinfo["valid"]:
                print("❌ 주문 수량 부족(최소 수량 미만) → 스킵")
                log_prediction(now_price, target_pr, target_min, target_roi, False)
                time.sleep(SLEEP_SEC); continue

            try:
                order = place_market_order(symbol, side, qinfo["quantity"])
                if not order: raise Exception("order None")
                # 체결가를 entry_price로
                entry_price = float(order.get("avgFillPrice", px_live))
            except Exception as e:
                print(f"❌ 진입 실패: {e}")
                log_event("ENTRY_FAILED", direction, now_price, 0, capital, capital, note=str(e))
                log_prediction(now_price, target_pr, target_min, target_roi, False)
                time.sleep(SLEEP_SEC); continue

            is_holding = True
            entered    = True
            gate_msg   = ("DISABLED" if not gate_enabled else ("PASS" if gate_pass else "REJECT"))
            print(f"🚀 ENTRY [{direction}] @ {entry_price:.6f}  (P_profit={p_profit*100:.2f}% {gate_msg})")
            log_event("ENTRY_MARKET", direction, entry_price, 0, capital, capital, f"min={target_min}")

            acct_roi = monitor_until_exit(entry_price, direction, qinfo["quantity"], capital)
            is_holding = False
            capital   *= (1.0 + acct_roi)

        log_prediction(now_price, target_pr, target_min, target_roi, entered)
        time.sleep(SLEEP_SEC)
