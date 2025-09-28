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
symbol     = "ETHUSDT"
CKPT_FILE  = f"{symbol}_deepar_model.ckpt"   # DeepAR 체크포인트
INPUT_CSV  = f"{symbol}_deepar_input.csv"    # 학습 스키마/통계 재사용(필수)
PROFIT_CLS_PATH = f"{symbol}_profit_cls.pt"  # Profit 분류기 (없으면 게이트 비활성)

seq_len    = 60
pred_len   = 60
interval   = "1m"

# ===== 전략(필수) =====
# 'A' = 스캘핑: SL=1*ATR, TP=2*ATR, Trail=1*ATR, BE=0.6*ATR, 타임스톱≈1.1*target_min
# 'B' = 트렌드: SL=1.5*ATR, TP 없음, Trail=2.5*ATR, BE=1*ATR, 타임스톱 느슨
# 'C' = 평균회귀: SL=1.2*ATR, TP=1.5*ATR, Trail=1*ATR, BE=0.6*ATR, 타임스톱≈1.5*target_min
STRATEGY   = "A"
ATR_N      = 14
PARTIAL_TP_RATIO = 0.5   # 첫 익절시 부분청산 비율(0~1). 0이면 부분익절 비활성

# ===== 수수료/슬리피지 (소수). 필요시 거래소 값으로 바꾸세요 =====
TAKER_FEE_ONEWAY = 0.0004
SLIPPAGE_ONEWAY  = 0.0005
FEE_ROUNDTRIP    = TAKER_FEE_ONEWAY * 2
SLIP_ROUNDTRIP   = SLIPPAGE_ONEWAY  * 2

# ===== 포지션/진입 =====
LEVERAGE       = 5         # 레버리지
ENTRY_PORTION  = 0.9       # 가용자본 대비 목표 진입 비율
MIN_PROFIT_USD = 0.01
thresh_roi     = 0.0001    # pxROI 문턱(소수). ex) 0.001 = 0.1%

# ===== Profit 게이트 =====
P_PROFIT_CUTOFF = 0.60

# ===== 로그 파일 =====
LOG_FILE            = f"{symbol}_trade_log.csv"
PREDICTION_LOG_FILE = f"{symbol}_predict_log.csv"

# ===== 실행 모드 =====
EVAL_ON_NEW_CLOSED_BAR  = True
POLL_MS                 = 10      # 실시간 가격 폴링(ms)

PRICE_DECIMALS_SAFE   = 2

# ===== Helpers =====
def pct(x: float, digits: int = 4, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)): return "nan"
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
    print(f"[THRESH] fee/side={pct(TAKER_FEE_ONEWAY)} slip/side={pct(SLIPPAGE_ONEWAY)} "
          f"roundtrip={pct(roundtrip_cost)} L=x{leverage}")
    print(f"[THRESH] fee_threshold(px) = {fee_threshold:.6f} ({pct(fee_threshold)})")
    print(f"[THRESH] user_threshold(px)= {abs(thresh_roi):.6f} ({pct(abs(thresh_roi))})")
    print(f"[THRESH] entry_threshold(chosen={chosen}) = {entry_threshold:.6f} ({pct(entry_threshold)})")

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
            "timestamp","event","direction","price","px_roi","acct_roi","capital","note"
        ]).to_csv(LOG_FILE, index=False)
    if not os.path.exists(PREDICTION_LOG_FILE):
        pd.DataFrame(columns=[
            "timestamp","now_price","target_price","pred_pct_roi",
            "target_min","pred_roi","entered"
        ]).to_csv(PREDICTION_LOG_FILE, index=False)

def log_event(event, direction, price, px_roi, acct_roi, capital, note=""):
    pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event, "direction": direction, "price": price,
        "px_roi": px_roi, "acct_roi": acct_roi, "capital": capital, "note": note
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
        return None, None, False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ProfitClassifier().to(device).eval()
    sd = torch.load(path, map_location=device)
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    model.load_state_dict(state)
    print(f"[INFO] Profit classifier loaded: {path}")
    return model, device, True

def profit_gate_probability(df_seq, vol_mu, vol_sd, profit_model, profit_device):
    if profit_model is None:
        return None
    tail = df_seq.tail(seq_len).copy()
    lr  = tail["log_return"].astype(np.float32).values
    vol = tail["volume"].astype(np.float32).values
    vz  = (vol - vol_mu) / (vol_sd + 1e-8)
    rv_series = pd.Series(df_seq["log_return"].astype(float)).rolling(seq_len).std(ddof=0).fillna(0.0)
    rv  = rv_series.values[-seq_len:].astype(np.float32)
    feats = np.stack([lr, vz, rv], axis=1).astype(np.float32)
    x = torch.from_numpy(feats).unsqueeze(0).to(profit_device)
    with torch.no_grad():
        p = torch.sigmoid(profit_model(x)).item()
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
    try:
        for b in client.futures_account_balance(recvWindow=10000):
            if b["asset"] == asset:
                return float(b["availableBalance"])
    except Exception:
        pass
    return 0.0

def get_symbol_filters(symbol):
    try:
        info  = client.futures_exchange_info()
        sym   = next(s for s in info["symbols"] if s["symbol"] == symbol)
        prec  = int(sym["quantityPrecision"])
        min_qty = float(next(f for f in sym["filters"] if f["filterType"]=="LOT_SIZE")["minQty"])
        return {"quantityPrecision":prec, "minQty":min_qty}
    except Exception:
        return {"quantityPrecision":3, "minQty":0.001}

def place_market_order(symbol, side, qty, reduce_only=False):
    params = dict(symbol=symbol, side=side, type="MARKET",
                  quantity=qty, recvWindow=10000, newOrderRespType="RESULT")
    if reduce_only:
        params["reduceOnly"] = True
    return client.futures_create_order(**params)

def calculate_safe_quantity(now_price, capital):
    pos_val = Decimal(str(capital)) * Decimal(str(ENTRY_PORTION)) * Decimal(str(LEVERAGE))
    qty = (pos_val / Decimal(str(now_price)))
    f = get_symbol_filters(symbol)
    prec, min_qty = f["quantityPrecision"], f["minQty"]
    qty = round(float(qty), prec)
    return {"quantity": qty, "valid": qty >= min_qty, "prec":prec, "minQty":min_qty}

def cleanup_orphaned_position():
    positions = []
    try:
        positions = client.futures_position_information(symbol=symbol)
    except Exception:
        return False
    for p in positions:
        amt = float(p["positionAmt"])
        if amt == 0: continue
        side  = "SELL" if amt > 0 else "BUY"
        try:
            mk    = place_market_order(symbol, side, abs(amt), reduce_only=True)
            price = float(mk.get("avgPrice") or mk.get("avgFillPrice") or 0) if mk else 0.0
        except Exception:
            price = 0.0
        cap = get_futures_balance()
        log_event("FORCE_EXIT", "LONG" if amt>0 else "SHORT", price, 0, 0, cap, note="orphan")
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
    for c in ["open","high","low","close","volume"]:
        df[c]  = pd.to_numeric(df[c], errors="coerce")
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df.dropna().reset_index(drop=True)

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
def predict(df_seq, now_price, dataset, model, n_samples=64):
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
    pct_steps = np.expm1(samples)
    avg       = pct_steps.mean(axis=0)
    cum       = np.cumsum(np.log1p(pct_steps), axis=1)
    prices    = now_price * np.exp(cum)
    prices_mean = prices.mean(axis=0)
    return prices_mean, avg

# ===== ATR & 동적 임계 =====
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
    elif strategy == "B":  # 트렌드
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

# ===== 실시간 엑싯 시그널 =====
def wait_exit_signal(entry_price, direction, thr, poll_ms=10, entry_ts_utc=None):
    """
    실시간 가격으로 TP/SL/Trail/BE/TimeStop 판정만 수행.
    반환: (reason, latest_price_at_decision)
    """
    EPS = 1e-9
    best_px = 0.0
    tp_hit  = False
    be_armed= False
    t0 = time.time() if entry_ts_utc is None else entry_ts_utc

    while True:
        px = get_live_prices(); latest = px["live"]
        if latest is None:
            time.sleep(max(0.001, poll_ms/1000.0)); continue

        px_roi = (latest - entry_price)/entry_price if direction=="LONG" else (entry_price - latest)/entry_price
        acct_roi_unreal = (px_roi * LEVERAGE) - (TAKER_FEE_ONEWAY + SLIPPAGE_ONEWAY)

        if px_roi > best_px: best_px = px_roi
        if (thr["tp_px"] is not None) and (best_px >= thr["tp_px"] - EPS): tp_hit = True
        if not be_armed and best_px >= max(thr["be_arm_px"], (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)/LEVERAGE*1.2) - EPS:
            be_armed = True

        # 타임 스톱
        if thr.get("time_stop_s") and (time.time() - t0) >= thr["time_stop_s"]:
            return "TIME_STOP", latest

        # 브레이크이븐 스톱
        if be_armed and px_roi <= 0.0 + EPS:
            return "BREAKEVEN_STOP", latest

        # 손절
        if px_roi <= -thr["sl_px"] - EPS or acct_roi_unreal <= -(thr["sl_px"]*LEVERAGE - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)) - EPS:
            return "STOP_LOSS", latest

        # 고정 TP
        if (thr["tp_px"] is not None):
            tp_acct_equiv = thr["tp_px"] * LEVERAGE - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
            if (px_roi >= thr["tp_px"] - EPS) or (acct_roi_unreal >= tp_acct_equiv - EPS):
                return "TAKE_PROFIT", latest

        # TP 이후 잠금(촘촘)
        if tp_hit and thr["trail_px"] is not None and (best_px - px_roi) >= min(thr["trail_px"], 0.0002) - EPS:
            return "TRAILING_TP_LOCK", latest

        # 일반 트레일링
        if thr["trail_px"] is not None and best_px > 0 and (best_px - px_roi) >= thr["trail_px"] - EPS:
            return "TRAILING_EXIT", latest

        time.sleep(max(0.001, poll_ms/1000.0))

# ===== 포지션 매니저 (부분익절 + 최종 청산) =====
def manage_position(entry_price, direction, qty_total, thr, poll_ms=10):
    """
    부분익절 1회(옵션) + 최종 청산까지 처리.
    반환: 마지막 청산 시점의 (px_roi, acct_roi, exit_price)
    """
    qty_remain = float(qty_total)
    tp1_done = False
    f = get_symbol_filters(symbol)
    min_qty, prec = f["minQty"], f["quantityPrecision"]

    entry_ts = time.time()

    while True:
        reason, latest = wait_exit_signal(entry_price, direction, thr, poll_ms=poll_ms, entry_ts_utc=entry_ts)

        # 부분익절
        if (reason == "TAKE_PROFIT") and (not tp1_done) and (PARTIAL_TP_RATIO > 0.0) and (qty_remain > min_qty):
            qty_tp1 = round(max(min_qty, qty_total * PARTIAL_TP_RATIO), prec)
            qty_tp1 = min(qty_tp1, qty_remain)
            exit_side = "SELL" if direction == "LONG" else "BUY"
            try:
                mk = place_market_order(symbol, exit_side, qty_tp1, reduce_only=True)
                exit_px = float(mk.get("avgPrice") or mk.get("avgFillPrice") or latest)
            except Exception:
                exit_px = latest

            px_roi = (exit_px - entry_price)/entry_price if direction=="LONG" else (entry_price - exit_px)/entry_price
            acct_roi = (px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
            log_event("TP1_PARTIAL", direction, exit_px, px_roi, acct_roi, get_futures_balance(),
                      note=f"qty={qty_tp1}/{qty_total}")
            qty_remain = round(qty_remain - qty_tp1, prec)
            tp1_done = True
            # 부분익절 후 계속 보유
            continue

        # 최종 청산
        qty_to_close = round(max(min_qty, qty_remain), prec)
        exit_side = "SELL" if direction == "LONG" else "BUY"
        try:
            mk = place_market_order(symbol, exit_side, qty_to_close, reduce_only=True)
            exit_px = float(mk.get("avgPrice") or mk.get("avgFillPrice") or latest)
        except Exception:
            exit_px = latest

        px_roi = (exit_px - entry_price)/entry_price if direction=="LONG" else (entry_price - exit_px)/entry_price
        acct_roi = (px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
        log_event(f"EXIT_{reason}", direction, exit_px, px_roi, acct_roi, get_futures_balance(),
                  note=f"qty={qty_to_close}/{qty_total}")
        return px_roi, acct_roi, exit_px


def get_net_position(symbol):
    try:
        pos = client.futures_position_information(symbol=symbol)
        # ONE-WAY 모드 가정: 해당 심볼만 필터
        p = next(x for x in pos if x["symbol"] == symbol)
        amt = float(p["positionAmt"])
        side = "LONG" if amt > 0 else ("SHORT" if amt < 0 else None)
        return amt, side
    except Exception:
        return 0.0, None

def ensure_oneway_mode():
    try:
        st = client.futures_get_position_mode()
        # dualSidePosition == True면 헤지 모드
        if st.get("dualSidePosition", False):
            client.futures_change_position_mode(dualSidePosition="false")
            print("[INFO] Position mode → ONE-WAY")
    except Exception as e:
        print(f"[WARN] ensure_oneway_mode: {e}")


# ===== Main =====
if __name__ == "__main__":
    ensure_margin_and_leverage(symbol, leverage=LEVERAGE, isolated=True)
    setup_logging()

    # 학습 입력 스키마/통계 재사용(필수)
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
    model = load_deepar_model(CKPT_FILE)
    profit_model, profit_device, gate_enabled = load_profit_classifier(PROFIT_CLS_PATH)

    print("[INFO] Live trading (ATR dynamic TP/SL/Trail + BE + Partial TP + TimeStop)")

    # 부트스트랩
    seed_df = fetch_ohlcv(symbol, seq_len + pred_len + 5)
    if seed_df.empty or len(seed_df) < (seq_len + pred_len + 1):
        raise RuntimeError("초기 히스토리 부족")
    window_len = seq_len + pred_len
    data_q  = deque(seed_df.tail(window_len).to_dict("records"), maxlen=window_len)

    last_close_time = int(data_q[-1]["close_time"])

    while True:
        try:
            # 새 확정봉 감지
            df2 = fetch_ohlcv(symbol, limit=2)
            if df2.empty:
                time.sleep(1); continue

            ct = int(df2.iloc[-1]["close_time"])
            if EVAL_ON_NEW_CLOSED_BAR and ct == last_close_time:
                time.sleep(1); continue

            last_close_time = ct
            new_bar = df2.iloc[-1]
            data_q.append(new_bar.to_dict())
            df_seq = pd.DataFrame(list(data_q))

            ts = pd.to_datetime(new_bar["close_time"], unit="ms", utc=True).tz_convert("Asia/Seoul")

            # 예측 기준 now_price = 직전 봉 종가
            now_price = df_seq["close"].iloc[-pred_len - 1]

            # 예측
            prices, rois = predict(df_seq, now_price, dataset, model)
            idx        = int(np.argmax(np.abs(rois)))
            target_roi = float(rois[idx])
            target_pr  = float(prices[idx])
            target_min = idx + 1
            direction  = "LONG" if target_roi > 0 else "SHORT"

            # Profit gate
            p_profit = profit_gate_probability(df_seq, VOL_MU, VOL_SD, profit_model, profit_device) if gate_enabled else None

            capital = get_futures_balance()
            print_prediction_log(rois, target_roi, target_min, capital, p_profit=p_profit)

            entered = False

            # 진입 문턱
            fee_threshold   = price_move_threshold_for_fees(LEVERAGE)
            entry_threshold = max(abs(thresh_roi), fee_threshold)
            gate_pass = True if not gate_enabled else (p_profit is not None and p_profit >= P_PROFIT_CUTOFF)

            # ATR 기반 동적 임계
            atr_px = compute_atr_px(df_seq.tail(100), n=ATR_N)
            hold_thr = build_thresholds(atr_px, fee_threshold, STRATEGY, target_min)

            # 기대 순이익(수수료/슬립 포함) 체크
            notional         = capital * ENTRY_PORTION * LEVERAGE
            expected_net_px  = max(0.0, abs(target_roi) - fee_threshold)
            expected_net_usd = notional * expected_net_px

            if (abs(target_roi) >= entry_threshold) and gate_pass and expected_net_usd >= MIN_PROFIT_USD:
                px_live = get_live_prices()["live"] or now_price
                qinfo   = calculate_safe_quantity(px_live, capital)
                if not qinfo["valid"]:
                    print(f"[{ts:%Y-%m-%d %H:%M}] ❌ 주문 수량 부족(최소 수량 미만) → 스킵")
                    log_prediction(now_price, target_pr, target_min, target_roi, False)
                    continue

                # MARKET 진입
                try:
                    side = "BUY" if direction == "LONG" else "SELL"
                    order = place_market_order(symbol, side, qinfo["quantity"], reduce_only=False)
                    entry_price = float(order.get("avgPrice") or order.get("avgFillPrice") or px_live)
                    print(f"[{ts:%Y-%m-%d %H:%M}] 🚀 ENTRY {direction} @ {entry_price:.6f} (MARKET) "
                          f"qty={qinfo['quantity']}")
                except Exception as e:
                    print(f"[{ts:%Y-%m-%d %H:%M}] ❌ 진입 실패: {e}")
                    log_event("ENTRY_FAILED", direction, now_price, 0, 0, capital, note=str(e))
                    log_prediction(now_price, target_pr, target_min, target_roi, False)
                    continue

                entered = True
                log_event("ENTRY", direction, entry_price, 0, 0, capital,
                          f"pred={pct(target_roi, signed=True)}@{target_min}m; thr={hold_thr}")

                # === 포지션 관리 (부분익절 + 최종 청산) ===
                px_roi, acct_roi, exit_px = manage_position(
                    entry_price, direction, qinfo["quantity"], hold_thr, poll_ms=POLL_MS
                )
                print(f"[{ts:%Y-%m-%d %H:%M}] ✅ DONE  entry={entry_price:.6f} exit={exit_px:.6f} "
                      f"pxROI={pct(px_roi, signed=True)} acctROI={pct(acct_roi, signed=True)}")
                capital = get_futures_balance()

            else:
                print(f"[{ts:%Y-%m-%d %H:%M}] ⏳ NO-ENTRY  pred={pct(target_roi, signed=True)}@{target_min}m  "
                      f"Eq≈${capital:.2f}")

            log_prediction(now_price, target_pr, target_min, target_roi, entered)

        except KeyboardInterrupt:
            print("\n[STOP] Interrupted by user.")
            break
        except Exception as e:
            print(f"[WARN] loop error: {e}")
            time.sleep(1)
            continue
