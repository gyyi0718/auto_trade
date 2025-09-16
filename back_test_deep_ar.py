
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import os, math, requests, numpy as np, pandas as pd, torch, torch.nn as nn
from collections import deque
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.deepar import DeepAR

# ================== USER CONFIG ==================
symbol      = "ETHUSDT"
interval    = "1m"   # 1분봉
OHLC_CSV    = f"{symbol}_today_1m.csv"        # ⬅️ 오늘 00:00~지금(KST) 저장될 CSV
CKPT_FILE   = f"{symbol}_deepar_model.ckpt"   # DeepAR 체크포인트(필수)
INPUT_CSV   = f"{symbol}_deepar_input.csv"    # 학습 스키마/통계 재사용(필수)
PROFIT_CLS_PATH = f"{symbol}_profit_cls.pt"   # Profit 게이트(옵션: 없으면 비활성)

# 모델/전략 파라미터
seq_len    = 60
pred_len   = 60
n_samples  = 64   # 느리면 32로 낮춰도 됨

# 수수료/슬리피지
TAKER_FEE_ONEWAY = 0.0004
SLIPPAGE_ONEWAY  = 0.0005
FEE_ROUNDTRIP    = TAKER_FEE_ONEWAY * 2
SLIP_ROUNDTRIP   = SLIPPAGE_ONEWAY  * 2

LEVERAGE       = 20
ENTRY_PORTION  = 1.0
MIN_PROFIT_USD = 0.01
thresh_roi     = 0.001   # px ROI 문턱(0.01%=0.0001)

# Risk controls (px 기준)
TP_PX    = 0.0030
SL_PX    = 0.0010
TRAIL_PX = 0.0015

# Profit 게이트(옵션)
P_PROFIT_CUTOFF = 0.60

# 초기자본 / 저장
INITIAL_CAPITAL = 100
EQUITY_CSV = f"{symbol}_today_equity.csv"
TRADES_CSV = f"{symbol}_today_trades.csv"

# ================== HELPERS ==================
def pct(x, d=4, signed=False):
    if x is None or (isinstance(x, float) and math.isnan(x)): return "nan"
    v = float(x) * 100
    return f"{v:+.{d}f}%" if signed else f"{v:.{d}f}%"

def price_move_threshold_for_fees(leverage: int) -> float:
    return (FEE_ROUNDTRIP + SLIP_ROUNDTRIP) / leverage

def binance_server_now():
    r = requests.get("https://fapi.binance.com/fapi/v1/time", timeout=10)
    r.raise_for_status()
    ms = r.json()["serverTime"]
    return pd.Timestamp(ms, unit="ms", tz="UTC")

def fetch_klines(symbol, start_ms=None, end_ms=None, interval="1m", limit=1000):
    """
    연속 klines 수집 (선물). start_ms/end_ms는 UTC ms.
    오늘 분량(<=1440개)이면 1~2회 호출로 끝남.
    """
    url = "https://fapi.binance.com/fapi/v1/klines"
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
    # 로그수익률
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df.dropna(inplace=True)
    return df.reset_index(drop=True)

def download_today_klines_to_csv(symbol, interval, out_csv):
    """
    오늘 00:00(KST) ~ 지금(KST) 분봉을 CSV로 저장.
    컬럼: close_time(ms,UTC), open, high, low, close, volume
    """
    server_now_utc = binance_server_now()
    now_kst = server_now_utc.tz_convert("Asia/Seoul")
    start_kst = now_kst.normalize()        # 오늘 00:00 KST
    start_utc = start_kst.tz_convert("UTC")
    start_ms  = int(start_utc.timestamp() * 1000)
    end_ms    = int(server_now_utc.timestamp() * 1000)

    df = fetch_klines(symbol, start_ms=start_ms, end_ms=end_ms, interval=interval, limit=1000)
    if df.empty:
        raise RuntimeError("다운로드된 오늘 분봉이 비어있습니다.")

    # 필요한 컬럼만 저장(백테스트 로더와 호환)
    to_save = df[["close_time","open","high","low","close","volume"]].copy()
    to_save.to_csv(out_csv, index=False)
    print(f"[SAVED] Today {interval} klines → {out_csv}  (rows={len(to_save)})")
    return out_csv, now_kst, start_kst

def load_ohlc_from_csv(path: str) -> pd.DataFrame:
    """
    CSV에서 OHLCV 로드. 필요한 컬럼: open, high, low, close, volume, close_time(ms)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = ["close_time","open","high","low","close","volume"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"CSV에 '{c}' 컬럼이 필요합니다.")
    for c in ["open","high","low","close","volume","close_time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    # 로그수익률
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df.dropna(inplace=True)
    return df.reset_index(drop=True)

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

# ================== EXIT (OHLC TOUCH) ==================
def step_exit_with_ohlc(entry_price, direction, bar_ohlc, state):
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
        if px_roi > best_px: best_px = px_roi
        if best_px + EPS >= TP_PX: tp_hit = True
        if not be_armed and best_px >= BREAKEVEN_ARM_PX - EPS: be_armed = True

        if be_armed and px_roi <= 0.0 + EPS:
            realized_px_roi = px_roi
            acct_roi = (realized_px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
            return True, acct_roi, "BREAKEVEN_STOP", realized_px_roi, {"best_px":best_px,"tp_hit":tp_hit,"be_armed":be_armed}
        if px_roi <= -SL_PX - EPS:
            realized_px_roi = px_roi
            acct_roi = (realized_px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
            return True, acct_roi, "STOP_LOSS", realized_px_roi, {"best_px":best_px,"tp_hit":tp_hit,"be_armed":be_armed}
        if px_roi >= TP_PX - EPS:
            realized_px_roi = px_roi
            acct_roi = (realized_px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
            return True, acct_roi, "TAKE_PROFIT", realized_px_roi, {"best_px":best_px,"tp_hit":tp_hit,"be_armed":be_armed}
        if tp_hit and (best_px - px_roi) >= POST_TP_TRAIL - EPS:
            realized_px_roi = px_roi
            acct_roi = (realized_px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
            return True, acct_roi, "TRAILING_TP_LOCK", realized_px_roi, {"best_px":best_px,"tp_hit":tp_hit,"be_armed":be_armed}
        if best_px > 0 and (best_px - px_roi) >= TRAIL_PX - EPS:
            realized_px_roi = px_roi
            acct_roi = (realized_px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
            return True, acct_roi, "TRAILING_EXIT", realized_px_roi, {"best_px":best_px,"tp_hit":tp_hit,"be_armed":be_armed}

    return False, 0.0, None, None, {"best_px":best_px,"tp_hit":tp_hit,"be_armed":be_armed}

# ================== MAIN ==================
if __name__ == "__main__":
    # 1) 오늘 00:00(KST)~지금(KST) 1분봉을 CSV로 저장
    OHLC_CSV, now_kst, start_kst = download_today_klines_to_csv(symbol, interval, OHLC_CSV)

    # 2) 모델/스키마 로드
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

    # 3) CSV 로드 (오늘 데이터만)
    df_hist = load_ohlc_from_csv(OHLC_CSV)
    if len(df_hist) < (seq_len + pred_len + 1):
        raise RuntimeError("오늘 데이터 길이가 짧아 예측 윈도우 확보 불가. 시간이 더 지난 후 실행하세요.")

    # 4) 오프라인 백테스트 (오늘만)
    tz = "Asia/Seoul"
    capital = INITIAL_CAPITAL
    equity_curve, trade_logs = [], []

    is_holding = False
    direction = None
    entry_price = None
    hold_state = {"best_px":0.0, "tp_hit":False, "be_armed":False}
    trades = 0
    wins = 0

    fee_threshold = price_move_threshold_for_fees(LEVERAGE)
    print(f"[INFO] TODAY OFFLINE BT  (KST {start_kst:%Y-%m-%d 00:00} ~ {now_kst:%Y-%m-%d %H:%M}) "
          f"Equity=${capital:.2f}, L=x{LEVERAGE}, Portion={ENTRY_PORTION*100:.0f}%")

    # 초기 윈도우: 오늘 데이터 내에서 바로 시작
    window_len = seq_len + pred_len
    data_q = deque(df_hist.iloc[:window_len].to_dict("records"), maxlen=window_len)

    for i in range(window_len, len(df_hist)):
        new_bar = df_hist.iloc[i].to_dict()
        data_q.append(new_bar)
        df_seq = pd.DataFrame(list(data_q))
        ts = pd.to_datetime(new_bar["close_time"], unit="ms", utc=True).tz_convert(tz)

        if not is_holding:
            now_price = float(df_seq["close"].iloc[-pred_len - 1])

            # 예측
            _, rois = predict_prices_and_pxroi(model, dataset, df_seq, now_price, pred_len, n_samples=n_samples)
            idx = int(np.argmax(np.abs(rois)))
            target_roi = float(rois[idx])
            target_min = idx + 1
            direction  = "LONG" if target_roi > 0 else "SHORT"

            # Profit 게이트
            p_profit = profit_gate_probability(df_seq, VOL_MU, VOL_SD, profit_model, profit_device) if gate_enabled else None
            gate_pass = True if not gate_enabled else (p_profit is not None and p_profit >= P_PROFIT_CUTOFF)

            # 진입 문턱 및 기대 순이익
            entry_threshold = max(abs(thresh_roi), fee_threshold)
            notional         = capital * ENTRY_PORTION * LEVERAGE
            expected_net_px  = max(0.0, abs(target_roi) - fee_threshold)
            expected_net_usd = notional * expected_net_px

            if (abs(target_roi) >= entry_threshold) and gate_pass and expected_net_usd >= MIN_PROFIT_USD:
                entry_price = now_price
                is_holding = True
                hold_state = {"best_px":0.0, "tp_hit":False, "be_armed":False}
        else:
            # 보유 중: OHLC 경로로 엑싯
            bar_ohlc = [float(new_bar["open"]), float(new_bar["high"]), float(new_bar["low"]), float(new_bar["close"])]
            exited, acct_roi, reason, realized_px_roi, hold_state = step_exit_with_ohlc(
                entry_price, direction, bar_ohlc, hold_state
            )
            if exited:
                capital_before = capital
                capital = capital * (1.0 + ENTRY_PORTION * acct_roi)
                trades += 1
                if acct_roi > 0: wins += 1

                trade_logs.append({
                    "timestamp": ts, "direction": direction, "reason": reason,
                    "realized_pxROI": realized_px_roi, "acctROI": acct_roi,
                    "capital_before": capital_before, "capital_after": capital
                })

                is_holding  = False
                direction   = None
                entry_price = None
                hold_state  = {"best_px":0.0, "tp_hit":False, "be_armed":False}

        equity_curve.append((ts, capital))

    # 종료 시 보유 중이면 마지막 종가로 정산
    if is_holding:
        last_close = float(df_hist.iloc[-1]["close"])
        px_roi = (last_close - entry_price) / entry_price if direction == "LONG" else (entry_price - last_close) / entry_price
        acct_roi = (px_roi * LEVERAGE) - (FEE_ROUNDTRIP + SLIP_ROUNDTRIP)
        capital_before = capital
        capital = capital * (1.0 + ENTRY_PORTION * acct_roi)
        trades += 1
        if acct_roi > 0: wins += 1
        ts_last = pd.to_datetime(df_hist.iloc[-1]["close_time"], unit="ms", utc=True).tz_convert(tz)
        trade_logs.append({
            "timestamp": ts_last, "direction": direction, "reason": "CLOSE_AT_END",
            "realized_pxROI": px_roi, "acctROI": acct_roi,
            "capital_before": capital_before, "capital_after": capital
        })

    # 저장 & 요약
    pd.DataFrame(equity_curve, columns=["timestamp","equity"]).to_csv(EQUITY_CSV, index=False)
    pd.DataFrame(trade_logs).to_csv(TRADES_CSV, index=False)

    total_ret = (capital / INITIAL_CAPITAL - 1.0) if INITIAL_CAPITAL > 0 else 0.0
    win_rate  = (wins / trades) if trades > 0 else 0.0

    print("\n================= SUMMARY (CSV, TODAY) =================")
    print(f"Symbol           : {symbol}")
    print(f"Period (KST)     : {start_kst:%Y-%m-%d 00:00} ~ {now_kst:%Y-%m-%d %H:%M}")
    print(f"Trades / Wins    : {trades} / {wins}  (Win% {pct(win_rate)})")
    print(f"Equity           : ${INITIAL_CAPITAL:.2f} → ${capital:.2f}  (Ret {pct(total_ret, signed=True)})")
    print(f"[SAVED] Equity → {EQUITY_CSV}")
    print(f"[SAVED] Trades → {TRADES_CSV}")
