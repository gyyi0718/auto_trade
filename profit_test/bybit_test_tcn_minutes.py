# realtime_monitor.py
# -*- coding: utf-8 -*-
"""
TCN 모델을 사용한 실시간 신호 모니터링
- 10분 예측 모델 전용
- CatBoost 제거
- 실시간 신호 표시
"""
import os
import time
import warnings
from datetime import datetime
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import requests
import certifi

warnings.filterwarnings("ignore")
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# ===== CONFIG =====
SYMBOLS = os.getenv("SYMBOLS", "EVAAUSDT,1000PEPEUSDT,AVNTUSDT,BTRUSDT,COAIUSDT").split(",")
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "2"))  # 스캔 간격(초)
TCN_CKPT = os.getenv("TCN_CKPT", "../multimodel/models/tcn_10min_fixed_best.ckpt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.55"))  # 신뢰도 임계값
USE_TESTNET = os.getenv("USE_TESTNET", "0") == "1"


# ===== TCN 모델 정의 =====
class Chomp1d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):
        return x[:, :, :-self.c].contiguous() if self.c > 0 else x


def wconv(i, o, k, d):
    pad = (k - 1) * d
    return nn.utils.weight_norm(nn.Conv1d(i, o, k, padding=pad, dilation=d))


class Block(nn.Module):
    def __init__(self, i, o, k, d, drop):
        super().__init__()
        self.c1 = wconv(i, o, k, d)
        self.h1 = Chomp1d((k - 1) * d)
        self.r1 = nn.ReLU()
        self.dr1 = nn.Dropout(drop)
        self.c2 = wconv(o, o, k, d)
        self.h2 = Chomp1d((k - 1) * d)
        self.r2 = nn.ReLU()
        self.dr2 = nn.Dropout(drop)
        self.ds = nn.Conv1d(i, o, 1) if i != o else None
        self.r = nn.ReLU()

    def forward(self, x):
        y = self.dr1(self.r1(self.h1(self.c1(x))))
        y = self.dr2(self.r2(self.h2(self.c2(y))))
        res = x if self.ds is None else self.ds(x)
        return self.r(y + res)


class TCN_MT(nn.Module):
    def __init__(self, in_f, hidden=128, levels=6, k=3, drop=0.1):
        super().__init__()
        L = []
        ch = in_f
        for i in range(levels):
            L.append(Block(ch, hidden, k, 2 ** i, drop))
            ch = hidden
        self.tcn = nn.Sequential(*L)
        self.head_cls = nn.Linear(hidden, 3)
        self.head_ttt = nn.Linear(hidden, 1)

    def forward(self, X):
        X = X.transpose(1, 2)
        H = self.tcn(X)[:, :, -1]
        return self.head_cls(H), self.head_ttt(H)


# ===== 모델 로드 =====
print(f"[INIT] 모델 로드 중: {TCN_CKPT}")
try:
    checkpoint = torch.load(TCN_CKPT, map_location="cpu")
    FEAT_COLS = checkpoint['feat_cols']
    META = checkpoint['meta']
    SEQ_LEN = META['seq_len']
    SCALER_MU = checkpoint['scaler_mu']
    SCALER_SD = checkpoint['scaler_sd']

    MODEL = TCN_MT(in_f=len(FEAT_COLS), hidden=128, levels=6, k=3, drop=0.2).eval()
    MODEL.load_state_dict(checkpoint['model'])
    print(f"   ✓ 모델 로드 완료 (seq_len={SEQ_LEN}, features={len(FEAT_COLS)})")
except Exception as e:
    print(f"   ✗ 모델 로드 실패: {e}")
    exit(1)


# ===== BYBIT API =====
class BybitPublic:
    def __init__(self, testnet: bool = False):
        self.base = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self.session = requests.Session()
        self.session.verify = certifi.where()

    def get_kline(self, symbol: str, interval: str, limit: int):
        url = f"{self.base}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        try:
            r = self.session.get(url, params=params, timeout=10)
            data = r.json()
            return ((data.get("result") or {}).get("list") or [])
        except:
            return []

    def get_ticker(self, symbol: str):
        url = f"{self.base}/v5/market/tickers"
        params = {"category": "linear", "symbol": symbol}
        try:
            r = self.session.get(url, params=params, timeout=5)
            data = r.json()
            rows = ((data.get("result") or {}).get("list") or [])
            return rows[0] if rows else {}
        except:
            return {}


API = BybitPublic(testnet=USE_TESTNET)


# ===== 데이터 가져오기 =====
def get_recent_data(symbol: str, minutes: int = 300) -> Optional[pd.DataFrame]:
    """최근 N분 데이터 가져오기"""
    lst = API.get_kline(symbol, "1", minutes)
    if not lst:
        return None

    rows = lst[::-1]  # 시간순 정렬
    df = pd.DataFrame([{
        "timestamp": pd.to_datetime(int(z[0]), unit="ms", utc=True),
        "open": float(z[1]),
        "high": float(z[2]),
        "low": float(z[3]),
        "close": float(z[4]),
        "volume": float(z[5]),
    } for z in rows])

    return df


# ===== 피처 생성 =====
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 생성"""
    g = df.copy().sort_values("timestamp")

    # 수익률
    g["ret1"] = np.log(g["close"]).diff()

    # 변동성
    for w in (5, 15, 30, 60):
        g[f"rv{w}"] = g["ret1"].rolling(w, min_periods=max(2, w // 3)).std()
        g[f"mom{w}"] = g["close"] / g["close"].ewm(span=w, adjust=False).mean() - 1.0

    # 거래량 z-score
    for w in (20, 60):
        mu = g["volume"].rolling(w, min_periods=max(2, w // 3)).mean()
        sd = g["volume"].rolling(w, min_periods=max(2, w // 3)).std().replace(0, np.nan)
        g[f"vz{w}"] = (g["volume"] - mu) / sd.fillna(1.0)

    # ATR
    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14, min_periods=5).mean()

    return g


# ===== 예측 함수 =====
@torch.no_grad()
def predict(symbol: str) -> dict:
    """실시간 예측"""
    # 디버그 모드
    DEBUG = os.getenv("DEBUG", "0") == "1"

    # 데이터 가져오기
    df = get_recent_data(symbol, SEQ_LEN + 100)
    if df is None or len(df) < SEQ_LEN:
        return {
            "symbol": symbol,
            "error": "데이터 부족",
            "direction": None,
            "confidence": 0.0
        }

    if DEBUG:
        print(f"\n[DEBUG][{symbol}] 원본 데이터: {len(df)}개")
        print(f"  가격 범위: {df['close'].min():.2f} ~ {df['close'].max():.2f}")

    # 피처 생성
    df_feat = make_features(df)
    if len(df_feat) < SEQ_LEN:
        return {
            "symbol": symbol,
            "error": "피처 생성 실패",
            "direction": None,
            "confidence": 0.0
        }

    if DEBUG:
        print(f"  피처 생성 후: {len(df_feat)}개")
        print(f"  NaN 개수: {df_feat[FEAT_COLS].isna().sum().sum()}")
        print(f"  피처 통계:\n{df_feat[FEAT_COLS].tail(1).T}")

    # 최근 SEQ_LEN 데이터 추출
    X = df_feat[FEAT_COLS].tail(SEQ_LEN).to_numpy(np.float32)

    if DEBUG:
        print(f"  입력 shape: {X.shape}")
        print(f"  입력 통계: mean={X.mean():.4f}, std={X.std():.4f}")

    # 표준화
    X = (X - SCALER_MU) / SCALER_SD

    if DEBUG:
        print(f"  스케일링 후: mean={X.mean():.4f}, std={X.std():.4f}")

    # 예측
    X_tensor = torch.from_numpy(X[None, ...])
    logits, time_pred = MODEL(X_tensor)

    if DEBUG:
        print(f"  Logits: {logits.cpu().numpy()[0]}")

    # 확률
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    prob_short, prob_flat, prob_long = probs

    if DEBUG:
        print(f"  확률: Short={prob_short:.3f}, Flat={prob_flat:.3f}, Long={prob_long:.3f}")

    # 방향
    pred_class = int(logits.argmax(dim=1).item())
    direction_map = {0: "Short", 1: "Flat", 2: "Long"}
    direction = direction_map[pred_class]

    # 신뢰도
    confidence = float(probs.max())

    # 목표 시간
    time_to_target = float(time_pred.cpu().item())

    # 현재 가격
    ticker = API.get_ticker(symbol)
    current_price = float(ticker.get("lastPrice", 0))

    # 변동성
    volatility = df_feat["ret1"].tail(60).abs().median() * 10000  # bps
    if not np.isfinite(volatility):
        volatility = 0.0

    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "prob_long": float(prob_long),
        "prob_flat": float(prob_flat),
        "prob_short": float(prob_short),
        "time_to_target": time_to_target,
        "current_price": current_price,
        "volatility_bps": float(volatility),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ===== 신호 판단 =====
def get_signal(result: dict) -> str:
    """매매 신호 생성"""
    if "error" in result:
        return "❌ 오류"

    direction = result["direction"]
    confidence = result["confidence"]

    if confidence < CONF_THRESHOLD:
        return f"⚠️  신호 약함 ({confidence:.1%})"

    if direction == "Long":
        return f"🟢 매수 ({confidence:.1%})"
    elif direction == "Short":
        return f"🔴 매도 ({confidence:.1%})"
    else:
        return f"⚪ 관망 ({confidence:.1%})"


# ===== 화면 출력 =====
def print_header():
    print("\n" + "=" * 100)
    print(f"{'심볼':^10} | {'가격':^12} | {'방향':^8} | {'신뢰도':^8} | "
          f"{'예상시간':^10} | {'변동성':^10} | {'신호':^20}")
    print("=" * 100)


def print_result(result: dict):
    """결과 출력"""
    symbol = result["symbol"]
    price = result.get("current_price", 0)
    direction = result.get("direction", "-")
    confidence = result.get("confidence", 0)
    time_min = result.get("time_to_target", 0)
    vol = result.get("volatility_bps", 0)
    signal = get_signal(result)

    # 방향 아이콘
    dir_icon = {"Long": "📈", "Short": "📉", "Flat": "➖"}.get(direction, "❓")

    print(f"{symbol:^10} | ${price:>10,.2f} | {dir_icon} {direction:^6} | "
          f"{confidence:>6.1%} | {time_min:>8.1f}분 | {vol:>8.1f}bps | {signal:^20}")


# ===== 메인 루프 =====
def main():
    print("\n" + "=" * 100)
    print(f"{'TCN 실시간 모니터링':^100}")
    print(f"{'모델: ' + TCN_CKPT:^100}")
    print(f"{'신뢰도 임계값: ' + f'{CONF_THRESHOLD:.0%}':^100}")
    print("=" * 100)

    try:
        loop_count = 0
        while True:
            loop_count += 1
            print(f"\n[스캔 #{loop_count}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print_header()

            for symbol in SYMBOLS:
                try:
                    result = predict(symbol.strip())
                    print_result(result)
                except Exception as e:
                    print(f"{symbol:^10} | 예측 오류: {e}")

            print("\n" + "-" * 100)
            print(f"다음 스캔까지 {INTERVAL_SEC}초 대기... (Ctrl+C로 종료)")
            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n프로그램을 종료합니다.")


if __name__ == "__main__":
    main()