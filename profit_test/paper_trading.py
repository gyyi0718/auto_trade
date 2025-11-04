# paper_trading.py
# -*- coding: utf-8 -*-
"""
TCN 모델 기반 페이퍼 트레이딩 시스템 (레버리지 포함)
- 실시간 신호 기반 자동 매매 시뮬레이션
- 레버리지 거래 시뮬레이션
- 포지션 관리 및 손익 계산
- 거래 내역 저장
"""
import os
import time
import json
import warnings
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
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
SYMBOLS = os.getenv("SYMBOLS",
                    "EVAAUSDT,ARIAUSDT,COAIUSDT,DBRUSDT,FUSDT,HANAUSDT,KDAUSDT,KGENUSDT,LIGHTUSDT,RECALLUSDT").split(
    ",")
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "2"))
TCN_CKPT = os.getenv("TCN_CKPT", "D:/ygy_work/coin/multimodel/models_5min/5min_2class_best.ckpt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.8"))
USE_TESTNET = os.getenv("USE_TESTNET", "0") == "1"

# 페이퍼 트레이딩 설정
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000"))  # 초기 자본 (USDT)
POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", "0.1"))  # 포지션 크기 (10%)
LEVERAGE = int(os.getenv("LEVERAGE", "20"))  # 레버리지 배율
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))  # 최대 동시 포지션
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))  # 손절 (2%)
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.03"))  # 익절 (3%)
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "30"))  # 최대 보유 시간
LIQUIDATION_BUFFER = float(os.getenv("LIQUIDATION_BUFFER", "0.8"))  # 청산 버퍼 (80%)
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "trades.json")


# ===== 데이터 클래스 =====
@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    direction: str  # "Long" or "Short"
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    leverage: int
    margin: float  # 실제 사용한 증거금
    liquidation_price: float  # 청산가

    def get_pnl(self, current_price: float) -> float:
        """손익 계산 (레버리지 적용)"""
        if self.direction == "Long":
            return (current_price - self.entry_price) * self.quantity
        else:  # Short
            return (self.entry_price - current_price) * self.quantity

    def get_pnl_pct(self, current_price: float) -> float:
        """손익률 계산 (증거금 기준)"""
        pnl = self.get_pnl(current_price)
        return (pnl / self.margin) * 100

    def get_roe(self, current_price: float) -> float:
        """ROE 계산 (레버리지 반영)"""
        if self.direction == "Long":
            price_change_pct = (current_price / self.entry_price - 1) * 100
        else:
            price_change_pct = (1 - current_price / self.entry_price) * 100
        return price_change_pct * self.leverage

    def get_liquidation_distance(self, current_price: float) -> float:
        """청산가까지 거리 (%)"""
        if self.direction == "Long":
            return (current_price - self.liquidation_price) / current_price * 100
        else:
            return (self.liquidation_price - current_price) / current_price * 100

    def should_close(self, current_price: float, current_time: datetime) -> tuple[bool, str]:
        """청산 여부 판단"""
        # 강제 청산 (레버리지 고려)
        if self.direction == "Long" and current_price <= self.liquidation_price:
            return True, "Liquidation"
        if self.direction == "Short" and current_price >= self.liquidation_price:
            return True, "Liquidation"

        # 손절
        if self.direction == "Long" and current_price <= self.stop_loss:
            return True, "Stop Loss"
        if self.direction == "Short" and current_price >= self.stop_loss:
            return True, "Stop Loss"

        # 익절
        if self.direction == "Long" and current_price >= self.take_profit:
            return True, "Take Profit"
        if self.direction == "Short" and current_price <= self.take_profit:
            return True, "Take Profit"

        # 시간 초과
        hold_minutes = (current_time - self.entry_time).total_seconds() / 60
        if hold_minutes >= MAX_HOLD_MINUTES:
            return True, "Time Limit"

        return False, ""


@dataclass
class Trade:
    """거래 기록"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int
    margin: float
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float
    roe: float
    exit_reason: str


class Account:
    """계좌 관리"""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.total_pnl = 0.0

    def get_available_balance(self) -> float:
        """사용 가능한 잔고 (증거금 차감)"""
        used_margin = sum(p.margin for p in self.positions.values())
        return self.balance - used_margin

    def get_total_value(self, prices: Dict[str, float]) -> float:
        """총 자산 (잔고 + 평가손익)"""
        unrealized_pnl = sum(
            p.get_pnl(prices.get(p.symbol, p.entry_price))
            for p in self.positions.values()
        )
        return self.balance + unrealized_pnl

    def can_open_position(self, symbol: str) -> bool:
        """포지션 진입 가능 여부"""
        if symbol in self.positions:
            return False
        if len(self.positions) >= MAX_POSITIONS:
            return False
        margin_needed = self.initial_capital * POSITION_SIZE_PCT
        if self.get_available_balance() < margin_needed:
            return False
        return True

    def open_position(self, symbol: str, direction: str, price: float):
        """포지션 진입 (레버리지 적용)"""
        # 증거금 (실제 사용할 자금)
        margin = self.initial_capital * POSITION_SIZE_PCT

        # 포지션 크기 (레버리지 적용)
        position_value = margin * LEVERAGE
        quantity = position_value / price

        # 손절/익절 가격 계산
        if direction == "Long":
            stop_loss = price * (1 - STOP_LOSS_PCT)
            take_profit = price * (1 + TAKE_PROFIT_PCT)
            # 청산가 계산 (Long): 진입가 * (1 - 1/레버리지 * 청산버퍼)
            liquidation_price = price * (1 - (1 / LEVERAGE) * LIQUIDATION_BUFFER)
        else:  # Short
            stop_loss = price * (1 + STOP_LOSS_PCT)
            take_profit = price * (1 - TAKE_PROFIT_PCT)
            # 청산가 계산 (Short): 진입가 * (1 + 1/레버리지 * 청산버퍼)
            liquidation_price = price * (1 + (1 / LEVERAGE) * LIQUIDATION_BUFFER)

        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=LEVERAGE,
            margin=margin,
            liquidation_price=liquidation_price
        )

        self.positions[symbol] = position
        print(f"\n{'=' * 90}")
        print(f"🔔 포지션 진입: {symbol}")
        print(f"   방향: {direction}")
        print(f"   레버리지: {LEVERAGE}x")
        print(f"   진입가: ${price:,.4f}")
        print(f"   수량: {quantity:.4f}")
        print(f"   증거금: ${margin:,.2f}")
        print(f"   포지션 크기: ${position_value:,.2f}")
        print(f"   손절가: ${stop_loss:,.4f} (-{STOP_LOSS_PCT * 100:.1f}%)")
        print(f"   익절가: ${take_profit:,.4f} (+{TAKE_PROFIT_PCT * 100:.1f}%)")
        print(f"   청산가: ${liquidation_price:,.4f} (거리: {position.get_liquidation_distance(price):.1f}%)")
        print(f"   사용 가능 잔고: ${self.get_available_balance():,.2f}")
        print(f"{'=' * 90}\n")

    def close_position(self, symbol: str, price: float, reason: str):
        """포지션 청산"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        pnl = position.get_pnl(price)
        pnl_pct = position.get_pnl_pct(price)
        roe = position.get_roe(price)

        # 강제 청산시 손실 제한
        if reason == "Liquidation":
            pnl = -position.margin  # 증거금 전액 손실
            pnl_pct = -100
            roe = -100

        # 거래 기록
        trade = Trade(
            symbol=symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=position.quantity,
            leverage=position.leverage,
            margin=position.margin,
            entry_time=position.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            exit_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pnl=pnl,
            pnl_pct=pnl_pct,
            roe=roe,
            exit_reason=reason
        )

        self.trades.append(trade)
        self.balance += pnl
        self.total_pnl += pnl

        # 이모지 선택
        if reason == "Liquidation":
            emoji = "💀"
        elif pnl > 0:
            emoji = "🟢"
        else:
            emoji = "🔴"

        print(f"\n{'=' * 90}")
        print(f"{emoji} 포지션 청산: {symbol}")
        print(f"   이유: {reason}")
        print(f"   레버리지: {position.leverage}x")
        print(f"   진입가: ${position.entry_price:,.4f}")
        print(f"   청산가: ${price:,.4f}")
        print(f"   증거금: ${position.margin:,.2f}")
        print(f"   손익: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        print(f"   ROE: {roe:+.2f}%")
        print(f"   보유 시간: {(datetime.now() - position.entry_time).total_seconds() / 60:.1f}분")
        print(f"   누적 손익: ${self.total_pnl:+,.2f} ({(self.total_pnl / self.initial_capital) * 100:+.2f}%)")
        print(f"   현재 잔고: ${self.balance:,.2f}")
        print(f"{'=' * 90}\n")

        del self.positions[symbol]

    def get_stats(self) -> dict:
        """통계 계산"""
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "max_pnl": 0.0,
                "min_pnl": 0.0,
            }

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        liquidations = [t for t in self.trades if t.exit_reason == "Liquidation"]

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "liquidations": len(liquidations),
            "win_rate": len(wins) / len(self.trades) * 100 if self.trades else 0,
            "total_pnl": self.total_pnl,
            "total_return_pct": (self.total_pnl / self.initial_capital) * 100,
            "avg_pnl": self.total_pnl / len(self.trades),
            "avg_roe": np.mean([t.roe for t in self.trades]),
            "max_pnl": max(t.pnl for t in self.trades),
            "min_pnl": min(t.pnl for t in self.trades),
            "max_roe": max(t.roe for t in self.trades),
            "min_roe": min(t.roe for t in self.trades),
            "avg_win": np.mean([t.pnl for t in wins]) if wins else 0.0,
            "avg_loss": np.mean([t.pnl for t in losses]) if losses else 0.0,
        }


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


# ===== TCN 모델 정의 (live_trading.py와 동일) =====
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


# ===== 모델 로드 (live_trading.py와 동일) =====
print(f"[INIT] 모델 로드 중: {TCN_CKPT}")
try:
    checkpoint = torch.load(TCN_CKPT, map_location="cpu", weights_only=False)
    FEAT_COLS = checkpoint['feat_cols']
    META = checkpoint.get('meta', {})
    SEQ_LEN = META.get('seq_len', 60)  # paper_trading 기본값 유지
    SCALER_MU = np.array(checkpoint.get('scaler_mu'), dtype=np.float32)
    SCALER_SD = np.array(checkpoint.get('scaler_sd'), dtype=np.float32)

    # 체크포인트에서 실제 모델 구조 자동 감지
    model_dict = checkpoint['model']

    # 1. TCN 레이어 수 감지
    max_layer = 0
    for key in model_dict.keys():
        if key.startswith('tcn.') and '.' in key[4:]:
            try:
                layer_num = int(key.split('.')[1])
                max_layer = max(max_layer, layer_num)
            except:
                pass
    levels = max_layer + 1

    # 2. Hidden channels 감지
    if 'tcn.0.c1.weight_v' in model_dict:
        hidden = model_dict['tcn.0.c1.weight_v'].shape[0]
    else:
        hidden = META.get('hidden', 32)

    # 3. Kernel size 감지
    if 'tcn.0.c1.weight_v' in model_dict:
        k = model_dict['tcn.0.c1.weight_v'].shape[2]
    else:
        k = META.get('k', 3)

    # 4. Dropout
    drop = META.get('dropout', 0.2)

    # 5. Head 타입 감지 (Single-task vs Multi-task)
    has_cls_head = 'head_cls.weight' in model_dict
    has_ttt_head = 'head_ttt.weight' in model_dict
    has_single_head = 'head.weight' in model_dict

    print(f"   📊 감지된 모델 구조:")
    print(f"      - Hidden channels: {hidden}")
    print(f"      - TCN levels: {levels}")
    print(f"      - Kernel size: {k}")
    print(f"      - Dropout: {drop}")
    print(f"      - Head type: {'Multi-task' if (has_cls_head or has_ttt_head) else 'Single-task'}")

    # 적절한 모델 클래스 선택
    if has_single_head:
        # Single-task 모델 (2-class)
        print(f"   ⚠️  경고: Single-task 모델이 감지되었습니다.")
        print(f"   이 모델은 방향만 예측하고 TP는 예측하지 않습니다.")


        # Single-task 모델 정의
        class TCN_SingleTask(nn.Module):
            def __init__(self, in_f, hidden, levels, k, drop):
                super().__init__()
                L = []
                ch = in_f
                for i in range(levels):
                    L.append(Block(ch, hidden, k, 2 ** i, drop))
                    ch = hidden
                self.tcn = nn.Sequential(*L)
                self.head = nn.Linear(hidden, 2)  # Single head for 2-class

            def forward(self, X):
                X = X.transpose(1, 2)
                H = self.tcn(X)[:, :, -1]
                return self.head(H)  # ✅ 단일 출력


        MODEL = TCN_SingleTask(in_f=len(FEAT_COLS), hidden=hidden, levels=levels, k=k, drop=drop).eval()
    else:
        # Multi-task 모델
        MODEL = TCN_MT(in_f=len(FEAT_COLS), hidden=hidden, levels=levels, k=k, drop=drop).eval()

    # 모델 로드 (strict=True로 정확한 매칭)
    MODEL.load_state_dict(checkpoint['model'], strict=True)
    '''
    print(f"   ✓ 모델 로드 완료")
    print(f"   ✓ SEQ_LEN: {SEQ_LEN}")
    print(f"   ✓ FEAT_COLS: {len(FEAT_COLS)}")
    print(f"\n[SCALER INFO]")
    print(f"  SCALER_MU type: {type(SCALER_MU)}")
    print(f"  SCALER_MU shape: {SCALER_MU.shape if hasattr(SCALER_MU, 'shape') else 'NO SHAPE'}")
    print(f"  SCALER_SD type: {type(SCALER_SD)}")
    print(f"  SCALER_SD shape: {SCALER_SD.shape if hasattr(SCALER_SD, 'shape') else 'NO SHAPE'}")

    if hasattr(SCALER_MU, '__len__'):
        print(f"  SCALER_MU[:5]: {SCALER_MU[:5]}")
        print(f"  SCALER_SD[:5]: {SCALER_SD[:5]}")
        print(f"  SCALER_MU mean: {SCALER_MU.mean():.6f}")
        print(f"  SCALER_SD mean: {SCALER_SD.mean():.6f}")
    else:
        print(f"  SCALER_MU: {SCALER_MU}")
        print(f"  SCALER_SD: {SCALER_SD}")
        '''
except Exception as e:
    print(f"   ✗ 모델 로드 실패: {e}")
    import traceback

    traceback.print_exc()
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

    rows = lst[::-1]
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
    """
    실시간 트레이딩용 Feature Engineering
    train_tcn_5minutes.py의 make_features()와 동일
    단, 단일 심볼용으로 간소화 (groupby 제거)
    """
    g = df.copy().sort_values("timestamp").reset_index(drop=True)

    # timestamp를 date로 변경 (train과 일치)
    if 'date' not in g.columns:
        g['date'] = pd.to_datetime(g['timestamp'], unit='ms', utc=True)

    # ===== 기본 수익률 =====
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g["logc"].diff().fillna(0.0)

    # ===== 변동성 =====
    for w in (8, 20, 40, 120):
        g[f"rv{w}"] = g["ret1"].rolling(w, min_periods=max(2, w // 3)).std()

    # ===== 모멘텀 =====
    for w in (8, 20, 40, 120):
        g[f"mom{w}"] = g["close"] / g["close"].ewm(span=w, adjust=False).mean() - 1.0

    # ===== 볼륨 분석 =====
    # 1. 거래량 Z-score
    for w in (20, 40, 120):
        mu = g["volume"].rolling(w, min_periods=max(2, w // 3)).mean()
        sd = g["volume"].rolling(w, min_periods=max(2, w // 3)).std().replace(0, np.nan)
        g[f"vz{w}"] = (g["volume"] - mu) / sd.fillna(1.0)

    # 2. 거래량 급등
    g["vol_spike"] = (g["volume"] / g["volume"].shift(1) - 1.0).fillna(0.0)

    # 3. 거래량 가속도
    g["vol_accel"] = g["vol_spike"].diff().fillna(0.0)

    # 4. 거래량-가격 상관관계
    for w in [8, 20]:
        g[f"vol_price_corr{w}"] = g["ret1"].rolling(w).corr(g["vol_spike"]).fillna(0.0)

    # ===== ATR =====
    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14, min_periods=5).mean()

    # ===== 가격 패턴 =====
    # 1. High-Low 스프레드
    g["hl_spread"] = (g["high"] - g["low"]) / g["close"]

    # 2. Close 위치 (High-Low 범위 내)
    g["close_position"] = (g["close"] - g["low"]) / (g["high"] - g["low"] + 1e-10)

    # 3. 캔들 바디 크기
    g["body_size"] = (g["close"] - g["open"]).abs() / g["open"]

    # 4. 위/아래 꼬리 길이
    g["upper_shadow"] = (g["high"] - g[["open", "close"]].max(axis=1)) / g["close"]
    g["lower_shadow"] = (g[["open", "close"]].min(axis=1) - g["low"]) / g["close"]

    # 5. 갭 (이전 종가 대비 현재 시가)
    g["gap"] = ((g["open"] - g["close"].shift(1)) / g["close"].shift(1)).fillna(0.0)

    # ===== 추세 분석 =====
    # 1. 수익률 (다양한 기간)
    for w in [2, 4, 8, 12, 24]:
        g[f"ret{w}"] = g["logc"].diff(w).fillna(0.0)

    # 2. 모멘텀 가속도
    g["mom_accel"] = g["mom8"].diff().fillna(0.0)

    # 3. 추세 강도 (수익률 절댓값)
    for w in [4, 8, 12]:
        g[f"trend_strength{w}"] = g[f"ret{w}"].abs()

    # 4. 가격이 이동평균 위/아래
    for w in [20, 40]:
        ma = g["close"].rolling(w, min_periods=w // 2).mean()
        g[f"above_ma{w}"] = ((g["close"] > ma).astype(float) - 0.5) * 2  # -1 ~ 1

    # ===== RSI =====
    for w in [14, 28]:
        delta = g["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(w, min_periods=w // 2).mean()
        avg_loss = loss.rolling(w, min_periods=w // 2).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        g[f"rsi{w}"] = 100 - (100 / (1 + rs))
        g[f"rsi{w}"] = (g[f"rsi{w}"] - 50) / 50  # -1 ~ 1 정규화

    # ===== MACD =====
    ema12 = g["close"].ewm(span=12, adjust=False).mean()
    ema26 = g["close"].ewm(span=26, adjust=False).mean()
    g["macd"] = (ema12 - ema26) / g["close"]
    g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
    g["macd_hist"] = g["macd"] - g["macd_signal"]

    # ===== 볼린저 밴드 =====
    for w in [20]:
        ma = g["close"].rolling(w, min_periods=w // 2).mean()
        std = g["close"].rolling(w, min_periods=w // 2).std()
        g[f"bb_upper{w}"] = (ma + 2 * std - g["close"]) / g["close"]
        g[f"bb_lower{w}"] = (g["close"] - (ma - 2 * std)) / g["close"]
        g[f"bb_width{w}"] = (4 * std) / ma

    # ===== 시간 패턴 =====
    hod = g["date"].dt.hour
    g["hod_sin"] = np.sin(2 * np.pi * hod / 24.0)
    g["hod_cos"] = np.cos(2 * np.pi * hod / 24.0)

    # 주요 시간대 더미 변수
    for h in [0, 6, 12, 18]:
        g[f"hour_{h}"] = (hod == h).astype(float)

    # 요일 효과
    dow = g["date"].dt.dayofweek
    g["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    g["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # 주말 여부
    g["is_weekend"] = (dow >= 5).astype(float)

    # ===== 최근 극값 =====
    # 최근 N시간 내 최고가/최저가 대비 현재 위치
    for w in [8, 24, 48]:
        recent_high = g["high"].rolling(w, min_periods=max(2, w // 3)).max()
        recent_low = g["low"].rolling(w, min_periods=max(2, w // 3)).min()
        g[f"price_vs_high{w}"] = (g["close"] - recent_high) / recent_high
        g[f"price_vs_low{w}"] = (g["close"] - recent_low) / recent_low

    # ===== Feature 이상치 클리핑 =====
    feats = [
        # 수익률
        "ret1", "ret2", "ret4", "ret8", "ret12", "ret24",
        # 변동성
        "rv8", "rv20", "rv40", "rv120",
        # 모멘텀
        "mom8", "mom20", "mom40", "mom120", "mom_accel",
        # 거래량
        "vz20", "vz40", "vz120",
        "vol_spike", "vol_accel",
        "vol_price_corr8", "vol_price_corr20",
        # ATR
        "atr14",
        # 가격 패턴
        "hl_spread", "close_position", "body_size",
        "upper_shadow", "lower_shadow", "gap",
        # 추세
        "trend_strength4", "trend_strength8", "trend_strength12",
        "above_ma20", "above_ma40",
        # RSI
        "rsi14", "rsi28",
        # MACD
        "macd", "macd_signal", "macd_hist",
        # 볼린저 밴드
        "bb_upper20", "bb_lower20", "bb_width20",
        # 시간
        "hod_sin", "hod_cos",
        "hour_0", "hour_6", "hour_12", "hour_18",
        "dow_sin", "dow_cos", "is_weekend",
        # 극값 대비 위치
        "price_vs_high8", "price_vs_low8",
        "price_vs_high24", "price_vs_low24",
        "price_vs_high48", "price_vs_low48"
    ]

    for feat in feats:
        if feat not in ['hod_sin', 'hod_cos', 'dow_sin', 'dow_cos']:  # 사인/코사인 제외
            q01 = g[feat].quantile(0.01)
            q99 = g[feat].quantile(0.99)
            g[feat] = g[feat].clip(q01, q99)

    return g


# ===== 예측 함수 =====
@torch.no_grad()
def predict(symbol: str) -> dict:
    """실시간 예측"""
    df = get_recent_data(symbol, SEQ_LEN + 100)
    if df is None or len(df) < SEQ_LEN:
        return {"symbol": symbol, "error": "데이터 부족", "direction": None, "confidence": 0.0}

    df_feat = make_features(df)
    if len(df_feat) < SEQ_LEN:
        return {"symbol": symbol, "error": "피처 생성 실패", "direction": None, "confidence": 0.0}

    try:
        X = df_feat[FEAT_COLS].tail(SEQ_LEN).to_numpy(np.float32)
        '''
        print(f"\n[LIVE INPUT DEBUG] {symbol}:")
        print(f"  X shape: {X.shape}")
        print(f"  BEFORE norm mean: {X.mean():.6f}, std: {X.std():.6f}")  # ← 추가!
        print(f"  BEFORE norm min/max: {X.min():.6f} / {X.max():.6f}")  # ← 추가!
        print(f"  Last row sample: {X[-1, :3]}")
        '''
    except KeyError as e:
        missing_cols = set(FEAT_COLS) - set(df_feat.columns)
        return {"symbol": symbol, "error": f"피처 누락: {missing_cols}", "direction": None, "confidence": 0.0}

    # ✅ 여기에 디버그 추가!
    '''
    print(f"\n[PAPER INPUT DEBUG] {symbol}:")
    print(f"  X shape: {X.shape}")
    print(f"  X mean: {X.mean():.6f}, std: {X.std():.6f}")
    print(f"  X min/max: {X.min():.6f} / {X.max():.6f}")
    print(f"  Last row sample: {X[-1, :3]}")
    '''
    # 정규화
    X = (X - SCALER_MU) / SCALER_SD
    X = np.clip(X, -10.0, 10.0)

    # ✅ 정규화 후에도!
    #print(f"  AFTER norm mean: {X.mean():.6f}, std: {X.std():.6f}")  # ← 수정!
    #print(f"  AFTER norm min/max: {X.min():.6f} / {X.max():.6f}")  # ← 수정!

    X_tensor = torch.from_numpy(X[None, ...])

    # 예측
    model_output = MODEL(X_tensor)
    if isinstance(model_output, tuple):
        logits = model_output[0]
    else:
        logits = model_output

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = int(logits.argmax(dim=1).item())

    # ✅ 예측 결과도
    #print(f"  pred_class: {pred_class}")
    #print(f"  logits: {logits.detach().numpy()[0]}")
    #print(f"  probs: {probs}\n")

    # 방향 매핑
    if len(probs) == 2:
        direction_map = {0: "Short", 1: "Long"}
    else:
        direction_map = {0: "Short", 1: "Flat", 2: "Long"}

    direction = direction_map[pred_class]
    confidence = float(probs.max())

    ticker = API.get_ticker(symbol)
    current_price = float(ticker.get("lastPrice", 0))

    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "current_price": current_price,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
# ===== 화면 출력 =====
def print_dashboard(account: Account, prices: Dict[str, float]):
    """대시보드 출력"""
    os.system('clear' if os.name == 'posix' else 'cls')

    print("\n" + "=" * 110)
    print(f"{'🎯 페이퍼 트레이딩 시스템 (레버리지 {LEVERAGE}x)':^110}")
    print("=" * 110)

    # 계좌 정보
    total_value = account.get_total_value(prices)
    unrealized_pnl = total_value - account.balance
    total_return = (total_value / account.initial_capital - 1) * 100

    print(f"\n💰 계좌 현황")
    print(f"   초기 자본:     ${account.initial_capital:>12,.2f}")
    print(f"   현재 잔고:     ${account.balance:>12,.2f}")
    print(f"   사용 가능:     ${account.get_available_balance():>12,.2f}")
    print(f"   평가 손익:     ${unrealized_pnl:>+12,.2f}")
    print(f"   총 자산:       ${total_value:>12,.2f}  ({total_return:>+6.2f}%)")
    print(
        f"   실현 손익:     ${account.total_pnl:>+12,.2f}  ({(account.total_pnl / account.initial_capital) * 100:>+6.2f}%)")

    # 포지션
    if account.positions:
        print(f"\n📍 보유 포지션 ({len(account.positions)}/{MAX_POSITIONS})")
        print(f"{'심볼':^12} | {'방향':^8} | {'진입가':^12} | {'현재가':^12} | {'손익(ROE)':^22} | {'청산가':^12} | {'보유':^8}")
        print("-" * 110)

        for symbol, pos in account.positions.items():
            current_price = prices.get(symbol, pos.entry_price)
            pnl = pos.get_pnl(current_price)
            roe = pos.get_roe(current_price)
            hold_min = (datetime.now() - pos.entry_time).total_seconds() / 60
            liq_dist = pos.get_liquidation_distance(current_price)

            emoji = "📈" if pos.direction == "Long" else "📉"
            pnl_emoji = "🟢" if pnl > 0 else "🔴"
            liq_warning = "⚠️" if liq_dist < 3 else ""

            print(f"{symbol:^12} | {emoji} {pos.direction:^6} | ${pos.entry_price:>10,.4f} | "
                  f"${current_price:>10,.4f} | {pnl_emoji} ${pnl:>+8,.2f} ({roe:>+6.1f}%) | "
                  f"${pos.liquidation_price:>10,.4f}{liq_warning} | {hold_min:>6.1f}분")
    else:
        print(f"\n📍 보유 포지션: 없음")

    # 통계
    stats = account.get_stats()
    if stats["total_trades"] > 0:
        print(f"\n📊 거래 통계")
        print(f"   총 거래:       {stats['total_trades']:>3}회")
        print(f"   승률:          {stats['win_rate']:>6.1f}% ({stats['wins']}승 {stats['losses']}패)")
        if stats['liquidations'] > 0:
            print(f"   강제 청산:     {stats['liquidations']:>3}회 💀")
        print(f"   평균 손익:     ${stats['avg_pnl']:>+12,.2f}")
        print(f"   평균 ROE:      {stats['avg_roe']:>+6.1f}%")
        print(f"   최대 수익:     ${stats['max_pnl']:>12,.2f}  (ROE: {stats['max_roe']:>+6.1f}%)")
        print(f"   최대 손실:     ${stats['min_pnl']:>12,.2f}  (ROE: {stats['min_roe']:>+6.1f}%)")
        if stats['wins'] > 0 and stats['losses'] > 0:
            rr = abs(stats['avg_win'] / stats['avg_loss'])
            print(f"   Risk/Reward:   {rr:>6.2f}")

    print("\n" + "=" * 110)


def save_trades(account: Account):
    """거래 내역 저장"""
    if not account.trades:
        return

    data = [asdict(t) for t in account.trades]
    with open(TRADE_LOG_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n💾 거래 내역 저장: {TRADE_LOG_FILE}")


# ===== 메인 루프 =====
def main():
    print("\n" + "=" * 110)
    print(f"{'🎯 페이퍼 트레이딩 시작':^110}")
    print(f"{'초기 자본: $' + f'{INITIAL_CAPITAL:,.2f}':^110}")
    print(f"{'레버리지: ' + f'{LEVERAGE}x (포지션 크기: {POSITION_SIZE_PCT * 100:.0f}%)':^110}")
    print(f"{'신뢰도 임계값: ' + f'{CONF_THRESHOLD:.0%}':^110}")
    print("=" * 110)

    account = Account(INITIAL_CAPITAL)

    try:
        loop_count = 0
        while True:
            loop_count += 1
            current_time = datetime.now()

            # 현재 가격 가져오기
            prices = {}
            for symbol in SYMBOLS:
                symbol = symbol.strip()
                ticker = API.get_ticker(symbol)
                prices[symbol] = float(ticker.get("lastPrice", 0))

            # 포지션 관리
            for symbol in list(account.positions.keys()):
                position = account.positions[symbol]
                current_price = prices.get(symbol, position.entry_price)

                # 청산 조건 확인
                should_close, reason = position.should_close(current_price, current_time)
                if should_close:
                    account.close_position(symbol, current_price, reason)
                else:
                    # 반대 신호로 청산
                    result = predict(symbol)
                    if result.get("confidence", 0) >= CONF_THRESHOLD:
                        signal_dir = result["direction"]
                        if (position.direction == "Long" and signal_dir == "Short") or \
                                (position.direction == "Short" and signal_dir == "Long"):
                            account.close_position(symbol, current_price, "Reverse Signal")

            # 대시보드 출력
            print_dashboard(account, prices)

            # 신호 스캔 테이블
            print(f"\n🔍 신호 스캔")
            print(f"{'심볼':^12} | {'가격':^12} | {'방향':^10} | {'신뢰도':^8} | {'신호':^20}")
            print("-" * 80)

            # 신호 스캔 및 진입
            for symbol in SYMBOLS:
                symbol = symbol.strip()

                result = predict(symbol)

                if "error" in result:
                    print(f"{symbol:^12} | {'N/A':^12} | {'오류':^10} | {'N/A':^8} | ❌ 데이터 부족")
                    continue

                direction = result["direction"]
                confidence = result["confidence"]
                price = result["current_price"]

                # 방향 이모지
                dir_icon = {"Long": "📈", "Short": "📉", "Flat": "➖"}.get(direction, "❓")

                # 신호 판단
                if confidence < CONF_THRESHOLD:
                    signal = f"⚠️  신호 약함 ({confidence:.1%})"
                elif direction == "Long":
                    signal = f"🟢 매수 신호 ({confidence:.1%})"
                elif direction == "Short":
                    signal = f"🔴 매도 신호 ({confidence:.1%})"
                else:
                    signal = f"⚪ 관망 ({confidence:.1%})"

                print(
                    f"{symbol:^12} | ${price:>10,.4f} | {dir_icon} {direction:^8} | {confidence:>6.1%} | {signal:^20}")

                # 진입 조건
                if account.can_open_position(symbol) and confidence >= CONF_THRESHOLD and direction in ["Long",
                                                                                                        "Short"]:
                    account.open_position(symbol, direction, price)

            print(f"\n[스캔 #{loop_count}] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"다음 스캔까지 {INTERVAL_SEC}초... (Ctrl+C로 종료)")

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n프로그램 종료 중...")
        print_dashboard(account, prices)
        save_trades(account)

        # 최종 통계
        stats = account.get_stats()
        if stats["total_trades"] > 0:
            print("\n" + "=" * 110)
            print(f"{'📊 최종 결과':^110}")
            print("=" * 110)
            final_balance = account.balance
            final_return = (final_balance / account.initial_capital - 1) * 100
            print(f"   최종 잔고:     ${final_balance:,.2f}")
            print(f"   총 수익률:     {final_return:+.2f}%")
            print(f"   총 거래:       {stats['total_trades']}회")
            print(f"   승률:          {stats['win_rate']:.1f}%")
            print(f"   평균 ROE:      {stats['avg_roe']:+.1f}%")
            if stats['liquidations'] > 0:
                print(f"   강제 청산:     {stats['liquidations']}회 💀")
            print("=" * 110)

        print("\n✅ 프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()