# paper_trading.py
# -*- coding: utf-8 -*-
"""
TCN 모델 기반 페이퍼 트레이딩 시스템 (심볼별 모델 사용)
- 각 심볼마다 별도의 딥러닝 모델 로드
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
                    "BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,BNBUSDT").split(
    ",")
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "2"))

# ✅ 심볼별 모델 경로 설정 (딕셔너리 형태)
# 방법 1: 개별 지정
MODEL_PATHS = {
    "BTCUSDT": "D:/ygy_work/coin/multimodel/models_5min_btc/5min_2class_best.ckpt",
    "ETHUSDT": "D:/ygy_work/coin/multimodel/models_5min_eth/5min_2class_best.ckpt",
    "SOLUSDT": "D:/ygy_work/coin/multimodel/models_5min_sol/5min_2class_best.ckpt",
    "DOGEUSDT": "D:/ygy_work/coin/multimodel/models_5min_doge/5min_2class_best.ckpt",
    "BNBUSDT": "D:/ygy_work/coin/multimodel/models_5min_bnb/5min_2class_best.ckpt",
    "XRPUSDT": "D:/ygy_work/coin/multimodel/models_5min_xrp/5min_2class_best.ckpt",
    "SAPIENUSDT": "D:/ygy_work/coin/multimodel/models_5min_sapien_v2/model_v2_best.pt",
    "FLMUSDT": "D:/ygy_work/coin/multimodel/models_5min_flm_v2/model_v2_best.pt",
    "SAPIENUSDT": "D:/ygy_work/coin/multimodel/models_5min_sapien/5min_2class_best.ckpt",
    "TRUMPUSDT": "D:/ygy_work/coin/multimodel/models_5min_trump/5min_2class_best.ckpt",
    "JELLYJELLYUSDT": "D:/ygy_work/coin/multimodel/models_minutes_jellyjelly/5min_2class_best.ckpt",
    "ARCUSDT": "D:/ygy_work/coin/multimodel/models_5min_arc/5min_2class_best.ckpt",
    "DASHUSDT": "D:/ygy_work/coin/multimodel/models_5min_arc/5min_2class_best.ckpt",
    "MMTUSDT": "D:/ygy_work/coin/multimodel/models_5min_mmt/5min_2class_best.ckpt",
    "AIAUSDT": "D:/ygy_work/coin/multimodel/models_5min_aia/5min_2class_best.ckpt",
    "GIGGLEUSDT": "D:/ygy_work/coin/multimodel/models_5min_giggle/5min_2class_best.ckpt",
    "XNOUSDT": "D:/ygy_work/coin/multimodel/models_5min_xno/5min_2class_best.ckpt",
    "KITEUSDT": "D:/ygy_work/coin/multimodel/models_5min_kite/5min_2class_best.ckpt",
    "SOONUSDT": "D:/ygy_work/coin/multimodel/models_5min_soon/5min_2class_best.ckpt",
    "ICPUSDT": "D:/ygy_work/coin/multimodel/models_5min_icp/5min_2class_best.ckpt",
    "ARUSDT": "D:/ygy_work/coin/multimodel/models_5min_ar/5min_2class_best.ckpt",
    "HUSDT": "D:/ygy_work/coin/multimodel/models_improved/model_h1.pt",
    "FLUXUSDT": "D:/ygy_work/coin/multimodel/models_5min_flux/5min_2class_best.ckpt",
    "WLFIUSDT": "D:/ygy_work/coin/multimodel/models_5min_wlfi/5min_2class_best.ckpt",
    "MELANIAUSDT": "D:/ygy_work/coin/multimodel/models_5min_melania/5min_2class_best.ckpt",
    "UNIUSDT": "D:/ygy_work/coin/multimodel/models_5min_uni/5min_2class_best.ckpt",
    "LSKUSDT": "D:/ygy_work/coin/multimodel/models_5min_lsk/5min_2class_best.ckpt",
    "XNOUSDT": "D:/ygy_work/coin/multimodel/models_5min_xno/5min_2class_best.ckpt",
    "PARTIUSDT": "D:/ygy_work/coin/multimodel/models_5min_parti/5min_2class_best.ckpt",
    "EVAAUSDT": "D:/ygy_work/coin/multimodel/models_5min_evaa/5min_2class_best.ckpt",
    "RESOLVUSDT": "D:/ygy_work/coin/multimodel/models_5min_resolv/5min_2class_best.ckpt"

}

# 방법 2: 패턴 기반 자동 생성 (선택사항)
# MODEL_DIR = "D:/ygy_work/coin/multimodel/models_5min/"
# MODEL_PATHS = {symbol: f"{MODEL_DIR}{symbol}_best.ckpt" for symbol in SYMBOLS}

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.6"))
USE_TESTNET = os.getenv("USE_TESTNET", "0") == "1"

# 페이퍼 트레이딩 설정
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000"))  # 초기 자본 (USDT)
POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", "0.2"))  # 포지션 크기 (10%)
LEVERAGE = int(os.getenv("LEVERAGE", "25"))  # 레버리지 배율
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))  # 최대 동시 포지션
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))  # 손절 (2%)
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.03"))  # 익절 (3%)
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "30"))  # 최대 보유 시간
LIQUIDATION_BUFFER = float(os.getenv("LIQUIDATION_BUFFER", "0.8"))  # 청산 버퍼 (80%)
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "trades.json")

# 🔧 가격 및 ROE 검증 설정 (ROE 2000% 버그 방지)
MAX_PRICE_CHANGE_PCT = 50.0  # 최대 가격 변동률 50%
MAX_ROE_LIMIT = 100.0  # ROE 상한선 ±100%
MIN_PRICE_RATIO = 0.5  # 최소 가격 비율 (진입가 대비 50%)
MAX_PRICE_RATIO = 2.0  # 최대 가격 비율 (진입가 대비 200%)
KLINES_CACHE = {}
CACHE_EXPIRY_SEC = 10  # 10초마다 갱신
MODEL_CONFIGS = {}

# ===== 가격 검증 함수 =====
def validate_price(current_price: float, entry_price: float, symbol: str) -> tuple[bool, str]:
    """가격 유효성 검증 - ROE 2000% 버그 방지"""

    # 1. 가격이 0이거나 음수인 경우
    if current_price <= 0:
        return False, f"Invalid price: {current_price}"

    # 2. 진입가 대비 너무 큰 변동
    price_ratio = current_price / entry_price
    if price_ratio < MIN_PRICE_RATIO or price_ratio > MAX_PRICE_RATIO:
        return False, f"Abnormal price change: {price_ratio:.2%} (entry: ${entry_price:.4f}, current: ${current_price:.4f})"

    # 3. 변동률 체크
    change_pct = abs(price_ratio - 1) * 100
    if change_pct > MAX_PRICE_CHANGE_PCT:
        return False, f"Price change too large: {change_pct:.1f}%"

    return True, ""


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
        """ROE 계산 (레버리지 반영) - 버그 수정 버전"""
        # 🔧 가격 검증 추가
        is_valid, error_msg = validate_price(current_price, self.entry_price, self.symbol)
        if not is_valid:
            print(f"⚠️  가격 검증 실패 ({self.symbol}): {error_msg}")
            return 0.0  # 비정상 데이터 무시

        if self.direction == "Long":
            price_change_pct = (current_price / self.entry_price - 1) * 100
        else:
            price_change_pct = (1 - current_price / self.entry_price) * 100

        roe = price_change_pct * self.leverage

        # 🔧 ROE 상한/하한 제한
        if roe > MAX_ROE_LIMIT:
            print(
                f"⚠️  ROE 상한 초과 ({self.symbol}): {roe:.1f}% -> {MAX_ROE_LIMIT:.1f}% (진입: ${self.entry_price:.4f}, 현재: ${current_price:.4f})")
            return MAX_ROE_LIMIT
        elif roe < -MAX_ROE_LIMIT:
            print(f"⚠️  ROE 하한 초과 ({self.symbol}): {roe:.1f}% -> {-MAX_ROE_LIMIT:.1f}%")
            return -MAX_ROE_LIMIT

        return roe

    def get_liquidation_distance(self, current_price: float) -> float:
        """청산가까지 거리 (%)"""
        # 🔧 가격 검증 추가
        if current_price <= 0:
            print(f"⚠️  청산거리 계산 실패 ({self.symbol}): Invalid price: {current_price}")
            return 0.0

        if self.direction == "Long":
            return (current_price - self.liquidation_price) / current_price * 100
        else:
            return (self.liquidation_price - current_price) / current_price * 100

    def should_close(self, current_price: float, current_time: datetime) -> tuple[bool, str]:
        """청산 여부 판단"""
        # 🔧 가격 검증
        is_valid, error_msg = validate_price(current_price, self.entry_price, self.symbol)
        if not is_valid:
            print(f"⚠️  비정상 가격 감지 ({self.symbol}): {error_msg}")
            return False, ""  # 비정상 가격일 경우 청산하지 않음

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
        self.invalid_price_count = 0  # 🔧 비정상 가격 카운터

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
        margin_needed = self.balance * POSITION_SIZE_PCT
        if self.get_available_balance() < margin_needed:
            return False
        return True

    def open_position(self, symbol: str, direction: str, price: float):
        """포지션 진입 (레버리지 적용)"""
        # 🔧 진입 가격 검증
        if price <= 0:
            print(f"⚠️  잘못된 진입가 ({symbol}): ${price:.4f} - 진입 취소")
            return

        # 증거금 (실제 사용할 자금)
        margin = self.balance * POSITION_SIZE_PCT

        # 포지션 크기 (레버리지 적용)
        position_value = margin * LEVERAGE
        quantity = position_value / price
        if quantity>200000:
            quantity =200000
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
        print(f"   청산가: ${liquidation_price:,.4f}")
        print(f"{'=' * 90}")

    def close_position(self, symbol: str, exit_price: float, reason: str):
        """포지션 청산"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # 🔧 청산 가격 검증
        is_valid, error_msg = validate_price(exit_price, position.entry_price, symbol)
        if not is_valid:
            print(f"⚠️  비정상 청산가 감지 ({symbol}): {error_msg}")
            print(f"    청산 취소 - 다음 스캔에서 재시도")
            self.invalid_price_count += 1
            return

        pnl = position.get_pnl(exit_price)
        pnl_pct = position.get_pnl_pct(exit_price)
        roe = position.get_roe(exit_price)
        roe = position.get_roe(exit_price)

        # 잔고 업데이트
        self.balance += pnl
        self.total_pnl += pnl

        # 거래 기록
        trade = Trade(
            symbol=symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
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

        # 포지션 제거
        del self.positions[symbol]

        # 출력
        emoji = "💀" if reason == "Liquidation" else ("🔴" if pnl < 0 else "🟢")
        print(f"\n{'=' * 90}")
        print(f"{emoji} 포지션 청산: {symbol}")
        print(f"   사유: {reason}")
        print(f"   진입가: ${position.entry_price:,.4f}")
        print(f"   청산가: ${exit_price:,.4f}")
        print(f"   손익: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        print(f"   ROE: {roe:+.2f}%")
        print(f"   현재 잔고: ${self.balance:,.2f}")
        print(f"{'=' * 90}")

    def get_stats(self) -> Dict:
        """통계 계산"""
        if not self.trades:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "avg_roe": 0.0,
                "max_pnl": 0.0,
                "min_pnl": 0.0,
                "max_roe": 0.0,
                "min_roe": 0.0,
                "liquidations": 0,
                "avg_win": 0.0,
                "avg_loss": 0.0
            }

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        liquidations = sum(1 for t in self.trades if t.exit_reason == "Liquidation")

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(self.trades) * 100) if self.trades else 0,
            "avg_pnl": sum(t.pnl for t in self.trades) / len(self.trades),
            "avg_roe": sum(t.roe for t in self.trades) / len(self.trades),
            "max_pnl": max(t.pnl for t in self.trades),
            "min_pnl": min(t.pnl for t in self.trades),
            "max_roe": max(t.roe for t in self.trades),
            "min_roe": min(t.roe for t in self.trades),
            "liquidations": liquidations,
            "avg_win": sum(t.pnl for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t.pnl for t in losses) / len(losses) if losses else 0,
            "invalid_prices": self.invalid_price_count  # 🔧 비정상 가격 카운트
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


class TCN_Single(nn.Module):
    """Single-task TCN (2-class)"""

    def __init__(self, in_f, hidden=128, levels=6, k=3, drop=0.1):
        super().__init__()
        L = []
        ch = in_f
        for i in range(levels):
            L.append(Block(ch, hidden, k, 2 ** i, drop))
            ch = hidden
        self.tcn = nn.Sequential(*L)
        self.head = nn.Linear(hidden, 2)  # 2-class

    def forward(self, X):
        X = X.transpose(1, 2)
        H = self.tcn(X)[:, :, -1]
        return self.head(H)


# ===== 심볼별 모델 로드 =====
print("\n" + "=" * 110)
print(f"{'🤖 심볼별 모델 로드 중...':^110}")
print("=" * 110)

MODELS = {}  # ✅ 심볼별 모델 저장
MODEL_CONFIGS = {}  # 각 모델의 설정 정보 저장

for symbol in SYMBOLS:
    symbol = symbol.strip()

    if symbol not in MODEL_PATHS:
        print(f"⚠️  {symbol}: 모델 경로가 지정되지 않음. 건너뜀.")
        continue

    model_path = MODEL_PATHS[symbol]

    if not os.path.exists(model_path):
        print(f"❌ {symbol}: 모델 파일 없음 ({model_path})")
        continue

    try:
        print(f"\n📦 {symbol} 모델 로드 중...")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        feat_cols = checkpoint['feat_cols']
        meta = checkpoint.get('meta', {})
        seq_len = meta.get('seq_len', 60)
        scaler_mu = checkpoint.get('scaler_mu')
        scaler_sd = checkpoint.get('scaler_sd')

        # 모델 구조 자동 감지
        model_dict = checkpoint['model']

        # TCN 레이어 수 감지
        max_layer = 0
        for key in model_dict.keys():
            if key.startswith('tcn.') and '.' in key[4:]:
                try:
                    layer_num = int(key.split('.')[1])
                    max_layer = max(max_layer, layer_num)
                except:
                    pass
        levels = max_layer + 1

        # Hidden channels 감지
        if 'tcn.0.c1.weight_v' in model_dict:
            hidden = model_dict['tcn.0.c1.weight_v'].shape[0]
        else:
            hidden = meta.get('hidden', 32)

        # Kernel size 감지
        if 'tcn.0.c1.weight_v' in model_dict:
            k = model_dict['tcn.0.c1.weight_v'].shape[2]
        else:
            k = meta.get('k', 3)

        drop = meta.get('dropout', 0.2)

        # Head 타입 감지
        has_cls_head = 'head_cls.weight' in model_dict
        has_single_head = 'head.weight' in model_dict

        # 적절한 모델 선택
        if has_single_head:
            model = TCN_Single(len(feat_cols), hidden, levels, k, drop)
            num_classes = 2
        else:
            model = TCN_MT(len(feat_cols), hidden, levels, k, drop)
            num_classes = 3

        model.load_state_dict(model_dict)
        model.eval()

        # 저장
        MODELS[symbol] = model
        MODEL_CONFIGS[symbol] = {
            'feat_cols': feat_cols,
            'seq_len': seq_len,
            'scaler_mu': scaler_mu,
            'scaler_sd': scaler_sd,
            'num_classes': num_classes,
            'hidden': hidden,
            'levels': levels,
            'k': k,
            'drop': drop
        }

        print(f"   ✅ 로드 완료")
        print(f"      - Hidden: {hidden}, Levels: {levels}, Kernel: {k}")
        print(f"      - Classes: {num_classes}, Features: {len(feat_cols)}, Seq: {seq_len}")

    except Exception as e:
        print(f"❌ {symbol}: 로드 실패 - {e}")

print(f"\n{'=' * 110}")
print(f"{'✅ 총 ' + str(len(MODELS)) + '개 심볼 모델 로드 완료':^110}")
print(f"{'=' * 110}\n")

if not MODELS:
    print("❌ 로드된 모델이 없습니다. 프로그램을 종료합니다.")
    exit(1)


# ===== API 클래스 =====
class BybitAPI:
    """Bybit API 클라이언트 (공통)"""

    def __init__(self):
        self.base_url = "https://api-testnet.bybit.com" if USE_TESTNET else "https://api.bybit.com"

    def get_ticker(self, symbol: str) -> dict:
        """현재 가격 조회"""
        try:
            url = f"{self.base_url}/v5/market/tickers"
            params = {"category": "linear", "symbol": symbol}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                return data
            return {}
        except Exception as e:
            print(f"⚠️  Ticker 조회 실패 ({symbol}): {e}")
            return {}

    def get_klines(self, symbol: str, interval: str = "5", limit: int = 200) -> pd.DataFrame:
        """과거 데이터 조회"""
        try:
            url = f"{self.base_url}/v5/market/kline"
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            ret_code = data.get("retCode", -1)
            if ret_code != 0:
                print(f"⚠️  API Error for {symbol}: retCode={ret_code}, msg={data.get('retMsg', 'Unknown')}")
                return pd.DataFrame()

            if data.get("retCode") == 0:
                klines = data["result"]["list"]
                if not klines:
                    print(f"⚠️  {symbol}: API returned empty klines list")
                    return pd.DataFrame()

                df = pd.DataFrame(klines,
                                  columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
                df = df.astype({
                    "open": float, "high": float, "low": float, "close": float, "volume": float
                })
                df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit='ms')
                df = df.sort_values("timestamp").reset_index(drop=True)
                return df
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"⚠️  네트워크 오류 ({symbol}): {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"⚠️  Klines 조회 실패 ({symbol}): {type(e).__name__}: {e}")
            return pd.DataFrame()


API = BybitAPI()


# ===== 특성 생성 (학습 코드와 완전 일치) =====
def generate_features(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """
    학습 시 사용한 make_features와 완전히 동일한 로직
    train_tcn_5minutes.py의 make_features 함수 기반
    """
    g = df.copy()

    # ===== 기본 수익률 =====
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g["logc"].diff().fillna(0.0)

    # ===== 변동성 =====
    for w in (8, 20, 40, 120):
        g[f"rv{w}"] = g["ret1"].rolling(w, min_periods=max(2, w // 3)).std().fillna(0.0)

    # ===== 모멘텀 =====
    for w in (8, 20, 40, 120):
        ema = g["close"].ewm(span=w, adjust=False).mean()
        g[f"mom{w}"] = (g["close"] / ema - 1.0).fillna(0.0)

    # ===== 볼륨 분석 =====
    # 1. 거래량 Z-score
    for w in (20, 40, 120):
        mu = g["volume"].rolling(w, min_periods=max(2, w // 3)).mean()
        sd = g["volume"].rolling(w, min_periods=max(2, w // 3)).std().replace(0, 1.0)
        g[f"vz{w}"] = ((g["volume"] - mu) / sd).fillna(0.0)

    # 2. 거래량 급등
    g["vol_spike"] = (g["volume"] / g["volume"].shift(1) - 1.0).fillna(0.0)

    # 3. 거래량 가속도
    g["vol_accel"] = g["vol_spike"].diff().fillna(0.0)

    # 4. 거래량-가격 상관관계
    for w in [8, 20]:
        vol_ret_corr = g["ret1"].rolling(w).corr(g["vol_spike"])
        g[f"vol_price_corr{w}"] = vol_ret_corr.fillna(0.0)

    # ===== ATR =====
    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14, min_periods=5).mean().fillna(0.0)

    # ===== 가격 패턴 =====
    g["hl_spread"] = ((g["high"] - g["low"]) / g["close"]).fillna(0.0)
    g["close_position"] = ((g["close"] - g["low"]) / (g["high"] - g["low"] + 1e-10)).fillna(0.5)
    g["body_size"] = ((g["close"] - g["open"]).abs() / g["open"]).fillna(0.0)
    g["upper_shadow"] = ((g["high"] - g[["open", "close"]].max(axis=1)) / g["close"]).fillna(0.0)
    g["lower_shadow"] = ((g[["open", "close"]].min(axis=1) - g["low"]) / g["close"]).fillna(0.0)

    # 갭
    prev_close_val = g["close"].shift(1)
    g["gap"] = ((g["open"] - prev_close_val) / prev_close_val).fillna(0.0)

    # ===== 추세 분석 =====
    # 1. 수익률 (다양한 기간)
    for w in [2, 4, 8, 12, 24]:
        g[f"ret{w}"] = g["logc"].diff(w).fillna(0.0)

    # 2. 모멘텀 가속도
    g["mom_accel"] = g["mom8"].diff().fillna(0.0)

    # 3. 추세 강도
    for w in [4, 8, 12]:
        g[f"trend_strength{w}"] = g[f"ret{w}"].abs()

    # 4. 가격이 이동평균 위/아래
    for w in [20, 40]:
        ma = g["close"].rolling(w, min_periods=w // 2).mean()
        g[f"above_ma{w}"] = ((g["close"] > ma).astype(float) - 0.5) * 2
        g[f"above_ma{w}"] = g[f"above_ma{w}"].fillna(0.0)

    # ===== RSI (정규화됨!) =====
    for w in [14, 28]:
        delta = g["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(w, min_periods=w // 2).mean()
        avg_loss = loss.rolling(w, min_periods=w // 2).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        # 중요: -1 ~ 1 범위로 정규화!
        g[f"rsi{w}"] = (100 - (100 / (1 + rs)) - 50) / 50
        g[f"rsi{w}"] = g[f"rsi{w}"].fillna(0.0)

    # ===== MACD (가격으로 정규화됨!) =====
    ema12 = g["close"].ewm(span=12, adjust=False).mean()
    ema26 = g["close"].ewm(span=26, adjust=False).mean()
    # 중요: 가격으로 나눔!
    g["macd"] = ((ema12 - ema26) / g["close"]).fillna(0.0)
    g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean().fillna(0.0)
    g["macd_hist"] = (g["macd"] - g["macd_signal"]).fillna(0.0)

    # ===== 볼린저 밴드 (상대값!) =====
    for w in [20]:
        ma = g["close"].rolling(w, min_periods=w // 2).mean()
        std = g["close"].rolling(w, min_periods=w // 2).std()
        # 중요: 상대값으로 변환!
        g[f"bb_upper{w}"] = ((ma + 2 * std - g["close"]) / g["close"]).fillna(0.0)
        g[f"bb_lower{w}"] = ((g["close"] - (ma - 2 * std)) / g["close"]).fillna(0.0)
        g[f"bb_width{w}"] = ((4 * std) / ma).fillna(0.0)

    # ===== 시간 패턴 =====
    if 'timestamp' in g.columns:
        hod = pd.to_datetime(g["timestamp"]).dt.hour
        g["hod_sin"] = np.sin(2 * np.pi * hod / 24.0)
        g["hod_cos"] = np.cos(2 * np.pi * hod / 24.0)

        for h in [0, 6, 12, 18]:
            g[f"hour_{h}"] = (hod == h).astype(float)

        dow = pd.to_datetime(g["timestamp"]).dt.dayofweek
        g["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        g["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        g["is_weekend"] = (dow >= 5).astype(float)

    # ===== 최근 극값 =====
    for w in [8, 24, 48]:
        recent_high = g["high"].rolling(w, min_periods=max(2, w // 3)).max()
        recent_low = g["low"].rolling(w, min_periods=max(2, w // 3)).min()
        g[f"price_vs_high{w}"] = ((g["close"] - recent_high) / recent_high).fillna(0.0)
        g[f"price_vs_low{w}"] = ((g["close"] - recent_low) / recent_low).fillna(0.0)

    # ===== Feature 이상치 클리핑 (학습 시와 동일) =====
    for feat in feat_cols:
        if feat in g.columns and feat not in ['hod_sin', 'hod_cos', 'dow_sin', 'dow_cos']:
            q01 = g[feat].quantile(0.01)
            q99 = g[feat].quantile(0.99)
            g[feat] = g[feat].clip(q01, q99)

    # ===== 특성 리스트 순서대로 반환 =====
    return g[feat_cols]


# ===== 추론 함수 (심볼별 모델 사용) =====
def predict(symbol: str, current_price: float = None, debug: bool = False) -> dict:
    """✅ 심볼별 모델로 추론"""
    symbol = symbol.strip()

    # 해당 심볼의 모델이 있는지 확인
    if symbol not in MODELS:
        return {
            "error": f"No model for {symbol}",
            "symbol": symbol
        }

    model = MODELS[symbol]
    config = MODEL_CONFIGS[symbol]

    # ✅ 최적화: Klines 캐시 사용
    current_time = time.time()
    cache_key = symbol

    # 캐시가 유효하면 사용
    if cache_key in KLINES_CACHE:
        cached_data, cached_time = KLINES_CACHE[cache_key]
        if current_time - cached_time < CACHE_EXPIRY_SEC:
            df = cached_data
            if debug:
                print(f"[DEBUG {symbol}] 캐시 사용 (age: {current_time - cached_time:.1f}초)")
        else:
            # 캐시 만료 - 새로 가져오기
            max_window = 120
            required_length = config['seq_len'] + max_window + 40
            limit = max(200, min(required_length, 1000))
            df = API.get_klines(symbol, interval="5", limit=limit)
            KLINES_CACHE[cache_key] = (df, current_time)
    else:
        # 캐시 없음 - 새로 가져오기
        max_window = 120
        required_length = config['seq_len'] + max_window + 40
        limit = max(200, min(required_length, 1000))
        df = API.get_klines(symbol, interval="5", limit=limit)
        KLINES_CACHE[cache_key] = (df, current_time)

    if debug:
        print(f"\n[DEBUG {symbol}] 데이터 조회 결과:")
        print(f"  - 요청한 limit: {limit}")
        print(f"  - df.empty: {df.empty}")
        print(f"  - len(df): {len(df) if not df.empty else 0}")
        print(f"  - 필요한 길이: {config['seq_len'] + 20}")
        print(f"  - seq_len: {config['seq_len']}")
        if not df.empty:
            print(f"  - df.columns: {df.columns.tolist()}")
            print(f"  - df.head(2):\n{df.head(2)}")

    if df.empty:
        return {
            "error": "API returned empty data",
            "symbol": symbol
        }

    if len(df) < config['seq_len'] + 20:
        return {
            "error": f"Insufficient data (got {len(df)}, need {config['seq_len'] + 20})",
            "symbol": symbol
        }

    # 특성 생성
    df_feat = generate_features(df, config['feat_cols'])
    df_feat = df_feat.dropna()

    if debug:
        print(f"\n[DEBUG {symbol}] 특성 생성 결과:")
        print(f"  - 요구되는 특성 수: {len(config['feat_cols'])}")
        print(f"  - 실제 생성된 특성 수: {len(df_feat.columns)}")
        print(f"  - 생성된 특성: {df_feat.columns.tolist()}")
        print(f"  - 요구되는 특성 (처음 10개): {config['feat_cols'][:10]}")

        # 누락된 특성 확인
        missing_cols = [c for c in config['feat_cols'] if c not in df_feat.columns]
        if missing_cols:
            print(f"  - ⚠️ 누락된 특성 ({len(missing_cols)}개): {missing_cols[:20]}")

        # ✅ 특성 값 범위 확인
        print(f"\n[DEBUG {symbol}] 특성 값 통계 (최근 데이터):")
        print(f"  - NaN 개수: {df_feat.isna().sum().sum()}")
        print(f"  - Inf 개수: {np.isinf(df_feat.values).sum()}")
        print(f"  - 최솟값: {df_feat.min().min():.4f}")
        print(f"  - 최댓값: {df_feat.max().max():.4f}")
        print(f"  - 평균: {df_feat.mean().mean():.4f}")

        # 요구되는 특성 중 처음 5개의 최근 값
        print(f"\n  요구 특성 최근 값 (마지막 행):")
        for col in config['feat_cols'][:5]:
            if col in df_feat.columns:
                val = df_feat[col].iloc[-1]
                print(f"    {col}: {val:.6f}")

    if len(df_feat) < config['seq_len']:
        return {
            "error": f"Insufficient features (got {len(df_feat)}, need {config['seq_len']})",
            "symbol": symbol
        }

    if len(df_feat.columns) != len(config['feat_cols']):
        return {
            "error": f"Feature mismatch (got {len(df_feat.columns)}, need {len(config['feat_cols'])})",
            "symbol": symbol
        }

    # 시퀀스 준비
    X = df_feat.iloc[-config['seq_len']:].values

    if debug:
        print(f"\n[DEBUG {symbol}] 정규화 전 데이터:")
        print(f"  - X shape: {X.shape}")
        print(f"  - X 통계: min={X.min():.4f}, max={X.max():.4f}, mean={X.mean():.4f}")
        print(f"  - scaler_mu: min={config['scaler_mu'].min():.4f}, max={config['scaler_mu'].max():.4f}")
        print(f"  - scaler_sd: min={config['scaler_sd'].min():.4f}, max={config['scaler_sd'].max():.4f}")

    # 안전한 정규화 (scaler_sd가 너무 작으면 보정)
    safe_scaler_sd = np.maximum(config['scaler_sd'], 0.01)  # 최소 0.01
    X = (X - config['scaler_mu']) / (safe_scaler_sd + 1e-8)

    # 극단값 클리핑 (정규화 후 -10 ~ 10 범위로 제한)
    X = np.clip(X, -10, 10)

    if debug:
        print(f"\n[DEBUG {symbol}] 정규화 후 데이터:")
        print(f"  - X 통계: min={X.min():.4f}, max={X.max():.4f}, mean={X.mean():.4f}")
        print(f"  - NaN 개수: {np.isnan(X).sum()}")
        print(f"  - Inf 개수: {np.isinf(X).sum()}")
        print(f"  - 클리핑 적용: -10 ~ 10")

    X_tensor = torch.FloatTensor(X).unsqueeze(0)

    # 추론
    with torch.no_grad():
        if config['num_classes'] == 2:
            # Single-task (2-class)
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
        else:
            # Multi-task (3-class)
            logits_cls, _ = model(X_tensor)
            probs = torch.softmax(logits_cls, dim=1).squeeze().numpy()

    if debug:
        print(f"\n[DEBUG {symbol}] 모델 출력:")
        if config['num_classes'] == 2:
            print(f"  - logits: [{logits.squeeze()[0]:.4f}, {logits.squeeze()[1]:.4f}]")
        else:
            print(
                f"  - logits: [{logits_cls.squeeze()[0]:.4f}, {logits_cls.squeeze()[1]:.4f}, {logits_cls.squeeze()[2]:.4f}]")
        print(f"  - probs: {probs}")
        print(f"  - confidence: {probs.max():.6f}")

    pred_class = int(probs.argmax())

    # 방향 매핑
    if config['num_classes'] == 2:
        direction_map = {0: "Short", 1: "Long"}
    else:
        direction_map = {0: "Short", 1: "Flat", 2: "Long"}

    direction = direction_map[pred_class]
    confidence = float(probs.max())

    # ✅ 최적화: current_price가 전달되면 API 호출 스킵
    if current_price is None:
        ticker = API.get_ticker(symbol)
        if ticker.get("retCode") == 0 and ticker.get("result", {}).get("list"):
            current_price = float(ticker["result"]["list"][0]["lastPrice"])
        else:
            # Ticker 실패 시 df의 마지막 close 가격 사용
            current_price = float(df['close'].iloc[-1])
            if debug:
                print(f"  - ⚠️ Ticker 조회 실패, df의 close 사용: ${current_price}")

    # 가격이 0이거나 너무 작으면 에러 반환
    if current_price <= 0 or current_price < 1e-10:
        return {
            "error": f"Invalid price: {current_price}",
            "symbol": symbol
        }

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
    print(f"{'🎯 페이퍼 트레이딩 시스템 (레버리지 ' + str(LEVERAGE) + 'x, ' + str(len(MODELS)) + '개 심볼 모델)':^110}")
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
        print(
            f"{'심볼':^12} | {'방향':^8} | {'진입가':^12} | {'현재가':^12} | {'손익(ROE)':^22} | {'청산가':^12} | {'보유':^8}")
        print("-" * 110)

        for symbol, pos in account.positions.items():
            current_price = prices.get(symbol, pos.entry_price)

            # 🔧 가격 유효성 검증
            if current_price <= 0:
                print(f"⚠️  {symbol}: 가격 데이터 없음, 진입가 사용")
                current_price = pos.entry_price

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
def test_api_connection():
    """API 연결 테스트"""
    print("\n🔍 API 연결 테스트 중...")

    # Public API 테스트
    print("   테스트 심볼: BTCUSDT")
    ticker = API.get_ticker("BTCUSDT")
    if ticker and ticker.get("retCode") == 0 and ticker.get("result", {}).get("list"):
        last_price = float(ticker["result"]["list"][0]["lastPrice"])
        print(f"   ✓ Ticker 조회 성공: ${last_price:,.2f}")
    else:
        print(f"   ✗ Ticker 조회 실패")
        print(f"   응답: {ticker}")
        return False

    df = API.get_klines("BTCUSDT", interval="5", limit=10)
    if not df.empty:
        print(f"   ✓ Klines 조회 성공: {len(df)}개 캔들")
        print(f"   최신 가격: ${df['close'].iloc[-1]:,.2f}")
    else:
        print(f"   ✗ Klines 조회 실패")
        return False

    print("\n✅ API 연결 정상\n")
    return True


def main():
    print("\n" + "=" * 110)
    print(f"{'🎯 페이퍼 트레이딩 시작':^110}")
    print(f"{'초기 자본: $' + f'{INITIAL_CAPITAL:,.2f}':^110}")
    print(f"{'레버리지: ' + f'{LEVERAGE}x (포지션 크기: {POSITION_SIZE_PCT * 100:.0f}%)':^110}")
    print(f"{'신뢰도 임계값: ' + f'{CONF_THRESHOLD:.0%}':^110}")
    print(f"{'사용 모델: ' + str(len(MODELS)) + '개 심볼별 모델':^110}")
    print("=" * 110)

    # API 연결 테스트
    if not test_api_connection():
        print("\n❌ API 연결에 실패했습니다. 프로그램을 종료합니다.")
        print("\n확인 사항:")
        print("   1. 인터넷 연결 확인")
        print("   2. Bybit API 서버 상태 확인")
        print("   3. 심볼이 올바른지 확인 (예: BTCUSDT)")
        return

    account = Account(INITIAL_CAPITAL)

    try:
        loop_count = 0
        while True:
            loop_count += 1
            current_time = datetime.now()

            # 현재 가격 가져오기 (모델이 있는 심볼만)
            prices = {}
            for symbol in MODELS.keys():
                ticker = API.get_ticker(symbol)
                if ticker.get("retCode") == 0 and ticker.get("result", {}).get("list"):
                    try:
                        price = float(ticker["result"]["list"][0]["lastPrice"])
                        if price > 0:  # 🔧 유효한 가격만 저장
                            prices[symbol] = price
                        else:
                            print(f"⚠️  {symbol}: 가격이 0입니다. 이전 가격 유지")
                            # 이전 가격이 있으면 유지, 없으면 skip
                            if symbol in prices:
                                pass  # 이전 가격 유지
                            # 새로운 심볼이면 가격 없이 skip
                    except (ValueError, KeyError, IndexError) as e:
                        print(f"⚠️  {symbol}: 가격 파싱 실패 - {e}")
                else:
                    # 조회 실패 시 이전 가격 유지 (없으면 skip)
                    if symbol not in prices:
                        print(f"⚠️  {symbol}: API 조회 실패, 가격 없음")

            # 포지션 관리
            for symbol in list(account.positions.keys()):
                position = account.positions[symbol]
                current_price = prices.get(symbol, position.entry_price)

                # 🔧 가격 유효성 검증
                if current_price <= 0:
                    print(f"⚠️  {symbol}: 가격 데이터 없음, 청산 판단 건너뜀")
                    continue

                # ✅ 1순위: 반대 신호 체크 (즉시 청산)
                result = predict(symbol, debug=False)
                if "error" not in result and result.get("confidence", 0) >= CONF_THRESHOLD:
                    signal_dir = result["direction"]
                    if (position.direction == "Long" and signal_dir == "Short") or                             (position.direction == "Short" and signal_dir == "Long"):
                        account.close_position(symbol, current_price, "Reverse Signal")
                        continue  # 청산 완료, 다음 포지션으로

                # 2순위: 일반 청산 조건 확인 (손절, 익절, 시간 초과 등)
                should_close, reason = position.should_close(current_price, current_time)
                if should_close:
                    account.close_position(symbol, current_price, reason)

            # 대시보드 출력
            print_dashboard(account, prices)

            # 신호 스캔 테이블
            print(f"\n🔍 신호 스캔 ({len(MODELS)}개 심볼)")
            print(f"{'심볼':^12} | {'가격':^12} | {'방향':^10} | {'신뢰도':^8} | {'신호':^20}")
            print("-" * 80)

            # 신호 스캔 및 진입 (모델이 있는 심볼만)
            debug_mode = (loop_count == 1)  # 첫 스캔만 디버그 모드
            for symbol in MODELS.keys():
                result = predict(symbol, debug=debug_mode)

                if "error" in result:
                    print(f"{symbol:^12} | {'N/A':^12} | {'오류':^10} | {'N/A':^8} | ❌ {result.get('error', '알 수 없음')}")
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