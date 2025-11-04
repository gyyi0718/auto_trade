# paper_trading_hourly_multi_model_vwap_fixed_v2.py
# -*- coding: utf-8 -*-
"""
TCN 모델 기반 페이퍼 트레이딩 시스템 (v2 - 동적 체크 주기)
- 포지션 없을 때: 60초 간격 (신호 스캔만)
- 포지션 있을 때: 2초 간격 (실시간 모니터링)
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
import torch.nn.functional as F
import requests
import certifi
import math

warnings.filterwarnings("ignore")
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# ===== CONFIG =====
SYMBOLS = os.getenv("SYMBOLS", "JELLYJELLYUSDT").split(",")

# ✅ 동적 체크 주기 설정
INTERVAL_NO_POSITION = int(os.getenv("INTERVAL_NO_POSITION", "60"))  # 포지션 없을 때: 60초
INTERVAL_WITH_POSITION = int(os.getenv("INTERVAL_WITH_POSITION", "2"))  # 포지션 있을 때: 2초

# ✅ 심볼별 모델 경로 설정
MODEL_PATHS = {
    "JELLYJELLYUSDT": "D:/ygy_work/coin/multimodel/models_hour_jellyjelly_vwap/hourly_realistic_best.ckpt"
}

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.55"))
USE_TESTNET = os.getenv("USE_TESTNET", "0") == "1"

# 페이퍼 트레이딩 설정
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000"))
POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", "0.15"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "1"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.03"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.05"))
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "1440"))  # 24 hours
LIQUIDATION_BUFFER = float(os.getenv("LIQUIDATION_BUFFER", "0.8"))
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "trades_hourly.json")

# 수수료 설정 (Bybit 기본값)
TAKER_FEE = float(os.getenv("TAKER_FEE", "0.00055"))
MAKER_FEE = float(os.getenv("MAKER_FEE", "0.0002"))
FUNDING_FEE_RATE = float(os.getenv("FUNDING_FEE_RATE", "0.0001"))


# ===== 데이터 클래스 =====
@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    direction: str
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    leverage: int
    margin: float
    liquidation_price: float
    model_confidence: float = 0.0

    def get_pnl(self, current_price: float) -> float:
        """손익 계산"""
        if self.direction == "Long":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def get_pnl_pct(self, current_price: float) -> float:
        """손익률 계산"""
        pnl = self.get_pnl(current_price)
        return (pnl / self.margin) * 100

    def get_roe(self, current_price: float) -> float:
        """ROE 계산"""
        if self.direction == "Long":
            price_change_pct = (current_price / self.entry_price - 1) * 100
        else:
            price_change_pct = (1 - current_price / self.entry_price) * 100
        return price_change_pct * self.leverage

    def get_liquidation_distance(self, current_price: float) -> float:
        """청산가까지 거리"""
        if self.direction == "Long":
            return (current_price - self.liquidation_price) / current_price * 100
        else:
            return (self.liquidation_price - current_price) / current_price * 100

    def should_close(self, current_price: float, current_time: datetime) -> tuple[bool, str]:
        """청산 여부 판단"""
        # 강제 청산
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
    model_confidence: float = 0.0
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    total_fee: float = 0.0
    net_pnl: float = 0.0


class Account:
    """계좌 관리"""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.total_pnl = 0.0

    def get_available_balance(self) -> float:
        """사용 가능한 잔고"""
        used_margin = sum(p.margin for p in self.positions.values())
        return self.balance - used_margin

    def get_total_value(self, prices: Dict[str, float]) -> float:
        """총 자산"""
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

    def open_position(self, symbol: str, direction: str, price: float,
                      model_confidence: float = 0.0):
        """포지션 진입"""
        margin = self.initial_capital * POSITION_SIZE_PCT
        position_value = margin * LEVERAGE
        quantity = position_value / price

        # 수수료 계산
        entry_fee = position_value * TAKER_FEE
        expected_exit_fee = position_value * TAKER_FEE
        total_expected_fee = entry_fee + expected_exit_fee

        if direction == "Long":
            stop_loss = price * (1 - STOP_LOSS_PCT)
            take_profit = price * (1 + TAKE_PROFIT_PCT)
            liquidation_price = price * (1 - (1 / LEVERAGE) * LIQUIDATION_BUFFER)
        else:
            stop_loss = price * (1 + STOP_LOSS_PCT)
            take_profit = price * (1 - TAKE_PROFIT_PCT)
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
            liquidation_price=liquidation_price,
            model_confidence=model_confidence
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
        print(f"   💰 진입 수수료: ${entry_fee:,.2f} ({TAKER_FEE:.3%})")
        print(f"   💰 예상 청산 수수료: ${expected_exit_fee:,.2f}")
        print(f"   💰 총 예상 수수료: ${total_expected_fee:,.2f}")
        fee_pct = total_expected_fee / position_value * 100
        print(f"   ⚖️  손익분기점: {fee_pct:+.2f}% (수수료 회복 필요)")
        print(f"   🛡️  손절가 (SL): ${stop_loss:,.4f} (-{STOP_LOSS_PCT * 100:.1f}%)")
        print(f"   🎯 익절가 (TP): ${take_profit:,.4f} (+{TAKE_PROFIT_PCT * 100:.1f}%)")
        print(f"   ⚠️  청산가: ${liquidation_price:,.4f}")
        print(f"   모델 신뢰도: {model_confidence:.1%}")
        print(f"   ⚡ 모니터링 모드 활성화: {INTERVAL_WITH_POSITION}초 간격으로 체크")
        print(f"{'=' * 90}")

    def close_position(self, symbol: str, exit_price: float, reason: str):
        """포지션 청산"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        pnl = position.get_pnl(exit_price)
        pnl_pct = position.get_pnl_pct(exit_price)
        roe = position.get_roe(exit_price)

        # 수수료 계산
        position_value = position.entry_price * position.quantity
        entry_fee = position_value * TAKER_FEE
        exit_fee = exit_price * position.quantity * TAKER_FEE
        total_fee = entry_fee + exit_fee

        # 순손익 (수수료 차감)
        net_pnl = pnl - total_fee
        net_pnl_pct = (net_pnl / position.margin) * 100

        # 순손익 반영
        self.balance += net_pnl
        self.total_pnl += net_pnl

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
            exit_reason=reason,
            model_confidence=position.model_confidence,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            total_fee=total_fee,
            net_pnl=net_pnl
        )
        self.trades.append(trade)
        del self.positions[symbol]

        emoji = "💀" if reason == "Liquidation" else ("🔴" if net_pnl < 0 else "🟢")
        print(f"\n{'=' * 90}")
        print(f"{emoji} 포지션 청산: {symbol}")
        print(f"   사유: {reason}")
        print(f"   진입가: ${position.entry_price:,.4f}")
        print(f"   청산가: ${exit_price:,.4f}")
        print(f"   손익 (수수료 전): ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        print(f"   💰 진입 수수료: ${entry_fee:,.2f}")
        print(f"   💰 청산 수수료: ${exit_fee:,.2f}")
        print(f"   💰 총 수수료: ${total_fee:,.2f}")
        print(f"   {'🟢' if net_pnl > 0 else '🔴'} 순손익: ${net_pnl:+,.2f} ({net_pnl_pct:+.2f}%)")
        print(f"   ROE (수수료 전): {roe:+.2f}%")
        print(f"   현재 잔고: ${self.balance:,.2f}")

        # 포지션이 모두 청산되면 스캔 모드로 전환
        if len(self.positions) == 0:
            print(f"   💤 스캔 모드로 전환: {INTERVAL_NO_POSITION}초 간격으로 체크")

        print(f"{'=' * 90}")

    def get_stats(self) -> Dict:
        """통계 계산"""
        if not self.trades:
            return {
                "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "avg_pnl": 0.0, "avg_net_pnl": 0.0, "avg_roe": 0.0,
                "max_pnl": 0.0, "min_pnl": 0.0, "max_net_pnl": 0.0, "min_net_pnl": 0.0,
                "max_roe": 0.0, "min_roe": 0.0, "liquidations": 0,
                "avg_win": 0.0, "avg_loss": 0.0, "total_fees": 0.0
            }

        # 순손익 기준 승패 계산
        wins = [t for t in self.trades if t.net_pnl > 0]
        losses = [t for t in self.trades if t.net_pnl <= 0]
        liquidations = sum(1 for t in self.trades if t.exit_reason == "Liquidation")
        total_fees = sum(t.total_fee for t in self.trades)

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(self.trades) * 100) if self.trades else 0,
            "avg_pnl": sum(t.pnl for t in self.trades) / len(self.trades),
            "avg_net_pnl": sum(t.net_pnl for t in self.trades) / len(self.trades),
            "avg_roe": sum(t.roe for t in self.trades) / len(self.trades),
            "max_pnl": max(t.pnl for t in self.trades),
            "min_pnl": min(t.pnl for t in self.trades),
            "max_net_pnl": max(t.net_pnl for t in self.trades),
            "min_net_pnl": min(t.net_pnl for t in self.trades),
            "max_roe": max(t.roe for t in self.trades),
            "min_roe": min(t.roe for t in self.trades),
            "liquidations": liquidations,
            "avg_win": sum(t.net_pnl for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t.net_pnl for t in losses) / len(losses) if losses else 0,
            "total_fees": total_fees
        }


# ===== TCN 모델 정의 (train_tcn_hourly_vwap.py와 동일) =====
class Chomp1d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):
        return x[:, :, :-self.c].contiguous() if self.c > 0 else x


def wconv(i, o, k, d):
    import torch.nn.utils as U
    pad = (k - 1) * d
    return U.weight_norm(nn.Conv1d(i, o, k, padding=pad, dilation=d))


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


class TCN_Simple(nn.Module):
    """Simple TCN model from train_tcn_hourly_vwap.py"""

    def __init__(self, in_f, hidden=32, levels=2, k=3, drop=0.2):
        super().__init__()
        L = []
        ch = in_f
        for i in range(levels):
            L.append(Block(ch, hidden, k, 2 ** i, drop))
            ch = hidden
        self.tcn = nn.Sequential(*L)

        # Batch Normalization
        self.bn = nn.BatchNorm1d(hidden)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(drop),
            nn.Linear(hidden, 2)
        )

    def forward(self, X):
        X = X.transpose(1, 2)
        H = self.tcn(X)[:, :, -1]
        H = self.bn(H)
        return self.head(H)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        # Linear projections
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.out(context)


class EnhancedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        # TCN part
        self.conv1 = wconv(in_ch, out_ch, kernel_size, dilation)
        self.chomp1 = Chomp1d((kernel_size - 1) * dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = wconv(out_ch, out_ch, kernel_size, dilation)
        self.chomp2 = Chomp1d((kernel_size - 1) * dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

        # Layer Normalization
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        # TCN forward
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(out + res)

        # Normalize
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)

        return out


class AdvancedTCN(nn.Module):
    """Advanced TCN with attention from train_tcn_hourly_vwap.py"""

    def __init__(self, in_f, hidden=64, levels=4, kernel_size=3, dropout=0.2, num_heads=4):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(in_f, hidden)

        # Multi-scale TCN branches
        self.tcn_blocks = nn.ModuleList()
        ch = hidden
        for i in range(levels):
            self.tcn_blocks.append(
                EnhancedBlock(ch, hidden, kernel_size, 2 ** i, dropout)
            )
            ch = hidden

        # Multi-head attention
        self.attention = MultiHeadAttention(hidden, num_heads, dropout)

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),  # TCN + Attention
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, 2)
        )

    def forward(self, x):
        # x: [batch, seq, features]
        batch_size, seq_len, _ = x.size()

        # Input projection
        x_proj = self.input_proj(x)  # [batch, seq, hidden]

        # TCN path
        x_tcn = x_proj.transpose(1, 2)  # [batch, hidden, seq]
        for block in self.tcn_blocks:
            x_tcn = block(x_tcn)
        x_tcn = x_tcn[:, :, -1]  # [batch, hidden] - last timestep

        # Attention path
        x_attn = self.attention(x_proj)  # [batch, seq, hidden]
        x_attn = x_attn.mean(dim=1)  # [batch, hidden] - global average

        # Fusion
        x_fused = torch.cat([x_tcn, x_attn], dim=1)  # [batch, hidden*2]
        x_fused = self.fusion(x_fused)  # [batch, hidden]

        # Classification
        return self.classifier(x_fused)


# ===== 특성 생성 =====
def make_features_hourly(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """
    train_tcn_hourly_vwap.py와 동일한 피처 생성
    """
    g = df.copy()

    # 기본 수익률
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g["logc"].diff().fillna(0.0)

    # 변동성
    for w in (8, 20, 40, 120):
        g[f"rv{w}"] = g["ret1"].rolling(w, min_periods=max(2, w // 3)).std().fillna(0.0)

    # 모멘텀
    for w in (8, 20, 40, 120):
        ema = g["close"].ewm(span=w, adjust=False).mean()
        g[f"mom{w}"] = ((g["close"] - ema) / ema).fillna(0.0)

    # 거래량 Z-score
    for w in (20, 40, 120):
        mu = g["volume"].rolling(w, min_periods=max(2, w // 3)).mean()
        sd = g["volume"].rolling(w, min_periods=max(2, w // 3)).std().replace(0, 1.0)
        g[f"vz{w}"] = ((g["volume"] - mu) / sd).fillna(0.0)

    # 거래량 급등
    vol_shift = g["volume"].shift(1)
    g["vol_spike"] = (g["volume"] / vol_shift - 1.0).fillna(0.0)

    # 거래량 가속도
    g["vol_accel"] = g["vol_spike"].diff().fillna(0.0)

    # 거래량-가격 상관관계
    for w in [8, 20]:
        g[f"vol_price_corr{w}"] = g["ret1"].rolling(w).corr(g["vol_spike"]).fillna(0.0)

    # ATR
    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14, min_periods=5).mean().fillna(0.0)

    # 가격 패턴
    g["hl_spread"] = (g["high"] - g["low"]) / g["close"]
    g["close_position"] = (g["close"] - g["low"]) / (g["high"] - g["low"] + 1e-10)
    g["body_size"] = (g["close"] - g["open"]).abs() / g["open"]
    g["upper_shadow"] = (g["high"] - g[["open", "close"]].max(axis=1)) / g["close"]
    g["lower_shadow"] = (g[["open", "close"]].min(axis=1) - g["low"]) / g["close"]

    prev_close = g["close"].shift(1)
    g["gap"] = ((g["open"] - prev_close) / prev_close).fillna(0.0)

    # 추세 분석
    for w in [2, 4, 8, 12, 24]:
        g[f"ret{w}"] = g["logc"].diff(w).fillna(0.0)

    g["mom_accel"] = g["mom8"].diff().fillna(0.0)

    for w in [4, 8, 12]:
        g[f"trend_strength{w}"] = g[f"ret{w}"].abs()

    for w in [20, 40]:
        ma = g["close"].rolling(w, min_periods=w // 2).mean()
        g[f"above_ma{w}"] = ((g["close"] > ma).astype(float) - 0.5) * 2

    # RSI
    for w in [14, 28]:
        delta = g["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(w, min_periods=w // 2).mean()
        avg_loss = loss.rolling(w, min_periods=w // 2).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        g[f"rsi{w}"] = (100 - 100 / (1 + rs)).fillna(50.0)

    # MACD
    ema12 = g["close"].ewm(span=12, adjust=False).mean()
    ema26 = g["close"].ewm(span=26, adjust=False).mean()
    g["macd"] = ((ema12 - ema26) / g["close"]).fillna(0.0)
    g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
    g["macd_hist"] = g["macd"] - g["macd_signal"]

    # 볼린저 밴드
    for w in [20]:
        ma = g["close"].rolling(w).mean()
        sd = g["close"].rolling(w).std()
        g[f"bb_upper{w}"] = ((ma + 2 * sd) / g["close"] - 1.0).fillna(0.0)
        g[f"bb_lower{w}"] = ((ma - 2 * sd) / g["close"] - 1.0).fillna(0.0)
        g[f"bb_width{w}"] = (sd / ma).fillna(0.0)

    # 시간 피처
    g["hour"] = pd.to_datetime(g["timestamp"]).dt.hour
    g["hod_sin"] = np.sin(2 * np.pi * g["hour"] / 24)
    g["hod_cos"] = np.cos(2 * np.pi * g["hour"] / 24)
    g["hour_0"] = (g["hour"] == 0).astype(float)
    g["hour_6"] = (g["hour"] == 6).astype(float)
    g["hour_12"] = (g["hour"] == 12).astype(float)
    g["hour_18"] = (g["hour"] == 18).astype(float)

    dow = pd.to_datetime(g["timestamp"]).dt.dayofweek
    g["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    g["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    g["is_weekend"] = (dow >= 5).astype(float)

    # 극값 대비 위치
    for w in [8, 24, 48]:
        high_w = g["high"].rolling(w).max()
        low_w = g["low"].rolling(w).min()
        g[f"price_vs_high{w}"] = ((high_w - g["close"]) / g["close"]).fillna(0.0)
        g[f"price_vs_low{w}"] = ((g["close"] - low_w) / g["close"]).fillna(0.0)

    # VWAP
    typical_price = (g["high"] + g["low"] + g["close"]) / 3.0
    for w in [6, 12, 24, 72]:
        pv = (typical_price * g["volume"]).rolling(w, min_periods=max(2, w // 3)).sum()
        v = g["volume"].rolling(w, min_periods=max(2, w // 3)).sum()
        vwap = (pv / v.replace(0, 1e-10)).fillna(0.0)
        g[f"vwap_dist{w}"] = ((g["close"] - vwap) / vwap).fillna(0.0).clip(-0.1, 0.1)

    pv24 = (typical_price * g["volume"]).rolling(24, min_periods=8).sum()
    v24 = g["volume"].rolling(24, min_periods=8).sum()
    vwap24 = (pv24 / v24.replace(0, 1e-10)).fillna(0.0)
    vwap24_shifted = vwap24.shift(6)
    g["vwap_slope"] = ((vwap24 / vwap24_shifted.replace(0, 1e-10)) - 1.0).fillna(0.0).clip(-0.05, 0.05)
    g["vwap_mom"] = ((g["close"] - vwap24) / vwap24).fillna(0.0).clip(-0.1, 0.1)

    # 피처 리스트
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
        "price_vs_high48", "price_vs_low48",
        # VWAP
        "vwap_dist6", "vwap_dist12", "vwap_dist24", "vwap_dist72",
        "vwap_slope", "vwap_mom"
    ]

    return g, feats


# ===== 모델 로딩 =====
MODELS = {}
MODEL_CONFIGS = {}

print("\n" + "=" * 110)
print(f"{'Hourly TCN Paper Trading System (Dynamic Interval)':^110}")
print("=" * 110)
print(f"API: {'Testnet' if USE_TESTNET else 'Mainnet'}")
print(f"Symbols: {SYMBOLS}")
print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Position Size: {POSITION_SIZE_PCT * 100:.1f}%")
print(f"Leverage: {LEVERAGE}x")
print(f"⏱️  Check Interval:")
print(f"   - No Position: {INTERVAL_NO_POSITION}s (Scanning Mode)")
print(f"   - With Position: {INTERVAL_WITH_POSITION}s (⚡ Fast Monitoring)")
print("=" * 110)

print(f"\nLoading hourly models...")
for symbol in SYMBOLS:
    symbol = symbol.strip()

    if symbol not in MODEL_PATHS:
        print(f"⚠️  {symbol}: No model path specified. Skipping.")
        continue

    model_path = MODEL_PATHS[symbol]

    if not os.path.exists(model_path):
        print(f"❌ {symbol}: Model file not found ({model_path})")
        continue

    try:
        print(f"\n📦 {symbol} Loading model...")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Checkpoint 구조 확인
        feat_cols = checkpoint.get('feat_cols', [])
        model_state = checkpoint.get('model', {})
        meta = checkpoint.get('meta', {})
        scaler_mu = checkpoint.get('scaler_mu')
        scaler_sd = checkpoint.get('scaler_sd')

        seq_len = meta.get('seq_len', 240)

        # ✅ 모델 타입 자동 감지
        state_keys = list(model_state.keys())

        is_simple_tcn = 'bn.weight' in state_keys
        is_advanced_tcn = 'attention.q_linear.weight' in state_keys

        if is_simple_tcn:
            # TCN_Simple 모델
            first_conv_weight = model_state.get('tcn.0.c1.weight_v')
            if first_conv_weight is not None:
                hidden = first_conv_weight.shape[0]
            else:
                hidden = 32

            levels = sum(1 for key in state_keys if key.startswith('tcn.') and '.c1.bias' in key)

            print(f"   🔍 Detected: TCN_Simple (hidden={hidden}, levels={levels})")
            model = TCN_Simple(in_f=len(feat_cols), hidden=hidden, levels=levels, drop=0.2)

        elif is_advanced_tcn:
            # AdvancedTCN 모델
            input_proj_weight = model_state.get('input_proj.weight')
            if input_proj_weight is not None:
                hidden = input_proj_weight.shape[0]
            else:
                hidden = 48

            levels = sum(1 for key in state_keys if key.startswith('tcn_blocks.') and '.conv1.weight_v' in key)

            print(f"   🔍 Detected: AdvancedTCN (hidden={hidden}, levels={levels})")
            model = AdvancedTCN(in_f=len(feat_cols), hidden=hidden, levels=levels, dropout=0.2, num_heads=4)

        else:
            print(f"❌ {symbol}: Unknown model architecture in checkpoint")
            continue

        # 모델 state_dict 로드
        model.load_state_dict(model_state)
        model.eval()

        MODELS[symbol] = model
        MODEL_CONFIGS[symbol] = {
            'feat_cols': feat_cols,
            'seq_len': seq_len,
            'scaler_mu': scaler_mu,
            'scaler_sd': scaler_sd
        }

        print(f"   ✅ Loaded successfully")
        print(f"      - Features: {len(feat_cols)}, Seq: {seq_len}")

    except Exception as e:
        print(f"❌ {symbol}: Failed to load - {e}")
        import traceback

        traceback.print_exc()

print(f"\n{'=' * 110}")
print(f"{'✅ Total ' + str(len(MODELS)) + ' models loaded':^110}")
print(f"{'=' * 110}\n")

if not MODELS:
    print("❌ No models loaded! Exiting...")
    exit(1)


# ===== API 클래스 =====
class BybitAPI:
    """Bybit API 클라이언트"""

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
                return data["result"]["list"][0]
            return {}
        except Exception as e:
            print(f"⚠️  Ticker query failed ({symbol}): {e}")
            return {}

    def get_klines(self, symbol: str, interval: str = "60", limit: int = 500) -> pd.DataFrame:
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

            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                klines = data["result"]["list"]
                df = pd.DataFrame(klines,
                                  columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
                df = df.astype({
                    "open": float, "high": float, "low": float, "close": float, "volume": float
                })
                df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit='ms')
                df = df.sort_values("timestamp").reset_index(drop=True)
                return df
            return pd.DataFrame()
        except Exception as e:
            print(f"⚠️  Klines query failed ({symbol}): {e}")
            return pd.DataFrame()


API = BybitAPI()


# ===== 추론 함수 =====
def predict_hourly(symbol: str, debug: bool = False) -> dict:
    """
    Hourly 모델을 사용한 예측
    """
    symbol = symbol.strip()

    if symbol not in MODELS:
        return {"error": f"No model for {symbol}", "symbol": symbol}

    model = MODELS[symbol]
    config = MODEL_CONFIGS[symbol]

    # 데이터 가져오기
    required_length = config['seq_len'] + 100
    limit = min(required_length, 1000)
    df = API.get_klines(symbol, interval="60", limit=limit)

    if df.empty or len(df) < config['seq_len'] + 20:
        return {"error": "Insufficient data", "symbol": symbol}

    # 특성 생성
    df_feat, feat_cols = make_features_hourly(df)

    # 필요한 특성만 선택
    df_feat = df_feat[config['feat_cols']]
    df_feat = df_feat.fillna(0.0)

    if len(df_feat) < config['seq_len']:
        return {"error": "Insufficient features after cleaning", "symbol": symbol}

    # 정규화
    X = df_feat.iloc[-config['seq_len']:].values
    mu = np.array(config['scaler_mu'])
    sd = np.array(config['scaler_sd'])
    X = (X - mu) / sd
    X = np.clip(X, -10, 10)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_tensor = torch.FloatTensor(X).unsqueeze(0)

    # 추론
    with torch.no_grad():
        logits = model(X_tensor)

        # Direction
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
        pred_class = int(probs.argmax())
        direction = "Long" if pred_class == 1 else "Short"
        confidence = float(probs.max())

    ticker = API.get_ticker(symbol)
    current_price = float(ticker.get("lastPrice", 0))

    if debug:
        print(f"\n[DEBUG {symbol}]")
        print(f"  Direction: {direction} (confidence: {confidence:.4f})")
        print(f"  Probs: Long={probs[1]:.4f}, Short={probs[0]:.4f}")

    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "current_price": current_price,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ===== 대시보드 =====
def print_dashboard(account: Account, prices: Dict[str, float], mode: str = "scanning"):
    """대시보드 출력"""
    os.system('cls' if os.name == 'nt' else 'clear')

    # 현재 모드 표시
    mode_icon = "💤" if mode == "scanning" else "⚡"
    mode_text = f"SCANNING MODE ({INTERVAL_NO_POSITION}s)" if mode == "scanning" else f"MONITORING MODE ({INTERVAL_WITH_POSITION}s)"

    total_value = account.get_total_value(prices)
    unrealized_pnl = total_value - account.balance
    total_pnl_pct = (total_value / account.initial_capital - 1) * 100
    used_margin = sum(p.margin for p in account.positions.values())

    print("\n" + "=" * 110)
    print(f"{'📊 ACCOUNT STATUS':^110}")
    print(f"{mode_icon + ' ' + mode_text:^110}")
    print("=" * 110)
    print(f"💵 Balance:        ${account.balance:>12,.2f}  |  "
          f"💰 Total Value: ${total_value:>12,.2f}  ({total_pnl_pct:>+6.2f}%)")
    print(f"📈 Unrealized:     ${unrealized_pnl:>+12,.2f}  |  "
          f"🔒 Used Margin: ${used_margin:>12,.2f}  "
          f"({used_margin / account.initial_capital * 100:>5.1f}%)")
    print(f"💎 Total PnL:      ${account.total_pnl:>+12,.2f}  |  "
          f"🎯 Positions:   {len(account.positions)}/{MAX_POSITIONS}")

    # 포지션 정보
    if account.positions:
        print(f"\n📍 Open Positions: {len(account.positions)}")
        print("-" * 130)
        print(f"{'Symbol':^12} | {'Direction':^10} | {'Entry Price':^12} | {'Current Price':^12} | "
              f"{'PnL':^20} | {'Confidence':^10} | {'Hold Time':^10}")
        print("-" * 130)

        for symbol, pos in account.positions.items():
            current_price = prices.get(symbol, pos.entry_price)
            pnl = pos.get_pnl(current_price)
            roe = pos.get_roe(current_price)

            # 수수료 계산
            position_value = pos.entry_price * pos.quantity
            entry_fee = position_value * TAKER_FEE
            exit_fee = current_price * pos.quantity * TAKER_FEE
            total_fee = entry_fee + exit_fee
            net_pnl = pnl - total_fee
            net_roe = (net_pnl / pos.margin) * 100

            hold_min = (datetime.now() - pos.entry_time).total_seconds() / 60

            emoji = "📈" if pos.direction == "Long" else "📉"
            pnl_emoji = "🟢" if net_pnl > 0 else "🔴"

            print(f"{symbol:^12} | {emoji} {pos.direction:^6} | ${pos.entry_price:>10,.4f} | "
                  f"${current_price:>10,.4f} | {pnl_emoji} ${net_pnl:>+8,.2f} ({net_roe:>+6.1f}%) | "
                  f"{pos.model_confidence:>8.1%} | {hold_min:>6.0f}분")
    else:
        print(f"\n📍 Open Positions: None")

    # 통계
    stats = account.get_stats()
    if stats["total_trades"] > 0:
        print(f"\n📊 Trading Statistics")
        print(f"   Total Trades:  {stats['total_trades']:>3}")
        print(f"   Win Rate:      {stats['win_rate']:>6.1f}% ({stats['wins']}W {stats['losses']}L)")
        if stats['liquidations'] > 0:
            print(f"   Liquidations:  {stats['liquidations']:>3} 💀")
        print(f"   💰 Total Fees: ${stats['total_fees']:>+12,.2f}")
        print(f"   Avg Net PnL:   ${stats['avg_net_pnl']:>+12,.2f}")
        print(f"   Avg ROE:       {stats['avg_roe']:>+6.1f}%")
        print(f"   Max Win:       ${stats['max_net_pnl']:>12,.2f}")
        print(f"   Max Loss:      ${stats['min_net_pnl']:>12,.2f}")

    print("\n" + "=" * 110)


def save_trades(account: Account):
    """거래 내역 저장"""
    if not account.trades:
        return

    data = [asdict(t) for t in account.trades]
    with open(TRADE_LOG_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n💾 Trades saved: {TRADE_LOG_FILE}")


# ===== 메인 루프 =====
def test_api_connection():
    """API 연결 테스트"""
    print("\n🔍 Testing API connection...")

    test_symbol = SYMBOLS[0]
    print(f"   Test symbol: {test_symbol}")
    ticker = API.get_ticker(test_symbol)
    if ticker and ticker.get("lastPrice"):
        print(f"   ✓ Ticker OK: ${float(ticker['lastPrice']):,.2f}")
    else:
        print(f"   ✗ Ticker failed")
        return False

    df = API.get_klines(test_symbol, interval="60", limit=10)
    if not df.empty:
        print(f"   ✓ Klines OK: {len(df)} candles")
        print(f"   Latest price: ${df['close'].iloc[-1]:,.2f}")
    else:
        print(f"   ✗ Klines failed")
        return False

    print("\n✅ API connection OK\n")
    return True


def main():
    print("\n" + "=" * 110)
    print(f"{'🎯 Paper Trading Started (Dynamic Interval)':^110}")
    print(f"{'Initial Capital: $' + f'{INITIAL_CAPITAL:,.2f}':^110}")
    print(f"{'Leverage: ' + f'{LEVERAGE}x (Position Size: {POSITION_SIZE_PCT * 100:.0f}%)':^110}")
    print(f"{'Confidence Threshold: ' + f'{CONF_THRESHOLD:.0%}':^110}")
    print(f"{'Models: ' + str(len(MODELS)):^110}")
    print("=" * 110)

    # 수수료 정보 표시
    position_size = INITIAL_CAPITAL * POSITION_SIZE_PCT * LEVERAGE
    expected_fee_per_trade = position_size * TAKER_FEE * 2
    fee_pct = (TAKER_FEE * 2) * 100

    print(f"\n💰 Fee Information:")
    print(f"   - Taker Fee: {TAKER_FEE:.3%} (Market order)")
    print(f"   - Expected fee per trade: ${expected_fee_per_trade:.2f}")
    print(f"   - Break-even: {fee_pct:.2f}% (round-trip)")
    print()

    print(f"⏱️  Dynamic Interval:")
    print(f"   - 💤 No Position: {INTERVAL_NO_POSITION}s (신호 스캔)")
    print(f"   - ⚡ With Position: {INTERVAL_WITH_POSITION}s (실시간 모니터링)")
    print()

    # API 연결 테스트
    if not test_api_connection():
        print("\n❌ API connection failed. Exiting.")
        return

    account = Account(INITIAL_CAPITAL)

    try:
        loop_count = 0
        last_signal_scan = 0  # 마지막 신호 스캔 시간

        while True:
            loop_count += 1
            current_time = datetime.now()

            # ✅ 동적 interval: 포지션 유무에 따라 체크 주기 변경
            has_positions = len(account.positions) > 0
            current_interval = INTERVAL_WITH_POSITION if has_positions else INTERVAL_NO_POSITION
            current_mode = "monitoring" if has_positions else "scanning"

            # 현재 가격 가져오기 (항상)
            prices = {}
            for symbol in MODELS.keys():
                ticker = API.get_ticker(symbol)
                prices[symbol] = float(ticker.get("lastPrice", 0))

            # ✅ 포지션 관리 (포지션이 있을 때만)
            if has_positions:
                for symbol in list(account.positions.keys()):
                    position = account.positions[symbol]
                    current_price = prices.get(symbol, position.entry_price)

                    # 청산 조건 확인
                    should_close, reason = position.should_close(current_price, current_time)
                    if should_close:
                        account.close_position(symbol, current_price, reason)
                    else:
                        # 반대 신호로 청산 (60초마다 한 번만 체크)
                        time_since_scan = time.time() - last_signal_scan
                        if time_since_scan >= INTERVAL_NO_POSITION:
                            result = predict_hourly(symbol, debug=False)
                            if "error" not in result and result.get("confidence", 0) >= CONF_THRESHOLD:
                                signal_dir = result["direction"]
                                if (position.direction == "Long" and signal_dir == "Short") or \
                                        (position.direction == "Short" and signal_dir == "Long"):
                                    account.close_position(symbol, current_price, "Reverse Signal")

            # 대시보드 출력
            print_dashboard(account, prices, current_mode)

            # ✅ 신호 스캔 (포지션 없거나 60초마다)
            should_scan = not has_positions or (time.time() - last_signal_scan >= INTERVAL_NO_POSITION)

            if should_scan:
                last_signal_scan = time.time()

                print(f"\n🔍 Signal Scan ({len(MODELS)} symbols)")
                print(f"{'Symbol':^12} | {'Price':^12} | {'Direction':^10} | {'Confidence':^10} | {'Signal':^20}")
                print("-" * 80)

                debug_mode = (loop_count == 1)
                for symbol in MODELS.keys():
                    result = predict_hourly(symbol, debug=debug_mode)

                    if "error" in result:
                        print(
                            f"{symbol:^12} | {'N/A':^12} | {'Error':^10} | {'N/A':^10} | ❌ {result.get('error', 'Unknown')}")
                        continue

                    direction = result["direction"]
                    confidence = result["confidence"]
                    price = result["current_price"]

                    dir_icon = {"Long": "📈", "Short": "📉"}.get(direction, "❓")

                    # 신호 판단
                    if confidence < CONF_THRESHOLD:
                        signal = f"⚠️  Weak signal"
                    elif direction == "Long":
                        signal = f"🟢 BUY signal"
                    else:
                        signal = f"🔴 SELL signal"

                    print(f"{symbol:^12} | ${price:>10,.4f} | {dir_icon} {direction:^8} | "
                          f"{confidence:>8.1%} | {signal}")

                    # 진입 조건
                    if account.can_open_position(symbol) and confidence >= CONF_THRESHOLD:
                        account.open_position(symbol, direction, price, confidence)
                        # 포지션 진입 시 즉시 모니터링 모드로 전환
                        has_positions = True

            print(f"\n[Check #{loop_count}] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Mode: {current_mode.upper()}")
            print(f"Next check in {current_interval}s... (Ctrl+C to exit)")

            time.sleep(current_interval)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        print_dashboard(account, prices)
        save_trades(account)

        # 최종 통계
        stats = account.get_stats()
        if stats["total_trades"] > 0:
            print("\n" + "=" * 110)
            print(f"{'📊 Final Results':^110}")
            print("=" * 110)
            final_balance = account.balance
            final_return = (final_balance / account.initial_capital - 1) * 100
            print(f"   Final Balance: ${final_balance:,.2f}")
            print(f"   Total Return:  {final_return:+.2f}%")
            print(f"   Total Trades:  {stats['total_trades']}")
            print(f"   Win Rate:      {stats['win_rate']:.1f}%")
            print(f"   Avg ROE:       {stats['avg_roe']:+.1f}%")
            if stats['liquidations'] > 0:
                print(f"   Liquidations:  {stats['liquidations']} 💀")
            print("=" * 110)

        print("\n✅ Program terminated.")


if __name__ == "__main__":
    main()