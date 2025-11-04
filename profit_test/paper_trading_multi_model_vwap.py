# paper_trading_multi_model_vwap.py
# -*- coding: utf-8 -*-
"""
TCN V2 모델 기반 페이퍼 트레이딩 시스템
- train_tcn_5minutes_vwap.py로 학습한 AdvancedTCN_vwap 모델 사용
- Rolling Window Normalization 적응
- Multi-task output 활용 (side, expected_bps, confidence)
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
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,BNBUSDT").split(",")
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "2"))

# ✅ 심볼별 V2 모델 경로 설정
MODEL_PATHS = {
    "BTCUSDT": "D:/ygy_work/coin/multimodel/models_5min_btc_vwap/model_vwap_best.pt",
    "ETHUSDT": "D:/ygy_work/coin/multimodel/models_5min_eth_vwap/model_vwap_best.pt",
    "SOLUSDT": "D:/ygy_work/coin/multimodel/models_5min_sol_vwap/model_vwap_best.pt",
    "DOGEUSDT": "D:/ygy_work/coin/multimodel/models_5min_doge_vwap/model_vwap_best.pt",
    "BNBUSDT": "D:/ygy_work/coin/multimodel/models_5min_bnb_vwap/model_vwap_best.pt",
    "XRPUSDT": "D:/ygy_work/coin/multimodel/models_5min_xrp_vwap/model_vwap_best.pt",
    "SAPIENUSDT": "D:/ygy_work/coin/multimodel/models_5min_sapien_vwap/model_vwap_best.pt",
    "ZEUSUSDT": "D:/ygy_work/coin/multimodel/models_5min_zeus_vwap/model_vwap_best.pt",
    "FLMUSDT": "D:/ygy_work/coin/multimodel/models_5min_flm_vwap/model_vwap_best.pt",
    "TRUMPUSDT": "D:/ygy_work/coin/multimodel/models_5min_trump_vwap/model_vwap_best.pt",
    "JELLYJELLYUSDT": "D:/ygy_work/coin/multimodel/models_5min_jellyjelly_vwap/model_v2_best.pt"

}

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.75"))
USE_TESTNET = os.getenv("USE_TESTNET", "0") == "1"

# 페이퍼 트레이딩 설정
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000"))
POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", "0.1"))
LEVERAGE = int(os.getenv("LEVERAGE", "20"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.03"))
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "30"))
LIQUIDATION_BUFFER = float(os.getenv("LIQUIDATION_BUFFER", "0.8"))
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "trades_vwap.json")

# 수수료 설정 (Bybit 기본값)
TAKER_FEE = float(os.getenv("TAKER_FEE", "0.00055"))  # 0.055% (Market order)
MAKER_FEE = float(os.getenv("MAKER_FEE", "0.0002"))  # 0.02% (Limit order)
FUNDING_FEE_RATE = float(os.getenv("FUNDING_FEE_RATE", "0.0001"))  # 0.01% (8시간마다)


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
    expected_bps: float = 0.0  # V2: 예상 수익률 (basis points)
    model_confidence: float = 0.0  # V2: 모델 신뢰도

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
    expected_bps: float = 0.0
    model_confidence: float = 0.0
    entry_fee: float = 0.0  # 진입 수수료
    exit_fee: float = 0.0  # 청산 수수료
    total_fee: float = 0.0  # 총 수수료
    net_pnl: float = 0.0  # 수수료 차감 후 순손익


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
                      expected_bps: float = 0.0, model_confidence: float = 0.0):
        """포지션 진입"""
        margin = self.initial_capital * POSITION_SIZE_PCT
        position_value = margin * LEVERAGE
        quantity = position_value / price

        # ✅ 수수료 계산
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
            expected_bps=expected_bps,
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
        print(f"   예상 수익: {expected_bps:.1f} bps")
        print(f"   모델 신뢰도: {model_confidence:.1%}")
        print(f"{'=' * 90}")

    def close_position(self, symbol: str, exit_price: float, reason: str):
        """포지션 청산"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        pnl = position.get_pnl(exit_price)
        pnl_pct = position.get_pnl_pct(exit_price)
        roe = position.get_roe(exit_price)

        # ✅ 수수료 계산
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
            expected_bps=position.expected_bps,
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

        # ✅ 순손익 기준 승패 계산
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


# ===== AdvancedTCN_vwap 모델 정의 (train_tcn_5minutes_vwap.py와 동일) =====
class GatedResidualBlock(nn.Module):
    """Gated Residual TCN Block"""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()

        pad = (kernel_size - 1) * dilation

        # Main path
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        )
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        )

        # Gate path
        self.gate = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, 1)
        )

        self.chomp1 = nn.ConstantPad1d((0, -pad), 0) if pad > 0 else nn.Identity()
        self.chomp2 = nn.ConstantPad1d((0, -pad), 0) if pad > 0 else nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

        # Layer norm
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.chomp2(out)

        # Gating
        gate = torch.sigmoid(self.gate(out))
        out = out * gate

        out = self.relu(out)
        out = self.dropout(out)

        # Residual
        res = x if self.downsample is None else self.downsample(x)
        out = out + res

        # Normalize
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)

        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self Attention"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        # Linear projections
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Output projection
        output = self.out_linear(context)
        output = self.dropout(output)

        # Residual + Layer norm
        output = self.layer_norm(x + output)

        return output


class AdvancedTCN_vwap(nn.Module):
    """
    V2: 멀티태스크 학습
    - 출력 1: 방향 (binary)
    - 출력 2: 기대 수익률 (regression)
    - 출력 3: 신뢰도 (regression)
    """

    def __init__(self, in_features, hidden=64, levels=4, dropout=0.2, num_heads=4):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(in_features, hidden)

        # TCN blocks
        self.tcn_blocks = nn.ModuleList()
        for i in range(levels):
            self.tcn_blocks.append(
                GatedResidualBlock(hidden, hidden, kernel_size=3,
                                   dilation=2 ** i, dropout=dropout)
            )

        # Attention
        self.attention = MultiHeadSelfAttention(hidden, num_heads, dropout)

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-task heads
        # Head 1: Direction (binary classification)
        self.head_direction = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2)
        )

        # Head 2: Expected return (regression)
        self.head_return = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

        # Head 3: Confidence (regression, 0~1)
        self.head_confidence = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Input projection
        x_proj = self.input_proj(x)

        # TCN path
        x_tcn = x_proj.transpose(1, 2)  # [B, H, S]
        for block in self.tcn_blocks:
            x_tcn = block(x_tcn)
        x_tcn = x_tcn[:, :, -1]  # [B, H]

        # Attention path
        x_attn = self.attention(x_proj)  # [B, S, H]
        x_attn = x_attn.mean(dim=1)  # [B, H]

        # Fusion
        x_fused = torch.cat([x_tcn, x_attn], dim=1)
        x_fused = self.fusion(x_fused)

        # Multi-task outputs
        direction_logits = self.head_direction(x_fused)
        expected_return = self.head_return(x_fused).squeeze(-1)
        confidence = self.head_confidence(x_fused).squeeze(-1)

        return direction_logits, expected_return, confidence


# ===== 특성 생성 V2 =====
def make_features_vwap(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """
    V2: train_tcn_5minutes_vwap.py와 동일한 피처 생성
    """
    g = df.copy()

    # 기본 수익률
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g["logc"].diff().fillna(0.0)

    # 변동성
    for w in (6, 12, 24, 72):
        g[f"rv{w}"] = g["ret1"].rolling(w, min_periods=max(2, w // 3)).std().fillna(0.0)

    # 모멘텀
    for w in (6, 12, 24, 72):
        ema = g["close"].ewm(span=w, adjust=False).mean()
        g[f"mom{w}"] = ((g["close"] - ema) / ema).fillna(0.0)

    # 거래량 Z-score
    for w in (12, 24, 72):
        mu = g["volume"].rolling(w, min_periods=max(2, w // 3)).mean()
        sd = g["volume"].rolling(w, min_periods=max(2, w // 3)).std().replace(0, 1.0)
        g[f"vz{w}"] = ((g["volume"] - mu) / sd).fillna(0.0).clip(-5, 5)

    # 거래량 변화
    g["vol_change"] = g["volume"].pct_change().fillna(0.0).clip(-2, 2)

    # ATR
    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=5).mean().fillna(0.0)
    g["atr_ratio"] = (atr / g["close"]).fillna(0.0).clip(0, 0.1)

    # 캔들스틱 패턴
    g["hl_ratio"] = ((g["high"] - g["low"]) / g["close"]).fillna(0.0).clip(0, 0.1)
    g["close_pos"] = ((g["close"] - g["low"]) / (g["high"] - g["low"] + 1e-10)).fillna(0.5).clip(0, 1)
    g["body_ratio"] = ((g["close"] - g["open"]).abs() / g["close"]).fillna(0.0).clip(0, 0.1)

    # RSI
    for w in (14,):
        delta = g["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(w, min_periods=w // 2).mean()
        avg_loss = loss.rolling(w, min_periods=w // 2).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        g[f"rsi{w}"] = ((rsi - 50) / 50).fillna(0.0).clip(-1, 1)

    # MACD
    ema12 = g["close"].ewm(span=12, adjust=False).mean()
    ema26 = g["close"].ewm(span=26, adjust=False).mean()
    g["macd"] = ((ema12 - ema26) / g["close"]).fillna(0.0).clip(-0.1, 0.1)

    # 시간 패턴
    if 'timestamp' in g.columns:
        hod = pd.to_datetime(g["timestamp"]).dt.hour
        g["hour_sin"] = np.sin(2 * np.pi * hod / 24.0)
        g["hour_cos"] = np.cos(2 * np.pi * hod / 24.0)

        dow = pd.to_datetime(g["timestamp"]).dt.dayofweek
        g["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        g["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # 최근 극값 대비 위치
    for w in (12, 24):
        high_max = g["high"].rolling(w, min_periods=max(2, w // 3)).max()
        low_min = g["low"].rolling(w, min_periods=max(2, w // 3)).min()
        g[f"vs_high{w}"] = ((g["close"] - high_max) / high_max).fillna(0.0).clip(-0.1, 0)
        g[f"vs_low{w}"] = ((g["close"] - low_min) / low_min).fillna(0.0).clip(0, 0.1)

    # 모멘텀 가속도
    g["mom_accel"] = g["mom6"].diff().fillna(0.0).clip(-0.1, 0.1)

    # 변동성 변화
    g["vol_change_rate"] = g["rv6"].pct_change().fillna(0.0).clip(-1, 1)

    # ===== VWAP 피처 =====
    # 대표 가격 (Typical Price)
    typical_price = (g["high"] + g["low"] + g["close"]) / 3.0

    # Rolling VWAP (여러 윈도우)
    for w in (12, 24, 72):  # 1시간, 2시간, 6시간
        # 누적 (price * volume)
        pv = (typical_price * g["volume"]).rolling(w, min_periods=max(2, w // 3)).sum()
        # 누적 volume
        v = g["volume"].rolling(w, min_periods=max(2, w // 3)).sum()
        vwap = (pv / v.replace(0, 1e-10)).fillna(0.0)

        # VWAP 대비 위치 (정규화)
        g[f"vwap_dist{w}"] = ((g["close"] - vwap) / vwap).fillna(0.0).clip(-0.1, 0.1)

    # VWAP 기울기 (24봉 기준)
    pv24 = (typical_price * g["volume"]).rolling(24, min_periods=8).sum()
    v24 = g["volume"].rolling(24, min_periods=8).sum()
    vwap24 = (pv24 / v24.replace(0, 1e-10)).fillna(0.0)
    vwap24_shifted = vwap24.shift(6)
    g["vwap_slope"] = ((vwap24 / vwap24_shifted.replace(0, 1e-10)) - 1.0).fillna(0.0).clip(-0.05, 0.05)

    # 거래량 가중 모멘텀
    g["vwap_mom"] = ((g["close"] - vwap24) / vwap24).fillna(0.0).clip(-0.1, 0.1)

    feats = [
        "ret1",
        "rv6", "rv12", "rv24", "rv72",
        "mom6", "mom12", "mom24", "mom72", "mom_accel",
        "vz12", "vz24", "vz72", "vol_change",
        "atr_ratio",
        "hl_ratio", "close_pos", "body_ratio",
        "rsi14", "macd",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "vs_high12", "vs_low12", "vs_high24", "vs_low24",
        "vol_change_rate",
        "vwap_dist12", "vwap_dist24", "vwap_dist72",
        "vwap_slope", "vwap_mom"
    ]

    # NaN 처리
    for feat in feats:
        g[feat] = g[feat].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    return g, feats


# ===== 심볼별 V2 모델 로드 =====
print("\n" + "=" * 110)
print(f"{'🤖 심볼별 V2 모델 로드 중...':^110}")
print("=" * 110)

MODELS = {}
MODEL_CONFIGS = {}

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
        print(f"\n📦 {symbol} V2 모델 로드 중...")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # V2 checkpoint 구조
        feat_cols = checkpoint.get('feat_cols', [])
        model_state = checkpoint.get('model_state', checkpoint.get('model', {}))

        # 메타 정보
        meta_path = os.path.join(os.path.dirname(model_path), "meta_vwap.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        else:
            meta = {}

        seq_len = meta.get('seq_len', 72)
        norm_window = meta.get('norm_window', 288)

        # AdvancedTCN_vwap 모델 생성
        model = AdvancedTCN_vwap(in_features=len(feat_cols), hidden=64,
                                 levels=4, dropout=0.2, num_heads=4)

        model.load_state_dict(model_state)
        model.eval()

        MODELS[symbol] = model
        MODEL_CONFIGS[symbol] = {
            'feat_cols': feat_cols,
            'seq_len': seq_len,
            'norm_window': norm_window
        }

        print(f"   ✅ 로드 완료")
        print(f"      - 모델: AdvancedTCN_vwap")
        print(f"      - Features: {len(feat_cols)}, Seq: {seq_len}")
        print(f"      - Norm Window: {norm_window}")

    except Exception as e:
        print(f"❌ {symbol}: 로드 실패 - {e}")
        import traceback

        traceback.print_exc()

print(f"\n{'=' * 110}")
print(f"{'✅ 총 ' + str(len(MODELS)) + '개 심볼 V2 모델 로드 완료':^110}")
print(f"{'=' * 110}\n")

if not MODELS:
    print("❌ 로드된 모델이 없습니다. 프로그램을 종료합니다.")
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
            print(f"⚠️  Klines 조회 실패 ({symbol}): {e}")
            return pd.DataFrame()


API = BybitAPI()


# ===== 추론 함수 V2 =====
def predict_vwap(symbol: str, debug: bool = False) -> dict:
    """
    AdvancedTCN_vwap 모델을 사용한 예측
    Rolling Window Normalization 적용
    """
    symbol = symbol.strip()

    if symbol not in MODELS:
        return {"error": f"No model for {symbol}", "symbol": symbol}

    model = MODELS[symbol]
    config = MODEL_CONFIGS[symbol]

    # 데이터 가져오기 (충분한 양)
    required_length = config['seq_len'] + config['norm_window'] + 100
    limit = min(required_length, 1000)
    df = API.get_klines(symbol, interval="5", limit=limit)

    if df.empty or len(df) < config['seq_len'] + 20:
        return {"error": "Insufficient data", "symbol": symbol}

    # 특성 생성 V2
    df_feat, feat_cols = make_features_vwap(df)

    # 필요한 특성만 선택
    df_feat = df_feat[config['feat_cols']]
    df_feat = df_feat.dropna()

    if len(df_feat) < config['seq_len']:
        return {"error": "Insufficient features after cleaning", "symbol": symbol}

    # Rolling Window Normalization
    # 최근 norm_window 데이터로 정규화
    norm_data = df_feat.iloc[-config['norm_window']:].values if len(df_feat) >= config[
        'norm_window'] else df_feat.values
    mu = np.mean(norm_data, axis=0)
    sd = np.std(norm_data, axis=0)
    sd = np.maximum(sd, 0.01)  # 최소값 설정

    # 시퀀스 준비 및 정규화
    X = df_feat.iloc[-config['seq_len']:].values
    X = (X - mu) / sd
    X = np.clip(X, -10, 10)  # 극단값 클리핑
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_tensor = torch.FloatTensor(X).unsqueeze(0)  # (1, T, F)

    # 추론
    with torch.no_grad():
        pred_direction, pred_return, pred_confidence = model(X_tensor)

        # Direction
        probs = torch.softmax(pred_direction, dim=1).squeeze().numpy()
        pred_class = int(probs.argmax())
        direction = "Long" if pred_class == 1 else "Short"

        # Expected BPS
        expected_bps = float(pred_return.squeeze().item())

        # Confidence
        confidence = float(pred_confidence.squeeze().item())

    ticker = API.get_ticker(symbol)
    current_price = float(ticker.get("lastPrice", 0))

    if debug:
        print(f"\n[DEBUG {symbol}]")
        print(f"  Direction: {direction} (prob: {probs.max():.4f})")
        print(f"  Expected BPS: {expected_bps:.2f}")
        print(f"  Confidence: {confidence:.4f}")

    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "expected_bps": expected_bps,
        "current_price": current_price,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ===== 화면 출력 =====
def print_dashboard(account: Account, prices: Dict[str, float]):
    """대시보드 출력"""
    os.system('clear' if os.name == 'posix' else 'cls')

    print("\n" + "=" * 110)
    print(f"{'🎯 페이퍼 트레이딩 시스템 V2 (레버리지 ' + str(LEVERAGE) + 'x)':^110}")
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
            f"{'심볼':^12} | {'방향':^8} | {'진입가':^12} | {'현재가':^12} | {'손익(수수료 포함)':^30} | {'예상BPS':^10} | {'신뢰도':^8} | {'보유':^8}")
        print("-" * 130)

        for symbol, pos in account.positions.items():
            current_price = prices.get(symbol, pos.entry_price)
            pnl = pos.get_pnl(current_price)
            roe = pos.get_roe(current_price)

            # ✅ 수수료 계산
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
                  f"${current_price:>10,.4f} | {pnl_emoji} ${net_pnl:>+8,.2f} ({net_roe:>+6.1f}%) 💰-${total_fee:.2f} | "
                  f"{pos.expected_bps:>8.1f} | {pos.model_confidence:>6.1%} | {hold_min:>6.1f}분")
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
        print(f"   💰 총 수수료:  ${stats['total_fees']:>+12,.2f}")
        print(f"   평균 손익 (수수료 전): ${stats['avg_pnl']:>+12,.2f}")
        print(f"   평균 순손익:   ${stats['avg_net_pnl']:>+12,.2f}")
        print(f"   평균 ROE (수수료 전): {stats['avg_roe']:>+6.1f}%")
        print(f"   최대 수익:     ${stats['max_pnl']:>12,.2f}  (ROE: {stats['max_roe']:>+6.1f}%)")
        print(f"   최대 손실:     ${stats['min_pnl']:>12,.2f}  (ROE: {stats['min_roe']:>+6.1f}%)")
        print(f"   최대 순수익:   ${stats['max_net_pnl']:>12,.2f}")
        print(f"   최대 순손실:   ${stats['min_net_pnl']:>12,.2f}")

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

    print("   테스트 심볼: BTCUSDT")
    ticker = API.get_ticker("BTCUSDT")
    if ticker and ticker.get("lastPrice"):
        print(f"   ✓ Ticker 조회 성공: ${float(ticker['lastPrice']):,.2f}")
    else:
        print(f"   ✗ Ticker 조회 실패")
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
    print(f"{'🎯 페이퍼 트레이딩 V2 시작':^110}")
    print(f"{'초기 자본: $' + f'{INITIAL_CAPITAL:,.2f}':^110}")
    print(f"{'레버리지: ' + f'{LEVERAGE}x (포지션 크기: {POSITION_SIZE_PCT * 100:.0f}%)':^110}")
    print(f"{'신뢰도 임계값: ' + f'{CONF_THRESHOLD:.0%}':^110}")
    print(f"{'사용 모델: ' + str(len(MODELS)) + '개 AdvancedTCN_vwap':^110}")
    print("=" * 110)

    # ✅ 수수료 정보 표시
    position_size = INITIAL_CAPITAL * POSITION_SIZE_PCT * LEVERAGE
    expected_fee_per_trade = position_size * TAKER_FEE * 2
    fee_pct = (TAKER_FEE * 2) * 100

    print(f"\n💰 수수료 정보:")
    print(f"   - Taker Fee: {TAKER_FEE:.3%} (Market order)")
    print(f"   - Maker Fee: {MAKER_FEE:.3%} (Limit order)")
    print(f"   - 포지션당 예상 총 수수료: ${expected_fee_per_trade:.2f}")
    print(f"   - 손익분기점: {fee_pct:.2f}% (양방향 수수료)")
    print()

    # API 연결 테스트
    if not test_api_connection():
        print("\n❌ API 연결에 실패했습니다. 프로그램을 종료합니다.")
        return

    account = Account(INITIAL_CAPITAL)

    try:
        loop_count = 0
        while True:
            loop_count += 1
            current_time = datetime.now()

            # 현재 가격 가져오기
            prices = {}
            for symbol in MODELS.keys():
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
                    result = predict_vwap(symbol, debug=False)
                    if "error" not in result and result.get("confidence", 0) >= CONF_THRESHOLD:
                        signal_dir = result["direction"]
                        if (position.direction == "Long" and signal_dir == "Short") or \
                                (position.direction == "Short" and signal_dir == "Long"):
                            account.close_position(symbol, current_price, "Reverse Signal")

            # 대시보드 출력
            print_dashboard(account, prices)

            # 신호 스캔
            print(f"\n🔍 신호 스캔 V2 ({len(MODELS)}개 심볼)")
            print(f"{'심볼':^12} | {'가격':^12} | {'방향':^10} | {'신뢰도':^8} | {'예상BPS':^10} | {'신호':^20}")
            print("-" * 90)

            debug_mode = (loop_count == 1)
            for symbol in MODELS.keys():
                result = predict_vwap(symbol, debug=debug_mode)

                if "error" in result:
                    print(
                        f"{symbol:^12} | {'N/A':^12} | {'오류':^10} | {'N/A':^8} | {'N/A':^10} | ❌ {result.get('error', '알 수 없음')}")
                    continue

                direction = result["direction"]
                confidence = result["confidence"]
                expected_bps = result["expected_bps"]
                price = result["current_price"]

                dir_icon = {"Long": "📈", "Short": "📉"}.get(direction, "❓")

                # 신호 판단
                if confidence < CONF_THRESHOLD:
                    signal = f"⚠️  신호 약함"
                elif direction == "Long":
                    signal = f"🟢 매수 신호"
                else:
                    signal = f"🔴 매도 신호"

                print(f"{symbol:^12} | ${price:>10,.4f} | {dir_icon} {direction:^8} | "
                      f"{confidence:>6.1%} | {expected_bps:>8.1f} | {signal}")

                # 진입 조건
                if account.can_open_position(symbol) and confidence >= CONF_THRESHOLD:
                    account.open_position(symbol, direction, price, expected_bps, confidence)

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