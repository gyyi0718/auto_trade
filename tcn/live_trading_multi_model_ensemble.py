# live_trading.py
# -*- coding: utf-8 -*-
"""
TCN 모델 기반 실전 자동 트레이딩 시스템 (심볼별 모델 사용)
⚠️  WARNING: 실제 자금을 사용합니다. 신중하게 사용하세요!
"""
import os
import time
import hmac
import hashlib
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
# API 인증 (필수!)
API_KEY = "Dlp4eJD6YFmO99T8vC"
API_SECRET = "YYYB5tMw2TWvfVF5wqi6lQRHqEIiDSpDJF1U"
USE_TESTNET = os.getenv("USE_TESTNET", "0") == "1"  # 기본값: Testnet

if not API_KEY or not API_SECRET:
    print("❌ ERROR: BYBIT_API_KEY 및 BYBIT_API_SECRET 환경변수를 설정하세요!")
    print("\n설정 방법 (Windows PowerShell):")
    print("  $env:BYBIT_API_KEY='your_api_key'")
    print("  $env:BYBIT_API_SECRET='your_api_secret'")
    print("  $env:USE_TESTNET='1'  # Testnet (기본값)")
    print("  $env:USE_TESTNET='0'  # Mainnet (실전)")
    print("\n설정 방법 (Linux/Mac):")
    print("  export BYBIT_API_KEY='your_api_key'")
    print("  export BYBIT_API_SECRET='your_api_secret'")
    print("  export USE_TESTNET=1  # Testnet (기본값)")
    print("  export USE_TESTNET=0  # Mainnet (실전)")
    print("\n포지션 모드 설정:")
    print("  $env:POSITION_MODE='hedge'     # Hedge Mode (양방향)")
    print("  $env:POSITION_MODE='one-way'   # One-Way Mode (단방향, 기본값)")
    print("\n⚠️  중요:")
    print("  - Testnet API Key: https://testnet.bybit.com 에서 발급")
    print("  - Mainnet API Key: https://www.bybit.com 에서 발급")
    print("  - Testnet과 Mainnet API Key는 서로 다릅니다!")
    print("  - API 권한: Contract Trading, Account Transfer 필수")
    print("  - 포지션 모드는 Bybit 웹사이트 설정과 일치해야 합니다!")
    exit(1)

# 거래 설정
SYMBOLS_ENV = os.getenv("SYMBOLS", "").strip()

INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "2"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.5"))  # 실전은 더 높게

# ✅ 심볼별 모델 경로 설정 (딕셔너리 형태)
MODEL_PATHS = {
    "BTCUSDT": "D:/ygy_work/coin/multimodel/models_5min_btc/5min_2class_best.ckpt",
    "ETHUSDT": "D:/ygy_work/coin/multimodel/models_5min_eth/5min_2class_best.ckpt",
    "SOLUSDT": "D:/ygy_work/coin/multimodel/models_5min_sol/5min_2class_best.ckpt",
    "DOGEUSDT": "D:/ygy_work/coin/multimodel/models_5min_doge/5min_2class_best.ckpt",
    "BNBUSDT": "D:/ygy_work/coin/multimodel/models_5min_bnb/5min_2class_best.ckpt",
    "XRPUSDT": "D:/ygy_work/coin/multimodel/models_5min_xrp/5min_2class_best.ckpt",
    "SAPIENUSDT": "D:/ygy_work/coin/multimodel/models_5min_sapien/5min_2class_best.ckpt",
    "FLMUSDT": "D:/ygy_work/coin/multimodel/models_5min_flm_v2/model_v2_best.pt",
    "TRUMPUSDT": "D:/ygy_work/coin/multimodel/models_5min_trump/5min_2class_best.ckpt",
    "JELLYJELLYUSDT": "D:/ygy_work/coin/multimodel/models_minutes_jellyjelly/5min_2class_best.ckpt",
    "ARCUSDT": "D:/ygy_work/coin/multimodel/models_5min_arc/5min_2class_best.ckpt",
    "DASHUSDT": "D:/ygy_work/coin/multimodel/models_5min_dash/5min_2class_best.ckpt",
    "MMTUSDT": "D:/ygy_work/coin/multimodel/models_5min_mmt/5min_2class_best.ckpt",
    "AIAUSDT": "D:/ygy_work/coin/multimodel/models_5min_aia/5min_2class_best.ckpt",
    "GIGGLEUSDT": "D:/ygy_work/coin/multimodel/models_5min_giggle/5min_2class_best.ckpt",
    "XNOUSDT": "D:/ygy_work/coin/multimodel/models_5min_xno/5min_2class_best.ckpt",
    "SOONUSDT": "D:/ygy_work/coin/multimodel/models_5min_soon/5min_2class_best.ckpt",
    "FLUXUSDT": "D:/ygy_work/coin/multimodel/models_improved/flux",
    "HUSDT": "D:/ygy_work/coin/multimodel/models_improved/h",
    "GIGGLEUSDT": "D:/ygy_work/coin/multimodel/models_improved/giggle"

}
if SYMBOLS_ENV:
    # 환경변수가 설정되어 있으면 사용
    SYMBOLS = [s.strip() for s in SYMBOLS_ENV.split(",") if s.strip()]
else:
    # 환경변수가 없으면 MODEL_PATHS의 모든 심볼 사용
    SYMBOLS = list(MODEL_PATHS.keys())
# 리스크 관리 (매우 중요!)
MARGIN_PER_POSITION = float(os.getenv("MARGIN_PER_POSITION", "100"))  # 포지션당 증거금
LEVERAGE = int(os.getenv("LEVERAGE", "20"))  # 레버리지 (실전은 낮게!)
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))  # 최대 동시 포지션
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))  # 손절 2%
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.03"))  # 익절 3%
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "500"))  # 일일 최대 손실
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "30"))  # 최대 보유 시간
MIN_HOLD_MINUTES = int(os.getenv("MIN_HOLD_MINUTES", "5"))  # 최소 보유 시간
LIQUIDATION_BUFFER = float(os.getenv("LIQUIDATION_BUFFER", "0.8"))  # 청산 버퍼 (80%)

# 포지션 모드
POSITION_MODE = os.getenv("POSITION_MODE", "one-way").lower()  # "one-way" 또는 "hedge"

# 로그
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "live_trades.json")
ORDER_LOG_FILE = os.getenv("ORDER_LOG_FILE", "orders.json")
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"


# ===== 데이터 클래스 =====
@dataclass
class Position:
    """포지션 정보 (개선된 버전)"""
    symbol: str
    direction: str  # "Long" or "Short"
    entry_price: float
    size: float  # 수량
    entry_time: datetime
    stop_loss: float
    take_profit: float
    leverage: int
    margin: float  # 실제 사용한 증거금
    liquidation_price: float  # 청산가
    unrealized_pnl: float = 0.0  # 평가 손익
    position_value: float = 0.0  # 포지션 가치

    def get_pnl(self, current_price: float) -> float:
        """손익 계산 (레버리지 적용)"""
        if self.direction == "Long":
            return (current_price - self.entry_price) * self.size
        else:  # Short
            return (self.entry_price - current_price) * self.size

    def get_pnl_pct(self, current_price: float) -> float:
        """손익률 계산 (증거금 기준)"""
        pnl = self.get_pnl(current_price)
        return (pnl / self.margin) * 100 if self.margin > 0 else 0

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
    size: float
    leverage: int
    margin: float
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float
    roe: float
    exit_reason: str


# ===== TCN 모델 정의 =====

# ===== 기존 TCN 모델 (단일 모델용) =====
class Chomp1d_Basic(nn.Module):
    """기본 Chomp1d (기존 모델용)"""

    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):
        return x[:, :, :-self.c].contiguous() if self.c > 0 else x


def wconv(i, o, k, d):
    """Weight Normalized Convolution"""
    pad = (k - 1) * d
    return nn.utils.weight_norm(nn.Conv1d(i, o, k, padding=pad, dilation=d))


class Block(nn.Module):
    """기본 TCN Block (기존 모델용)"""

    def __init__(self, i, o, k, d, drop):
        super().__init__()
        self.c1 = wconv(i, o, k, d)
        self.h1 = Chomp1d_Basic((k - 1) * d)
        self.r1 = nn.ReLU()
        self.dr1 = nn.Dropout(drop)
        self.c2 = wconv(o, o, k, d)
        self.h2 = Chomp1d_Basic((k - 1) * d)
        self.r2 = nn.ReLU()
        self.dr2 = nn.Dropout(drop)
        self.ds = nn.Conv1d(i, o, 1) if i != o else None
        self.r = nn.ReLU()

    def forward(self, x):
        y = self.dr1(self.r1(self.h1(self.c1(x))))
        y = self.dr2(self.r2(self.h2(self.c2(y))))
        res = x if self.ds is None else self.ds(x)
        return self.r(y + res)


# ===== 개선된 TCN 모델 (앙상블용) =====
class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        squeeze = x.mean(dim=2)
        excitation = self.fc(squeeze).unsqueeze(2)
        return x * excitation


class Chomp1d(nn.Module):
    """Chomp1d (개선된 모델용)"""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class EnhancedTCNBlock(nn.Module):
    """TCN Block with SE and Layer Norm"""

    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        padding = (kernel - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=padding)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=padding)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.se = SqueezeExcitation(out_ch, reduction=4)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.se(out)
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(out + res)

        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)

        return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""

    def __init__(self, hidden, num_heads, dropout):
        super().__init__()
        if hidden % num_heads != 0:
            for h in range(num_heads, 0, -1):
                if hidden % h == 0:
                    num_heads = h
                    break

        self.attn = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))


class ImprovedTCN(nn.Module):
    """개선된 TCN with SE, Attention, Uncertainty"""

    def __init__(self, in_f, hidden=64, levels=4, dropout=0.3, num_heads=4, num_classes=2):
        super().__init__()

        if hidden % num_heads != 0:
            hidden = max(num_heads, (hidden // num_heads) * num_heads)

        self.input_proj = nn.Sequential(
            nn.Linear(in_f, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        self.tcn_blocks = nn.ModuleList()
        ch = hidden
        for i in range(levels):
            self.tcn_blocks.append(
                EnhancedTCNBlock(ch, hidden, 3, 2 ** i, dropout)
            )
            ch = hidden

        self.scale_weights = nn.Parameter(torch.ones(levels))
        self.attention = MultiHeadAttention(hidden, num_heads, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, num_classes)
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_uncertainty=False):
        batch_size, seq_len, _ = x.size()

        x_proj = self.input_proj(x)

        x_tcn = x_proj.transpose(1, 2)
        tcn_features = []
        for block in self.tcn_blocks:
            x_tcn = block(x_tcn)
            tcn_features.append(x_tcn[:, :, -1])

        weights = torch.nn.functional.softmax(self.scale_weights, dim=0)
        x_tcn_fused = sum(w * f for w, f in zip(weights, tcn_features))

        x_attn = self.attention(x_proj)
        x_attn = x_attn.mean(dim=1)

        x_fused = torch.cat([x_tcn_fused, x_attn], dim=1)
        x_fused = self.fusion(x_fused)

        logits = self.classifier(x_fused)

        if return_uncertainty:
            uncertainty = self.uncertainty_head(x_fused)
            return logits, uncertainty

        return logits


class ModelEnsemble:
    """앙상블 모델 래퍼"""

    def __init__(self, models):
        self.models = models

    def predict(self, x, use_tta=False, n_tta=5):
        """앙상블 예측"""
        device = x.device
        all_preds = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                if use_tta:
                    tta_preds = []
                    for _ in range(n_tta):
                        noise = torch.randn_like(x) * 0.01
                        pred = model(x + noise)
                        tta_preds.append(torch.softmax(pred, dim=1))
                    pred = torch.stack(tta_preds).mean(0)
                else:
                    pred = torch.softmax(model(x), dim=1)
                all_preds.append(pred)

        # 평균
        ensemble_pred = torch.stack(all_preds).mean(0)

        # 불확실성 (표준편차)
        uncertainty = torch.stack(all_preds).std(0).max(1)[0]

        return ensemble_pred, uncertainty


# ===== 모델 로드 =====
print("\n🤖 심볼별 모델 로드 중...")
MODELS = {}
MODEL_CONFIGS = {}

for symbol in SYMBOLS:
    symbol = symbol.strip()
    if symbol not in MODEL_PATHS:
        print(f"⚠️  {symbol}: 모델 경로 미설정")
        continue

    model_path_or_dir = MODEL_PATHS[symbol]

    # 디렉토리인지 파일인지 확인
    if os.path.isdir(model_path_or_dir):
        # ===== 디렉토리 - 앙상블 방식 =====
        model_dir = model_path_or_dir
        meta_path = os.path.join(model_dir, "meta.json")

        if not os.path.exists(meta_path):
            print(f"❌ {symbol}: meta.json 파일 없음 - {meta_path}")
            continue

        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            feat_cols = meta['feat_cols']
            seq_len = meta['seq_len']
            scaler_mu = np.array(meta['scaler_mu'], dtype=np.float32)
            scaler_sd = np.array(meta['scaler_sd'], dtype=np.float32)
            n_models = meta['n_models']
            hidden = meta['hidden']
            levels = meta['levels']
            dropout = meta['dropout']
            num_classes = 2  # train_tcn_improved.py는 2-class

            # 앙상블 모델들 로드
            models = []
            for i in range(n_models):
                model_file = os.path.join(model_dir, f"model_{i + 1}.pt")
                if not os.path.exists(model_file):
                    print(f"⚠️  {symbol}: 모델 파일 없음 - {model_file}")
                    continue

                # 각 모델의 실제 hidden size 추출
                state_dict = torch.load(model_file, map_location="cpu", weights_only=True)

                if 'input_proj.0.weight' in state_dict:
                    actual_hidden = state_dict['input_proj.0.weight'].shape[0]
                else:
                    print(f"❌ {symbol} model_{i + 1}: 로드 실패")
                    continue

                model = ImprovedTCN(
                    in_f=len(feat_cols),
                    hidden=actual_hidden,  # ← 실제 값 사용!
                    levels=levels,
                    dropout=dropout,
                    num_heads=4,
                    num_classes=num_classes
                )
                model.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
                model.eval()
                models.append(model)

            if len(models) == 0:
                print(f"❌ {symbol}: 앙상블 모델 로드 실패")
                continue

            # 앙상블 래퍼
            ensemble = ModelEnsemble(models)

            MODELS[symbol] = ensemble
            MODEL_CONFIGS[symbol] = {
                'feat_cols': feat_cols,
                'seq_len': seq_len,
                'scaler_mu': scaler_mu,
                'scaler_sd': scaler_sd,
                'is_single_task': True,
                'num_classes': num_classes,
                'is_ensemble': True,
                'n_models': len(models)
            }

            print(f"✅ {symbol}: 앙상블 모델 로드 완료 "
                  f"({len(models)}개 모델, {num_classes}-class, levels={levels}, hidden={hidden})")

        except Exception as e:
            import traceback

            print(f"❌ {symbol}: 앙상블 모델 로드 실패 - {e}")
            if DEBUG_MODE:
                traceback.print_exc()

    elif os.path.isfile(model_path_or_dir):
        # ===== 파일 - 기존 단일 모델 방식 =====
        model_path = model_path_or_dir

        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            feat_cols = checkpoint['feat_cols']
            meta = checkpoint.get('meta', {})
            seq_len = meta.get('seq_len', 60)
            scaler_mu = checkpoint.get('scaler_mu')
            scaler_sd = checkpoint.get('scaler_sd')

            model_dict = checkpoint['model']

            # 모델 구조 감지
            max_layer = max(
                [int(k.split('.')[1]) for k in model_dict.keys()
                 if k.startswith('tcn.') and len(k.split('.')) > 2],
                default=0)
            levels = max_layer + 1
            hidden = model_dict['tcn.0.c1.weight_v'].shape[0] if 'tcn.0.c1.weight_v' in model_dict else 32
            k = model_dict['tcn.0.c1.weight_v'].shape[2] if 'tcn.0.c1.weight_v' in model_dict else 3
            drop = meta.get('dropout', 0.2)

            # 모델 타입 및 클래스 수 감지
            if 'head.weight' in model_dict:
                # Single-task 모델
                num_classes = model_dict['head.weight'].shape[0]


                class TCN_SingleTask(nn.Module):
                    def __init__(self, in_f, hidden, levels, k, drop, num_classes):
                        super().__init__()
                        L = []
                        ch = in_f
                        for i in range(levels):
                            L.append(Block(ch, hidden, k, 2 ** i, drop))
                            ch = hidden
                        self.tcn = nn.Sequential(*L)
                        self.head = nn.Linear(hidden, num_classes)

                    def forward(self, X):
                        X = X.transpose(1, 2)
                        H = self.tcn(X)[:, :, -1]
                        return self.head(H)


                model = TCN_SingleTask(len(feat_cols), hidden, levels, k, drop, num_classes)

            elif 'head_cls.weight' in model_dict:
                # Multi-task 모델
                num_classes = model_dict['head_cls.weight'].shape[0]


                class TCN_MT(nn.Module):
                    def __init__(self, in_f, hidden, levels, k, drop, num_classes):
                        super().__init__()
                        L = []
                        ch = in_f
                        for i in range(levels):
                            L.append(Block(ch, hidden, k, 2 ** i, drop))
                            ch = hidden
                        self.tcn = nn.Sequential(*L)
                        self.head_cls = nn.Linear(hidden, num_classes)
                        self.head_ttt = nn.Linear(hidden, 1)

                    def forward(self, X):
                        X = X.transpose(1, 2)
                        H = self.tcn(X)[:, :, -1]
                        return self.head_cls(H), self.head_ttt(H)


                model = TCN_MT(len(feat_cols), hidden, levels, k, drop, num_classes)
            else:
                # 기본값 (3-class Multi-task)
                num_classes = 3


                class TCN_MT(nn.Module):
                    def __init__(self, in_f, hidden, levels, k, drop):
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


                model = TCN_MT(len(feat_cols), hidden, levels, k, drop)

            model.load_state_dict(model_dict)
            model.eval()

            MODELS[symbol] = model
            MODEL_CONFIGS[symbol] = {
                'feat_cols': feat_cols,
                'seq_len': seq_len,
                'scaler_mu': scaler_mu,
                'scaler_sd': scaler_sd,
                'is_single_task': 'head.weight' in model_dict,
                'num_classes': num_classes,
                'is_ensemble': False
            }

            class_type = f"{num_classes}-class"
            task_type = 'Single-task' if MODEL_CONFIGS[symbol]['is_single_task'] else 'Multi-task'
            print(f"✅ {symbol}: {task_type} {class_type} 모델 로드 완료 "
                  f"(levels={levels}, hidden={hidden}, k={k})")

        except Exception as e:
            import traceback

            print(f"❌ {symbol}: 모델 로드 실패 - {e}")
            if DEBUG_MODE:
                traceback.print_exc()

    else:
        print(f"❌ {symbol}: 경로가 유효하지 않음 - {model_path_or_dir}")
        print(f"   디렉토리 존재: {os.path.isdir(model_path_or_dir)}")
        print(f"   파일 존재: {os.path.isfile(model_path_or_dir)}")

if not MODELS:
    print("\n❌ ERROR: 로드된 모델이 없습니다!")
    print("확인 사항:")
    print("   1. MODEL_PATHS에 심볼별 모델 경로가 올바르게 설정되어 있는지 확인")
    print("   2. 모델 파일/디렉토리가 실제로 존재하는지 확인")
    print("   3. SYMBOLS 환경변수에 설정된 심볼과 MODEL_PATHS의 키가 일치하는지 확인")
    print(f"\n현재 설정:")
    print(f"   SYMBOLS 환경변수: {SYMBOLS_ENV if SYMBOLS_ENV else '(미설정 - 모든 모델 사용)'}")
    print(f"   시도한 심볼: {', '.join(SYMBOLS)}")
    print(f"   MODEL_PATHS에 있는 심볼: {', '.join(MODEL_PATHS.keys())}")
    exit(1)

print(f"\n✅ 총 {len(MODELS)}개 모델 로드 완료")
print(f"   사용 가능 심볼: {', '.join(MODELS.keys())}")

# 환경변수에 설정된 심볼 중 로드 실패한 것들 안내
failed_symbols = [s for s in SYMBOLS if s not in MODELS]
if failed_symbols:
    print(f"\n⚠️  로드 실패한 심볼 ({len(failed_symbols)}개): {', '.join(failed_symbols)}")

# ===== 모델 로드 (앙상블 각 모델의 hidden size 자동 추출) =====
print("\n🤖 심볼별 모델 로드 중...")
MODELS = {}
MODEL_CONFIGS = {}

for symbol in SYMBOLS:
    symbol = symbol.strip()
    if symbol not in MODEL_PATHS:
        print(f"⚠️  {symbol}: 모델 경로 미설정")
        continue

    model_path_or_dir = MODEL_PATHS[symbol]

    # 디렉토리인지 파일인지 확인
    if os.path.isdir(model_path_or_dir):
        # ===== 디렉토리 - 앙상블 방식 =====
        model_dir = model_path_or_dir
        meta_path = os.path.join(model_dir, "meta.json")

        if not os.path.exists(meta_path):
            print(f"❌ {symbol}: meta.json 파일 없음 - {meta_path}")
            continue

        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            feat_cols = meta['feat_cols']
            seq_len = meta['seq_len']
            scaler_mu = np.array(meta['scaler_mu'], dtype=np.float32)
            scaler_sd = np.array(meta['scaler_sd'], dtype=np.float32)
            n_models = meta['n_models']
            # hidden은 meta에서 읽지 않음 - 각 모델에서 추출
            levels = meta['levels']
            dropout = meta['dropout']
            num_classes = 2  # train_tcn_improved.py는 2-class

            # ✅ 앙상블 모델들 로드 (각 모델의 hidden size 자동 추출)
            models = []
            for i in range(n_models):
                model_file = os.path.join(model_dir, f"model_{i + 1}.pt")
                if not os.path.exists(model_file):
                    print(f"⚠️  {symbol}: 모델 파일 없음 - {model_file}")
                    continue

                # 🔧 각 모델 파일에서 실제 hidden size 추출
                state_dict = torch.load(model_file, map_location="cpu", weights_only=True)

                if 'input_proj.0.weight' in state_dict:
                    actual_hidden, actual_in_features = state_dict['input_proj.0.weight'].shape

                    # Feature 개수 검증
                    if actual_in_features != len(feat_cols):
                        print(f"⚠️  {symbol} model_{i + 1}.pt: Feature 개수 불일치 "
                              f"(모델={actual_in_features}, meta={len(feat_cols)})")
                        continue

                    if DEBUG_MODE:
                        print(f"   📊 model_{i + 1}.pt: hidden={actual_hidden}, features={actual_in_features}")
                else:
                    print(f"❌ {symbol} model_{i + 1}.pt: input_proj.0.weight 없음")
                    continue

                # 실제 hidden size로 모델 생성
                model = ImprovedTCN(
                    in_f=len(feat_cols),
                    hidden=actual_hidden,  # ✅ 실제 값 사용!
                    levels=levels,
                    dropout=dropout,
                    num_heads=4,
                    num_classes=num_classes
                )
                model.load_state_dict(state_dict)
                model.eval()
                models.append(model)

            if len(models) == 0:
                print(f"❌ {symbol}: 앙상블 모델 로드 실패")
                continue

            # 앙상블 래퍼
            ensemble = ModelEnsemble(models)

            MODELS[symbol] = ensemble
            MODEL_CONFIGS[symbol] = {
                'feat_cols': feat_cols,
                'seq_len': seq_len,
                'scaler_mu': scaler_mu,
                'scaler_sd': scaler_sd,
                'is_single_task': True,
                'num_classes': num_classes,
                'is_ensemble': True,
                'n_models': len(models)
            }

            print(f"✅ {symbol}: 앙상블 모델 로드 완료 "
                  f"({len(models)}개 모델, {num_classes}-class, levels={levels})")

        except Exception as e:
            import traceback

            print(f"❌ {symbol}: 앙상블 모델 로드 실패 - {e}")
            if DEBUG_MODE:
                traceback.print_exc()

    elif os.path.isfile(model_path_or_dir):
        # ===== 파일 - 기존 단일 모델 방식 =====
        model_path = model_path_or_dir

        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            feat_cols = checkpoint['feat_cols']
            meta = checkpoint.get('meta', {})
            seq_len = meta.get('seq_len', 60)
            scaler_mu = checkpoint.get('scaler_mu')
            scaler_sd = checkpoint.get('scaler_sd')

            model_dict = checkpoint['model']

            # 모델 구조 감지
            max_layer = max(
                [int(k.split('.')[1]) for k in model_dict.keys()
                 if k.startswith('tcn.') and len(k.split('.')) > 2],
                default=0)
            levels = max_layer + 1
            hidden = model_dict['tcn.0.c1.weight_v'].shape[0] if 'tcn.0.c1.weight_v' in model_dict else 32
            k = model_dict['tcn.0.c1.weight_v'].shape[2] if 'tcn.0.c1.weight_v' in model_dict else 3
            drop = meta.get('dropout', 0.2)

            # 모델 타입 및 클래스 수 감지
            if 'head.weight' in model_dict:
                # Single-task 모델
                num_classes = model_dict['head.weight'].shape[0]


                class TCN_SingleTask(nn.Module):
                    def __init__(self, in_f, hidden, levels, k, drop, num_classes):
                        super().__init__()
                        L = []
                        ch = in_f
                        for i in range(levels):
                            L.append(Block(ch, hidden, k, 2 ** i, drop))
                            ch = hidden
                        self.tcn = nn.Sequential(*L)
                        self.head = nn.Linear(hidden, num_classes)

                    def forward(self, X):
                        X = X.transpose(1, 2)
                        H = self.tcn(X)[:, :, -1]
                        return self.head(H)


                model = TCN_SingleTask(len(feat_cols), hidden, levels, k, drop, num_classes)

            elif 'head_cls.weight' in model_dict:
                # Multi-task 모델
                num_classes = model_dict['head_cls.weight'].shape[0]


                class TCN_MT(nn.Module):
                    def __init__(self, in_f, hidden, levels, k, drop, num_classes):
                        super().__init__()
                        L = []
                        ch = in_f
                        for i in range(levels):
                            L.append(Block(ch, hidden, k, 2 ** i, drop))
                            ch = hidden
                        self.tcn = nn.Sequential(*L)
                        self.head_cls = nn.Linear(hidden, num_classes)
                        self.head_ttt = nn.Linear(hidden, 1)

                    def forward(self, X):
                        X = X.transpose(1, 2)
                        H = self.tcn(X)[:, :, -1]
                        return self.head_cls(H), self.head_ttt(H)


                model = TCN_MT(len(feat_cols), hidden, levels, k, drop, num_classes)
            else:
                # 기본값 (3-class Multi-task)
                num_classes = 3


                class TCN_MT(nn.Module):
                    def __init__(self, in_f, hidden, levels, k, drop):
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


                model = TCN_MT(len(feat_cols), hidden, levels, k, drop)

            model.load_state_dict(model_dict)
            model.eval()

            MODELS[symbol] = model
            MODEL_CONFIGS[symbol] = {
                'feat_cols': feat_cols,
                'seq_len': seq_len,
                'scaler_mu': scaler_mu,
                'scaler_sd': scaler_sd,
                'is_single_task': 'head.weight' in model_dict,
                'num_classes': num_classes,
                'is_ensemble': False
            }

            class_type = f"{num_classes}-class"
            task_type = 'Single-task' if MODEL_CONFIGS[symbol]['is_single_task'] else 'Multi-task'
            print(f"✅ {symbol}: {task_type} {class_type} 모델 로드 완료 "
                  f"(levels={levels}, hidden={hidden}, k={k})")

        except Exception as e:
            import traceback

            print(f"❌ {symbol}: 모델 로드 실패 - {e}")
            if DEBUG_MODE:
                traceback.print_exc()

    else:
        print(f"❌ {symbol}: 경로가 유효하지 않음 - {model_path_or_dir}")
        print(f"   디렉토리 존재: {os.path.isdir(model_path_or_dir)}")
        print(f"   파일 존재: {os.path.isfile(model_path_or_dir)}")

if not MODELS:
    print("\n❌ ERROR: 로드된 모델이 없습니다!")
    print("확인 사항:")
    print("   1. MODEL_PATHS에 심볼별 모델 경로가 올바르게 설정되어 있는지 확인")
    print("   2. 모델 파일/디렉토리가 실제로 존재하는지 확인")
    print("   3. SYMBOLS 환경변수에 설정된 심볼과 MODEL_PATHS의 키가 일치하는지 확인")
    print(f"\n현재 설정:")
    print(f"   SYMBOLS 환경변수: {SYMBOLS_ENV if SYMBOLS_ENV else '(미설정 - 모든 모델 사용)'}")
    print(f"   시도한 심볼: {', '.join(SYMBOLS)}")
    print(f"   MODEL_PATHS에 있는 심볼: {', '.join(MODEL_PATHS.keys())}")
    exit(1)

print(f"\n✅ 총 {len(MODELS)}개 모델 로드 완료")
print(f"   사용 가능 심볼: {', '.join(MODELS.keys())}")

# 환경변수에 설정된 심볼 중 로드 실패한 것들 안내
failed_symbols = [s for s in SYMBOLS if s not in MODELS]
if failed_symbols:
    print(f"\n⚠️  로드 실패한 심볼 ({len(failed_symbols)}개): {', '.join(failed_symbols)}")


# ===== Bybit API =====
class BybitAPI:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._instrument_cache = {}  # 심볼 정보 캐시

        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"

        self.recv_window = "5000"

    def _generate_signature(self, timestamp: str, params_str: str) -> str:
        """서명 생성"""
        # signature = HMAC_SHA256(timestamp + api_key + recv_window + params_str)
        sign_str = timestamp + self.api_key + self.recv_window + params_str

        if DEBUG_MODE:
            print(f"[DEBUG] Sign string: {sign_str}")

        signature = hmac.new(
            bytes(self.api_secret, 'utf-8'),
            bytes(sign_str, 'utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature

    def _request(self, method: str, endpoint: str, params: dict = None, sign: bool = False) -> dict:
        """API 요청"""
        if params is None:
            params = {}

        headers = {
            "Content-Type": "application/json"
        }

        if sign:
            timestamp = str(int(time.time() * 1000))

            # 파라미터 문자열 생성
            if method == "GET":
                # GET: 쿼리 파라미터를 알파벳 순으로 정렬
                params_str = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())]) if params else ""
            else:
                # POST: JSON body
                params_str = json.dumps(params) if params else ""

            # 서명 생성
            signature = self._generate_signature(timestamp, params_str)

            # 헤더 설정
            headers.update({
                'X-BAPI-API-KEY': self.api_key,
                'X-BAPI-SIGN': signature,
                'X-BAPI-TIMESTAMP': timestamp,
                'X-BAPI-RECV-WINDOW': self.recv_window
            })

            if DEBUG_MODE:
                print(f"\n[DEBUG] {method} {endpoint}")
                print(f"[DEBUG] Timestamp: {timestamp}")
                print(f"[DEBUG] Params: {params}")
                print(f"[DEBUG] Params str: {params_str}")
                print(f"[DEBUG] Signature: {signature}")

        try:
            url = f"{self.base_url}{endpoint}"

            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")

            if DEBUG_MODE:
                print(f"[DEBUG] Status: {response.status_code}")
                print(f"[DEBUG] Response: {response.text[:500]}")

            data = response.json()
            return data

        except Exception as e:
            return {"retCode": -1, "retMsg": str(e)}

    def get_ticker(self, symbol: str) -> dict:
        """현재가 조회"""
        return self._request("GET", "/v5/market/tickers", {"category": "linear", "symbol": symbol})

    def get_klines(self, symbol: str, interval: str = "5", limit: int = 200) -> pd.DataFrame:
        """캔들 데이터 조회"""
        result = self._request("GET", "/v5/market/kline", {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        })

        if result.get("retCode") != 0:
            return pd.DataFrame()

        data = result["result"]["list"]
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df = df.astype({
            "timestamp": "int64",
            "open": "float",
            "high": "float",
            "low": "float",
            "close": "float",
            "volume": "float",
            "turnover": "float"
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.sort_values("timestamp").reset_index(drop=True)

    def get_balance(self) -> float:
        """USDT 잔고 조회"""
        result = self._request("GET", "/v5/account/wallet-balance", {
            "accountType": "UNIFIED"
        }, sign=True)

        if result.get("retCode") != 0:
            if DEBUG_MODE:
                print(f"[DEBUG] Balance API Error: {result}")
            return 0.0

        try:
            coins = result["result"]["list"][0]["coin"]
            usdt = next((c for c in coins if c["coin"] == "USDT"), None)
            return float(usdt["walletBalance"]) if usdt else 0.0
        except (KeyError, IndexError, TypeError) as e:
            if DEBUG_MODE:
                print(f"[DEBUG] Balance parsing error: {e}")
            return 0.0

    def get_positions(self) -> List[Position]:
        """포지션 조회 (개선된 버전)"""
        result = self._request("GET", "/v5/position/list", {"category": "linear", "settleCoin": "USDT"}, sign=True)

        if result.get("retCode") != 0:
            return []

        positions = []
        try:
            for item in result["result"]["list"]:
                size = float(item.get("size", 0))
                if size == 0:
                    continue

                side = item.get("side")
                direction = "Long" if side == "Buy" else "Short"
                entry_price = float(item.get("avgPrice", 0))
                symbol = item.get("symbol")
                leverage = int(item.get("leverage", LEVERAGE))
                unrealized_pnl = float(item.get("unrealisedPnl", 0))

                # 증거금 계산
                position_value = size * entry_price
                margin = position_value / leverage

                # 청산가 계산
                liq_price_str = item.get("liqPrice", "0")
                if liq_price_str and liq_price_str != "":
                    liquidation_price = float(liq_price_str)
                else:
                    # API에서 청산가를 제공하지 않으면 직접 계산
                    if direction == "Long":
                        liquidation_price = entry_price * (1 - (1 / leverage) * LIQUIDATION_BUFFER)
                    else:
                        liquidation_price = entry_price * (1 + (1 / leverage) * LIQUIDATION_BUFFER)

                # 손절/익절 가격 계산
                if direction == "Long":
                    stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                    take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
                else:
                    stop_loss = entry_price * (1 + STOP_LOSS_PCT)
                    take_profit = entry_price * (1 - TAKE_PROFIT_PCT)

                # 포지션 생성 시간 (API에서 제공하지 않으면 현재 시간 사용)
                created_time_str = item.get("createdTime", "")
                if created_time_str:
                    try:
                        # 밀리초 timestamp를 초 단위로 변환
                        created_timestamp = int(created_time_str)
                        if created_timestamp > 9999999999:  # 밀리초 단위
                            created_timestamp = created_timestamp / 1000
                        entry_time = datetime.fromtimestamp(created_timestamp)

                        if DEBUG_MODE:
                            print(f"[DEBUG] {symbol} createdTime: {created_time_str} -> {entry_time}")
                    except (ValueError, OSError) as e:
                        if DEBUG_MODE:
                            print(f"[DEBUG] {symbol} createdTime 파싱 실패: {e}")
                        entry_time = datetime.now()
                else:
                    entry_time = datetime.now()

                positions.append(Position(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    size=size,
                    entry_time=entry_time,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=leverage,
                    margin=margin,
                    liquidation_price=liquidation_price,
                    unrealized_pnl=unrealized_pnl,
                    position_value=position_value
                ))

        except Exception as e:
            print(f"❌ 포지션 파싱 오류: {e}")

        return positions

    def get_instrument_info(self, symbol: str) -> dict:
        """심볼 정보 조회 (최소 수량, 수량 단위 등) - 캐시 사용"""
        # 캐시 확인
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        result = self._request("GET", "/v5/market/instruments-info", {
            "category": "linear",
            "symbol": symbol
        })

        if result.get("retCode") != 0 or not result.get("result", {}).get("list"):
            # 기본값 반환
            default_info = {
                "minOrderQty": 0.01,
                "qtyStep": 0.01,
                "minNotionalValue": 5.0
            }
            self._instrument_cache[symbol] = default_info
            return default_info

        info = result["result"]["list"][0]
        lot_size_filter = info.get("lotSizeFilter", {})

        instrument_info = {
            "minOrderQty": float(lot_size_filter.get("minOrderQty", 0.01)),
            "qtyStep": float(lot_size_filter.get("qtyStep", 0.01)),
            "maxOrderQty": float(lot_size_filter.get("maxOrderQty", 1000000)),
            "minNotionalValue": float(lot_size_filter.get("minNotionalValue", 5.0))
        }

        # 캐시 저장
        self._instrument_cache[symbol] = instrument_info

        if DEBUG_MODE:
            print(f"ℹ️  {symbol} 정보: minQty={instrument_info['minOrderQty']}, "
                  f"qtyStep={instrument_info['qtyStep']}, "
                  f"minNotional=${instrument_info['minNotionalValue']}")

        return instrument_info

    def set_trading_stop(self, symbol: str, side: str, stop_loss: float = None,
                         take_profit: float = None) -> tuple[bool, str]:
        """TP/SL 설정"""
        params = {
            "category": "linear",
            "symbol": symbol,
            "positionIdx": 0  # One-way mode
        }

        if POSITION_MODE == "hedge":
            params["positionIdx"] = 1 if side == "Buy" else 2

        if stop_loss:
            params["stopLoss"] = str(stop_loss)

        if take_profit:
            params["takeProfit"] = str(take_profit)

        if not stop_loss and not take_profit:
            return False, "TP 또는 SL이 필요합니다"

        result = self._request("POST", "/v5/position/trading-stop", params, sign=True)

        success = result.get("retCode") == 0
        msg = result.get("retMsg", "알 수 없는 오류")

        return success, msg

    def set_leverage(self, symbol: str, leverage: int) -> tuple[bool, str]:
        """레버리지 설정"""
        result = self._request("POST", "/v5/position/set-leverage", {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        }, sign=True)

        success = result.get("retCode") == 0
        msg = result.get("retMsg", "알 수 없는 오류")

        return success, msg

    def place_order(self, symbol: str, side: str, qty: float, order_type: str = "Market",
                    price: Optional[float] = None, reduce_only: bool = False) -> Optional[str]:
        """주문 생성"""
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": "GTC"  # Good Till Cancel
        }

        if reduce_only:
            params["reduceOnly"] = True  # boolean 타입

        if order_type == "Limit" and price:
            params["price"] = str(price)

        if POSITION_MODE == "hedge":
            params["positionIdx"] = 1 if side == "Buy" else 2
        else:
            params["positionIdx"] = 0  # One-way mode

        result = self._request("POST", "/v5/order/create", params, sign=True)

        if result.get("retCode") == 0:
            order_id = result["result"]["orderId"]
            if DEBUG_MODE:
                print(f"✅ 주문 성공: {symbol} | {side} | {qty} | Order ID: {order_id}")
            return order_id
        else:
            ret_code = result.get("retCode")
            ret_msg = result.get("retMsg", "알 수 없는 오류")
            print(f"❌ 주문 실패: {symbol} | {side} | {qty}")
            print(f"   에러 코드: {ret_code} | 메시지: {ret_msg}")

            # 일반적인 에러 해석
            if "insufficient" in ret_msg.lower() or "balance" in ret_msg.lower():
                print(f"   💡 잔고 부족 - 사용 가능한 증거금을 확인하세요")
            elif "qty" in ret_msg.lower() or "size" in ret_msg.lower():
                print(f"   💡 수량 오류 - 최소/최대 수량 또는 수량 단위를 확인하세요")
            elif "leverage" in ret_msg.lower():
                print(f"   💡 레버리지 오류 - 허용된 레버리지 범위를 확인하세요")
            elif "risk limit" in ret_msg.lower():
                print(f"   💡 리스크 한도 초과 - 포지션 크기를 줄이거나 레버리지를 낮추세요")
            elif "sign" in ret_msg.lower():
                print(f"   💡 API 서명 오류 - API Key와 Secret을 확인하세요")
            elif "reduce" in ret_msg.lower() or "position" in ret_msg.lower():
                print(f"   💡 포지션 오류 - 청산할 포지션이 없거나 수량이 맞지 않습니다")

            return None


API = BybitAPI(API_KEY, API_SECRET, USE_TESTNET)


# ===== 포지션 관리자 =====
class PositionManager:
    """포지션 관리 클래스"""

    def __init__(self):
        self.positions_entry_time: Dict[str, datetime] = {}  # 심볼별 진입 시간 추적
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.daily_reset_time = datetime.now().date()

    def reset_daily_stats(self):
        """일일 통계 초기화"""
        current_date = datetime.now().date()
        if current_date != self.daily_reset_time:
            self.daily_pnl = 0.0
            self.daily_reset_time = current_date

    def can_open_position(self, symbol: str, direction: str, current_positions: List[Position]) -> bool:
        """포지션 진입 가능 여부 (같은 방향이면 유지, 반대 방향이면 청산 후 진입)"""
        # 같은 심볼의 기존 포지션 확인
        existing = next((p for p in current_positions if p.symbol == symbol), None)

        if existing:
            # 같은 방향이면 진입하지 않음 (기존 포지션 유지)
            if existing.direction == direction:
                if DEBUG_MODE:
                    print(f"ℹ️  {symbol}: 같은 방향 포지션 이미 보유 중 ({direction}) - 유지")
                return False
            # 반대 방향이면 진입 가능 (메인 루프에서 청산 처리)
            return True

        # 최대 포지션 수 확인
        if len(current_positions) >= MAX_POSITIONS:
            return False

        # 일일 손실 한도 확인
        if abs(self.daily_pnl) >= MAX_DAILY_LOSS:
            print(f"⚠️  일일 손실 한도 도달: ${self.daily_pnl:.2f}")
            return False

        return True

    def open_position(self, symbol: str, direction: str, price: float) -> bool:
        """포지션 진입"""
        try:
            # 가격 유효성 검사
            if price <= 0 or price is None or not np.isfinite(price):
                print(f"❌ {symbol}: 유효하지 않은 가격 (${price})")
                return False

            # 레버리지 설정 (실패 시 경고만 출력하고 계속 진행)
            success, msg = API.set_leverage(symbol, LEVERAGE)
            if not success:
                # 이미 설정되어 있거나 레버리지 변경이 필요없는 경우
                if "leverage not modified" in msg.lower() or "same" in msg.lower():
                    if DEBUG_MODE:
                        print(f"ℹ️  {symbol}: 레버리지 이미 설정됨")
                else:
                    print(f"⚠️  {symbol}: 레버리지 설정 실패 - {msg} (기존 레버리지 사용)")

            # 심볼 정보 조회
            instrument_info = API.get_instrument_info(symbol)
            min_qty = instrument_info["minOrderQty"]
            qty_step = instrument_info["qtyStep"]
            min_notional = instrument_info["minNotionalValue"]

            # 포지션 크기 계산
            position_value = MARGIN_PER_POSITION * LEVERAGE
            qty = position_value / price

            # 최소 주문 금액 확인
            notional_value = qty * price
            if notional_value < min_notional:
                print(f"❌ {symbol}: 주문 금액 부족 (${notional_value:.2f} < ${min_notional:.2f})")
                return False

            # 최소 수량 확인
            if qty < min_qty:
                print(f"❌ {symbol}: 수량 부족 ({qty:.4f} < {min_qty:.4f})")
                return False

            # ✅ 개선: qty를 step 단위로 올림 (반올림이 아닌 올림)
            # 이렇게 하면 반올림 후 최소 수량보다 작아지는 문제 방지
            import math
            steps = math.ceil(qty / qty_step)
            qty = steps * qty_step

            # ✅ 부동소수점 정밀도 문제 해결
            # qty_step의 소수점 자리수 계산
            step_str = f"{qty_step:.10f}".rstrip('0')
            if '.' in step_str:
                decimals = len(step_str.split('.')[1])
            else:
                decimals = 0

            # 해당 자리수로 반올림
            qty = round(qty, decimals)

            # 최종 qty 검증
            if qty < min_qty:
                print(f"❌ {symbol}: 반올림 후 수량 부족 ({qty:.4f} < {min_qty:.4f})")
                return False

            # 최종 주문 금액 재확인
            final_notional = qty * price
            if final_notional < min_notional:
                print(f"❌ {symbol}: 최종 주문 금액 부족 (${final_notional:.2f} < ${min_notional:.2f})")
                return False

            if DEBUG_MODE:
                print(f"\n[주문 상세] {symbol}")
                print(f"  포지션 크기: ${position_value:.2f}")
                print(f"  계산된 수량: {position_value / price:.8f}")
                print(f"  qtyStep: {qty_step}")
                print(f"  반올림 후: {qty}")
                print(f"  최종 금액: ${final_notional:.2f}")

            # TP/SL 가격 계산
            if direction == "Long":
                stop_loss = price * (1 - STOP_LOSS_PCT)
                take_profit = price * (1 + TAKE_PROFIT_PCT)
            else:  # Short
                stop_loss = price * (1 + STOP_LOSS_PCT)
                take_profit = price * (1 - TAKE_PROFIT_PCT)

            # 주문 생성
            side = "Buy" if direction == "Long" else "Sell"
            order_id = API.place_order(symbol, side, qty)

            if order_id:
                # 포지션 진입 성공 - TP/SL 설정
                time.sleep(0.5)  # 주문 체결 대기

                success, msg = API.set_trading_stop(symbol, side, stop_loss, take_profit)
                if success:
                    if DEBUG_MODE:
                        print(f"   ✅ TP/SL 설정 완료: SL=${stop_loss:,.4f}, TP=${take_profit:,.4f}")
                else:
                    print(f"   ⚠️  TP/SL 설정 실패: {msg}")
                    print(f"   수동 설정: SL=${stop_loss:,.4f}, TP=${take_profit:,.4f}")

                self.positions_entry_time[symbol] = datetime.now()

                print(f"🟢 포지션 진입: {symbol} | {direction} | ${price:,.4f} | 수량: {qty} | "
                      f"레버리지: {LEVERAGE}x | 증거금: ${MARGIN_PER_POSITION}")
                print(f"   📍 SL: ${stop_loss:,.4f} ({-STOP_LOSS_PCT * 100:.1f}%) | "
                      f"TP: ${take_profit:,.4f} ({TAKE_PROFIT_PCT * 100:.1f}%)")

                return True

        except Exception as e:
            print(f"❌ {symbol}: 포지션 진입 실패 - {e}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()

        return False

    def close_position(self, position: Position, reason: str) -> bool:
        """포지션 청산"""
        try:
            # 반대 방향으로 주문 (청산)
            side = "Sell" if position.direction == "Long" else "Buy"

            # 정확한 포지션 수량 사용
            qty = position.size

            if DEBUG_MODE:
                print(f"🔄 청산 시도: {position.symbol} | {side} | qty={qty} | reason={reason}")

            order_id = API.place_order(position.symbol, side, qty, reduce_only=True)

            if order_id:
                exit_time = datetime.now()
                entry_time_str = self.positions_entry_time.get(position.symbol, position.entry_time)
                if isinstance(entry_time_str, datetime):
                    entry_time_str = entry_time_str.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    entry_time_str = str(entry_time_str)

                # 현재가로 손익 계산
                ticker = API.get_ticker(position.symbol)
                if ticker.get("retCode") == 0 and ticker["result"]["list"]:
                    current_price = float(ticker["result"]["list"][0]["lastPrice"])
                else:
                    current_price = position.entry_price  # 조회 실패 시 진입가 사용

                pnl = position.get_pnl(current_price)
                pnl_pct = position.get_pnl_pct(current_price)
                roe = position.get_roe(current_price)

                # 거래 기록 저장
                trade = Trade(
                    symbol=position.symbol,
                    direction=position.direction,
                    entry_price=position.entry_price,
                    exit_price=current_price,
                    size=position.size,
                    leverage=position.leverage,
                    margin=position.margin,
                    entry_time=entry_time_str,
                    exit_time=exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    roe=roe,
                    exit_reason=reason
                )

                self.trades.append(trade)
                self.daily_pnl += pnl

                # 진입 시간 삭제
                if position.symbol in self.positions_entry_time:
                    del self.positions_entry_time[position.symbol]

                emoji = "🟢" if pnl > 0 else "🔴"
                print(f"{emoji} 포지션 청산: {position.symbol} | {position.direction} | "
                      f"진입: ${position.entry_price:,.4f} → 청산: ${current_price:,.4f} | "
                      f"손익: ${pnl:+,.2f} ({roe:+.1f}%) | 사유: {reason}")

                return True
            else:
                print(f"⚠️  {position.symbol}: 청산 주문 실패 - 수동으로 청산하세요")

        except Exception as e:
            print(f"❌ {position.symbol}: 포지션 청산 실패 - {e}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()

        return False

    def save_trades(self):
        """거래 내역 저장"""
        if not self.trades:
            return

        data = [asdict(t) for t in self.trades]
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n💾 거래 내역 저장: {TRADE_LOG_FILE}")

    def get_stats(self) -> dict:
        """거래 통계"""
        if not self.trades:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "avg_roe": 0,
                "max_pnl": 0,
                "min_pnl": 0,
                "max_roe": 0,
                "min_roe": 0,
                "liquidations": 0,
                "avg_win": 0,
                "avg_loss": 0
            }

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        liquidations = [t for t in self.trades if t.exit_reason == "Liquidation"]

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(self.trades) * 100) if self.trades else 0,
            "avg_pnl": sum(t.pnl for t in self.trades) / len(self.trades),
            "avg_roe": sum(t.roe for t in self.trades) / len(self.trades),
            # 최대 수익: 승리한 거래에서만 계산
            "max_pnl": max(t.pnl for t in wins) if wins else 0,
            # 최대 손실: 손실 거래에서만 계산
            "min_pnl": min(t.pnl for t in losses) if losses else 0,
            # 최대 ROE: 승리한 거래에서만 계산
            "max_roe": max(t.roe for t in wins) if wins else 0,
            # 최소 ROE: 손실 거래에서만 계산
            "min_roe": min(t.roe for t in losses) if losses else 0,
            "avg_win": sum(t.pnl for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t.pnl for t in losses) / len(losses) if losses else 0,
            "liquidations": len(liquidations)
        }


manager = PositionManager()


# ===== 예측 함수 =====
def predict(symbol: str, debug: bool = False) -> dict:
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

    # 데이터 가져오기 (동적으로 필요한 만큼 요청)
    # seq_len + 최대 window(120) + 여유(40) = 충분한 데이터
    max_window = 120  # rv120, mom120, vz120 등에서 사용
    required_length = config['seq_len'] + max_window + 40
    limit = max(200, min(required_length, 1000))  # 최소 200, 최대 1000
    df = API.get_klines(symbol, interval="5", limit=limit)

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
        # 숫자형 컬럼만 선택해서 안전하게 체크
        try:
            numeric_df = df_feat.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 0:
                print(f"  - Inf 개수: {np.isinf(numeric_df.values).sum()}")
                print(f"  - 최솟값: {numeric_df.min().min():.4f}")
                print(f"  - 최댓값: {numeric_df.max().max():.4f}")
                print(f"  - 평균: {numeric_df.mean().mean():.4f}")
            else:
                print(f"  - ⚠️ 숫자형 컬럼이 없습니다")
        except Exception as e:
            print(f"  - ⚠️ 통계 계산 오류: {e}")

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
        # 앙상블 모델 vs 단일 모델 구분
        is_ensemble = config.get('is_ensemble', False)

        if is_ensemble:
            # 앙상블 모델: predict 메서드 사용
            probs_tensor, uncertainty = model.predict(X_tensor, use_tta=False)
            probs = probs_tensor.squeeze().numpy()
            logits = None  # 앙상블은 이미 softmax된 확률 반환
        elif config['num_classes'] == 2:
            # 단일 모델 - Single-task (2-class)
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
        else:
            # 단일 모델 - Multi-task (3-class)
            logits_cls, _ = model(X_tensor)
            probs = torch.softmax(logits_cls, dim=1).squeeze().numpy()
            logits = logits_cls

    if debug:
        print(f"\n[DEBUG {symbol}] 모델 출력:")
        is_ensemble = config.get('is_ensemble', False)

        if is_ensemble:
            print(f"  - 앙상블 모델 (이미 softmax 적용)")
            if 'uncertainty' in locals():
                print(f"  - uncertainty: {uncertainty.item():.6f}")
        elif config['num_classes'] == 2:
            if logits is not None:
                print(f"  - logits: [{logits.squeeze()[0]:.4f}, {logits.squeeze()[1]:.4f}]")
        else:
            if logits is not None:
                print(f"  - logits: [{logits.squeeze()[0]:.4f}, {logits.squeeze()[1]:.4f}, {logits.squeeze()[2]:.4f}]")

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

    # Live API 응답 구조에 맞게 가격 가져오기
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

    if debug:
        print(f"⚠️  {symbol}: 가격 조회 실패")

    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "current_price": current_price,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def calculate_rsi(series, period=14):
    """RSI 계산"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(series, fast=12, slow=26, signal=9):
    """MACD 계산"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return (macd - macd_signal).fillna(0)


def generate_features(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """
    train_tcn_improved.py의 add_advanced_features와 완전히 동일한 로직
    """
    g = df.copy()

    # symbol 컬럼이 없으면 추가 (단일 심볼 처리)
    if 'symbol' not in g.columns:
        g['symbol'] = 'TEMP'

    # date 컬럼 처리
    if 'date' not in g.columns and 'timestamp' in g.columns:
        g['date'] = pd.to_datetime(g['timestamp'])
    elif 'date' in g.columns:
        g['date'] = pd.to_datetime(g['date'])
    else:
        g['date'] = pd.to_datetime(df.index)

    g = g.sort_values(["symbol", "date"]).reset_index(drop=True)

    # ===== 1. 기본 수익률 =====
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g.groupby("symbol")["logc"].diff().fillna(0.0)

    # ===== 2. 시장 미세구조 =====
    g['spread'] = (g['high'] - g['low']) / (g['close'] + 1e-8)
    g['body_ratio'] = np.abs(g['close'] - g['open']) / (g['high'] - g['low'] + 1e-8)
    g['upper_shadow'] = (g['high'] - np.maximum(g['open'], g['close'])) / (g['high'] - g['low'] + 1e-8)
    g['lower_shadow'] = (np.minimum(g['open'], g['close']) - g['low']) / (g['high'] - g['low'] + 1e-8)

    # ===== 3. 변동성 (다중 시간대) =====
    for w in (6, 12, 24, 72, 144):
        g[f"rv{w}"] = g.groupby("symbol")["ret1"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).std()
        ).fillna(0.0)

    # ===== 4. 고급 모멘텀 =====
    for w in (6, 12, 24, 72):
        # EMA 기반 모멘텀
        ema = g.groupby("symbol")["close"].transform(
            lambda s: s.ewm(span=w, adjust=False).mean()
        )
        g[f"mom{w}"] = (g["close"] / (ema + 1e-8) - 1.0).fillna(0.0)

        # RSI
        g[f"rsi{w}"] = g.groupby("symbol")["close"].transform(
            lambda s: calculate_rsi(s, w)
        ) / 100.0 - 0.5  # Normalize to [-0.5, 0.5]

    # MACD
    for w in (12, 24):
        g[f"macd{w}"] = g.groupby("symbol")["close"].transform(
            lambda s: calculate_macd(s, fast=w, slow=w * 2, signal=w // 2)
        ) / (g["close"] + 1e-8)

    # ===== 5. 볼륨 프로파일 =====
    # Z-score
    for w in (12, 24, 72):
        mu = g.groupby("symbol")["volume"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean()
        )
        sd = g.groupby("symbol")["volume"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).std()
        ).replace(0, 1.0)
        g[f"vz{w}"] = ((g["volume"] - mu) / sd).fillna(0.0)

    # VWAP
    g["vwap_num"] = (g["close"] * g["volume"]).groupby(g["symbol"]).transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    g["vwap_den"] = g.groupby("symbol")["volume"].transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    g["vwap20"] = g["vwap_num"] / (g["vwap_den"] + 1e-8)
    g["vwap_dist"] = (g["close"] - g["vwap20"]) / (g["vwap20"] + 1e-8)
    g = g.drop(["vwap_num", "vwap_den"], axis=1)

    # 매수/매도 압력
    g['buy_pressure'] = np.where(g['close'] > g['open'], g['volume'], 0)
    g['sell_pressure'] = np.where(g['close'] < g['open'], g['volume'], 0)

    for w in (12, 24):
        buy_vol = g.groupby("symbol")['buy_pressure'].transform(lambda s: s.rolling(w).sum())
        sell_vol = g.groupby("symbol")['sell_pressure'].transform(lambda s: s.rolling(w).sum())
        g[f'pressure_ratio{w}'] = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-8)

    # 거래량 급등
    g["vol_spike"] = g.groupby("symbol")["volume"].transform(
        lambda s: s / (s.shift(1) + 1e-8) - 1.0
    ).fillna(0.0)

    # ===== 6. 가격 패턴 =====
    g["hl_ratio"] = (g["high"] - g["low"]) / (g["close"] + 1e-8)
    g["co_ratio"] = (g["close"] - g["open"]) / (g["open"] + 1e-8)

    # 연속 상승/하락
    g["up_streak"] = g.groupby("symbol")["ret1"].transform(
        lambda s: (s > 0).astype(int).groupby((s <= 0).cumsum()).cumsum()
    )
    g["down_streak"] = g.groupby("symbol")["ret1"].transform(
        lambda s: (s < 0).astype(int).groupby((s >= 0).cumsum()).cumsum()
    )

    # ===== 7. 변동성 레짐 =====
    g['vol_regime'] = g.groupby("symbol")['rv24'].transform(
        lambda s: pd.qcut(s.fillna(s.median()), q=5, labels=False, duplicates='drop')
    ).fillna(2).astype(float) / 4.0  # Normalize to [0, 1]

    # ===== 8. 시간 기반 피처 =====
    g['hour'] = g['date'].dt.hour / 23.0  # Normalize
    g['day_of_week'] = g['date'].dt.dayofweek / 6.0
    g['is_market_hours'] = g['date'].dt.hour.between(9, 16).astype(float)

    # ===== 9. 크로스오버 신호 =====
    for w1, w2 in [(6, 12), (12, 24), (24, 72)]:
        ema1 = g.groupby("symbol")["close"].transform(lambda s: s.ewm(span=w1).mean())
        ema2 = g.groupby("symbol")["close"].transform(lambda s: s.ewm(span=w2).mean())
        g[f"cross_{w1}_{w2}"] = (ema1 - ema2) / (ema2 + 1e-8)

    # ===== 10. ATR (Average True Range) =====
    g['tr'] = np.maximum(
        g['high'] - g['low'],
        np.maximum(
            np.abs(g['high'] - g['close'].shift(1)),
            np.abs(g['low'] - g['close'].shift(1))
        )
    )
    for w in (12, 24):
        g[f'atr{w}'] = g.groupby("symbol")['tr'].transform(
            lambda s: s.rolling(w).mean()
        ) / (g['close'] + 1e-8)

    # ATR 평균 (tr의 rolling mean이 atr)
    g['atr'] = g.groupby("symbol")['tr'].transform(
        lambda s: s.rolling(14).mean()
    ) / (g['close'] + 1e-8)

    # NaN 처리
    for col in g.columns:
        if col not in ["symbol", "date", "timestamp", "open", "high", "low", "close", "volume", "turnover"]:
            g[col] = g[col].fillna(0.0)
            # Clip extreme values
            g[col] = g[col].clip(-10, 10)

    # 특성 리스트 순서대로 반환
    return g[feat_cols]


def print_dashboard(balance: float, positions: List[Position], prices: Dict[str, float]):
    """대시보드 출력 (개선된 버전)"""
    os.system('cls' if os.name == 'nt' else 'clear')

    print("\n" + "=" * 140)
    print(f"{'📊 실전 트레이딩 대시보드':^140}")
    print("=" * 140)

    # 계좌 정보
    used_margin = sum(p.margin for p in positions)
    unrealized_pnl = sum(p.unrealized_pnl for p in positions)
    total_value = balance + unrealized_pnl
    available = balance - used_margin

    print(f"\n💰 계좌 정보")
    print(f"   총 잔고:       ${balance:>12,.2f}")
    print(f"   사용 증거금:   ${used_margin:>12,.2f}")
    print(f"   가용 잔고:     ${available:>12,.2f}")
    print(f"   평가 손익:     ${unrealized_pnl:>+12,.2f}")
    print(f"   총 자산:       ${total_value:>12,.2f}")
    print(f"   일일 실현손익: ${manager.daily_pnl:>+12,.2f}")

    # 포지션
    if positions:
        print(f"\n📍 보유 포지션 ({len(positions)}/{MAX_POSITIONS})")
        print(
            f"{'심볼':^12} | {'방향':^8} | {'진입가':^12} | {'현재가':^12} | {'손익(ROE)':^22} | "
            f"{'SL':^12} | {'TP':^12} | {'청산가':^12} | {'보유':^8}")
        print("-" * 140)

        for pos in positions:
            current_price = prices.get(pos.symbol, pos.entry_price)
            pnl = pos.get_pnl(current_price)
            roe = pos.get_roe(current_price)
            liq_dist = pos.get_liquidation_distance(current_price)

            # 보유 시간 계산
            if pos.symbol in manager.positions_entry_time:
                hold_min = (datetime.now() - manager.positions_entry_time[pos.symbol]).total_seconds() / 60
            else:
                hold_min = (datetime.now() - pos.entry_time).total_seconds() / 60

            emoji = "📈" if pos.direction == "Long" else "📉"
            pnl_emoji = "🟢" if pnl > 0 else "🔴"
            liq_warning = "⚠️" if liq_dist < 3 else ""

            print(f"{pos.symbol:^12} | {emoji} {pos.direction:^6} | ${pos.entry_price:>10,.4f} | "
                  f"${current_price:>10,.4f} | {pnl_emoji} ${pnl:>+8,.2f} ({roe:>+6.1f}%) | "
                  f"${pos.stop_loss:>10,.4f} | ${pos.take_profit:>10,.4f} | "
                  f"${pos.liquidation_price:>10,.4f}{liq_warning} | {hold_min:>6.1f}분")
    else:
        print(f"\n📍 보유 포지션: 없음")

    # 통계
    stats = manager.get_stats()
    if stats["total_trades"] > 0:
        print(f"\n📊 거래 통계")
        print(f"   총 거래:       {stats['total_trades']:>3}회")
        print(f"   승률:          {stats['win_rate']:>6.1f}% ({stats['wins']}승 {stats['losses']}패)")
        if stats['liquidations'] > 0:
            print(f"   강제 청산:     {stats['liquidations']:>3}회 💀")
        print(f"   평균 손익:     ${stats['avg_pnl']:>+12,.2f}")
        print(f"   평균 ROE:      {stats['avg_roe']:>+6.1f}%")

        # 최대 수익: 승리가 있을 때만 표시
        if stats['wins'] > 0:
            print(f"   최대 수익:     ${stats['max_pnl']:>12,.2f}  (ROE: {stats['max_roe']:>+6.1f}%)")

        # 최대 손실: 손실이 있을 때만 표시
        if stats['losses'] > 0:
            print(f"   최대 손실:     ${stats['min_pnl']:>12,.2f}  (ROE: {stats['min_roe']:>+6.1f}%)")

        if stats['wins'] > 0 and stats['losses'] > 0:
            rr = abs(stats['avg_win'] / stats['avg_loss'])
            print(f"   Risk/Reward:   {rr:>6.2f}")

    print("\n" + "=" * 140)


def test_api_connection():
    """API 연결 테스트"""
    print("\n🔍 API 연결 테스트 중...")

    # Public API 테스트
    print("   테스트 심볼: BTCUSDT")
    ticker = API.get_ticker("BTCUSDT")
    if ticker.get("retCode") == 0 and ticker["result"]["list"]:
        price = float(ticker["result"]["list"][0]["lastPrice"])
        print(f"   ✓ Ticker 조회 성공: ${price:,.2f}")
    else:
        print(f"   ✗ Ticker 조회 실패: {ticker.get('retMsg')}")
        return False

    df = API.get_klines("BTCUSDT", interval="5", limit=10)
    if not df.empty:
        print(f"   ✓ Klines 조회 성공: {len(df)}개 캔들")
        print(f"   최신 가격: ${df['close'].iloc[-1]:,.2f}")
    else:
        print(f"   ✗ Klines 조회 실패")
        return False

    # Private API 테스트
    balance = API.get_balance()
    if balance >= 0:
        print(f"   ✓ 잔고 조회 성공: ${balance:,.2f} USDT")
        if balance == 0:
            print(f"   ⚠️  잔고가 0입니다. 입금 후 거래를 시작하세요.")
    else:
        print(f"   ✗ 잔고 조회 실패")
        print(f"\n   확인 사항:")
        print(f"      1. API 권한: Contract Trading, Account Transfer")
        print(f"      2. API 모드: {'Mainnet' if not USE_TESTNET else 'Testnet'}")
        print(f"      3. IP 화이트리스트 설정 확인")
        return False

    # 심볼 정보 사전 로드
    print("\n📋 심볼 정보 로드 중...")
    loaded_count = 0
    for symbol in MODELS.keys():
        try:
            info = API.get_instrument_info(symbol)
            loaded_count += 1
            if DEBUG_MODE:
                print(f"   ✓ {symbol}: minQty={info['minOrderQty']}, qtyStep={info['qtyStep']}")
        except Exception as e:
            print(f"   ⚠️  {symbol}: 정보 로드 실패 - {e}")

    print(f"   ✓ {loaded_count}/{len(MODELS)}개 심볼 정보 로드 완료")

    print("\n✅ API 연결 정상\n")
    return True


# ===== 메인 루프 =====
def main():
    # 경고 메시지
    mode = "🧪 TESTNET" if USE_TESTNET else "🔴 LIVE"
    print("\n" + "=" * 110)
    print(f"{'⚠️  실전 자동 트레이딩 시스템  ⚠️':^110}")
    print(f"{mode:^110}")
    print("=" * 110)
    print("\n⚠️  경고:")
    print("   - 이 프로그램은 실제 자금을 사용합니다")
    print("   - 투자 손실의 위험이 있습니다")
    print("   - 본인 책임 하에 사용하세요")
    print("\nAPI 설정:")
    print(f"   - API Key: {API_KEY[:8]}...{API_KEY[-4:]} (길이: {len(API_KEY)})")
    print(f"   - API Secret: {'*' * 8}...{'*' * 4} (길이: {len(API_SECRET)})")
    print(f"   - 모드: {'Testnet' if USE_TESTNET else 'Mainnet (실전!)'}")
    print(f"   - 베이스 URL: {API.base_url}")

    # API 키 유효성 간단 체크
    if len(API_KEY) < 10 or len(API_SECRET) < 10:
        print("\n❌ ERROR: API Key 또는 Secret이 너무 짧습니다!")
        print("   환경변수가 제대로 설정되었는지 확인하세요.")
        print("\n현재 값:")
        print(f"   BYBIT_API_KEY 길이: {len(API_KEY)}")
        print(f"   BYBIT_API_SECRET 길이: {len(API_SECRET)}")
        return
    print("\n거래 설정:")
    print(f"   - 포지션 모드: {POSITION_MODE.upper()}")
    print(f"   - 레버리지: {LEVERAGE}x")
    print(f"   - 증거금/포지션: ${MARGIN_PER_POSITION}")
    print(f"   - 포지션 크기: ${MARGIN_PER_POSITION * LEVERAGE} (증거금 × 레버리지)")
    print(f"   - 최대 포지션: {MAX_POSITIONS}개")
    print(f"   - 일일 최대 손실: ${MAX_DAILY_LOSS}")
    print(f"   - 신뢰도 임계값: {CONF_THRESHOLD:.0%}")
    print(f"   - 반대 신호 청산: 즉시 청산 후 전환 (공격적)")
    print(f"   - 사용 모델: {len(MODELS)}개 심볼별 모델")

    # 확인
    print("\n계속하시겠습니까? (y/n): ", end='')
    if input().lower() != 'y':
        print("프로그램을 종료합니다.")
        return

    # API 연결 테스트
    if not test_api_connection():
        print("\n❌ API 연결 실패. 프로그램을 종료합니다.")
        print("\n확인 사항:")
        print("   1. API Key와 Secret이 올바른지 확인")
        print("   2. API 권한 설정 확인 (Contract, Position, Wallet)")
        print("   3. IP 화이트리스트 설정 확인")
        print("   4. Testnet/Mainnet 모드 확인")
        return

    print("\n시작합니다...\n")

    try:
        loop_count = 0
        while True:
            loop_count += 1
            current_time = datetime.now()

            # 일일 통계 리셋
            manager.reset_daily_stats()

            # 계좌 정보
            balance = API.get_balance()
            positions = API.get_positions()

            # 현재 가격 가져오기 (모델이 있는 심볼만)
            prices = {}
            for symbol in MODELS.keys():
                ticker = API.get_ticker(symbol)
                if ticker.get("retCode") == 0 and ticker["result"]["list"]:
                    prices[symbol] = float(ticker["result"]["list"][0]["lastPrice"])

            # 포지션 관리
            for position in list(positions):  # 리스트 복사로 순회 중 수정 방지
                current_price = prices.get(position.symbol, position.entry_price)

                # ✅ 수정: entry_time 동기화
                # 1. manager에 기록된 시간이 있으면 사용
                if position.symbol in manager.positions_entry_time:
                    entry_time_to_use = manager.positions_entry_time[position.symbol]
                else:
                    # 2. API에서 가져온 시간 사용
                    entry_time_to_use = position.entry_time
                    # 3. manager에 등록 (다음 루프부터 사용)
                    manager.positions_entry_time[position.symbol] = position.entry_time

                # 보유 시간 계산
                hold_minutes = (current_time - entry_time_to_use).total_seconds() / 60

                # 현재 손익
                pnl = position.get_pnl(current_price)
                roe = position.get_roe(current_price)

                # 청산 조건 디버그
                print(f"\n[체크] {position.symbol} | 보유: {hold_minutes:.1f}분 | 손익: ${pnl:+.2f} ({roe:+.1f}%)")
                print(f"       진입: ${position.entry_price:.4f} | 현재: ${current_price:.4f}")
                print(f"       SL: ${position.stop_loss:.4f} | TP: ${position.take_profit:.4f}")

                # ✅ 청산 조건 직접 체크 (hold_minutes 사용)
                should_close = False
                reason = ""

                # 1. 청산가
                if position.direction == "Long" and current_price <= position.liquidation_price:
                    should_close, reason = True, "Liquidation"
                elif position.direction == "Short" and current_price >= position.liquidation_price:
                    should_close, reason = True, "Liquidation"
                # 2. 손절
                elif position.direction == "Long" and current_price <= position.stop_loss:
                    should_close, reason = True, "Stop Loss"
                elif position.direction == "Short" and current_price >= position.stop_loss:
                    should_close, reason = True, "Stop Loss"
                # 3. 익절
                elif position.direction == "Long" and current_price >= position.take_profit:
                    should_close, reason = True, "Take Profit"
                elif position.direction == "Short" and current_price <= position.take_profit:
                    should_close, reason = True, "Take Profit"
                # 4. 시간 초과
                elif hold_minutes >= MAX_HOLD_MINUTES:
                    should_close, reason = True, "Time Limit"

                if should_close:
                    print(f"       ➡️  청산 조건 충족: {reason}")
                    manager.close_position(position, reason)
                else:
                    print(f"       ✓ 유지")

                # ✅ 반대 신호 청산 제거 - TP/SL/시간으로만 관리

            # 포지션 업데이트 (청산 후)
            new_positions = API.get_positions()

            # Bybit 자동 청산 감지
            if len(new_positions) < len(positions):
                for old_pos in positions:
                    if not any(p.symbol == old_pos.symbol for p in new_positions):
                        print(f"\n🔔 {old_pos.symbol} 포지션이 사라졌습니다!")
                        print(f"   💡 Bybit가 TP/SL을 자동 실행했을 가능성이 있습니다")

                        # ✅ 손익 계산 및 기록
                        try:
                            # 현재가 조회
                            ticker = API.get_ticker(old_pos.symbol)
                            if ticker.get("retCode") == 0 and ticker["result"]["list"]:
                                current_price = float(ticker["result"]["list"][0]["lastPrice"])
                            else:
                                current_price = old_pos.entry_price  # 조회 실패 시 진입가 사용

                            # 손익 계산
                            pnl = old_pos.get_pnl(current_price)
                            roe = old_pos.get_roe(current_price)

                            # ✅ daily_pnl에 반영 (가장 중요!)
                            manager.daily_pnl += pnl

                            # 거래 기록 저장
                            entry_time_str = manager.positions_entry_time.get(old_pos.symbol, old_pos.entry_time)
                            if isinstance(entry_time_str, datetime):
                                entry_time_str = entry_time_str.strftime("%Y-%m-%d %H:%M:%S")
                            else:
                                entry_time_str = str(entry_time_str)

                            exit_time = datetime.now()
                            trade = Trade(
                                symbol=old_pos.symbol,
                                direction=old_pos.direction,
                                entry_price=old_pos.entry_price,
                                exit_price=current_price,
                                size=old_pos.size,
                                leverage=old_pos.leverage,
                                margin=old_pos.margin,
                                entry_time=entry_time_str,
                                exit_time=exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                                pnl=pnl,
                                pnl_pct=old_pos.get_pnl_pct(current_price),
                                roe=roe,
                                exit_reason="Bybit Auto Close (TP/SL)"
                            )
                            manager.trades.append(trade)

                            # 진입 시간 삭제
                            if old_pos.symbol in manager.positions_entry_time:
                                del manager.positions_entry_time[old_pos.symbol]

                            emoji = "🟢" if pnl > 0 else "🔴"
                            print(f"   {emoji} 손익 기록: ${pnl:+,.2f} (ROE: {roe:+.1f}%)")
                            print(f"   📊 Bybit 웹사이트 > Orders > Closed에서 확인하세요")

                        except Exception as e:
                            print(f"   ⚠️  손익 계산 실패: {e}")
                            print(f"   수동으로 Bybit에서 확인하세요")

            positions = new_positions

            # ✅ 일일 손실 한도 체크 (포지션 청산 후)
            if manager.daily_pnl < -MAX_DAILY_LOSS:
                print(f"\n{'=' * 110}")
                print(f"{'⛔ 일일 손실 한도 초과 감지':^110}")
                print(f"{'=' * 110}")
                print(f"   현재 일일 손익: ${manager.daily_pnl:+,.2f}")
                print(f"   손실 한도: -${MAX_DAILY_LOSS:,.2f}")
                print(f"\n🚨 모든 포지션을 청산하고 프로그램을 종료합니다...")

                # 모든 포지션 강제 청산
                positions = API.get_positions()
                for position in positions:
                    print(f"\n📤 강제 청산 중: {position.symbol} ({position.direction})")
                    manager.close_position(position, "Daily Loss Limit")
                    time.sleep(1)

                # 최종 상태 출력
                final_balance = API.get_balance()
                print(f"\n{'=' * 110}")
                print(f"{'💀 일일 거래 종료':^110}")
                print(f"{'=' * 110}")
                print(f"   최종 잔고: ${final_balance:,.2f}")
                print(f"   일일 손익: ${manager.daily_pnl:+,.2f}")
                print(f"   손실률: {(manager.daily_pnl / final_balance * 100):+.2f}%")
                print(f"{'=' * 110}")
                manager.save_trades()
                print("\n⚠️  일일 손실 한도에 도달하여 프로그램을 종료합니다.")
                print("   내일 다시 시작하세요.")
                return

            # 대시보드 출력
            print_dashboard(balance, positions, prices)

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

                # 가격이 0이면 스킵
                if price <= 0:
                    print(f"{symbol:^12} | {'가격조회실패':^12} | {'오류':^10} | {'N/A':^8} | ❌ 가격 조회 실패")
                    continue

                # 기존 포지션 확인
                existing = next((p for p in positions if p.symbol == symbol), None)
                position_status = ""
                if existing:
                    if existing.direction == direction:
                        position_status = f" [보유: {existing.direction}]"
                    else:
                        position_status = f" [전환: {existing.direction}→{direction}]"

                # 방향 이모지
                dir_icon = {"Long": "📈", "Short": "📉", "Flat": "➖"}.get(direction, "❓")

                # 신호 판단
                if confidence < CONF_THRESHOLD:
                    signal = f"⚠️  신호 약함 ({confidence:.1%}){position_status}"
                elif direction == "Long":
                    signal = f"🟢 매수 신호 ({confidence:.1%}){position_status}"
                elif direction == "Short":
                    signal = f"🔴 매도 신호 ({confidence:.1%}){position_status}"
                else:
                    signal = f"⚪ 관망 ({confidence:.1%}){position_status}"

                # 가격 표시 형식 (매우 작은 가격 대응)
                if price < 0.0001:
                    price_str = f"{price:.8f}"
                elif price < 1:
                    price_str = f"{price:.6f}"
                else:
                    price_str = f"{price:,.4f}"

                print(
                    f"{symbol:^12} | ${price_str:>10} | {dir_icon} {direction:^8} | {confidence:>6.1%} | {signal:^20}")

                # ✅ 반대 신호 청산 체크 (신호 강도와 무관하게 항상 체크)
                existing = next((p for p in positions if p.symbol == symbol), None)
                if existing and direction in ["Long", "Short"] and existing.direction != direction:
                    # 반대 신호면 무조건 청산 후 전환
                    current_roe = existing.get_roe(price)
                    current_pnl = existing.get_pnl(price)

                    print(f"\n🔄 {symbol}: 반대 신호 감지 - 즉시 청산 후 전환")
                    print(f"   방향: {existing.direction} → {direction}")
                    print(f"   현재 손익: ${current_pnl:+.2f} (ROE: {current_roe:+.1f}%)")

                    manager.close_position(existing, "Reverse Signal")
                    time.sleep(1)
                    positions = API.get_positions()

                # 진입 조건 (신호가 강할 때만)
                if confidence >= CONF_THRESHOLD and direction in ["Long", "Short"]:
                    # 가격 유효성 재확인
                    if price <= 0 or not np.isfinite(price):
                        print(f"⚠️  {symbol}: 유효하지 않은 가격 (${price}) - 진입 스킵")
                        continue

                    # 기존 포지션 확인
                    existing = next((p for p in positions if p.symbol == symbol), None)

                    # 같은 방향 포지션이 있으면 스킵
                    if existing and existing.direction == direction:
                        if DEBUG_MODE:
                            print(f"ℹ️  {symbol}: 같은 방향 포지션 보유 중 ({direction}) - 유지")
                        continue

                    # 반대 방향 포지션은 이미 위에서 청산 체크했으므로 계속 진행

                    # 새로운 포지션 진입 (기존 포지션이 없거나 청산 후)
                    if manager.can_open_position(symbol, direction, positions):
                        if manager.open_position(symbol, direction, price):
                            time.sleep(1)  # 주문 처리 대기
                            positions = API.get_positions()  # 포지션 업데이트

            print(f"\n[스캔 #{loop_count}] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"다음 스캔까지 {INTERVAL_SEC}초... (Ctrl+C로 종료)")

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n프로그램 종료 중...")

        # 최종 상태
        balance = API.get_balance()
        positions = API.get_positions()
        prices = {}
        for symbol in MODELS.keys():
            ticker = API.get_ticker(symbol)
            if ticker.get("retCode") == 0 and ticker["result"]["list"]:
                prices[symbol] = float(ticker["result"]["list"][0]["lastPrice"])

        print_dashboard(balance, positions, prices)
        manager.save_trades()

        # 최종 통계
        stats = manager.get_stats()
        if stats["total_trades"] > 0:
            print("\n" + "=" * 110)
            print(f"{'📊 최종 결과':^110}")
            print("=" * 110)
            print(f"   최종 잔고:     ${balance:,.2f}")
            print(f"   일일 손익:     ${manager.daily_pnl:+,.2f}")
            print(f"   총 거래:       {stats['total_trades']}회")
            print(f"   승률:          {stats['win_rate']:.1f}%")
            print(f"   평균 ROE:      {stats['avg_roe']:+.1f}%")
            if stats['liquidations'] > 0:
                print(f"   강제 청산:     {stats['liquidations']}회 💀")
            print("=" * 110)

        if positions:
            print("\n⚠️  주의: 아직 포지션이 남아있습니다!")
            for pos in positions:
                pnl = pos.get_pnl(prices.get(pos.symbol, pos.entry_price))
                print(f"   - {pos.symbol}: {pos.direction} | ${pnl:+,.2f}")

        print("\n✅ 프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()