# live_trading_multi_model_v2.py
# -*- coding: utf-8 -*-
"""
AdvancedTCN_V2 모델 기반 실전 자동 트레이딩 시스템
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
import torch.nn.functional as F
import requests
import certifi
import math

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
    print("\n설정 방법:")
    print("  export BYBIT_API_KEY='your_api_key'")
    print("  export BYBIT_API_SECRET='your_api_secret'")
    print("  export USE_TESTNET=1  # Testnet (기본값)")
    print("  export USE_TESTNET=0  # Mainnet (실전)")
    print("\n포지션 모드 설정:")
    print("  export POSITION_MODE=hedge     # Hedge Mode (양방향, 기본값)")
    print("  export POSITION_MODE=one-way   # One-Way Mode (단방향)")
    print("\n⚠️  중요:")
    print("  - Testnet API Key: https://testnet.bybit.com 에서 발급")
    print("  - Mainnet API Key: https://www.bybit.com 에서 발급")
    print("  - Testnet과 Mainnet API Key는 서로 다릅니다!")
    print("  - 포지션 모드는 Bybit 웹사이트 설정과 일치해야 합니다!")
    exit(1)

# 거래 설정
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,BNBUSDT").split(",")
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "2"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.65"))  # 실전은 더 높게

# ✅ 심볼별 V2 모델 경로
'''
MODEL_PATHS = {
    "BTCUSDT": "D:/ygy_work/coin/multimodel/models_5min_btc_v2/model_v2_best.pt",
    "ETHUSDT": "D:/ygy_work/coin/multimodel/models_5min_eth_v2/model_v2_best.pt",
    "SOLUSDT": "D:/ygy_work/coin/multimodel/models_5min_sol_v2/model_v2_best.pt",
    "DOGEUSDT": "D:/ygy_work/coin/multimodel/models_5min_doge_v2/model_v2_best.pt",
    "BNBUSDT": "D:/ygy_work/coin/multimodel/models_5min_bnb_v2/model_v2_best.pt",
    "XRPUSDT": "D:/ygy_work/coin/multimodel/models_5min_xrp_v2/model_v2_best.pt",
    "SAPIENUSDT": "D:/ygy_work/coin/multimodel/models_5min_sapien_v2/model_v2_best.pt",
    "FLMUSDT": "D:/ygy_work/coin/multimodel/models_5min_flm_v2/model_v2_best.pt",
    "TRUMPUSDT": "D:/ygy_work/coin/multimodel/models_5min_trump_v2/model_v2_best.pt"

}
'''
MODEL_PATHS = {
    "BTCUSDT": "D:/ygy_work/coin/multimodel/models_5min_btc/5min_2class_best.ckpt",
    "ETHUSDT": "D:/ygy_work/coin/multimodel/models_5min_eth/5min_2class_best.ckpt",
    "SOLUSDT": "D:/ygy_work/coin/multimodel/models_5min_sol/5min_2class_best.ckpt",
    "DOGEUSDT": "D:/ygy_work/coin/multimodel/models_5min_doge/5min_2class_best.ckpt",
    "BNBUSDT": "D:/ygy_work/coin/multimodel/models_5min_bnb/5min_2class_best.ckpt",
    "XRPUSDT": "D:/ygy_work/coin/multimodel/models_5min_xrp/5min_2class_best.ckpt",
    "SAPIENUSDT": "D:/ygy_work/coin/multimodel/models_5min_sapien_v2/model_v2_best.pt",
    "FLMUSDT": "D:/ygy_work/coin/multimodel/models_5min_flm_v2/model_v2_best.pt",
    "TRUMPUSDT": "D:/ygy_work/coin/multimodel/models_5min_trump_v2/model_v2_best.pt"

}

# 리스크 관리 (매우 중요!)
MARGIN_PER_POSITION = float(os.getenv("MARGIN_PER_POSITION", "30"))  # 포지션당 증거금
LEVERAGE = int(os.getenv("LEVERAGE", "10"))  # 레버리지 (실전은 낮게!)
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))  # 최대 동시 포지션
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))  # 손절 2%s
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.03"))  # 익절 3%
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "500"))  # 일일 최대 손실
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "30"))

# 수수료 설정 (Bybit 기본값)
TAKER_FEE = float(os.getenv("TAKER_FEE", "0.00055"))  # 0.055% (Market order)
MAKER_FEE = float(os.getenv("MAKER_FEE", "0.0002"))  # 0.02% (Limit order)
FUNDING_FEE_RATE = float(os.getenv("FUNDING_FEE_RATE", "0.0001"))  # 0.01% (8시간마다)

# 포지션 모드
POSITION_MODE = os.getenv("POSITION_MODE", "one-way").lower()  # "one-way" 또는 "hedge"

# 로그
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "live_trades_v2.json")
ORDER_LOG_FILE = os.getenv("ORDER_LOG_FILE", "orders_v2.json")
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"


# ===== 데이터 클래스 =====
@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    side: str  # "Buy" or "Sell"
    size: float
    entry_price: float
    leverage: int
    unrealized_pnl: float
    position_value: float

    def get_direction(self) -> str:
        return "Long" if self.side == "Buy" else "Short"


@dataclass
class Order:
    """주문 정보"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    qty: float
    price: float
    status: str
    timestamp: str


@dataclass
class Trade:
    """거래 기록"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    leverage: int
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float
    exit_reason: str
    margin: float = 0.0
    roe: float = 0.0
    expected_bps: float = 0.0  # V2 추가
    model_confidence: float = 0.0  # V2 추가
    entry_fee: float = 0.0  # 진입 수수료
    exit_fee: float = 0.0  # 청산 수수료
    total_fee: float = 0.0  # 총 수수료
    net_pnl: float = 0.0  # 수수료 차감 후 순손익


# ===== AdvancedTCN_V2 모델 정의 =====
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


class AdvancedTCN_V2(nn.Module):
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
def make_features_v2(df: pd.DataFrame) -> tuple:
    """
    V2: train_tcn_5minutes_v2.py와 동일한 피처 생성
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
        "vol_change_rate"
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
        meta_path = os.path.join(os.path.dirname(model_path), "meta_v2.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        else:
            meta = {}

        seq_len = meta.get('seq_len', 72)
        norm_window = meta.get('norm_window', 288)

        # AdvancedTCN_V2 모델 생성
        model = AdvancedTCN_V2(in_features=len(feat_cols), hidden=64,
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
        print(f"      - 모델: AdvancedTCN_V2")
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


# ===== Bybit API 클래스 =====
class BybitAPI:
    """Bybit Private API"""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self.session = requests.Session()
        self.session.verify = certifi.where()
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

    def _request(self, method: str, endpoint: str, params: dict = None) -> dict:
        """API 요청"""
        if params is None:
            params = {}

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
        headers = {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-SIGN': signature,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': self.recv_window,
            'Content-Type': 'application/json'
        }

        url = f"{self.base}{endpoint}"

        if DEBUG_MODE:
            print(f"\n[DEBUG] {method} {endpoint}")
            print(f"[DEBUG] Timestamp: {timestamp}")
            print(f"[DEBUG] Params: {params}")
            print(f"[DEBUG] Params str: {params_str}")
            print(f"[DEBUG] Signature: {signature}")

        try:
            if method == "GET":
                r = self.session.get(url, params=params, headers=headers, timeout=10)
            else:
                r = self.session.post(url, json=params, headers=headers, timeout=10)

            if DEBUG_MODE:
                print(f"[DEBUG] Status: {r.status_code}")
                print(f"[DEBUG] Response: {r.text[:500]}")

            if r.status_code == 401:
                print(f"❌ 인증 실패 (401)")
                print(f"   API Key: {self.api_key[:8]}...{self.api_key[-4:]}")
                print(f"   Endpoint: {endpoint}")
                print(f"   모드: {'Testnet' if 'testnet' in self.base else 'Mainnet'}")
                print(f"   응답: {r.text[:200] if r.text else '(비어있음)'}")
                return {}

            if r.status_code != 200:
                print(f"❌ HTTP Error {r.status_code}: {r.text[:200]}")
                return {}

            data = r.json()

            # retCode 확인
            ret_code = data.get("retCode", -1)

            # 110043: leverage not modified - 이미 레버리지가 설정되어 있음 (정상)
            if ret_code == 110043:
                if DEBUG_MODE:
                    print(f"[DEBUG] 레버리지 이미 설정됨 (정상)")
                return data.get("result", {})

            if ret_code != 0:
                print(f"❌ API Error [{ret_code}]: {data.get('retMsg', 'Unknown error')}")
                if DEBUG_MODE:
                    print(f"[DEBUG] Full response: {data}")
                return {}

            return data.get("result", {})

        except json.JSONDecodeError as e:
            print(f"❌ JSON Decode Error: {e}")
            print(f"   Response: {r.text[:200]}")
            return {}
        except Exception as e:
            print(f"❌ Request Error: {e}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
            return {}

    def get_balance(self) -> float:
        """계좌 잔고 조회"""
        result = self._request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})
        if not result:
            print("⚠️  잔고 조회 실패 - API 인증 정보를 확인하세요")
            return 0.0

        try:
            coins = result.get("list", [{}])[0].get("coin", [])
            for coin in coins:
                if coin.get("coin") == "USDT":
                    balance = float(coin.get("walletBalance", 0))
                    print(f"✓ 잔고 조회 성공: ${balance:,.2f}")
                    return balance
        except Exception as e:
            print(f"❌ 잔고 파싱 오류: {e}")

        return 0.0

    def get_positions(self) -> List[Position]:
        """포지션 조회"""
        result = self._request("GET", "/v5/position/list", {
            "category": "linear",
            "settleCoin": "USDT"
        })
        if not result:
            return []

        positions = []
        for item in result.get("list", []):
            size = float(item.get("size", 0))
            if size == 0:
                continue

            positions.append(Position(
                symbol=item["symbol"],
                side=item["side"],
                size=size,
                entry_price=float(item.get("avgPrice", 0)),
                leverage=int(item.get("leverage", 1)),
                unrealized_pnl=float(item.get("unrealisedPnl", 0)),
                position_value=float(item.get("positionValue", 0))
            ))
        return positions

    def get_instrument_info(self, symbol: str) -> dict:
        """심볼 정보 조회 (최소 주문 수량 등)"""
        url = f"{self.base}/v5/market/instruments-info"
        try:
            r = self.session.get(url, params={"category": "linear", "symbol": symbol}, timeout=5)
            data = r.json()
            if data.get("retCode") == 0:
                items = data.get("result", {}).get("list", [])
                if items:
                    return items[0]
        except:
            pass
        return {}

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """레버리지 설정"""
        result = self._request("POST", "/v5/position/set-leverage", {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        })

        # retCode가 없거나 110043(leverage not modified)이면 성공으로 처리
        # 이미 레버리지가 설정되어 있으면 변경할 필요 없음
        if not result:
            # API 호출 실패
            return False

        # 성공으로 간주 (이미 설정되어 있어도 OK)
        if DEBUG_MODE:
            print(f"[DEBUG] 레버리지 설정 완료 또는 이미 설정됨: {symbol} {leverage}x")
        return True

    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = "Market", price: float = None,
                    stop_loss: float = None, take_profit: float = None) -> Optional[str]:
        """주문 생성 (TP/SL 포함)"""
        # positionIdx 결정
        if POSITION_MODE == "one-way":
            position_idx = 0
        else:  # hedge mode
            position_idx = 1 if side == "Buy" else 2

        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "positionIdx": position_idx
        }

        # Market 주문이 아닐 때만 가격 추가
        if order_type == "Limit" and price:
            params["price"] = str(price)
            params["timeInForce"] = "GTC"

        # ✅ TP/SL 설정 추가 (tick size 고려)
        # 심볼 정보에서 tick size 가져오기
        instrument = self.get_instrument_info(symbol)
        price_filter = instrument.get("priceFilter", {})
        tick_size = float(price_filter.get("tickSize", "0.01"))

        # tick size로 소수점 자리수 결정
        if tick_size >= 1:
            decimal_places = 0
        else:
            decimal_places = len(str(tick_size).split('.')[-1].rstrip('0'))

        if stop_loss:
            # tick size에 맞춰 반올림
            stop_loss_rounded = round(stop_loss / tick_size) * tick_size
            stop_loss_rounded = round(stop_loss_rounded, decimal_places)
            params["stopLoss"] = str(stop_loss_rounded)
            print(f"   🛡️  Stop Loss: ${stop_loss_rounded:.{decimal_places}f}")
        if take_profit:
            # tick size에 맞춰 반올림
            take_profit_rounded = round(take_profit / tick_size) * tick_size
            take_profit_rounded = round(take_profit_rounded, decimal_places)
            params["takeProfit"] = str(take_profit_rounded)
            print(f"   🎯 Take Profit: ${take_profit_rounded:.{decimal_places}f}")

        if DEBUG_MODE:
            print(f"[DEBUG] 주문 생성: {symbol} {side} {qty} (positionIdx={position_idx})")
            if stop_loss or take_profit:
                print(f"[DEBUG] TP/SL: SL={stop_loss}, TP={take_profit}")

        result = self._request("POST", "/v5/order/create", params)
        order_id = result.get("orderId")

        if order_id:
            print(f"✓ 주문 생성 성공: {order_id}")
        else:
            print(f"✗ 주문 생성 실패")

        return order_id

    def close_position(self, symbol: str, side: str, qty: float) -> Optional[str]:
        """포지션 청산"""
        # 포지션 방향의 반대로 주문
        close_side = "Sell" if side == "Buy" else "Buy"
        return self.place_order(symbol, close_side, qty, "Market")

    def get_ticker(self, symbol: str) -> dict:
        """현재가 조회 (Public)"""
        url = f"{self.base}/v5/market/tickers"
        try:
            r = self.session.get(url, params={"category": "linear", "symbol": symbol}, timeout=5)
            data = r.json()
            rows = ((data.get("result") or {}).get("list") or [])
            return rows[0] if rows else {}
        except:
            return {}

    def get_kline(self, symbol: str, interval: str, limit: int):
        """K라인 조회 (Public) - list 반환"""
        url = f"{self.base}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        try:
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            return ((data.get("result") or {}).get("list") or [])
        except:
            return []


# API 초기화
API = BybitAPI(API_KEY, API_SECRET, testnet=USE_TESTNET)


# ===== 추론 함수 V2 =====
def predict_v2(symbol: str, debug: bool = False) -> dict:
    """
    AdvancedTCN_V2 모델을 사용한 예측
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
    klines = API.get_kline(symbol, interval="5", limit=limit)

    if not klines or len(klines) < config['seq_len'] + 20:
        return {"error": "Insufficient data", "symbol": symbol}

    # list를 DataFrame으로 변환
    rows = klines[::-1]  # 역순 정렬
    df = pd.DataFrame([{
        "timestamp": pd.to_datetime(int(z[0]), unit="ms", utc=True),
        "open": float(z[1]),
        "high": float(z[2]),
        "low": float(z[3]),
        "close": float(z[4]),
        "volume": float(z[5]),
    } for z in rows])

    if df.empty or len(df) < config['seq_len'] + 20:
        return {"error": "Insufficient data after conversion", "symbol": symbol}

    # 특성 생성 V2
    df_feat, feat_cols = make_features_v2(df)

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
        "current_price": current_price
    }


# ===== 트레이드 매니저 =====
class TradeManager:
    """거래 관리"""

    def __init__(self):
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        self.position_entry_times: Dict[str, datetime] = {}
        self.initial_balance = 0.0  # 초기 잔고 저장

    def set_initial_balance(self, balance: float):
        """프로그램 시작 시 초기 잔고 설정"""
        if self.initial_balance == 0.0:
            self.initial_balance = balance
            print(f"✓ 초기 잔고 설정: ${balance:,.2f}")

    def check_daily_loss_limit(self, current_total_value: float) -> bool:
        """일일 손실 한도 체크"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            # 날짜가 바뀌면 초기 잔고를 현재 총 자산으로 재설정
            self.initial_balance = current_total_value
            self.daily_pnl = 0.0
            self.last_reset_date = today
            print(f"\n📅 날짜 변경: 초기 잔고 재설정 ${current_total_value:,.2f}")

        # 실제 일일 손익 계산
        actual_daily_pnl = current_total_value - self.initial_balance if self.initial_balance > 0 else 0

        if actual_daily_pnl <= -MAX_DAILY_LOSS:
            print(f"\n⛔ 일일 손실 한도 도달: ${actual_daily_pnl:.2f}")
            return False
        return True

    def can_open_position(self, symbol: str, positions: List[Position], balance: float = 0) -> bool:
        """포지션 진입 가능 여부"""
        # 최대 포지션 수 체크 (같은 심볼은 제외)
        other_symbols_count = len([p for p in positions if p.symbol != symbol])
        if other_symbols_count >= MAX_POSITIONS:
            return False

        # 일일 손실 한도 체크 (실제 총 자산 기준)
        unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        current_total_value = balance + unrealized_pnl
        if not self.check_daily_loss_limit(current_total_value):
            return False

        return True

    def open_position(self, symbol: str, direction: str, price: float,
                      expected_bps: float = 0.0, confidence: float = 0.0) -> bool:
        """포지션 진입"""
        print(f"\n{'=' * 80}")
        print(f"💡 포지션 진입 시도: {symbol} {direction}")

        # 기존 포지션 확인 및 청산
        existing_positions = API.get_positions()
        for pos in existing_positions:
            if pos.symbol == symbol:
                existing_direction = pos.get_direction()

                # 같은 방향이면 진입하지 않음
                if existing_direction == direction:
                    print(f"⚠️  이미 같은 방향({direction}) 포지션 보유 중 | 손익: ${pos.unrealized_pnl:+,.2f}")
                    print(f"   진입을 건너뜁니다.")
                    print(f"{'=' * 80}\n")
                    return False

                # 반대 방향이면 청산 후 진입
                print(f"⚠️  반대 방향 포지션 발견: {existing_direction} | 손익: ${pos.unrealized_pnl:+,.2f}")
                print(f"   청산 후 {direction} 포지션으로 진입합니다...")

                # 기존 포지션 청산
                close_order_id = API.close_position(pos.symbol, pos.side, pos.size)
                if close_order_id:
                    print(f"✓ 기존 포지션 청산 완료 (주문 ID: {close_order_id})")
                    time.sleep(2)  # 청산 완료 대기
                else:
                    print(f"⚠️  기존 포지션 청산 실패 - 진입을 중단합니다")
                    print(f"{'=' * 80}\n")
                    return False

        # 레버리지 설정
        if not API.set_leverage(symbol, LEVERAGE):
            print(f"⚠️  레버리지 설정 실패했지만 계속 진행 (이미 설정되어 있을 수 있음)")

        # 심볼 정보 조회
        instrument = API.get_instrument_info(symbol)
        lot_size_filter = instrument.get("lotSizeFilter", {})
        min_order_qty = float(lot_size_filter.get("minOrderQty", 0.01))
        max_order_qty = float(lot_size_filter.get("maxMktOrderQty", 1000000000))
        qty_step = float(lot_size_filter.get("qtyStep", 0.01))

        # 포지션 크기 계산 (증거금 * 레버리지)
        position_value = MARGIN_PER_POSITION * LEVERAGE
        qty = position_value / price

        # qtyStep에 맞춰 조정
        qty = round(qty / qty_step) * qty_step

        # 최소/최대 수량 체크
        if qty < min_order_qty:
            qty = min_order_qty
        elif qty > max_order_qty:
            qty = max_order_qty

        # 최종 반올림 (소수점 처리)
        if qty_step >= 1:
            qty = int(qty)
        else:
            decimal_places = len(str(qty_step).split('.')[-1].rstrip('0'))
            qty = round(qty, decimal_places)

        print(f"   증거금: ${MARGIN_PER_POSITION:,.2f}")
        print(f"   레버리지: {LEVERAGE}x")
        print(f"   포지션 크기: ${position_value:,.2f}")
        print(f"   계산된 수량: {qty}")
        print(f"   수량 단위: {qty_step}")
        print(f"   진입가: ${price:,.4f}")
        print(f"   예상 수익: {expected_bps:.1f} bps")
        print(f"   모델 신뢰도: {confidence:.1%}")

        # ✅ 수수료 계산 (Market order이므로 Taker fee)
        entry_fee = position_value * TAKER_FEE
        expected_exit_fee = position_value * TAKER_FEE
        total_expected_fee = entry_fee + expected_exit_fee
        print(f"   💰 진입 수수료: ${entry_fee:,.2f} ({TAKER_FEE:.3%})")
        print(f"   💰 예상 청산 수수료: ${expected_exit_fee:,.2f}")
        print(f"   💰 총 예상 수수료: ${total_expected_fee:,.2f}")

        # 수수료를 고려한 손익분기점
        fee_pct = total_expected_fee / position_value * 100
        print(f"   ⚖️  손익분기점: {fee_pct:+.2f}% (수수료 회복 필요)")

        # ✅ TP/SL 가격 계산
        if direction == "Long":
            stop_loss_price = price * (1 - STOP_LOSS_PCT)
            take_profit_price = price * (1 + TAKE_PROFIT_PCT)
        else:  # Short
            stop_loss_price = price * (1 + STOP_LOSS_PCT)
            take_profit_price = price * (1 - TAKE_PROFIT_PCT)

        # 주문 생성 (TP/SL 포함)
        side = "Buy" if direction == "Long" else "Sell"
        order_id = API.place_order(symbol, side, qty, "Market",
                                   stop_loss=stop_loss_price,
                                   take_profit=take_profit_price)

        if not order_id:
            print(f"❌ 주문 실패: {symbol}")
            print(f"{'=' * 80}\n")
            return False

        # 진입 시간 기록
        self.position_entry_times[symbol] = datetime.now()

        print(f"✅ 포지션 진입 성공!")
        print(f"{'=' * 80}\n")
        return True

    def close_position(self, position: Position, exit_reason: str):
        """포지션 청산"""
        try:
            # 반대 주문 실행
            side = "Sell" if position.side == "Buy" else "Buy"
            order_id = API.close_position(position.symbol, position.side, position.size)

            if order_id:
                ticker = API.get_ticker(position.symbol)
                exit_price = float(ticker.get("lastPrice", position.entry_price))

                # 손익 계산 (수수료 제외)
                if position.get_direction() == "Long":
                    pnl = (exit_price - position.entry_price) * position.size
                else:
                    pnl = (position.entry_price - exit_price) * position.size

                # ✅ 수수료 계산
                position_value = position.entry_price * position.size
                entry_fee = position_value * TAKER_FEE  # 진입 시 수수료
                exit_fee = exit_price * position.size * TAKER_FEE  # 청산 시 수수료
                total_fee = entry_fee + exit_fee

                # 순손익 (수수료 차감)
                net_pnl = pnl - total_fee
                net_pnl_pct = (net_pnl / MARGIN_PER_POSITION) * 100
                net_roe = net_pnl_pct  # 순ROE

                # 기존 손익률 (수수료 미포함 - 참고용)
                pnl_pct = (pnl / MARGIN_PER_POSITION) * 100
                roe = pnl_pct

                self.daily_pnl += net_pnl  # 순손익 반영

                # 진입 시간 가져오기
                entry_time = self.position_entry_times.get(position.symbol)

                # 거래 기록
                trade = Trade(
                    symbol=position.symbol,
                    direction=position.get_direction(),
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    size=position.size,
                    leverage=position.leverage,
                    entry_time=entry_time.isoformat() if entry_time else "",
                    exit_time=datetime.now().isoformat(),
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=exit_reason,
                    margin=MARGIN_PER_POSITION,
                    roe=roe,
                    entry_fee=entry_fee,
                    exit_fee=exit_fee,
                    total_fee=total_fee,
                    net_pnl=net_pnl
                )
                self.trades.append(trade)
                self.save_trades()

                # 진입 시간 삭제
                if position.symbol in self.position_entry_times:
                    del self.position_entry_times[position.symbol]

                emoji = "💀" if exit_reason == "Liquidation" else ("🔴" if net_pnl < 0 else "🟢")
                print(f"\n{'=' * 80}")
                print(f"{emoji} 포지션 청산: {position.symbol}")
                print(f"   주문 ID: {order_id}")
                print(f"   사유: {exit_reason}")
                print(f"   진입가: ${position.entry_price:,.4f}")
                print(f"   청산가: ${exit_price:,.4f}")
                print(f"   손익 (수수료 전): ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
                print(f"   💰 진입 수수료: ${entry_fee:,.2f}")
                print(f"   💰 청산 수수료: ${exit_fee:,.2f}")
                print(f"   💰 총 수수료: ${total_fee:,.2f}")
                print(f"   {'🟢' if net_pnl > 0 else '🔴'} 순손익: ${net_pnl:+,.2f} ({net_pnl_pct:+.2f}%)")
                print(f"   ROE (순): {net_roe:+.2f}%")
                print(f"   일일 손익: ${self.daily_pnl:+,.2f}")
                print(f"{'=' * 80}\n")
        except Exception as e:
            print(f"⚠️  포지션 청산 실패 ({position.symbol}): {e}")

    def save_trades(self):
        """거래 내역 저장"""
        if not self.trades:
            return

        data = [asdict(t) for t in self.trades]
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def save_trades(self):
        """거래 내역 저장"""
        if not self.trades:
            return

        data = [asdict(t) for t in self.trades]
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n💾 거래 내역 저장: {TRADE_LOG_FILE}")


manager = TradeManager()


# ===== API 연결 테스트 =====
def test_api_connection() -> bool:
    """API 연결 테스트"""
    print("\n🔍 API 연결 테스트 중...")

    # 잔고 조회
    balance = API.get_balance()
    if balance > 0:
        print(f"   ✓ 잔고 조회 성공: ${balance:,.2f}")
    else:
        print(f"   ✗ 잔고 조회 실패")
        return False

    # Ticker 조회
    ticker = API.get_ticker("BTCUSDT")
    if ticker and ticker.get("lastPrice"):
        print(f"   ✓ Ticker 조회 성공: ${float(ticker['lastPrice']):,.2f}")
    else:
        print(f"   ✗ Ticker 조회 실패")
        return False

    # Klines 조회
    klines = API.get_kline("BTCUSDT", interval="5", limit=10)
    if klines:
        print(f"   ✓ Klines 조회 성공: {len(klines)}개 캔들")
    else:
        print(f"   ✗ Klines 조회 실패")
        return False

    print("\n✅ API 연결 정상\n")
    return True


# ===== 대시보드 출력 =====
def print_dashboard(balance: float, positions: List[Position], manager: TradeManager, prices: Dict[str, float]):
    """대시보드 출력"""
    os.system('clear' if os.name == 'posix' else 'cls')

    print("\n" + "=" * 110)
    print(f"{'⚡ 실전 자동 트레이딩 V2 (레버리지 ' + str(LEVERAGE) + 'x)':^110}")
    print("=" * 110)

    # 계좌 정보
    unrealized_pnl = sum(p.unrealized_pnl for p in positions)
    total_value = balance + unrealized_pnl
    used_margin = len(positions) * MARGIN_PER_POSITION

    # 실제 일일 손익 = 현재 총 자산 - 초기 잔고
    actual_daily_pnl = total_value - manager.initial_balance if manager.initial_balance > 0 else manager.daily_pnl

    print(f"\n💰 계좌 현황")
    print(f"   현재 잔고:     ${balance:>12,.2f}")
    print(f"   사용 증거금:   ${used_margin:>12,.2f} ({len(positions)}/{MAX_POSITIONS} 포지션)")
    print(f"   사용 가능:     ${balance - used_margin:>12,.2f}")
    print(f"   평가 손익:     ${unrealized_pnl:>+12,.2f}")
    print(f"   총 자산:       ${total_value:>12,.2f}")
    print(f"   일일 손익:     ${actual_daily_pnl:>+12,.2f}")
    if manager.initial_balance > 0:
        daily_return_pct = (actual_daily_pnl / manager.initial_balance) * 100
        print(f"   일일 수익률:   {daily_return_pct:>+12.2f}%")

    # 포지션
    if positions:
        print(f"\n📍 보유 포지션 ({len(positions)}/{MAX_POSITIONS})")
        print(f"{'심볼':^12} | {'방향':^8} | {'진입가':^12} | {'현재가':^12} | {'손익 (수수료 포함)':^30} | {'레버리지':^10}")
        print("-" * 110)

        for pos in positions:
            current_price = prices.get(pos.symbol, pos.entry_price)

            # ✅ 예상 청산 수수료 계산
            position_value = pos.entry_price * pos.size
            entry_fee = position_value * TAKER_FEE
            exit_fee = current_price * pos.size * TAKER_FEE
            total_fee = entry_fee + exit_fee

            # 순손익 = 미실현 손익 - 수수료
            net_unrealized_pnl = pos.unrealized_pnl - total_fee
            net_roe = (net_unrealized_pnl / MARGIN_PER_POSITION) * 100

            emoji = "📈" if pos.get_direction() == "Long" else "📉"
            pnl_emoji = "🟢" if net_unrealized_pnl > 0 else "🔴"

            print(f"{pos.symbol:^12} | {emoji} {pos.get_direction():^6} | ${pos.entry_price:>10,.4f} | "
                  f"${current_price:>10,.4f} | {pnl_emoji} ${net_unrealized_pnl:>+8,.2f} ({net_roe:>+6.1f}%) 💰-${total_fee:.2f} | "
                  f"{pos.leverage:>8}x")
    else:
        print(f"\n📍 보유 포지션: 없음")

    # 통계
    if manager.trades:
        # ✅ 순손익 기준으로 승패 계산
        wins = sum(1 for t in manager.trades if t.net_pnl > 0)
        losses = sum(1 for t in manager.trades if t.net_pnl <= 0)
        total_trades = len(manager.trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        # 순손익 통계
        avg_net_pnl = sum(t.net_pnl for t in manager.trades) / total_trades if total_trades > 0 else 0
        avg_net_roe = (avg_net_pnl / MARGIN_PER_POSITION) * 100
        max_net_pnl = max([t.net_pnl for t in manager.trades]) if manager.trades else 0
        min_net_pnl = min([t.net_pnl for t in manager.trades]) if manager.trades else 0
        total_fees = sum(t.total_fee for t in manager.trades)

        # 수수료 전 손익 (참고용)
        avg_pnl = sum(t.pnl for t in manager.trades) / total_trades if total_trades > 0 else 0

        print(f"\n📊 거래 통계")
        print(f"   총 거래:       {total_trades:>3}회")
        print(f"   승률:          {win_rate:>6.1f}% ({wins}승 {losses}패)")
        print(f"   💰 총 수수료:  ${total_fees:>+12,.2f}")
        print(f"   평균 손익 (수수료 전): ${avg_pnl:>+12,.2f}")
        print(f"   평균 순손익:   ${avg_net_pnl:>+12,.2f}")
        print(f"   평균 순ROE:    {avg_net_roe:>+6.1f}%")
        print(f"   최대 순수익:   ${max_net_pnl:>12,.2f}")
        print(f"   최대 순손실:   ${min_net_pnl:>12,.2f}")

        if wins > 0 and losses > 0:
            avg_win = sum(t.net_pnl for t in manager.trades if t.net_pnl > 0) / wins
            avg_loss = sum(t.net_pnl for t in manager.trades if t.net_pnl <= 0) / losses
            rr = abs(avg_win / avg_loss)
            print(f"   Risk/Reward:   {rr:>6.2f}")

    print("\n" + "=" * 110)


def main():
    # 경고 메시지
    mode = "🧪 TESTNET" if USE_TESTNET else "🔴 LIVE"
    print("\n" + "=" * 100)
    print(f"{'⚠️  실전 자동 트레이딩 시스템 V2  ⚠️':^100}")
    print(f"{mode:^100}")
    print("=" * 100)
    print("\n⚠️  경고:")
    print("   - 이 프로그램은 실제 자금을 사용합니다")
    print("   - 투자 손실의 위험이 있습니다")
    print("   - 본인 책임 하에 사용하세요")
    print("\nAPI 설정:")
    print(f"   - API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    print(f"   - 모드: {'Testnet' if USE_TESTNET else 'Mainnet (실전!)'}")
    print("\n거래 설정:")
    print(f"   - 포지션 모드: {POSITION_MODE.upper()}")
    print(f"   - 레버리지: {LEVERAGE}x")
    print(f"   - 증거금/포지션: ${MARGIN_PER_POSITION}")
    print(f"   - 포지션 크기: ${MARGIN_PER_POSITION * LEVERAGE} (증거금 × 레버리지)")
    print(f"   - 최대 포지션: {MAX_POSITIONS}개")
    print(f"   - 일일 최대 손실: ${MAX_DAILY_LOSS}")
    print(f"   - 신뢰도 임계값: {CONF_THRESHOLD:.0%}")
    print(f"   - 모델: AdvancedTCN_V2 ({len(MODELS)}개 심볼)")
    print(f"\n💰 수수료:")
    print(f"   - Taker Fee: {TAKER_FEE:.3%} (Market order)")
    print(f"   - Maker Fee: {MAKER_FEE:.3%} (Limit order)")
    print(f"   - 포지션당 예상 총 수수료: ${MARGIN_PER_POSITION * LEVERAGE * TAKER_FEE * 2:.2f}")
    print(f"   - 손익분기점: {(TAKER_FEE * 2) * 100:.2f}% (양방향 수수료)")

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

            # 계좌 정보
            balance = API.get_balance()
            positions = API.get_positions()

            # 첫 실행 시 초기 잔고 설정
            if loop_count == 1:
                manager.set_initial_balance(balance)

            # 현재 가격 가져오기
            prices = {}
            for symbol in MODELS.keys():
                ticker = API.get_ticker(symbol)
                prices[symbol] = float(ticker.get("lastPrice", 0))

            # 포지션 관리
            for position in positions:
                current_price = prices.get(position.symbol, position.entry_price)

                # 손절/익절 체크
                if position.get_direction() == "Long":
                    stop_loss = position.entry_price * (1 - STOP_LOSS_PCT)
                    take_profit = position.entry_price * (1 + TAKE_PROFIT_PCT)

                    if current_price <= stop_loss:
                        manager.close_position(position, "Stop Loss")
                        continue
                    if current_price >= take_profit:
                        manager.close_position(position, "Take Profit")
                        continue
                else:  # Short
                    stop_loss = position.entry_price * (1 + STOP_LOSS_PCT)
                    take_profit = position.entry_price * (1 - TAKE_PROFIT_PCT)

                    if current_price >= stop_loss:
                        manager.close_position(position, "Stop Loss")
                        continue
                    if current_price <= take_profit:
                        manager.close_position(position, "Take Profit")
                        continue

                # 반대 신호로 청산
                result = predict_v2(position.symbol, debug=False)
                if "error" not in result and result.get("confidence", 0) >= CONF_THRESHOLD:
                    signal_dir = result["direction"]
                    if (position.get_direction() == "Long" and signal_dir == "Short") or \
                            (position.get_direction() == "Short" and signal_dir == "Long"):
                        manager.close_position(position, "Reverse Signal")

            # 대시보드 출력
            print_dashboard(balance, positions, manager, prices)

            # 신호 스캔
            print(f"\n🔍 신호 스캔 V2 ({len(MODELS)}개 심볼)")
            print(f"{'심볼':^12} | {'가격':^12} | {'방향':^10} | {'신뢰도':^8} | {'예상BPS':^10} | {'신호':^20}")
            print("-" * 90)

            debug_mode = (loop_count == 1)
            for symbol in MODELS.keys():
                result = predict_v2(symbol, debug=debug_mode)

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
                if manager.can_open_position(symbol, positions, balance) and confidence >= CONF_THRESHOLD:
                    if manager.open_position(symbol, direction, price, expected_bps, confidence):
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
            prices[symbol] = float(ticker.get("lastPrice", 0))

        print_dashboard(balance, positions, manager, prices)
        manager.save_trades()

        # 최종 통계
        if manager.trades:
            # ✅ 순손익 기준 통계
            wins = sum(1 for t in manager.trades if t.net_pnl > 0)
            losses = sum(1 for t in manager.trades if t.net_pnl <= 0)
            total_trades = len(manager.trades)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

            avg_net_pnl = sum(t.net_pnl for t in manager.trades) / total_trades if total_trades > 0 else 0
            avg_net_roe = (avg_net_pnl / MARGIN_PER_POSITION) * 100
            total_fees = sum(t.total_fee for t in manager.trades)

            # 실제 일일 손익 계산
            unrealized_pnl = sum(p.unrealized_pnl for p in positions)
            total_value = balance + unrealized_pnl
            actual_daily_pnl = total_value - manager.initial_balance if manager.initial_balance > 0 else 0

            print("\n" + "=" * 110)
            print(f"{'📊 최종 결과':^110}")
            print("=" * 110)
            print(f"   초기 잔고:     ${manager.initial_balance:,.2f}")
            print(f"   최종 잔고:     ${balance:,.2f}")
            print(f"   총 자산:       ${total_value:,.2f}")
            print(f"   일일 손익:     ${actual_daily_pnl:+,.2f}")
            print(f"   총 거래:       {total_trades}회")
            print(f"   승률:          {win_rate:.1f}% ({wins}승 {losses}패)")
            print(f"   💰 총 수수료:  ${total_fees:,.2f}")
            print(f"   평균 순손익:   ${avg_net_pnl:+,.2f}")
            print(f"   평균 순ROE:    {avg_net_roe:+.1f}%")
            print("=" * 110)

        if positions:
            print("\n⚠️  주의: 아직 포지션이 남아있습니다!")
            for pos in positions:
                print(f"   - {pos.symbol}: {pos.get_direction()} | ${pos.unrealized_pnl:+,.2f}")

        print("\n✅ 프로그램이 종료되었습니다.")


if __name__ == "__main__":
    print(f"현재 코드 설정: {POSITION_MODE}")
    main()