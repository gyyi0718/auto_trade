# bithumb_live_trading_v2_api1.0_fixed.py
# -*- coding: utf-8 -*-
"""
AdvancedTCN_V2 모델 기반 빗썸 현물 자동 트레이딩 시스템 (API 1.0 - 수정 완료)
⚠️  WARNING: 실제 자금을 사용합니다. 신중하게 사용하세요!
빗썸은 현물 거래만 가능하므로 Long(매수) 포지션만 지원합니다.

주요 수정사항:
1. 빗썸 API 1.0 정확한 인증 방식 적용 (pybithumb 참조)
2. base64 인코딩 추가
3. HMAC-SHA512 서명 방식 수정
"""
import os
import time
import hmac
import hashlib
import json
import warnings
import urllib.parse
import base64
import math
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

warnings.filterwarnings("ignore")

# 테스트 모드
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

# ===== CONFIG =====
API_KEY = os.getenv("BITHUMB_API_KEY", "e94fbe7305e50003e7a62de0fb6c0248")
API_SECRET = os.getenv("BITHUMB_API_SECRET", "393b3ad30c006963f053dcc269b7814a")

if not API_KEY or not API_SECRET:
    print("❌ ERROR: BITHUMB_API_KEY 및 BITHUMB_API_SECRET 환경변수를 설정하세요!")
    if not DRY_RUN:
        exit(1)

# 거래 설정
SYMBOLS = os.getenv("SYMBOLS", "TRUMP,YB,SYRUP,DBR,FF,PUMP").split(",")
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "5"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.55"))

# ✅ 심볼별 V2 모델 경로
MODEL_PATHS = {
    "TRUMP": "D:/ygy_work/coin/bithumb/models_5min_trump_v2/model_v2_best.pt",
    "YB": "D:/ygy_work/coin/bithumb/models_5min_yb_v2/model_v2_best.pt",
    "SYRUP": "D:/ygy_work/coin/bithumb/models_5min_syrup_v2/model_v2_best.pt",
    "DBR": "D:/ygy_work/coin/bithumb/models_5min_dbr_v2/model_v2_best.pt",
    "FF": "D:/ygy_work/coin/bithumb/models_5min_ff_v2/model_v2_best.pt",
    "PUMP": "D:/ygy_work/coin/bithumb/models_5min_pump_v2/model_v2_best.pt",
}

# 리스크 관리
INVESTMENT_PER_POSITION = float(os.getenv("INVESTMENT_PER_POSITION", "50000"))  # ✅ 50,000원으로 조정
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.03"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.05"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "300000"))
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "60"))
MIN_ORDER_AMOUNT = float(os.getenv("MIN_ORDER_AMOUNT", "5000"))  # ✅ 최소 주문 금액 (KRW)

# 로그
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "bithumb_trades_v2.json")
ORDER_LOG_FILE = os.getenv("ORDER_LOG_FILE", "bithumb_orders_v2.json")
POSITION_LOG_FILE = os.getenv("POSITION_LOG_FILE", "bithumb_positions_v2.json")
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"

# ===== 빗썸 API 1.0 엔드포인트 =====
BASE_URL = "https://api.bithumb.com"


# ===== 데이터 클래스 =====
@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    size: float
    entry_price: float
    investment: float
    unrealized_pnl: float
    entry_time: str

    def get_direction(self) -> str:
        return "Long"


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
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float
    exit_reason: str
    investment: float = 0.0
    expected_bps: float = 0.0
    model_confidence: float = 0.0


# ===== AdvancedTCN_V2 모델 =====
class GatedResidualBlock(nn.Module):
    """Gated Residual TCN Block"""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()

        pad = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        )
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        )

        self.gate = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, 1)
        )

        self.chomp1 = nn.ConstantPad1d((0, -pad), 0) if pad > 0 else nn.Identity()
        self.chomp2 = nn.ConstantPad1d((0, -pad), 0) if pad > 0 else nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.chomp2(out)

        gate = torch.sigmoid(self.gate(out))
        out = out * gate

        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        out = out + res

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

        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        output = self.out_linear(context)
        output = self.dropout(output)

        output = self.layer_norm(x + output)

        return output


class AdvancedTCN_V2(nn.Module):
    """V2: 멀티태스크 학습"""

    def __init__(self, in_features=10, hidden=64, levels=4, dropout=0.2, num_heads=4):
        super().__init__()

        self.input_proj = nn.Linear(in_features, hidden)

        self.tcn_blocks = nn.ModuleList()
        for i in range(levels):
            self.tcn_blocks.append(
                GatedResidualBlock(hidden, hidden, kernel_size=3,
                                   dilation=2 ** i, dropout=dropout)
            )

        self.attention = MultiHeadSelfAttention(hidden, num_heads, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.head_direction = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2)
        )

        self.head_return = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

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

        x_proj = self.input_proj(x)

        x_tcn = x_proj.transpose(1, 2)
        for block in self.tcn_blocks:
            x_tcn = block(x_tcn)
        x_tcn = x_tcn[:, :, -1]

        x_attn = self.attention(x_proj)
        x_attn = x_attn.mean(dim=1)

        x_fused = torch.cat([x_tcn, x_attn], dim=1)
        x_fused = self.fusion(x_fused)

        direction_logits = self.head_direction(x_fused)
        expected_return = self.head_return(x_fused).squeeze(-1)
        confidence = self.head_confidence(x_fused).squeeze(-1)

        return direction_logits, expected_return, confidence


# ===== 빗썸 API 1.0 클래스 (수정 완료) =====
class BithumbAPI_v1_Fixed:
    """
    빗썸 API 1.0 (pybithumb 참조한 정확한 인증 방식)
    """

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret.encode('utf-8')  # bytes로 변환
        self.session = requests.Session()

    def _get_nonce(self) -> str:
        """Nonce 생성 (밀리초 타임스탬프)"""
        return str(int(time.time() * 1000))

    def _generate_signature(self, endpoint: str, params: Dict = None) -> Tuple[str, str]:
        """
        빗썸 API 1.0 서명 생성 (pybithumb 방식)

        서명 생성 방법:
        1. query_string = endpoint + chr(0) + urlencode(params) + chr(0) + nonce
        2. HMAC-SHA512로 서명 (api_secret을 bytes로)
        3. hexdigest()를 encode('utf-8')한 후 base64 인코딩
        """
        nonce = self._get_nonce()

        # 파라미터를 쿼리 스트링으로 변환
        if params:
            query_string = urllib.parse.urlencode(params)
        else:
            query_string = ""

        # 서명할 데이터: endpoint + chr(0) + query_string + chr(0) + nonce
        sign_data = endpoint + chr(0) + query_string + chr(0) + nonce

        # HMAC-SHA512 서명
        h = hmac.new(
            self.api_secret,
            sign_data.encode('utf-8'),
            hashlib.sha512
        )

        # hexdigest()를 다시 인코딩한 후 base64
        signature = base64.b64encode(h.hexdigest().encode('utf-8'))

        return signature.decode('utf-8'), nonce

    def _request_public(self, endpoint: str) -> Dict:
        """Public API 요청"""
        url = f"{BASE_URL}{endpoint}"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('status') != '0000':
                error_msg = data.get('message', 'Unknown error')
                if DEBUG_MODE:
                    print(f"[API ERROR] {endpoint}: {error_msg}")
                return {'status': 'error', 'message': error_msg}

            return data

        except Exception as e:
            if DEBUG_MODE:
                print(f"[REQUEST ERROR] {endpoint}: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _request_private(self, endpoint: str, params: Dict = None) -> Dict:
        """Private API 요청"""
        if params is None:
            params = {}

        # endpoint 추가
        params['endpoint'] = endpoint

        # 서명 생성
        signature, nonce = self._generate_signature(endpoint, params)

        # HTTP 헤더
        headers = {
            'Api-Key': self.api_key,
            'Api-Sign': signature,
            'Api-Nonce': nonce,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        url = f"{BASE_URL}{endpoint}"

        try:
            response = self.session.post(
                url,
                headers=headers,
                data=urllib.parse.urlencode(params),
                timeout=10
            )

            response.raise_for_status()
            data = response.json()

            if data.get('status') != '0000':
                error_msg = data.get('message', 'Unknown error')
                error_code = data.get('status', 'Unknown')
                if DEBUG_MODE:
                    print(f"[API ERROR] {endpoint}")
                    print(f"   Status Code: {error_code}")
                    print(f"   Message: {error_msg}")
                    print(f"   Params: {params}")
                # 에러 메시지에 status code 포함
                return {'status': 'error', 'message': f"[{error_code}] {error_msg}"}

            return data

        except Exception as e:
            if DEBUG_MODE:
                print(f"[REQUEST ERROR] {endpoint}: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_ticker(self, symbol: str) -> Dict:
        """현재가 조회 (Public API)"""
        endpoint = f"/public/ticker/{symbol}_KRW"
        result = self._request_public(endpoint)

        if result.get('status') == 'error':
            return {}

        data = result.get('data', {})

        try:
            return {
                'last_price': float(data.get('closing_price', 0)),
                'volume': float(data.get('units_traded_24H', 0)),
                'high': float(data.get('max_price', 0)),
                'low': float(data.get('min_price', 0)),
                'opening_price': float(data.get('opening_price', 0))
            }
        except (ValueError, TypeError):
            return {}

    def get_candlestick(self, symbol: str, interval: str = "5m") -> List[Dict]:
        """캔들스틱 데이터 조회"""
        endpoint = f"/public/candlestick/{symbol}_KRW/{interval}"
        result = self._request_public(endpoint)

        if result.get('status') == 'error':
            return []

        candles = result.get('data', [])
        return candles

    def get_balance(self) -> Tuple[float, Dict[str, float]]:
        """잔고 조회 (Private API)"""
        endpoint = "/info/balance"
        params = {'currency': 'ALL'}

        result = self._request_private(endpoint, params)

        if result.get('status') == 'error':
            return 0.0, {}

        data = result.get('data', {})

        # KRW 잔고 (사용 가능 금액)
        try:
            krw_balance = float(data.get('available_krw', 0))
        except (ValueError, TypeError):
            krw_balance = 0.0

        # 코인 잔고
        balances = {}
        for symbol in SYMBOLS:
            key = f"available_{symbol.lower()}"
            try:
                qty = float(data.get(key, 0))
                if qty > 0:
                    balances[symbol] = qty
            except (ValueError, TypeError):
                continue

        return krw_balance, balances

    def place_market_buy(self, symbol: str, krw_amount: float) -> Optional[str]:
        """
        시장가 매수 (수정됨)

        Args:
            symbol: 코인 심볼
            krw_amount: 매수 금액 (KRW)
        """
        # ✅ FIX: 현재가 조회 후 코인 수량 계산
        ticker = self.get_ticker(symbol)
        if not ticker or 'last_price' not in ticker:
            print(f"   [매수 실패] {symbol}: 현재가 조회 실패")
            return None

        current_price = ticker['last_price']
        coin_units = krw_amount / current_price  # 코인 수량 계산

        endpoint = "/trade/market_buy"
        params = {
            'order_currency': symbol,
            'payment_currency': 'KRW',
            'units': f"{coin_units:.8f}"  # ✅ 코인 수량으로 전달!
        }

        if DEBUG_MODE:
            print(f"   [매수 요청 상세]")
            print(f"      현재가: ₩{current_price:,.2f}")
            print(f"      투자금: ₩{krw_amount:,.0f}")
            print(f"      수량: {coin_units:.8f} {symbol}")
            print(f"      예상: ₩{coin_units * current_price:,.0f}")

        result = self._request_private(endpoint, params)

        if result.get('status') == 'error':
            error_msg = result.get('message', 'Unknown error')
            print(f"   [매수 실패] {symbol}: {error_msg}")
            if DEBUG_MODE and '[5600]' in error_msg:
                print(f"      디버그: units={coin_units:.8f}, price={current_price:,.2f}")
            return None

        order_id = result.get('order_id')
        if not order_id:
            data = result.get('data', {})
            order_id = data.get('order_id')

        return order_id

    def place_market_sell(self, symbol: str, units: float) -> Optional[str]:
        """
        시장가 매도

        Args:
            symbol: 코인 심볼
            units: 매도 수량
        """
        endpoint = "/trade/market_sell"
        params = {
            'order_currency': symbol,
            'payment_currency': 'KRW',
            'units': str(units)
        }

        result = self._request_private(endpoint, params)

        if result.get('status') == 'error':
            error_msg = result.get('message', 'Unknown error')
            print(f"   [매도 실패] {symbol}: {error_msg}")
            return None

        order_id = result.get('order_id')
        if not order_id:
            data = result.get('data', {})
            order_id = data.get('order_id')

        return order_id


# ===== 전역 변수 =====
API = BithumbAPI_v1_Fixed(API_KEY, API_SECRET)
MODELS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== API 연결 테스트 =====
def test_api_connection() -> bool:
    """API 연결 테스트"""
    print("\n🔌 빗썸 API 1.0 연결 테스트 중...")

    # 1. Public API 테스트
    print("   1. Public API (ticker) 테스트...")
    ticker = API.get_ticker("BTC")
    if not ticker or 'last_price' not in ticker:
        print("      ❌ Public API 실패")
        return False
    print(f"      ✅ BTC 현재가: ₩{ticker['last_price']:,.0f}")

    # 2. Private API 테스트
    print("   2. Private API (balance) 테스트...")
    krw_balance, balances = API.get_balance()

    print(f"      ✅ KRW 잔고: ₩{krw_balance:,.0f}")
    if balances:
        print(f"      ✅ 보유 코인: {list(balances.keys())}")

    print("\n✅ 빗썸 API 1.0 연결 성공!\n")
    return True


# ===== 모델 로딩 =====
def load_models():
    """심볼별 모델 로딩"""
    print("\n📦 모델 로딩 중...")

    for symbol, path in MODEL_PATHS.items():
        if symbol not in SYMBOLS:
            continue

        if not os.path.exists(path):
            print(f"   ⚠️  {symbol}: 모델 파일 없음 ({path})")
            continue

        try:
            print(f"   🔄 {symbol} 로딩 중...")

            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

            if isinstance(checkpoint, dict):
                if 'model_state' in checkpoint:
                    state_dict = checkpoint['model_state']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif any(k.startswith(('tcn.', 'attention.', 'fc')) for k in checkpoint.keys()):
                    state_dict = checkpoint
                else:
                    print(f"   ❌ {symbol}: 알 수 없는 체크포인트 형식")
                    continue
            else:
                state_dict = checkpoint

            if 'feat_cols' in checkpoint:
                in_features = len(checkpoint['feat_cols'])
            else:
                in_features = 10

            model = AdvancedTCN_V2(
                in_features=in_features,
                hidden=64,
                levels=4,
                dropout=0.2,
                num_heads=4
            )
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()

            MODELS[symbol] = model
            print(f"   ✅ {symbol}: 모델 로드 완료")

        except Exception as e:
            print(f"   ❌ {symbol}: 모델 로드 실패 - {str(e)[:100]}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()

    if not MODELS:
        print("\n❌ ERROR: 로드된 모델이 없습니다!")
        exit(1)

    print(f"\n✅ {len(MODELS)}개 모델 로드 완료\n")


# ===== 피처 생성 =====
def make_features_v2_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """학습 시와 동일한 V2 피처 생성"""
    g = df.copy()

    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g["logc"].diff().fillna(0.0)

    for w in (6, 12, 24, 72):
        g[f"rv{w}"] = g["ret1"].rolling(w, min_periods=max(2, w // 3)).std().fillna(0.0)

    for w in (6, 12, 24, 72):
        ema = g["close"].ewm(span=w, adjust=False).mean()
        g[f"mom{w}"] = ((g["close"] - ema) / ema).fillna(0.0)

    for w in (12, 24, 72):
        mu = g["volume"].rolling(w, min_periods=max(2, w // 3)).mean()
        sd = g["volume"].rolling(w, min_periods=max(2, w // 3)).std().replace(0, 1.0)
        g[f"vz{w}"] = ((g["volume"] - mu) / sd).fillna(0.0).clip(-5, 5)

    g["vol_change"] = g["volume"].pct_change().fillna(0.0).clip(-2, 2)

    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=5).mean().fillna(0.0)
    g["atr_ratio"] = (atr / g["close"]).fillna(0.0).clip(0, 0.1)

    g["hl_ratio"] = ((g["high"] - g["low"]) / g["close"]).fillna(0.0).clip(0, 0.1)
    g["close_pos"] = ((g["close"] - g["low"]) / (g["high"] - g["low"] + 1e-10)).fillna(0.5).clip(0, 1)
    g["body_ratio"] = ((g["close"] - g["open"]).abs() / g["close"]).fillna(0.0).clip(0, 0.1)

    delta = g["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    g["rsi14"] = ((rsi - 50) / 50).fillna(0.0).clip(-1, 1)

    ema12 = g["close"].ewm(span=12, adjust=False).mean()
    ema26 = g["close"].ewm(span=26, adjust=False).mean()
    g["macd"] = ((ema12 - ema26) / g["close"]).fillna(0.0).clip(-0.1, 0.1)

    if 'timestamp' in g.columns:
        hod = pd.to_datetime(g["timestamp"]).dt.hour
        dow = pd.to_datetime(g["timestamp"]).dt.dayofweek
    else:
        now = datetime.now()
        hod = pd.Series([now.hour] * len(g))
        dow = pd.Series([now.weekday()] * len(g))

    g["hour_sin"] = np.sin(2 * np.pi * hod / 24.0)
    g["hour_cos"] = np.cos(2 * np.pi * hod / 24.0)
    g["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    g["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    for w in (12, 24):
        high_max = g["high"].rolling(w, min_periods=max(2, w // 3)).max()
        low_min = g["low"].rolling(w, min_periods=max(2, w // 3)).min()
        g[f"vs_high{w}"] = ((g["close"] - high_max) / high_max).fillna(0.0).clip(-0.1, 0)
        g[f"vs_low{w}"] = ((g["close"] - low_min) / low_min).fillna(0.0).clip(0, 0.1)

    g["mom_accel"] = g["mom6"].diff().fillna(0.0).clip(-0.1, 0.1)
    g["vol_change_rate"] = g["rv6"].pct_change().fillna(0.0).clip(-1, 1)

    g = g.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    return g


def get_recent_candles(symbol: str, interval: str = "5m", limit: int = 200) -> pd.DataFrame:
    """최근 캔들 데이터 가져오기"""
    try:
        candles = API.get_candlestick(symbol, interval)

        if not candles:
            return pd.DataFrame()

        candles = candles[:limit]

        df = pd.DataFrame(
            candles,
            columns=['timestamp', 'open', 'close', 'high', 'low', 'volume']
        )

        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    except Exception as e:
        if DEBUG_MODE:
            print(f"[ERROR] {symbol} 캔들 데이터 로드 실패: {e}")
        return pd.DataFrame()


def prepare_features(df: pd.DataFrame, feat_cols: List[str], seq_len: int = 72) -> Optional[torch.Tensor]:
    """특성 준비"""
    if len(df) < seq_len + 100:
        return None

    df_feat = make_features_v2_for_prediction(df)

    available_feats = [f for f in feat_cols if f in df_feat.columns]
    if len(available_feats) < len(feat_cols):
        if DEBUG_MODE:
            print(f"[WARN] 일부 피처 누락: {len(available_feats)}/{len(feat_cols)}")

    features = df_feat[available_feats].iloc[-seq_len:].values
    features = np.nan_to_num(features, 0.0)

    tensor = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)

    return tensor


def predict_v2(symbol: str, debug: bool = False) -> Dict:
    """V2 모델로 예측"""
    if symbol not in MODELS:
        return {"error": "모델 없음"}

    df = get_recent_candles(symbol, interval="5m", limit=300)

    if df.empty or len(df) < 150:
        return {"error": "데이터 부족"}

    feat_cols = [
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

    X = prepare_features(df, feat_cols, seq_len=72)

    if X is None:
        return {"error": "특성 준비 실패"}

    model = MODELS[symbol]

    with torch.no_grad():
        direction_logits, expected_return, confidence = model(X)

        probs = F.softmax(direction_logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_confidence = probs[0, pred_class].item()

        expected_bps = abs(expected_return[0].item() * 1000)
        model_confidence = confidence[0].item()

    direction = "Long" if pred_class == 0 else "Short"
    current_price = float(df.iloc[-1]['close'])

    if debug:
        print(f"\n[DEBUG] {symbol}")
        print(f"   방향: {direction}")
        print(f"   방향 신뢰도: {pred_confidence:.1%}")
        print(f"   모델 신뢰도: {model_confidence:.1%}")
        print(f"   예상 BPS: {expected_bps:.1f}")
        print(f"   현재가: ₩{current_price:,.0f}")

    return {
        "direction": direction,
        "confidence": model_confidence,
        "expected_bps": expected_bps,
        "current_price": current_price
    }


# ===== 포지션 매니저 =====
class PositionManager:
    """포지션 관리"""

    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.load_trades()
        self.load_positions()  # 포지션 로드

    def load_trades(self):
        """거래 기록 로드"""
        if os.path.exists(TRADE_LOG_FILE):
            try:
                with open(TRADE_LOG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.trades = [Trade(**t) for t in data]

                    today = datetime.now().strftime("%Y-%m-%d")
                    self.daily_pnl = sum(
                        t.pnl for t in self.trades
                        if t.exit_time.startswith(today)
                    )
            except Exception as e:
                if DEBUG_MODE:
                    print(f"[WARN] 거래 기록 로드 실패: {e}")

    def save_trades(self):
        """거래 기록 저장"""
        try:
            with open(TRADE_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump([asdict(t) for t in self.trades], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] 거래 기록 저장 실패: {e}")

    def load_positions(self):
        """저장된 포지션 로드"""
        if os.path.exists(POSITION_LOG_FILE):
            try:
                with open(POSITION_LOG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for p in data:
                        position = Position(**p)
                        self.positions[position.symbol] = position

                if self.positions:
                    print(f"\n✅ {len(self.positions)}개 포지션 로드:")
                    for symbol, pos in self.positions.items():
                        print(f"   - {symbol}: {pos.size:.6f}개 (진입가: ₩{pos.entry_price:,.0f})")

            except Exception as e:
                if DEBUG_MODE:
                    print(f"[WARN] 포지션 로드 실패: {e}")

    def save_positions(self):
        """현재 포지션 저장"""
        try:
            with open(POSITION_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump([asdict(p) for p in self.positions.values()],
                          f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] 포지션 저장 실패: {e}")

    def can_open_position(self, symbol: str, current_positions: Dict[str, Position]) -> bool:
        """포지션 오픈 가능 여부"""
        if symbol in current_positions:
            return False

        if len(current_positions) >= MAX_POSITIONS:
            return False

        if self.daily_pnl < -MAX_DAILY_LOSS:
            print(f"   ⚠️  일일 손실 한도 도달 (₩{self.daily_pnl:,.0f})")
            return False

        return True

    def open_position(self, symbol: str, direction: str, price: float,
                      expected_bps: float, confidence: float, krw_balance: float = 0) -> bool:
        """포지션 오픈"""
        if direction != "Long":
            return False

        if DRY_RUN:
            print(f"\n🧪 [DRY RUN] 매수 신호!")
            print(f"   심볼:     {symbol}")
            print(f"   가격:     ₩{price:,.0f}")
            print(f"   투자금:   ₩{INVESTMENT_PER_POSITION:,.0f}")
            print(f"   신뢰도:   {confidence:.1%}")
            print(f"   (실제 주문은 실행되지 않음)")
            return False

        # ✅ 최소 주문 금액 체크
        if INVESTMENT_PER_POSITION < MIN_ORDER_AMOUNT:
            print(f"   ⚠️  {symbol} 주문 금액이 최소 주문 금액(₩{MIN_ORDER_AMOUNT:,.0f})보다 작습니다")
            return False

        # ✅ 잔고 체크 (수수료 0.25% + 안전 마진 고려)
        fee_rate = 0.0025  # 빗썸 기본 수수료 0.25%
        safety_margin = 1.01  # 1% 안전 마진
        required_balance = INVESTMENT_PER_POSITION * (1 + fee_rate) * safety_margin

        if krw_balance > 0 and krw_balance < required_balance:
            print(f"   ⚠️  {symbol} 잔고 부족: ₩{krw_balance:,.0f} < ₩{required_balance:,.0f} (필요)")
            return False

        # ✅ 실제 주문 금액 (잔고가 부족하면 조정)
        actual_order_amount = INVESTMENT_PER_POSITION
        if krw_balance > 0:
            # 수수료를 고려한 최대 주문 가능 금액
            max_order_amount = krw_balance / (1 + fee_rate + 0.01)  # 1% 안전 마진
            actual_order_amount = min(INVESTMENT_PER_POSITION, max_order_amount)
            actual_order_amount = int(actual_order_amount)  # 정수로 변환

            if actual_order_amount < MIN_ORDER_AMOUNT:
                print(f"   ⚠️  {symbol} 조정된 주문 금액(₩{actual_order_amount:,.0f})이 최소 금액 미달")
                return False

        order_id = API.place_market_buy(symbol, actual_order_amount)

        if not order_id:
            print(f"   ❌ {symbol} 매수 주문 실패")
            return False

        qty = actual_order_amount / price

        position = Position(
            symbol=symbol,
            size=qty,
            entry_price=price,
            investment=actual_order_amount,
            unrealized_pnl=0.0,
            entry_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        self.positions[symbol] = position
        self.save_positions()  # 포지션 저장

        print(f"\n✅ {symbol} 매수 완료!")
        print(f"   주문 ID:  {order_id}")
        print(f"   가격:     ₩{price:,.0f}")
        print(f"   수량:     {qty:.6f}")
        print(f"   투자금:   ₩{actual_order_amount:,.0f}")
        if actual_order_amount != INVESTMENT_PER_POSITION:
            print(f"   (목표 투자금 ₩{INVESTMENT_PER_POSITION:,.0f}에서 조정됨)")

        return True

    def close_position(self, position: Position, reason: str, current_price: float) -> bool:
        """포지션 청산"""
        if DRY_RUN:
            print(f"\n🧪 [DRY RUN] 청산 신호!")
            print(f"   심볼:     {position.symbol}")
            print(f"   사유:     {reason}")
            print(f"   (실제 주문은 실행되지 않음)")
            return False

        order_id = API.place_market_sell(position.symbol, position.size)

        if not order_id:
            print(f"   ❌ {position.symbol} 매도 주문 실패")
            return False

        pnl = (current_price - position.entry_price) * position.size
        pnl_pct = (current_price - position.entry_price) / position.entry_price

        trade = Trade(
            symbol=position.symbol,
            direction="Long",
            entry_price=position.entry_price,
            exit_price=current_price,
            size=position.size,
            entry_time=position.entry_time,
            exit_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            investment=position.investment
        )

        self.trades.append(trade)
        self.daily_pnl += pnl
        self.save_trades()

        del self.positions[position.symbol]

        print(f"\n✅ {position.symbol} 청산 완료!")
        print(f"   주문 ID:  {order_id}")
        print(f"   진입:     ₩{position.entry_price:,.0f}")
        print(f"   청산:     ₩{current_price:,.0f}")
        print(f"   손익:     ₩{pnl:+,.0f} ({pnl_pct:+.2%})")
        print(f"   사유:     {reason}")

        return True

    def update_positions(self, prices: Dict[str, float]):
        """포지션 평가손익 업데이트"""
        for symbol, position in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                position.unrealized_pnl = (current_price - position.entry_price) * position.size


# ===== 대시보드 =====
def print_dashboard(krw_balance: float, balances: Dict[str, float],
                    manager: PositionManager, prices: Dict[str, float]):
    """대시보드 출력"""
    os.system('clear' if os.name == 'posix' else 'cls')

    print("\n" + "=" * 110)
    print(f"{'🤖 빗썸 자동 트레이딩 시스템 V2 (API 1.0 Fixed)':^110}")
    print("=" * 110)

    total_investment = sum(p.investment for p in manager.positions.values())
    total_unrealized = sum(p.unrealized_pnl for p in manager.positions.values())
    total_value = krw_balance + total_investment + total_unrealized

    print(f"\n💰 계좌 정보")
    print(f"   KRW 잔고:      ₩{krw_balance:>12,.0f}")
    print(f"   보유 자산:     ₩{total_investment:>12,.0f}")
    print(f"   평가 손익:     ₩{total_unrealized:>+12,.0f}")
    print(f"   총 자산:       ₩{total_value:>12,.0f}")
    print(f"   일일 손익:     ₩{manager.daily_pnl:>+12,.0f}")

    if manager.positions:
        print(f"\n📍 보유 포지션: {len(manager.positions)}개")
        print(f"{'심볼':^10} | {'방향':^8} | {'진입가':^13} | {'현재가':^13} | "
              f"{'손익(%)':^12} | {'수량':^13}")
        print("-" * 90)

        for symbol, pos in manager.positions.items():
            current_price = prices.get(symbol, pos.entry_price)
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price

            print(f"{symbol:^10} | {'Long':^8} | ₩{pos.entry_price:>11,.0f} | "
                  f"₩{current_price:>11,.0f} | {pnl_pct:>+10.2%} | {pos.size:>13.6f}")
    else:
        print(f"\n📍 보유 포지션: 없음")

    if manager.trades:
        wins = sum(1 for t in manager.trades if t.pnl > 0)
        losses = sum(1 for t in manager.trades if t.pnl <= 0)
        total_trades = len(manager.trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = sum(t.pnl for t in manager.trades) / total_trades if total_trades > 0 else 0
        max_pnl = max([t.pnl for t in manager.trades]) if manager.trades else 0
        min_pnl = min([t.pnl for t in manager.trades]) if manager.trades else 0

        print(f"\n📊 거래 통계")
        print(f"   총 거래:       {total_trades:>3}회")
        print(f"   승률:          {win_rate:>6.1f}% ({wins}승 {losses}패)")
        print(f"   평균 손익:     ₩{avg_pnl:>+12,.0f}")
        print(f"   최대 수익:     ₩{max_pnl:>12,.0f}")
        print(f"   최대 손실:     ₩{min_pnl:>12,.0f}")

    print("\n" + "=" * 110)


# ===== 메인 =====
manager = PositionManager()


# ===== 포지션 동기화 =====
def sync_positions_with_api(manager: PositionManager, balances: Dict[str, float],
                            prices: Dict[str, float]):
    """API 잔고와 저장된 포지션 동기화"""
    api_symbols = set(balances.keys())
    manager_symbols = set(manager.positions.keys())

    # API에 있지만 manager에 없는 경우 → 경고만 (자동 추가 안함)
    for symbol in api_symbols - manager_symbols:
        if symbol in SYMBOLS and symbol in prices:
            qty = balances[symbol]
            price = prices[symbol]
            value = qty * price

            print(f"\n❌ 경고: {symbol} 포지션 감지됨 (프로그램 외부에서 매수)")
            print(f"   수량: {qty:.6f}개")
            print(f"   현재가: ₩{price:,.0f}")
            print(f"   평가액: ₩{value:,.0f}")
            print(f"\n   ⚠️  이 포지션은 프로그램이 관리하지 않습니다!")
            print(f"   이유: 진입가를 알 수 없어 손익 계산 불가")
            print(f"\n   해결책:")
            print(f"     1) 빗썸 앱/웹에서 수동으로 관리")
            print(f"     2) 수동 청산 후 프로그램으로 재매수")
            print(f"     3) 무시하고 프로그램 사용 (새 거래만)")

            # 포지션 추가 안함! 무시함

    # manager에 있지만 API에 없는 경우 → 제거
    for symbol in manager_symbols - api_symbols:
        print(f"\n⚠️  {symbol} 포지션 제거 (API에 없음, 수동 청산됨?)")
        del manager.positions[symbol]

    # 수량 불일치 → 업데이트
    for symbol in manager_symbols & api_symbols:
        manager_qty = manager.positions[symbol].size
        api_qty = balances[symbol]

        if abs(manager_qty - api_qty) > 0.00001:
            print(f"\n⚠️  {symbol} 수량 불일치: {manager_qty:.6f} → {api_qty:.6f}")
            manager.positions[symbol].size = api_qty
            manager.positions[symbol].investment = api_qty * manager.positions[symbol].entry_price

    if api_symbols != manager_symbols or len(manager_symbols & api_symbols) > 0:
        manager.save_positions()


def main():
    mode = "🧪 DRY RUN (테스트)" if DRY_RUN else "🔴 LIVE TRADING"
    print("\n" + "=" * 100)
    print(f"{'⚠️  빗썸 현물 자동 트레이딩 시스템 V2 (API 1.0 Fixed)  ⚠️':^100}")
    print(f"{mode:^100}")
    print("=" * 100)

    if DRY_RUN:
        print("\n💡 DRY RUN 모드:")
        print("   - 실제 거래는 실행되지 않습니다")
        print("   - 신호만 확인하고 테스트합니다")
        print("   - 실전 거래를 원하면: export DRY_RUN=0")
    else:
        print("\n⚠️  경고:")
        print("   - 이 프로그램은 실제 자금을 사용합니다")
        print("   - 투자 손실의 위험이 있습니다")
        print("   - 본인 책임 하에 사용하세요")

    print("\n거래 설정:")
    print(f"   - API 버전: 빗썸 API 1.0 (Fixed)")
    print(f"   - 투자금/종목: ₩{INVESTMENT_PER_POSITION:,.0f}")
    print(f"   - 최대 보유 종목: {MAX_POSITIONS}개")
    print(f"   - 신뢰도 임계값: {CONF_THRESHOLD:.0%}")
    print(f"   - 모델: AdvancedTCN_V2 ({len(MODEL_PATHS)}개 심볼)")

    print("\n계속하시겠습니까? (y/n): ", end='')
    if input().lower() != 'y':
        print("프로그램을 종료합니다.")
        return

    load_models()

    if not DRY_RUN:
        if not test_api_connection():
            print("\n❌ API 연결 실패. 프로그램을 종료합니다.")
            return
    else:
        print("\n🧪 DRY RUN 모드: API 연결 테스트 건너뜀")

    print("\n시작합니다...\n")

    try:
        loop_count = 0
        while True:
            loop_count += 1
            current_time = datetime.now()

            if not DRY_RUN:
                krw_balance, balances = API.get_balance()
            else:
                krw_balance = 0.0
                balances = {}

            prices = {}
            for symbol in SYMBOLS:
                ticker = API.get_ticker(symbol)
                if ticker:
                    prices[symbol] = ticker['last_price']

            # 첫 실행 시 포지션 동기화
            if loop_count == 1 and not DRY_RUN:
                sync_positions_with_api(manager, balances, prices)

            current_positions = {}
            if not DRY_RUN:
                for symbol, qty in balances.items():
                    if symbol in SYMBOLS and qty > 0:
                        price = prices.get(symbol, 0)
                        if price > 0:
                            position = Position(
                                symbol=symbol,
                                size=qty,
                                entry_price=price,
                                investment=qty * price,
                                unrealized_pnl=0,
                                entry_time=""
                            )
                            current_positions[symbol] = position

            if not DRY_RUN:
                manager.update_positions(prices)

                for symbol, position in list(manager.positions.items()):
                    current_price = prices.get(symbol, position.entry_price)

                    stop_loss = position.entry_price * (1 - STOP_LOSS_PCT)
                    take_profit = position.entry_price * (1 + TAKE_PROFIT_PCT)

                    if current_price <= stop_loss:
                        manager.close_position(position, "Stop Loss", current_price)
                        continue

                    if current_price >= take_profit:
                        manager.close_position(position, "Take Profit", current_price)
                        continue

                    result = predict_v2(symbol, debug=False)
                    if "error" not in result and result.get("confidence", 0) >= CONF_THRESHOLD:
                        signal_dir = result["direction"]
                        if signal_dir == "Short":
                            manager.close_position(position, "Reverse Signal", current_price)

            if not DRY_RUN:
                print_dashboard(krw_balance, balances, manager, prices)
            else:
                os.system('clear' if os.name == 'posix' else 'cls')
                print("\n" + "=" * 90)
                print(f"{'🧪 DRY RUN - 신호 테스트 모드 (API 1.0 Fixed)':^90}")
                print("=" * 90)

            print(f"\n🔍 신호 스캔 V2 ({len(MODELS)}개 심볼)")
            print(f"{'심볼':^12} | {'가격':^15} | {'방향':^10} | {'신뢰도':^8} | "
                  f"{'예상BPS':^10} | {'신호':^20}")
            print("-" * 90)

            debug_mode = (loop_count == 1)
            for symbol in MODELS.keys():
                result = predict_v2(symbol, debug=debug_mode)

                if "error" in result:
                    print(f"{symbol:^12} | {'N/A':^15} | {'오류':^10} | {'N/A':^8} | "
                          f"{'N/A':^10} | ❌ {result.get('error', '알 수 없음')}")
                    continue

                direction = result["direction"]
                confidence = result["confidence"]
                expected_bps = result["expected_bps"]
                price = result["current_price"]

                dir_icon = {"Long": "📈", "Short": "📉"}.get(direction, "❓")

                if confidence < CONF_THRESHOLD:
                    signal = f"⚠️  신호 약함"
                elif direction == "Long":
                    signal = f"🟢 매수 신호"
                else:
                    signal = f"🔴 매도 신호 (보류)"

                print(f"{symbol:^12} | ₩{price:>13,.0f} | {dir_icon} {direction:^8} | "
                      f"{confidence:>6.1%} | {expected_bps:>8.1f} | {signal}")

                if direction == "Long" and confidence >= CONF_THRESHOLD:
                    if DRY_RUN:
                        print(f"   🧪 [테스트] {symbol} 매수 조건 충족 (실제 거래 안함)")
                    elif manager.can_open_position(symbol, current_positions):
                        # ✅ krw_balance를 전달
                        if manager.open_position(symbol, direction, price, expected_bps, confidence, krw_balance):
                            krw_balance, balances = API.get_balance()  # 잔고 업데이트

            print(f"\n[스캔 #{loop_count}] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"다음 스캔까지 {INTERVAL_SEC}초... (Ctrl+C로 종료)")

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n프로그램 종료 중...")

        if not DRY_RUN:
            manager.save_positions()  # 포지션 저장
            krw_balance, balances = API.get_balance()
            prices = {}
            for symbol in SYMBOLS:
                ticker = API.get_ticker(symbol)
                if ticker:
                    prices[symbol] = ticker['last_price']

            print_dashboard(krw_balance, balances, manager, prices)
            manager.save_trades()

        print("\n✅ 프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()