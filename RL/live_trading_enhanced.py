# live_trading_rl_enhanced.py
# -*- coding: utf-8 -*-
"""
강화학습 모델 기반 실전 자동 트레이딩 시스템 (Enhanced 버전)
⚠️  WARNING: 실제 자금을 사용합니다. 신중하게 사용하세요!

- train_rl_enhanced.py로 학습한 PPO 모델 사용
- EnhancedCryptoTradingEnv와 동일한 feature & observation 구조
- Bybit API로 실시간 거래
- 3가지 액션: LONG, SHORT, CLOSE
- 레버리지 거래
- TP/SL 자동 설정
"""
import os
import time
import hmac
import hashlib
import json
import warnings
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from enum import IntEnum
import numpy as np
import pandas as pd
import requests
import certifi
import torch
from stable_baselines3 import PPO

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*Gym.*")
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# ===== CONFIG =====
# API 인증 (필수!)
API_KEY = "Dlp4eJD6YFmO99T8vC"
API_SECRET = "YYYB5tMw2TWvfVF5wqi6lQRHqEIiDSpDJF1U"

USE_TESTNET = os.getenv("USE_TESTNET", "0") == "1"

if not API_KEY or not API_SECRET:
    print("❌ ERROR: API_KEY 및 API_SECRET을 설정하세요!")
    exit(1)

# 거래 설정
MODEL_PATH = "rl_models_enhanced/BTCUSDT_5min_best/best_model.zip"  # 학습된 모델 경로
SYMBOL = "BTCUSDT"
INTERVAL_MINUTES = 5  # 캔들 간격 (분)
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "10"))  # 스캔 주기 (초)

# 리스크 관리
MARGIN_PER_POSITION = float(os.getenv("MARGIN_PER_POSITION", "200"))  # 포지션당 증거금
LEVERAGE = int(os.getenv("LEVERAGE", "10"))  # 레버리지 배율
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.05"))  # 손절 (5%)
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.08"))  # 익절 (8%)
LIQUIDATION_BUFFER = float(os.getenv("LIQUIDATION_BUFFER", "0.8"))  # 청산 버퍼
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "500"))  # 일일 최대 손실

# 로그
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "live_trades_rl_enhanced.json")
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"

# 기술적 지표 계산용 상수
WINDOW_SIZE = 30  # 모델 입력 윈도우 크기


# ===== 액션 정의 =====
class Actions(IntEnum):
    """강화학습 액션"""
    LONG = 0  # 롱 포지션
    SHORT = 1  # 숏 포지션
    CLOSE = 2  # 청산


# ===== 포지션 정보 =====
@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    leverage: int
    margin: float
    liquidation_price: float
    unrealized_pnl: float = 0.0
    position_value: float = 0.0

    def get_pnl(self, current_price: float) -> float:
        """손익 계산"""
        if self.direction == "long":
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size

    def get_roe(self, current_price: float) -> float:
        """ROE 계산 (레버리지 반영)"""
        if self.direction == "long":
            price_change_pct = (current_price / self.entry_price - 1) * 100
        else:
            price_change_pct = (1 - current_price / self.entry_price) * 100
        return price_change_pct * self.leverage

    def should_close(self, current_price: float) -> Tuple[bool, str]:
        """청산 여부 판단 - TP/SL만 체크"""
        # 청산가
        if self.direction == "long" and current_price <= self.liquidation_price:
            return True, "Liquidation"
        if self.direction == "short" and current_price >= self.liquidation_price:
            return True, "Liquidation"

        # 손절
        if self.direction == "long" and current_price <= self.stop_loss:
            print(f"[청산 조건] {self.symbol}: 손절 발동 (현재: ${current_price:.4f}, SL: ${self.stop_loss:.4f})")
            return True, "Stop Loss"
        if self.direction == "short" and current_price >= self.stop_loss:
            print(f"[청산 조건] {self.symbol}: 손절 발동 (현재: ${current_price:.4f}, SL: ${self.stop_loss:.4f})")
            return True, "Stop Loss"

        # 익절
        if self.direction == "long" and current_price >= self.take_profit:
            print(f"[청산 조건] {self.symbol}: 익절 발동 (현재: ${current_price:.4f}, TP: ${self.take_profit:.4f})")
            return True, "Take Profit"
        if self.direction == "short" and current_price <= self.take_profit:
            print(f"[청산 조건] {self.symbol}: 익절 발동 (현재: ${current_price:.4f}, TP: ${self.take_profit:.4f})")
            return True, "Take Profit"

        return False, ""


@dataclass
class Trade:
    """완료된 거래 기록"""
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
    roe: float
    exit_reason: str


# ===== Bybit API =====
class BybitAPI:
    """Bybit API 클라이언트"""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._instrument_cache = {}
        self.base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self.recv_window = "5000"

    def _generate_signature(self, timestamp: str, params_str: str) -> str:
        sign_str = timestamp + self.api_key + self.recv_window + params_str
        return hmac.new(
            bytes(self.api_secret, 'utf-8'),
            bytes(sign_str, 'utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _request(self, method: str, endpoint: str, params: dict = None, sign: bool = False) -> dict:
        if params is None:
            params = {}

        headers = {"Content-Type": "application/json"}

        if sign:
            timestamp = str(int(time.time() * 1000))
            if method == "GET":
                params_str = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())]) if params else ""
            else:
                params_str = json.dumps(params) if params else ""

            signature = self._generate_signature(timestamp, params_str)
            headers.update({
                'X-BAPI-API-KEY': self.api_key,
                'X-BAPI-SIGN': signature,
                'X-BAPI-TIMESTAMP': timestamp,
                'X-BAPI-RECV-WINDOW': self.recv_window
            })

        try:
            url = f"{self.base_url}{endpoint}"
            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            return response.json()
        except Exception as e:
            return {"retCode": -1, "retMsg": str(e)}

    def get_ticker(self, symbol: str) -> dict:
        """현재 가격 조회"""
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
            "volume": "float"
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.sort_values("timestamp").reset_index(drop=True)

    def get_balance(self) -> float:
        """잔고 조회"""
        result = self._request("GET", "/v5/account/wallet-balance", {
            "accountType": "UNIFIED"
        }, sign=True)

        if result.get("retCode") != 0:
            return 0.0

        try:
            coins = result["result"]["list"][0]["coin"]
            usdt = next((c for c in coins if c["coin"] == "USDT"), None)
            return float(usdt["walletBalance"]) if usdt else 0.0
        except:
            return 0.0

    def get_positions(self) -> Optional[Position]:
        """포지션 조회 (단일 심볼만)"""
        result = self._request("GET", "/v5/position/list", {
            "category": "linear",
            "symbol": SYMBOL,
            "settleCoin": "USDT"
        }, sign=True)

        if result.get("retCode") != 0:
            return None

        try:
            for item in result["result"]["list"]:
                size = float(item.get("size", 0))
                if size == 0:
                    continue

                side = item.get("side")
                direction = "long" if side == "Buy" else "short"
                entry_price = float(item.get("avgPrice", 0))
                symbol = item.get("symbol")
                leverage = int(item.get("leverage", LEVERAGE))
                unrealized_pnl = float(item.get("unrealisedPnl", 0))

                position_value = size * entry_price
                margin = position_value / leverage

                # ✅ API에서 실제 TP/SL 가져오기
                stop_loss_str = item.get("stopLoss", "")
                take_profit_str = item.get("takeProfit", "")

                if stop_loss_str and stop_loss_str != "":
                    stop_loss = float(stop_loss_str)
                else:
                    if direction == "long":
                        stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                    else:
                        stop_loss = entry_price * (1 + STOP_LOSS_PCT)

                if take_profit_str and take_profit_str != "":
                    take_profit = float(take_profit_str)
                else:
                    if direction == "long":
                        take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
                    else:
                        take_profit = entry_price * (1 - TAKE_PROFIT_PCT)

                liq_price_str = item.get("liqPrice", "")
                if liq_price_str and liq_price_str != "":
                    liquidation_price = float(liq_price_str)
                else:
                    if direction == "long":
                        liquidation_price = entry_price * (1 - (1 / leverage) * LIQUIDATION_BUFFER)
                    else:
                        liquidation_price = entry_price * (1 + (1 / leverage) * LIQUIDATION_BUFFER)

                entry_time = datetime.now()

                return Position(
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
                )

        except Exception as e:
            print(f"❌ 포지션 파싱 오류: {e}")

        return None

    def get_instrument_info(self, symbol: str) -> dict:
        """심볼 정보 조회"""
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        result = self._request("GET", "/v5/market/instruments-info", {
            "category": "linear",
            "symbol": symbol
        })

        if result.get("retCode") != 0 or not result.get("result", {}).get("list"):
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

        self._instrument_cache[symbol] = instrument_info
        return instrument_info

    def set_leverage(self, symbol: str, leverage: int) -> tuple:
        """레버리지 설정"""
        result = self._request("POST", "/v5/position/set-leverage", {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        }, sign=True)

        success = result.get("retCode") == 0
        msg = result.get("retMsg", "")
        return success, msg

    def place_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> Optional[str]:
        """주문 실행"""
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "GTC",
            "positionIdx": 0
        }

        if reduce_only:
            params["reduceOnly"] = True

        result = self._request("POST", "/v5/order/create", params, sign=True)

        if result.get("retCode") == 0:
            return result["result"]["orderId"]
        else:
            print(f"❌ 주문 실패: {result.get('retMsg')}")
            return None

    def set_trading_stop(self, symbol: str, side: str, stop_loss: float = None, take_profit: float = None):
        """TP/SL 설정"""
        params = {
            "category": "linear",
            "symbol": symbol,
            "positionIdx": 0
        }

        if stop_loss:
            params["stopLoss"] = str(stop_loss)
        if take_profit:
            params["takeProfit"] = str(take_profit)

        if not stop_loss and not take_profit:
            return False

        result = self._request("POST", "/v5/position/trading-stop", params, sign=True)
        return result.get("retCode") == 0


API = BybitAPI(API_KEY, API_SECRET, USE_TESTNET)


# ===== 데이터 전처리 (Enhanced 버전) =====
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    🔥 train_rl_enhanced.py와 동일한 확장된 기술적 지표 계산
    """
    df = df.copy()

    # 기본 지표
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volume_norm'] = np.log1p(df['volume'])

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'] / 100

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # SMA
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_ratio_20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['sma_ratio_50'] = (df['close'] - df['sma_50']) / df['sma_50']

    # MACD
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    # 변동성
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['volatility_norm'] = df['volatility'] / df['volatility'].rolling(window=100).mean()

    # 🔥 새로운 지표들 (train_rl_enhanced.py와 동일)
    # 1. 가격 모멘텀 (여러 구간)
    for period in [5, 10, 20, 30]:
        df[f'momentum_{period}'] = df['close'].pct_change(period)

    # 2. 거래량 모멘텀
    df['volume_momentum'] = df['volume'].pct_change(5)
    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # 3. High-Low 스프레드
    df['hl_spread'] = (df['high'] - df['low']) / df['close']
    df['hl_spread_ma'] = df['hl_spread'].rolling(10).mean()

    # 4. 캔들 패턴
    df['body_size'] = abs(df['close'] - df['open']) / df['open']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']

    # 5. 트렌드 강도
    df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / df['sma_50']

    df = df.fillna(0).replace([np.inf, -np.inf], 0)

    return df


# 전역 변수로 정규화 통계 저장
feature_columns = [
    'returns', 'log_returns', 'volume_norm', 'rsi', 'bb_position',
    'sma_ratio_20', 'sma_ratio_50',
    'macd', 'macd_signal', 'macd_hist',
    'atr_pct', 'volatility_norm',
    'momentum_5', 'momentum_10', 'momentum_20', 'momentum_30',
    'volume_momentum', 'volume_ma_ratio',
    'hl_spread', 'hl_spread_ma',
    'body_size', 'upper_shadow', 'lower_shadow',
    'trend_strength'
]

feature_median = None
feature_iqr = None


def initialize_feature_stats(df: pd.DataFrame):
    """
    🔥 feature 정규화 통계 초기화
    train_rl_enhanced.py와 동일한 Robust Scaling 사용
    """
    global feature_median, feature_iqr

    feature_data = df[feature_columns]

    # 중앙값과 IQR(Interquartile Range) 사용
    feature_median = feature_data.median()
    q75 = feature_data.quantile(0.75)
    q25 = feature_data.quantile(0.25)
    feature_iqr = (q75 - q25) + 1e-3  # std보다 안정적

    print("\n📊 Feature 정규화 통계 초기화 완료")
    print(f"   Feature 수: {len(feature_columns)}")
    print(f"   중앙값 범위: [{feature_median.min():.4f}, {feature_median.max():.4f}]")
    print(f"   IQR 범위: [{feature_iqr.min():.4f}, {feature_iqr.max():.4f}]")


def get_observation(df: pd.DataFrame, position: Optional[Position], current_price: float,
                    balance: float, initial_balance: float = 10000) -> np.ndarray:
    """
    🔥 train_rl_enhanced.py의 EnhancedCryptoTradingEnv와 동일한 observation 생성
    """
    # 최근 WINDOW_SIZE개 데이터 추출
    if len(df) < WINDOW_SIZE:
        padding_size = WINDOW_SIZE - len(df)
        padding = np.zeros((padding_size, len(feature_columns)))
        obs_data = np.vstack([padding, df[feature_columns].values])
    else:
        obs_data = df[feature_columns].iloc[-WINDOW_SIZE:].values

    # 🔥 Robust Scaling 적용 (train_rl_enhanced.py와 동일)
    if feature_median is not None and feature_iqr is not None:
        obs_data = (obs_data - feature_median.values) / feature_iqr.values
        obs_data = np.clip(obs_data, -5, 5)  # 더 좁은 범위로 clip
    else:
        # 초기화 전에는 기본 정규화
        feature_mean = obs_data.mean(axis=0)
        feature_std = obs_data.std(axis=0) + 1e-8
        obs_data = (obs_data - feature_mean) / feature_std
        obs_data = np.clip(obs_data, -5, 5)

    # 🔥 포지션 정보 (6개 채널) - train_rl_enhanced.py와 동일
    position_info = np.zeros((WINDOW_SIZE, 6))

    if position:
        if position.direction == 'long':
            position_info[:, 0] = 1  # long indicator
            position_info[:, 2] = (current_price - position.entry_price) / (position.entry_price + 1e-8)  # unrealized PnL
        elif position.direction == 'short':
            position_info[:, 1] = 1  # short indicator
            position_info[:, 2] = (position.entry_price - current_price) / (position.entry_price + 1e-8)

        # 보유 기간 계산 (초 단위를 스텝으로 변환)
        holding_seconds = (datetime.now() - position.entry_time).total_seconds()
        holding_steps = holding_seconds / (INTERVAL_MINUTES * 60)
        position_info[:, 4] = holding_steps / 100.0  # 정규화된 보유 기간

    position_info[:, 3] = balance / initial_balance  # 자산 비율

    # 최대 낙폭은 실시간에서 계산하기 어려우므로 0으로 설정
    position_info[:, 5] = 0.0

    # 결합
    obs = np.concatenate([obs_data, position_info], axis=1)
    return obs.astype(np.float32)


# ===== 트레이딩 매니저 =====
class TradingManager:
    def __init__(self):
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.initial_balance = None  # 시작 잔고 저장

    def set_initial_balance(self, balance: float):
        """초기 잔고 설정"""
        if self.initial_balance is None:
            self.initial_balance = balance
            print(f"💰 초기 잔고: ${balance:,.2f}")

    def can_open_position(self, current_position: Optional[Position]) -> bool:
        """포지션 진입 가능 여부"""
        if current_position is not None:
            return False
        if abs(self.daily_pnl) >= MAX_DAILY_LOSS:
            print(f"⚠️  일일 손실 한도 도달: ${self.daily_pnl:+,.2f}")
            return False
        return True

    def open_position(self, direction: str, price: float) -> bool:
        """포지션 진입"""
        try:
            if price <= 0 or not np.isfinite(price):
                return False

            # 레버리지 설정
            success, msg = API.set_leverage(SYMBOL, LEVERAGE)
            if not success and "not modified" not in msg.lower():
                print(f"⚠️  레버리지 설정 실패: {msg}")

            # 심볼 정보
            instrument_info = API.get_instrument_info(SYMBOL)
            min_qty = instrument_info["minOrderQty"]
            qty_step = instrument_info["qtyStep"]
            min_notional = instrument_info["minNotionalValue"]

            # 수량 계산
            notional = MARGIN_PER_POSITION * LEVERAGE
            qty = notional / price

            # 수량 조정
            qty = max(qty, min_qty)
            qty = round(qty / qty_step) * qty_step
            qty = round(qty, 8)

            if qty * price < min_notional:
                print(f"⚠️  최소 거래 금액 미달: ${qty * price:.2f} < ${min_notional:.2f}")
                return False

            # 주문 실행
            side = "Buy" if direction == "long" else "Sell"
            order_id = API.place_order(SYMBOL, side, qty)

            if not order_id:
                return False

            time.sleep(1)

            # TP/SL 설정
            if direction == "long":
                sl = price * (1 - STOP_LOSS_PCT)
                tp = price * (1 + TAKE_PROFIT_PCT)
            else:
                sl = price * (1 + STOP_LOSS_PCT)
                tp = price * (1 - TAKE_PROFIT_PCT)

            API.set_trading_stop(SYMBOL, side, stop_loss=sl, take_profit=tp)

            print(f"\n✅ 포지션 진입: {direction.upper()}")
            print(f"   진입가:  ${price:.4f}")
            print(f"   수량:    {qty:.4f}")
            print(f"   레버리지: {LEVERAGE}x")
            print(f"   손절가:  ${sl:.4f}")
            print(f"   익절가:  ${tp:.4f}")

            return True

        except Exception as e:
            print(f"❌ 포지션 진입 실패: {e}")
            return False

    def close_position(self, position: Position, reason: str) -> bool:
        """포지션 청산"""
        try:
            side = "Sell" if position.direction == "long" else "Buy"
            order_id = API.place_order(SYMBOL, side, position.size, reduce_only=True)

            if not order_id:
                return False

            time.sleep(0.5)

            # 청산 확인
            ticker = API.get_ticker(SYMBOL)
            if ticker.get("retCode") == 0 and ticker.get("result", {}).get("list"):
                exit_price = float(ticker["result"]["list"][0]["lastPrice"])
                pnl = position.get_pnl(exit_price)
                roe = position.get_roe(exit_price)

                # 거래 기록
                trade = Trade(
                    symbol=position.symbol,
                    direction=position.direction,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    size=position.size,
                    leverage=position.leverage,
                    margin=position.margin,
                    entry_time=position.entry_time.isoformat(),
                    exit_time=datetime.now().isoformat(),
                    pnl=pnl,
                    roe=roe,
                    exit_reason=reason
                )

                self.trades.append(trade)
                self.daily_pnl += pnl

                print(f"\n✅ 포지션 청산: {position.direction.upper()}")
                print(f"   진입가:    ${position.entry_price:.4f}")
                print(f"   청산가:    ${exit_price:.4f}")
                print(f"   손익:      ${pnl:+,.2f}")
                print(f"   ROE:       {roe:+.2f}%")
                print(f"   사유:      {reason}")

            return True

        except Exception as e:
            print(f"❌ 포지션 청산 실패: {e}")
            return False

    def save_trades(self):
        """거래 내역 저장"""
        try:
            with open(TRADE_LOG_FILE, 'w') as f:
                json.dump([asdict(t) for t in self.trades], f, indent=2)
            print(f"\n💾 거래 내역 저장: {TRADE_LOG_FILE}")
        except Exception as e:
            print(f"❌ 거래 내역 저장 실패: {e}")

    def get_stats(self) -> Dict:
        """거래 통계"""
        if not self.trades:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "avg_roe": 0.0,
                "max_pnl": 0.0,
                "min_pnl": 0.0
            }

        wins = sum(1 for t in self.trades if t.pnl > 0)
        losses = len(self.trades) - wins
        avg_pnl = sum(t.pnl for t in self.trades) / len(self.trades)
        avg_roe = sum(t.roe for t in self.trades) / len(self.trades)
        max_pnl = max(t.pnl for t in self.trades)
        min_pnl = min(t.pnl for t in self.trades)

        return {
            "total_trades": len(self.trades),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(self.trades) * 100,
            "avg_pnl": avg_pnl,
            "avg_roe": avg_roe,
            "max_pnl": max_pnl,
            "min_pnl": min_pnl
        }


# ===== 대시보드 =====
def clear_screen():
    """터미널 화면 지우기"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_dashboard(balance: float, position: Optional[Position], current_price: float,
                    action: int, action_probs: np.ndarray, manager: TradingManager,
                    loop_count: int = 0):
    """대시보드 출력"""
    clear_screen()

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("\n" + "=" * 110)
    print(f"{'🤖 강화학습 라이브 트레이딩 (PPO Enhanced)':^110}")
    print(f"{'스캔 #' + str(loop_count) + ' | ' + current_time + ' | 다음: ' + str(SCAN_INTERVAL_SEC) + '초 후':^110}")
    if USE_TESTNET:
        print(f"{'⚠️  TESTNET 모드':^110}")
    else:
        print(f"{'🔴 MAINNET 모드':^110}")
    print("=" * 110)

    # 💰 계좌 정보
    cash_balance = balance
    margin_used = 0
    position_pnl = 0
    position_value = 0

    if position:
        margin_used = position.margin
        position_pnl = position.unrealized_pnl
        position_value = margin_used + position_pnl

    total_equity = cash_balance + position_value

    print(f"\n💰 계좌 정보")
    print(f"   현금 잔고:       ${cash_balance:>10,.2f}")
    if position:
        print(f"   포지션 증거금:   ${margin_used:>10,.2f}")
        pnl_color = "🟢" if position_pnl >= 0 else "🔴"
        print(f"   포지션 손익:     {pnl_color} ${position_pnl:>+10,.2f}")
        print(f"   포지션 평가액:   ${position_value:>10,.2f}")
    print(f"   " + "-" * 40)
    print(f"   총 자산:         ${total_equity:>10,.2f}")
    print(f"   일일 손익:       ${manager.daily_pnl:>+10,.2f}")
    print(f"   현재가:          ${current_price:>10,.4f}")

    # 📍 포지션 정보
    if position:
        pnl = position.get_pnl(current_price)
        roe = position.get_roe(current_price)
        emoji = "📈" if position.direction == "long" else "📉"
        pnl_emoji = "🟢" if pnl > 0 else "🔴"

        print(f"\n📍 보유 포지션")
        print(f"   방향:      {emoji} {position.direction.upper()}")
        print(f"   진입가:    ${position.entry_price:,.4f}")
        print(f"   현재가:    ${current_price:,.4f}")
        print(f"   수량:      {position.size:,.4f}")
        print(f"   증거금:    ${position.margin:,.2f}")
        print(f"   손익:      {pnl_emoji} ${pnl:+,.2f} (ROE: {roe:+.2f}%)")
        print(f"   청산가:    ${position.liquidation_price:,.4f}")
        print(f"   손절가:    ${position.stop_loss:,.4f}")
        print(f"   익절가:    ${position.take_profit:,.4f}")
    else:
        print(f"\n📍 보유 포지션: 없음")

    # 🤖 모델 예측
    print(f"\n🤖 모델 예측 (PPO Enhanced)")
    action_names = ["LONG", "SHORT", "CLOSE"]
    print(f"   추천 액션:  {action_names[int(action)]}")
    print(f"   확률 분포:")
    for i, name in enumerate(action_names):
        bar_length = int(action_probs[i] * 30)
        bar = "█" * bar_length
        print(f"      {name:5s}: {bar:30s} {action_probs[i] * 100:5.1f}%")

    # 📊 거래 통계
    stats = manager.get_stats()
    if stats["total_trades"] > 0:
        print(f"\n📊 거래 통계")
        print(f"   총 거래:    {stats['total_trades']:>3}회")
        print(f"   승률:       {stats['win_rate']:>6.1f}% ({stats['wins']}승 {stats['losses']}패)")
        print(f"   평균 손익:  ${stats['avg_pnl']:>+12,.2f}")
        print(f"   평균 ROE:   {stats['avg_roe']:>+6.1f}%")
        print(f"   최대 수익:  ${stats['max_pnl']:>12,.2f}")
        print(f"   최대 손실:  ${stats['min_pnl']:>12,.2f}")

    print("\n" + "=" * 110)


# ===== 메인 함수 =====
def main():
    """메인 트레이딩 루프"""
    print("\n" + "=" * 110)
    print(f"{'🚀 강화학습 기반 라이브 트레이딩 시작 (Enhanced)':^110}")
    if USE_TESTNET:
        print(f"{'⚠️  TESTNET 모드':^110}")
    else:
        print(f"{'🔴 MAINNET 모드':^110}")
    print("=" * 110)
    print(f"\n설정:")
    print(f"   심볼:         {SYMBOL}")
    print(f"   레버리지:     {LEVERAGE}x")
    print(f"   포지션 증거금: ${MARGIN_PER_POSITION:,.2f}")
    print(f"   모델:         {MODEL_PATH}")
    print(f"   스캔 주기:    {SCAN_INTERVAL_SEC}초")
    print(f"   손절:         {STOP_LOSS_PCT * 100}%")
    print(f"   익절:         {TAKE_PROFIT_PCT * 100}%")

    # 모델 로드
    print(f"\n🤖 모델 로드 중...")
    try:
        model = PPO.load(MODEL_PATH)
        print(f"✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    manager = TradingManager()

    # 캔들 데이터 캐시
    df_cache = pd.DataFrame()
    last_kline_update = 0

    # feature 정규화 통계 초기화 플래그
    feature_stats_initialized = False

    try:
        loop_count = 0
        while True:
            loop_count += 1

            # 잔고 조회
            balance = API.get_balance()

            # 초기 잔고 설정 (한 번만)
            if manager.initial_balance is None:
                manager.set_initial_balance(balance)

            # 현재 포지션 조회
            position = API.get_positions()

            # 현재 가격 조회
            ticker = API.get_ticker(SYMBOL)
            if ticker.get("retCode") != 0 or not ticker.get("result", {}).get("list"):
                print(f"❌ 가격 조회 실패")
                time.sleep(SCAN_INTERVAL_SEC)
                continue

            current_price = float(ticker["result"]["list"][0]["lastPrice"])

            # 캔들 데이터 업데이트 (매 분마다)
            if time.time() - last_kline_update > 60 or df_cache.empty:
                df = API.get_klines(SYMBOL, str(INTERVAL_MINUTES), limit=200)
                if not df.empty:
                    df_cache = calculate_features(df)
                    last_kline_update = time.time()

                    # 🔥 feature 정규화 통계 초기화 (첫 번째만)
                    if not feature_stats_initialized and len(df_cache) >= 100:
                        initialize_feature_stats(df_cache)
                        feature_stats_initialized = True

            if df_cache.empty:
                print(f"❌ 데이터 부족")
                time.sleep(SCAN_INTERVAL_SEC)
                continue

            # 포지션 청산 체크 (TP/SL)
            if position:
                should_close, reason = position.should_close(current_price)
                if should_close:
                    manager.close_position(position, reason)
                    time.sleep(1)
                    position = API.get_positions()  # 재조회

            # 🔥 Observation 생성 (Enhanced 버전)
            obs = get_observation(df_cache, position, current_price, balance,
                                  manager.initial_balance or balance)

            # 모델 예측
            action, _states = model.predict(obs, deterministic=True)
            action = int(action)

            # 액션 확률 계산
            try:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    distribution = model.policy.get_distribution(obs_tensor)
                    action_probs = distribution.distribution.probs.cpu().numpy()[0]
            except:
                action_probs = np.array([0.33, 0.33, 0.34])

            # 대시보드 출력
            print_dashboard(balance, position, current_price, action, action_probs, manager, loop_count)

            # 액션 실행
            if action == Actions.LONG:
                if manager.can_open_position(position):
                    if manager.open_position("long", current_price):
                        time.sleep(1)
            elif action == Actions.SHORT:
                if manager.can_open_position(position):
                    if manager.open_position("short", current_price):
                        time.sleep(1)
            elif action == Actions.CLOSE:
                if position:
                    manager.close_position(position, "RL Model Signal")
                    time.sleep(1)

            time.sleep(SCAN_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n프로그램 종료 중...")

        # 미결제 포지션 청산
        position = API.get_positions()
        if position:
            ticker = API.get_ticker(SYMBOL)
            if ticker.get("retCode") == 0 and ticker.get("result", {}).get("list"):
                current_price = float(ticker["result"]["list"][0]["lastPrice"])
                manager.close_position(position, "Manual Close")

        # 거래 내역 저장
        manager.save_trades()

        # 최종 통계
        balance = API.get_balance()
        stats = manager.get_stats()
        if stats["total_trades"] > 0:
            print("\n" + "=" * 110)
            print(f"{'📊 최종 결과':^110}")
            print("=" * 110)
            print(f"   최종 잔고:  ${balance:,.2f}")
            print(f"   일일 손익:  ${manager.daily_pnl:+,.2f}")
            print(f"   총 거래:    {stats['total_trades']}회")
            print(f"   승률:       {stats['win_rate']:.1f}%")
            print(f"   평균 ROE:   {stats['avg_roe']:+.1f}%")
            print("=" * 110)

        if position:
            print("\n⚠️  주의: 아직 포지션이 남아있습니다!")

        print("\n✅ 프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()