# live_trading_multi_exchange.py
# -*- coding: utf-8 -*-
"""
멀티 거래소 TCN 모델 기반 실제 거래 시스템
- 바이비트/바이낸스 각각의 모델 사용
- 4개 거래소 데이터 (스팟/퓨처) 기반 예측
- 실시간 신호 기반 자동 매매 (실제 거래)
- 거래소별 On/Off 스위치 기능
"""
import os
import time
import json
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ccxt
from sklearn.preprocessing import RobustScaler
import sys

warnings.filterwarnings("ignore")


# 터미널 화면 클리어 함수
def clear_screen():
    """운영체제에 따라 화면 클리어"""
    os.system('cls' if os.name == 'nt' else 'clear')


# ===== CONFIG =====
# 거래소별 API 키 설정
API_KEYS = {
    'bybit': {
        'apiKey': 'YOUR_BYBIT_API_KEY',
        'secret': 'YOUR_BYBIT_SECRET'
    },
    'binance': {
        'apiKey': 'YOUR_BINANCE_API_KEY',
        'secret': 'YOUR_BINANCE_SECRET'
    }
}

# 거래소별 거래 활성화 설정 (True: 실거래, False: 페이퍼 트레이딩)
TRADING_ENABLED = {
    'bybit_future': True,  # 바이비트 퓨처 거래 활성화
    'binance_future': True  # 바이낸스 퓨처 거래 활성화
}

# 거래할 심볼
SYMBOL = "EVAA/USDT"

# 타겟 거래소별 모델 경로
MODEL_PATHS = {
    'bybit_future': '../multimodel/models_multi_exchange/tcn_bybit_future_EVAA_USDT_1m.pth',
    'binance_future': '../multimodel/models_multi_exchange/tcn_binance_future_EVAA_USDT_1m.pth'
}

# 실행 설정
INTERVAL_SEC = 10  # 스캔 간격
CONF_THRESHOLD = 0.4  # 신뢰도 임계값

# 거래 설정
INITIAL_CAPITAL = 2000.0  # 초기 자본 (USDT)
POSITION_SIZE_PCT = 0.2  # 포지션 크기 (20%)
LEVERAGE = 20  # 레버리지
MAX_POSITIONS = 2  # 최대 동시 포지션
STOP_LOSS_PCT = 0.02  # 손절 2%
TAKE_PROFIT_PCT = 0.04  # 익절 4%
MAX_HOLD_MINUTES = 60  # 최대 보유 시간

# 모델 설정
LOOKBACK = 100  # 시퀀스 길이
TIMEFRAME = '1m'

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ===== TCN 모델 정의 =====
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=0, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=0, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.padding = padding

    def forward(self, x):
        x_padded = nn.functional.pad(x, (self.padding, 0))
        out = self.conv1(x_padded)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = nn.functional.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)

        if out.size(2) != res.size(2):
            out = out[:, :, :res.size(2)]

        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=4, kernel_size=3, dropout=0.2, num_classes=3):
        super(TCNModel, self).__init__()

        layers = []
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else hidden_dim
            out_channels = hidden_dim
            padding = (kernel_size - 1) * dilation_size

            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size, padding=padding, dropout=dropout
            ))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x[:, :, -1]
        x = self.fc(x)
        return x


# ===== 데이터 수집 =====
class MultiExchangeDataCollector:
    def __init__(self, debug=False):
        self.debug = debug
        self.exchanges = {
            'bybit_spot': ccxt.bybit({'options': {'defaultType': 'spot'}, 'enableRateLimit': True}),
            'bybit_future': ccxt.bybit({'options': {'defaultType': 'swap'}, 'enableRateLimit': True}),
            'binance_spot': ccxt.binance({'options': {'defaultType': 'spot'}, 'enableRateLimit': True}),
            'binance_future': ccxt.binance({'options': {'defaultType': 'future'}, 'enableRateLimit': True})
        }

        # 마켓 로드 (한 번만)
        self.markets = {}
        for exchange_name, exchange in self.exchanges.items():
            try:
                self.markets[exchange_name] = exchange.load_markets()
                if self.debug:
                    print(f"✅ {exchange_name}: Loaded {len(self.markets[exchange_name])} markets")
            except Exception as e:
                if self.debug:
                    print(f"❌ {exchange_name}: Failed to load markets - {e}")
                self.markets[exchange_name] = {}

    def fetch_ohlcv(self, exchange_name, symbol, timeframe, limit=100):
        try:
            exchange = self.exchanges[exchange_name]

            # 심볼 형식 변환
            symbol_modified = symbol.replace('/', '')  # 모든 거래소에서 슬래시 제거

            # 심볼 존재 여부 확인
            if symbol_modified not in self.markets.get(exchange_name, {}):
                if self.debug:
                    print(f"❌ {exchange_name}: Symbol {symbol_modified} not found in markets")
                return None

            # OHLCV 데이터 가져오기
            ohlcv = exchange.fetch_ohlcv(symbol_modified, timeframe, limit=limit)

            if not ohlcv or len(ohlcv) == 0:
                if self.debug:
                    print(f"❌ {exchange_name}: No OHLCV data returned")
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            df.columns = [f'{col}_{exchange_name}' for col in df.columns]

            if self.debug:
                print(f"✅ {exchange_name}: Fetched {len(df)} candles")

            return df

        except Exception as e:
            if self.debug:
                print(f"❌ {exchange_name}: Error fetching OHLCV - {e}")
            return None

    def fetch_all_data(self, symbol, timeframe, limit=100):
        dfs = []
        errors = []

        for exchange_name in self.exchanges.keys():
            df = self.fetch_ohlcv(exchange_name, symbol, timeframe, limit)
            if df is not None and len(df) > 0:
                dfs.append(df)
            else:
                errors.append(exchange_name)
            time.sleep(0.3)

        if len(dfs) == 0:
            error_msg = f"No data collected from any exchange. Failed: {', '.join(errors)}"
            if self.debug:
                print(f"❌ {error_msg}")
            raise ValueError(error_msg)

        if self.debug and len(errors) > 0:
            print(f"⚠️  Data collected from {len(dfs)}/{len(self.exchanges)} exchanges. Failed: {', '.join(errors)}")

        combined_df = pd.concat(dfs, axis=1, join='inner')
        combined_df = combined_df.sort_index()

        if self.debug:
            print(f"✅ Combined dataframe: {len(combined_df)} rows, {len(combined_df.columns)} columns")

        return combined_df


# ===== 피처 생성 =====
class MultiExchangeFeatureEngine:
    def __init__(self):
        self.exchange_types = ['bybit_spot', 'bybit_future', 'binance_spot', 'binance_future']

    def add_technical_indicators(self, df):
        result_df = df.copy()

        for exchange in self.exchange_types:
            close_col = f'close_{exchange}'
            high_col = f'high_{exchange}'
            low_col = f'low_{exchange}'
            volume_col = f'volume_{exchange}'

            if close_col not in df.columns:
                continue

            # Returns
            result_df[f'returns_{exchange}'] = df[close_col].pct_change()

            # Moving Averages
            for window in [5, 10, 20, 50]:
                result_df[f'sma_{window}_{exchange}'] = df[close_col].rolling(window).mean()
                result_df[f'ema_{window}_{exchange}'] = df[close_col].ewm(span=window).mean()

            # Volatility
            result_df[f'volatility_{exchange}'] = df[close_col].rolling(20).std()

            # RSI
            delta = df[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result_df[f'rsi_{exchange}'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df[close_col].ewm(span=12).mean()
            exp2 = df[close_col].ewm(span=26).mean()
            result_df[f'macd_{exchange}'] = exp1 - exp2
            result_df[f'macd_signal_{exchange}'] = result_df[f'macd_{exchange}'].ewm(span=9).mean()

            # Bollinger Bands
            sma_20 = df[close_col].rolling(20).mean()
            std_20 = df[close_col].rolling(20).std()
            result_df[f'bb_upper_{exchange}'] = sma_20 + (std_20 * 2)
            result_df[f'bb_lower_{exchange}'] = sma_20 - (std_20 * 2)
            result_df[f'bb_width_{exchange}'] = result_df[f'bb_upper_{exchange}'] - result_df[f'bb_lower_{exchange}']

            # Volume indicators
            result_df[f'volume_sma_{exchange}'] = df[volume_col].rolling(20).mean()
            result_df[f'volume_ratio_{exchange}'] = df[volume_col] / result_df[f'volume_sma_{exchange}']

        # Cross-exchange features
        if all(f'close_{ex}' in df.columns for ex in ['bybit_future', 'binance_future']):
            result_df['price_spread'] = df['close_bybit_future'] - df['close_binance_future']
            result_df['price_spread_pct'] = result_df['price_spread'] / df['close_bybit_future']

        if all(f'close_{ex}' in df.columns for ex in ['bybit_spot', 'bybit_future']):
            result_df['basis_bybit'] = df['close_bybit_future'] - df['close_bybit_spot']
            result_df['basis_bybit_pct'] = result_df['basis_bybit'] / df['close_bybit_spot']

        if all(f'close_{ex}' in df.columns for ex in ['binance_spot', 'binance_future']):
            result_df['basis_binance'] = df['close_binance_future'] - df['close_binance_spot']
            result_df['basis_binance_pct'] = result_df['basis_binance'] / df['close_binance_spot']

        result_df = result_df.ffill().bfill()
        return result_df

    def create_sequences(self, df, lookback):
        feature_cols = [col for col in df.columns if not col.startswith(('timestamp', 'open_', 'high_', 'low_'))]
        feature_data = df[feature_cols].values

        if len(feature_data) < lookback:
            return None

        sequence = feature_data[-lookback:]
        scaler = RobustScaler()
        sequence_scaled = scaler.fit_transform(sequence)

        return sequence_scaled


# ===== 실제 거래 실행기 =====
class LiveTradeExecutor:
    def __init__(self, api_keys: Dict, trading_enabled: Dict):
        self.api_keys = api_keys
        self.trading_enabled = trading_enabled
        self.exchanges = {}

        # 거래소 초기화
        if trading_enabled.get('bybit_future', False):
            self.exchanges['bybit_future'] = ccxt.bybit({
                'apiKey': api_keys['bybit']['apiKey'],
                'secret': api_keys['bybit']['secret'],
                'options': {'defaultType': 'swap'},
                'enableRateLimit': True
            })
            print("✅ Bybit Future 거래 활성화")

        if trading_enabled.get('binance_future', False):
            self.exchanges['binance_future'] = ccxt.binance({
                'apiKey': api_keys['binance']['apiKey'],
                'secret': api_keys['binance']['secret'],
                'options': {'defaultType': 'future'},
                'enableRateLimit': True
            })
            print("✅ Binance Future 거래 활성화")

    def set_leverage(self, exchange_name: str, symbol: str, leverage: int):
        """레버리지 설정"""
        if exchange_name not in self.exchanges:
            return False

        try:
            exchange = self.exchanges[exchange_name]
            symbol_modified = symbol.replace('/', '')

            if exchange_name == 'bybit_future':
                exchange.set_leverage(leverage, symbol_modified)
            elif exchange_name == 'binance_future':
                exchange.fapiPrivate_post_leverage({
                    'symbol': symbol_modified,
                    'leverage': leverage
                })

            print(f"✅ {exchange_name} 레버리지 {leverage}x 설정 완료")
            return True
        except Exception as e:
            print(f"❌ {exchange_name} 레버리지 설정 실패: {e}")
            return False

    def get_balance(self, exchange_name: str) -> float:
        """잔고 조회 (USDT)"""
        if exchange_name not in self.exchanges:
            return 0.0

        try:
            exchange = self.exchanges[exchange_name]
            balance = exchange.fetch_balance()
            return float(balance['USDT']['free'])
        except Exception as e:
            print(f"❌ {exchange_name} 잔고 조회 실패: {e}")
            return 0.0

    def get_positions(self, exchange_name: str, symbol: str) -> Dict:
        """현재 포지션 조회"""
        if exchange_name not in self.exchanges:
            return {}

        try:
            exchange = self.exchanges[exchange_name]
            symbol_modified = symbol.replace('/', '')

            if exchange_name == 'bybit_future':
                positions = exchange.fetch_positions([symbol_modified])
            elif exchange_name == 'binance_future':
                positions = exchange.fapiPrivate_get_positionrisk({'symbol': symbol_modified})

            # 포지션 정보 파싱
            for pos in positions:
                if float(pos.get('contracts', 0)) > 0 or float(pos.get('positionAmt', 0)) != 0:
                    return {
                        'size': float(pos.get('contracts', pos.get('positionAmt', 0))),
                        'side': pos.get('side', 'long' if float(pos.get('positionAmt', 0)) > 0 else 'short'),
                        'entry_price': float(pos.get('entryPrice', 0)),
                        'unrealized_pnl': float(pos.get('unrealizedPnl', 0))
                    }

            return {}
        except Exception as e:
            print(f"❌ {exchange_name} 포지션 조회 실패: {e}")
            return {}

    def open_position(self, exchange_name: str, symbol: str, direction: str,
                      size_usdt: float, current_price: float) -> bool:
        """포지션 오픈"""
        if exchange_name not in self.exchanges:
            print(f"⚠️  {exchange_name} 거래 비활성화 상태")
            return False

        try:
            exchange = self.exchanges[exchange_name]
            symbol_modified = symbol.replace('/', '')

            # 주문 수량 계산 (USDT -> 코인 수량)
            quantity = (size_usdt * LEVERAGE) / current_price

            # 소수점 자리수 조정 (거래소별로 다를 수 있음)
            if exchange_name == 'bybit_future':
                quantity = round(quantity, 2)
            elif exchange_name == 'binance_future':
                quantity = round(quantity, 3)

            # 주문 실행
            side = 'buy' if direction == "Long" else 'sell'

            print(f"🔄 {exchange_name} 주문 실행 중...")
            print(f"   심볼: {symbol}, 방향: {direction}, 수량: {quantity}, 가격: ${current_price:.4f}")

            order = exchange.create_market_order(
                symbol=symbol_modified,
                side=side,
                amount=quantity
            )

            print(f"✅ {exchange_name} 포지션 오픈 성공!")
            print(f"   주문 ID: {order.get('id', 'N/A')}")
            return True

        except Exception as e:
            print(f"❌ {exchange_name} 포지션 오픈 실패: {e}")
            return False

    def close_position(self, exchange_name: str, symbol: str, direction: str) -> bool:
        """포지션 청산"""
        if exchange_name not in self.exchanges:
            print(f"⚠️  {exchange_name} 거래 비활성화 상태")
            return False

        try:
            exchange = self.exchanges[exchange_name]
            symbol_modified = symbol.replace('/', '')

            # 현재 포지션 조회
            position = self.get_positions(exchange_name, symbol)
            if not position:
                print(f"⚠️  {exchange_name} 청산할 포지션 없음")
                return False

            # 반대 방향 주문으로 청산
            side = 'sell' if direction == "Long" else 'buy'
            quantity = abs(position['size'])

            print(f"🔄 {exchange_name} 포지션 청산 중...")
            print(f"   심볼: {symbol}, 방향: {direction}, 수량: {quantity}")

            order = exchange.create_market_order(
                symbol=symbol_modified,
                side=side,
                amount=quantity,
                params={'reduceOnly': True}  # 청산 전용
            )

            print(f"✅ {exchange_name} 포지션 청산 성공!")
            print(f"   주문 ID: {order.get('id', 'N/A')}")
            print(f"   실현 손익: ${position['unrealized_pnl']:.2f}")
            return True

        except Exception as e:
            print(f"❌ {exchange_name} 포지션 청산 실패: {e}")
            return False

    def set_stop_loss_take_profit(self, exchange_name: str, symbol: str,
                                  direction: str, stop_loss: float, take_profit: float) -> bool:
        """손절/익절 설정"""
        if exchange_name not in self.exchanges:
            return False

        try:
            exchange = self.exchanges[exchange_name]
            symbol_modified = symbol.replace('/', '')

            # 현재 포지션 조회
            position = self.get_positions(exchange_name, symbol)
            if not position:
                return False

            quantity = abs(position['size'])

            # 손절 주문
            if stop_loss > 0:
                side_sl = 'sell' if direction == "Long" else 'buy'
                exchange.create_order(
                    symbol=symbol_modified,
                    type='stop_market',
                    side=side_sl,
                    amount=quantity,
                    params={
                        'stopPrice': stop_loss,
                        'reduceOnly': True
                    }
                )

            # 익절 주문
            if take_profit > 0:
                side_tp = 'sell' if direction == "Long" else 'buy'
                exchange.create_order(
                    symbol=symbol_modified,
                    type='take_profit_market',
                    side=side_tp,
                    amount=quantity,
                    params={
                        'stopPrice': take_profit,
                        'reduceOnly': True
                    }
                )

            print(f"✅ {exchange_name} 손절/익절 설정 완료")
            print(f"   손절: ${stop_loss:.4f}, 익절: ${take_profit:.4f}")
            return True

        except Exception as e:
            print(f"❌ {exchange_name} 손절/익절 설정 실패: {e}")
            return False


# ===== 모델 로더 =====
class ModelLoader:
    def __init__(self):
        self.models = {}
        self.input_dims = {}

    def load_model(self, exchange, path):
        checkpoint = torch.load(path, map_location=device)

        # 다양한 체크포인트 형식 지원
        if 'input_dim' in checkpoint:
            input_dim = checkpoint['input_dim']
        elif 'config' in checkpoint and 'input_dim' in checkpoint['config']:
            input_dim = checkpoint['config']['input_dim']
        else:
            # input_dim이 없으면 모델에서 추출 시도
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # 첫 번째 conv 레이어에서 input_dim 추출
            for key in state_dict.keys():
                if 'network.0.conv1.weight' in key:
                    input_dim = state_dict[key].shape[1]
                    break
            else:
                # 기본값 사용 (피처 엔진 기반 계산)
                # 4개 거래소 * (5 기본 + 4*4 이동평균 + 1 변동성 + 1 RSI + 2 MACD + 3 볼린저 + 2 볼륨) + 크로스 피처
                # 대략 120~150 정도
                print(f"⚠️  {exchange}: input_dim을 찾을 수 없어서 자동 계산합니다...")
                input_dim = None  # 자동 계산하도록 설정

        if input_dim is None:
            # 실제 데이터로 피처 개수 계산
            try:
                print(f"⏳ {exchange}: 실제 데이터로 input_dim 계산 중...")
                temp_collector = MultiExchangeDataCollector(debug=True)
                temp_feature_engine = MultiExchangeFeatureEngine()
                temp_df = temp_collector.fetch_all_data(SYMBOL, TIMEFRAME, limit=LOOKBACK + 50)
                temp_df = temp_feature_engine.add_technical_indicators(temp_df)
                feature_cols = [col for col in temp_df.columns if
                                not col.startswith(('timestamp', 'open_', 'high_', 'low_'))]
                input_dim = len(feature_cols)
                print(f"✅ {exchange}: 자동 계산된 input_dim={input_dim}")
            except Exception as e:
                print(f"❌ {exchange}: input_dim 자동 계산 실패: {e}")
                import traceback
                traceback.print_exc()
                raise ValueError(f"Cannot determine input_dim for {exchange}")

        # 모델 생성
        model = TCNModel(input_dim=input_dim).to(device)

        # state_dict 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        self.models[exchange] = model
        self.input_dims[exchange] = input_dim
        print(f"✅ Loaded {exchange} model (input_dim={input_dim})")


# ===== 예측 시스템 =====
class PredictionSystem:
    def __init__(self, model_loader, data_collector, feature_engine):
        self.model_loader = model_loader
        self.data_collector = data_collector
        self.feature_engine = feature_engine

    def predict(self, symbol, exchange, debug=False):
        try:
            df = self.data_collector.fetch_all_data(symbol, TIMEFRAME, limit=LOOKBACK + 50)
            df = self.feature_engine.add_technical_indicators(df)
            sequence = self.feature_engine.create_sequences(df, LOOKBACK)

            if sequence is None:
                return {"error": "Insufficient data"}

            expected_dim = self.model_loader.input_dims.get(exchange)
            if sequence.shape[1] != expected_dim:
                return {"error": f"Feature mismatch: {sequence.shape[1]} != {expected_dim}"}

            model = self.model_loader.models[exchange]
            X = torch.FloatTensor(sequence).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(X)
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()

            direction = ["Short", "Hold", "Long"][pred_class]

            # 현재가
            price_key = f'close_{exchange}'
            current_price = df[price_key].iloc[-1]

            return {
                "direction": direction,
                "confidence": confidence,
                "current_price": current_price,
                "pred_class": pred_class
            }

        except Exception as e:
            if debug:
                print(f"Prediction error for {exchange}: {e}")
            return {"error": str(e)}


# ===== 포지션 관리 (실거래용) =====
@dataclass
class LivePosition:
    symbol: str
    exchange: str
    direction: str
    entry_price: float
    size_usdt: float
    leverage: int
    entry_time: datetime
    stop_loss: float
    take_profit: float

    def get_roe(self, current_price):
        if self.direction == "Long":
            return ((current_price - self.entry_price) / self.entry_price) * 100 * self.leverage
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100 * self.leverage

    def get_pnl(self, current_price):
        roe = self.get_roe(current_price)
        return self.size_usdt * (roe / 100)

    def should_close(self, current_price, current_time):
        # 손절 체크
        if self.direction == "Long" and current_price <= self.stop_loss:
            return True, "Stop Loss"
        if self.direction == "Short" and current_price >= self.stop_loss:
            return True, "Stop Loss"

        # 익절 체크
        if self.direction == "Long" and current_price >= self.take_profit:
            return True, "Take Profit"
        if self.direction == "Short" and current_price <= self.take_profit:
            return True, "Take Profit"

        # 최대 보유 시간 체크
        hold_minutes = (current_time - self.entry_time).total_seconds() / 60
        if hold_minutes >= MAX_HOLD_MINUTES:
            return True, "Max Hold Time"

        return False, ""


class LiveAccount:
    def __init__(self, executor: LiveTradeExecutor):
        self.executor = executor
        self.positions: Dict[str, LivePosition] = {}
        self.trades = []
        self.exchange_stats = {}

        # 거래소별 초기 자본 기록
        for exchange in TRADING_ENABLED.keys():
            if TRADING_ENABLED[exchange]:
                balance = executor.get_balance(exchange)
                self.exchange_stats[exchange] = {
                    'initial_capital': balance,
                    'trades': []
                }

    def can_open_position(self, key):
        # 이미 같은 포지션이 있는지 체크
        if key in self.positions:
            return False

        # 최대 포지션 수 체크
        if len(self.positions) >= MAX_POSITIONS:
            return False

        return True

    def open_position(self, symbol, exchange, direction, current_price):
        key = f"{exchange}_{symbol}"

        if not self.can_open_position(key):
            return False

        # 거래 활성화 확인
        if not TRADING_ENABLED.get(exchange, False):
            print(f"⚠️  {exchange} 거래 비활성화 상태 (TRADING_ENABLED={TRADING_ENABLED.get(exchange, False)})")
            return False

        # 잔고 확인
        balance = self.executor.get_balance(exchange)
        size_usdt = balance * POSITION_SIZE_PCT

        if size_usdt < 10:  # 최소 주문 금액
            print(f"⚠️  {exchange} 잔고 부족 (${balance:.2f})")
            return False

        # 손절/익절 가격 계산
        if direction == "Long":
            stop_loss = current_price * (1 - STOP_LOSS_PCT)
            take_profit = current_price * (1 + TAKE_PROFIT_PCT)
        else:
            stop_loss = current_price * (1 + STOP_LOSS_PCT)
            take_profit = current_price * (1 - TAKE_PROFIT_PCT)

        # 레버리지 설정
        self.executor.set_leverage(exchange, symbol, LEVERAGE)

        # 실제 주문 실행
        success = self.executor.open_position(exchange, symbol, direction, size_usdt, current_price)

        if success:
            # 손절/익절 설정
            self.executor.set_stop_loss_take_profit(exchange, symbol, direction, stop_loss, take_profit)

            # 포지션 기록
            position = LivePosition(
                symbol=symbol,
                exchange=exchange,
                direction=direction,
                entry_price=current_price,
                size_usdt=size_usdt,
                leverage=LEVERAGE,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            self.positions[key] = position
            print(f"✅ 포지션 오픈: {key} | {direction} | ${current_price:.4f} | ${size_usdt:.2f}")
            return True

        return False

    def close_position(self, key, current_price, reason):
        if key not in self.positions:
            return False

        position = self.positions[key]

        # 실제 청산 실행
        success = self.executor.close_position(position.exchange, position.symbol, position.direction)

        if success:
            # 거래 기록
            roe = position.get_roe(current_price)
            pnl = position.get_pnl(current_price)

            trade_record = {
                'exchange': position.exchange,
                'symbol': position.symbol,
                'direction': position.direction,
                'entry_price': position.entry_price,
                'exit_price': current_price,
                'size_usdt': position.size_usdt,
                'roe': roe,
                'pnl': pnl,
                'reason': reason,
                'entry_time': position.entry_time,
                'exit_time': datetime.now()
            }

            self.trades.append(trade_record)
            if position.exchange in self.exchange_stats:
                self.exchange_stats[position.exchange]['trades'].append(trade_record)

            # 포지션 제거
            del self.positions[key]

            emoji = "🟢" if pnl > 0 else "🔴"
            print(f"{emoji} 포지션 청산: {key} | {reason} | ROE: {roe:+.1f}% | PNL: ${pnl:+.2f}")
            return True

        return False

    def get_total_value(self):
        """현재 총 자산 가치"""
        total = 0.0
        for exchange in self.exchange_stats.keys():
            balance = self.executor.get_balance(exchange)
            total += balance
        return total

    def get_exchange_value(self, exchange):
        """거래소별 자산 가치"""
        return self.executor.get_balance(exchange)


# ===== 대시보드 =====
def print_dashboard(account: LiveAccount):
    print(f"\n{'=' * 100}")
    print(f"{'💰 계좌 현황':^100}")
    print(f"{'=' * 100}")
    print(f"{'거래소':^15} | {'거래모드':^12} | {'초기자본':^12} | {'현재가치':^12} | {'수익률':^10} | {'거래수':^8}")
    print("-" * 100)

    for exchange in TRADING_ENABLED.keys():
        mode = "🟢 실거래" if TRADING_ENABLED[exchange] else "⚪ 비활성"

        if exchange in account.exchange_stats:
            stats = account.exchange_stats[exchange]
            init_cap = stats['initial_capital']
            current_val = account.get_exchange_value(exchange)
            return_pct = (current_val / init_cap - 1) * 100 if init_cap > 0 else 0
            num_trades = len(stats['trades'])

            emoji = "🟢" if return_pct > 0 else "🔴" if return_pct < 0 else "⚪"
            print(
                f"{exchange:^15} | {mode:^12} | ${init_cap:>10,.2f} | ${current_val:>10,.2f} | {emoji}{return_pct:>+6.2f}% | {num_trades:^8}")
        else:
            print(f"{exchange:^15} | {mode:^12} | {'N/A':^12} | {'N/A':^12} | {'N/A':^10} | {'N/A':^8}")

    if account.positions:
        print(f"\n{'=' * 100}")
        print(f"{'📈 보유 포지션':^100} ({len(account.positions)}개)")
        print(f"{'=' * 100}")
        print(f"{'거래소':^15} | {'심볼':^12} | {'방향':^6} | {'진입가':^12} | {'손절가':^12} | {'익절가':^12}")
        print("-" * 100)

        for key, pos in account.positions.items():
            print(f"{pos.exchange:^15} | {pos.symbol:^12} | {pos.direction:^6} | "
                  f"${pos.entry_price:>10,.4f} | ${pos.stop_loss:>10,.4f} | ${pos.take_profit:>10,.4f}")

    if account.trades:
        recent_trades = account.trades[-3:]
        print(f"\n{'=' * 100}")
        print(f"{'📝 최근 거래':^100} ({len(recent_trades)}개)")
        print(f"{'=' * 100}")
        print(f"{'거래소':^15} | {'방향':^6} | {'진입가':^12} | {'청산가':^12} | {'ROE':^8} | {'PNL':^12} | {'사유':^15}")
        print("-" * 100)

        for trade in recent_trades:
            emoji = "🟢" if trade['pnl'] > 0 else "🔴"
            print(f"{trade['exchange']:^15} | {trade['direction']:^6} | "
                  f"${trade['entry_price']:>10,.4f} | ${trade['exit_price']:>10,.4f} | "
                  f"{trade['roe']:>+6.1f}% | {emoji} ${trade['pnl']:>+8.2f} | {trade['reason']:^15}")


# ===== 메인 =====
def main():
    print("\n" + "=" * 100)
    print(f"{'🎯 멀티 거래소 실제 거래 시스템':^100}")
    print("=" * 100)

    # 심볼 검증
    print(f"\n심볼 검증 중: {SYMBOL}")
    print("-" * 100)

    temp_exchanges = {
        'bybit_future': ccxt.bybit({'options': {'defaultType': 'swap'}}),
        'binance_future': ccxt.binance({'options': {'defaultType': 'future'}})
    }

    symbol_found = {}
    for exchange_name, exchange in temp_exchanges.items():
        try:
            markets = exchange.load_markets()
            symbol_modified = SYMBOL.replace('/', '')
            if symbol_modified in markets and markets[symbol_modified].get('active', False):
                symbol_found[exchange_name] = True
                print(f"✅ {exchange_name}: {SYMBOL} 존재 및 활성화")
            else:
                symbol_found[exchange_name] = False
                print(f"❌ {exchange_name}: {SYMBOL} 찾을 수 없음")
        except Exception as e:
            symbol_found[exchange_name] = False
            print(f"❌ {exchange_name}: 에러 - {e}")

    # 거래 활성화된 거래소 중 심볼이 없는 경우 경고
    active_exchanges_without_symbol = [
        ex for ex in TRADING_ENABLED.keys()
        if TRADING_ENABLED[ex] and not symbol_found.get(ex, False)
    ]

    if active_exchanges_without_symbol:
        print(f"\n⚠️  경고: 다음 거래소에서 {SYMBOL}을(를) 거래할 수 없습니다:")
        for ex in active_exchanges_without_symbol:
            print(f"   - {ex}")

        print(f"\n💡 해결 방법:")
        print(f"   1. 코드 상단의 SYMBOL 변수를 다른 심볼로 변경")
        print(f"   2. find_tradable_symbols.py 실행하여 거래 가능한 심볼 확인")
        print(f"   3. 추천: BTC/USDT, ETH/USDT, SOL/USDT 등 메이저 코인 사용")

        proceed = input(f"\n계속하시겠습니까? (yes/no): ")
        if proceed.lower() != 'yes':
            print("프로그램 종료")
            return

    print("=" * 100)
    print(f"   심볼: {SYMBOL}")
    print(f"   레버리지: {LEVERAGE}x")
    print(f"   포지션 크기: {POSITION_SIZE_PCT * 100:.0f}%")
    print(f"   신뢰도 임계값: {CONF_THRESHOLD:.1%}")
    print("\n거래소 활성화 상태:")
    for exchange, enabled in TRADING_ENABLED.items():
        status = "🟢 실거래" if enabled else "⚪ 비활성"
        symbol_status = "✅" if symbol_found.get(exchange, False) else "❌"
        print(f"   {exchange}: {status} {symbol_status}")
    print("=" * 100)

    # 경고 메시지
    print("\n⚠️  경고: 실제 거래가 활성화되어 있습니다!")
    print("⚠️  API 키가 올바르게 설정되었는지 확인하세요.")
    print("⚠️  손실 위험이 있으니 신중하게 사용하세요.\n")

    confirm = input("계속하시겠습니까? (yes/no): ")
    if confirm.lower() != 'yes':
        print("프로그램 종료")
        return

    # 모델 로드
    model_loader = ModelLoader()
    for exchange, path in MODEL_PATHS.items():
        try:
            model_loader.load_model(exchange, path)
        except Exception as e:
            print(f"❌ Failed to load {exchange} model: {e}")

    if not model_loader.models:
        print("\n❌ No models loaded. Exiting.")
        return

    # 시스템 초기화
    data_collector = MultiExchangeDataCollector(debug=True)  # 디버그 모드 활성화
    feature_engine = MultiExchangeFeatureEngine()
    predictor = PredictionSystem(model_loader, data_collector, feature_engine)

    # 거래 실행기 초기화
    executor = LiveTradeExecutor(API_KEYS, TRADING_ENABLED)
    account = LiveAccount(executor)

    print("\n✅ 시스템 초기화 완료\n")

    try:
        loop_count = 0
        while True:
            loop_count += 1
            current_time = datetime.now()

            # 화면 클리어
            if loop_count > 1:
                clear_screen()

            # 헤더 출력
            print("=" * 100)
            print(f"{'🎯 멀티 거래소 실제 거래':^100}")
            print(f"{'심볼: ' + SYMBOL + ' | 레버리지: ' + str(LEVERAGE) + 'x | 신뢰도: ' + f'{CONF_THRESHOLD:.0%}':^100}")
            scan_info = f'[스캔 #{loop_count}] {current_time.strftime("%Y-%m-%d %H:%M:%S")}'
            print(f"{scan_info:^100}")
            print("=" * 100)

            # 포지션 관리 (청산 조건 체크)
            for key in list(account.positions.keys()):
                position = account.positions[key]

                # 실제 포지션 정보 조회
                live_pos = executor.get_positions(position.exchange, position.symbol)
                if not live_pos:
                    # 포지션이 없으면 (수동 청산 or 손절/익절 실행됨)
                    del account.positions[key]
                    print(f"⚠️  {key} 포지션이 이미 청산되었습니다")
                    continue

                current_price = live_pos.get('entry_price', position.entry_price)

                # 반대 신호 체크
                result = predictor.predict(position.symbol, position.exchange, debug=False)
                if "error" not in result and result["confidence"] >= CONF_THRESHOLD:
                    signal_dir = result["direction"]
                    if (position.direction == "Long" and signal_dir == "Short") or \
                            (position.direction == "Short" and signal_dir == "Long"):
                        account.close_position(key, current_price, "Reverse Signal")
                        continue

                # 일반 청산 조건
                should_close, reason = position.should_close(current_price, current_time)
                if should_close:
                    account.close_position(key, current_price, reason)

            # 대시보드
            print_dashboard(account)

            # 신호 스캔
            print(f"\n{'=' * 100}")
            print(f"{'🔍 신호 스캔':^100}")
            print(f"{'=' * 100}")
            print(f"{'거래소':^15} | {'심볼':^12} | {'가격':^12} | {'방향':^8} | {'신뢰도':^8} | {'신호':^20}")
            print("-" * 100)

            for exchange in model_loader.models.keys():
                result = predictor.predict(SYMBOL, exchange, debug=False)

                if "error" in result:
                    print(f"{exchange:^15} | {SYMBOL:^12} | {'N/A':^12} | {'—':^8} | {'—':^8} | ⚠️ 데이터 없음")
                    continue

                direction = result["direction"]
                confidence = result["confidence"]
                price = result["current_price"]

                dir_icon = {"Long": "📈", "Short": "📉", "Hold": "➖"}[direction]

                if confidence < CONF_THRESHOLD:
                    signal = f"⚠️ 약함 ({confidence:.1%})"
                elif direction == "Long":
                    signal = f"🟢 매수 ({confidence:.1%})"
                elif direction == "Short":
                    signal = f"🔴 매도 ({confidence:.1%})"
                else:
                    signal = f"⚪ 관망 ({confidence:.1%})"

                print(f"{exchange:^15} | {SYMBOL:^12} | ${price:>10,.4f} | "
                      f"{dir_icon}  {direction:^6} | {confidence:>6.1%} | {signal}")

                # 진입 조건
                key = f"{exchange}_{SYMBOL}"
                if (account.can_open_position(key) and
                        confidence >= CONF_THRESHOLD and
                        direction in ["Long", "Short"] and
                        price > 0 and
                        TRADING_ENABLED.get(exchange, False)):
                    account.open_position(SYMBOL, exchange, direction, price)

            # 대기 시간 표시
            print(f"\n⏳ 다음 스캔까지 {INTERVAL_SEC}초... (Ctrl+C로 종료)")
            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n프로그램 종료 중...")

        # 모든 포지션 청산
        print("\n모든 포지션을 청산하시겠습니까? (yes/no): ")
        close_all = input()
        if close_all.lower() == 'yes':
            for key in list(account.positions.keys()):
                position = account.positions[key]
                live_pos = executor.get_positions(position.exchange, position.symbol)
                if live_pos:
                    current_price = live_pos.get('entry_price', position.entry_price)
                    account.close_position(key, current_price, "Manual Close")

        print_dashboard(account)

        # 최종 통계
        if account.trades:
            wins = sum(1 for t in account.trades if t['pnl'] > 0)
            losses = sum(1 for t in account.trades if t['pnl'] < 0)
            win_rate = wins / len(account.trades) * 100 if account.trades else 0
            avg_roe = sum(t['roe'] for t in account.trades) / len(account.trades)
            total_pnl = sum(t['pnl'] for t in account.trades)

            print("\n" + "=" * 100)
            print(f"{'📊 최종 결과':^100}")
            print("=" * 100)
            print(f"   총 거래: {len(account.trades)}회 (승: {wins}, 패: {losses})")
            print(f"   승률: {win_rate:.1f}%")
            print(f"   평균 ROE: {avg_roe:+.1f}%")
            print(f"   총 손익: ${total_pnl:+.2f}")
            print("=" * 100)

        print("\n✅ 프로그램 종료")


if __name__ == "__main__":
    main()