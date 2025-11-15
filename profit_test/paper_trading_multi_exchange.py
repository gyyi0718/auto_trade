# paper_trading_multi_exchange.py
# -*- coding: utf-8 -*-
"""
멀티 거래소 TCN 모델 기반 페이퍼 트레이딩 시스템
- 바이비트/바이낸스 각각의 모델 사용
- 4개 거래소 데이터 (스팟/퓨처) 기반 예측
- 실시간 신호 기반 자동 매매 시뮬레이션
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
# 거래할 심볼
# 환경변수에서 심볼 가져오기 (기본값: EVAA/USDT)
SYMBOL = os.getenv('TRADING_SYMBOL', 'EVAA/USDT')

# 심볼에서 / 제거하여 파일명에 사용
symbol_for_filename = SYMBOL.replace('/', '_')

# 환경변수에서 모델 디렉토리 경로 가져오기 (기본값 유지)
MODEL_DIR = os.getenv('MODEL_DIR', '../multimodel/models_multi_exchange')

# 타겟 거래소별 모델 경로 (동적으로 생성)
MODEL_PATHS = {
    'bybit_future': os.path.join(MODEL_DIR, f'tcn_bybit_future_{symbol_for_filename}_1m.pth'),
    'binance_future': os.path.join(MODEL_DIR, f'tcn_binance_future_{symbol_for_filename}_1m.pth')
}

print(f"Trading Symbol: {SYMBOL}")
print(f"Model Paths: {MODEL_PATHS}")
# 실행 설정
INTERVAL_SEC = 10  # 스캔 간격
CONF_THRESHOLD = 0.4  # 신뢰도 임계값 (3클래스이므로 33%보다 높으면 의미있음)

# 페이퍼 트레이딩 설정
INITIAL_CAPITAL = 2000.0  # 초기 자본 (USDT)
POSITION_SIZE_PCT = 0.2  # 포지션 크기 (20%)
LEVERAGE = 20  # 레버리지
MAX_POSITIONS = 2  # 최대 동시 포지션
STOP_LOSS_PCT = 0.02  # 손절 2%
TAKE_PROFIT_PCT = 0.04  # 익절 4%
MAX_HOLD_MINUTES = 60  # 최대 보유 시간

# 모델 설정
LOOKBACK = 100  # 시퀀스 길이 (학습 시와 동일해야 함)
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
    def __init__(self):
        self.exchanges = {
            'bybit_spot': ccxt.bybit({'options': {'defaultType': 'spot'}}),
            'bybit_future': ccxt.bybit({'options': {'defaultType': 'swap'}}),
            'binance_spot': ccxt.binance({'options': {'defaultType': 'spot'}}),
            'binance_future': ccxt.binance({'options': {'defaultType': 'future'}})
        }

    def fetch_ohlcv(self, exchange_name, symbol, timeframe, limit=100):
        try:
            exchange = self.exchanges[exchange_name]

            # 심볼 형식 변환
            if exchange_name in ['binance_future']:
                symbol_modified = symbol.replace('/', '')  # BTC/USDT -> BTCUSDT
            elif exchange_name in ['bybit_spot', 'bybit_future']:
                symbol_modified = symbol.replace('/', '')  # BTC/USDT -> BTCUSDT (Bybit도 슬래시 없음)
            else:
                symbol_modified = symbol

            ohlcv = exchange.fetch_ohlcv(symbol_modified, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            df.columns = [f'{col}_{exchange_name}' for col in df.columns]

            return df
        except Exception as e:
            # 에러 메시지 출력 안 함 (조용한 실패)
            return None

    def fetch_all_data(self, symbol, timeframe, limit=100):
        dfs = []
        for exchange_name in self.exchanges.keys():
            df = self.fetch_ohlcv(exchange_name, symbol, timeframe, limit)
            if df is not None:
                dfs.append(df)
            time.sleep(0.3)

        if len(dfs) == 0:
            raise ValueError("No data collected")

        combined_df = pd.concat(dfs, axis=1, join='inner')
        combined_df = combined_df.sort_index()
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
            result_df[f'bb_upper_{exchange}'] = sma_20 + (2 * std_20)
            result_df[f'bb_lower_{exchange}'] = sma_20 - (2 * std_20)
            result_df[f'bb_width_{exchange}'] = (result_df[f'bb_upper_{exchange}'] - result_df[
                f'bb_lower_{exchange}']) / sma_20

            # ATR
            high_low = df[high_col] - df[low_col]
            high_close = np.abs(df[high_col] - df[close_col].shift())
            low_close = np.abs(df[low_col] - df[close_col].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            result_df[f'atr_{exchange}'] = true_range.rolling(14).mean()

            # Volume indicators
            result_df[f'volume_ma_{exchange}'] = df[volume_col].rolling(20).mean()
            result_df[f'volume_ratio_{exchange}'] = df[volume_col] / result_df[f'volume_ma_{exchange}']

        return result_df

    def add_spread_features(self, df):
        result_df = df.copy()

        # 거래소 간 스프레드
        if 'close_bybit_spot' in df.columns and 'close_binance_spot' in df.columns:
            result_df['spread_spot'] = df['close_bybit_spot'] - df['close_binance_spot']
            result_df['spread_spot_pct'] = (df['close_bybit_spot'] / df['close_binance_spot'] - 1) * 100

        if 'close_bybit_future' in df.columns and 'close_binance_future' in df.columns:
            result_df['spread_future'] = df['close_bybit_future'] - df['close_binance_future']
            result_df['spread_future_pct'] = (df['close_bybit_future'] / df['close_binance_future'] - 1) * 100

        # 스팟-퓨처 프리미엄
        if 'close_bybit_spot' in df.columns and 'close_bybit_future' in df.columns:
            result_df['premium_bybit'] = df['close_bybit_future'] - df['close_bybit_spot']
            result_df['premium_bybit_pct'] = (df['close_bybit_future'] / df['close_bybit_spot'] - 1) * 100

        if 'close_binance_spot' in df.columns and 'close_binance_future' in df.columns:
            result_df['premium_binance'] = df['close_binance_future'] - df['close_binance_spot']
            result_df['premium_binance_pct'] = (df['close_binance_future'] / df['close_binance_spot'] - 1) * 100

        # 크로스 스프레드
        if 'close_bybit_spot' in df.columns and 'close_binance_future' in df.columns:
            result_df['cross_spread_1'] = df['close_binance_future'] - df['close_bybit_spot']
            result_df['cross_spread_1_pct'] = (df['close_binance_future'] / df['close_bybit_spot'] - 1) * 100

        if 'close_binance_spot' in df.columns and 'close_bybit_future' in df.columns:
            result_df['cross_spread_2'] = df['close_bybit_future'] - df['close_binance_spot']
            result_df['cross_spread_2_pct'] = (df['close_bybit_future'] / df['close_binance_spot'] - 1) * 100

        # 스프레드 변화율
        spread_cols = [col for col in result_df.columns if 'spread' in col or 'premium' in col]
        for col in spread_cols:
            if col.endswith('_pct'):
                result_df[f'{col}_change'] = result_df[col].diff()
                result_df[f'{col}_ma5'] = result_df[col].rolling(5).mean()
                result_df[f'{col}_ma20'] = result_df[col].rolling(20).mean()

        return result_df

    def process(self, df):
        df = self.add_technical_indicators(df)
        df = self.add_spread_features(df)
        df = df.dropna()
        return df


# ===== 모델 로더 =====
class ModelLoader:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_cols = {}

    def load_model(self, target_exchange, model_path):
        """모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=device)

        # 하이퍼파라미터 추출
        hp = checkpoint['hyperparameters']

        # 모델 생성
        model = TCNModel(
            input_dim=hp['input_dim'],
            hidden_dim=hp['hidden_dim'],
            num_layers=hp['num_layers'],
            kernel_size=hp['kernel_size'],
            dropout=hp['dropout'],
            num_classes=3
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        self.models[target_exchange] = model
        self.scalers[target_exchange] = checkpoint['scaler']
        self.feature_cols[target_exchange] = checkpoint['feature_cols']

        print(f"✓ Loaded model for {target_exchange}: {model_path}")
        print(f"  Features: {len(self.feature_cols[target_exchange])}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")


# ===== 예측 시스템 =====
class PredictionSystem:
    def __init__(self, model_loader, data_collector, feature_engine):
        self.model_loader = model_loader
        self.data_collector = data_collector
        self.feature_engine = feature_engine

    def predict(self, symbol, target_exchange, debug=False):
        """예측 수행"""
        try:
            # 데이터 수집
            raw_df = self.data_collector.fetch_all_data(symbol, TIMEFRAME, limit=LOOKBACK + 50)

            if raw_df is None or len(raw_df) < LOOKBACK:
                return {"error": "Insufficient data"}

            # 피처 생성
            df = self.feature_engine.process(raw_df)

            if len(df) < LOOKBACK:
                return {"error": "Insufficient data after processing"}

            # 현재 가격 먼저 확인 (0이면 중단)
            close_col = f'close_{target_exchange}'
            if close_col not in df.columns:
                return {"error": f"Price column {close_col} not found"}

            current_price = float(df[close_col].iloc[-1])
            if current_price <= 0:
                return {"error": "Invalid price (zero or negative)"}

            # 피처 추출
            feature_cols = self.model_loader.feature_cols[target_exchange]

            # 피처 컬럼 검증
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                return {"error": f"Missing features: {len(missing_cols)} columns"}

            X = df[feature_cols].values[-LOOKBACK:]

            # 정규화
            scaler = self.model_loader.scalers[target_exchange]
            X = scaler.transform(X)
            X = X.reshape(1, LOOKBACK, -1)
            X_tensor = torch.FloatTensor(X).to(device)

            # 예측
            model = self.model_loader.models[target_exchange]
            with torch.no_grad():
                output = model(X_tensor)
                probs = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()

            # 레이블 변환 (0: Short, 1: Hold, 2: Long)
            direction_map = {0: "Short", 1: "Hold", 2: "Long"}
            direction = direction_map[pred_class]

            result = {
                "symbol": symbol,
                "target_exchange": target_exchange,
                "direction": direction,
                "confidence": confidence,
                "current_price": current_price,
                "probabilities": {
                    "Short": float(probs[0, 0]),
                    "Hold": float(probs[0, 1]),
                    "Long": float(probs[0, 2])
                },
                "raw_output": output[0].tolist(),  # 디버그용
                "pred_class": pred_class  # 디버그용
            }

            if debug:
                print(f"\n🔍 Debug - {target_exchange}")
                print(f"   Direction: {direction}")
                print(f"   Confidence: {confidence:.2%}")
                print(f"   Probabilities: Short={probs[0, 0]:.2%}, Hold={probs[0, 1]:.2%}, Long={probs[0, 2]:.2%}")
                print(f"   Current Price: ${current_price:.4f}")
                print(f"   Raw output: {output[0].tolist()}")
                print(f"   Pred class: {pred_class}")

            return result

        except Exception as e:
            if debug:
                print(f"⚠️ Prediction error for {target_exchange}: {e}")
                import traceback
                traceback.print_exc()
            return {"error": str(e)}


# ===== 포지션 관리 =====
@dataclass
class Position:
    symbol: str
    exchange: str
    direction: str
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    leverage: int
    margin: float

    def get_pnl(self, current_price: float) -> float:
        if self.direction == "Long":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def get_roe(self, current_price: float) -> float:
        if self.direction == "Long":
            price_change_pct = (current_price / self.entry_price - 1) * 100
        else:
            price_change_pct = (1 - current_price / self.entry_price) * 100
        return price_change_pct * self.leverage

    def should_close(self, current_price: float, current_time: datetime) -> tuple[bool, str]:
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
    symbol: str
    exchange: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    roe: float
    reason: str


class Account:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

        # 거래소별 통계
        self.exchange_stats = {
            'bybit_future': {
                'initial_capital': initial_capital / 2,
                'balance': initial_capital / 2,
                'positions': {},
                'trades': []
            },
            'binance_future': {
                'initial_capital': initial_capital / 2,
                'balance': initial_capital / 2,
                'positions': {},
                'trades': []
            }
        }

    def can_open_position(self, key: str) -> bool:
        return key not in self.positions and len(self.positions) < MAX_POSITIONS

    def open_position(self, symbol: str, exchange: str, direction: str, price: float):
        # 가격 검증
        if price <= 0:
            return

        # 거래소별 잔고에서 마진 차감
        if exchange not in self.exchange_stats:
            return

        exchange_balance = self.exchange_stats[exchange]['balance']
        margin = exchange_balance * POSITION_SIZE_PCT

        if margin <= 0:
            return

        quantity = (margin * LEVERAGE) / price

        if direction == "Long":
            stop_loss = price * (1 - STOP_LOSS_PCT)
            take_profit = price * (1 + TAKE_PROFIT_PCT)
        else:
            stop_loss = price * (1 + STOP_LOSS_PCT)
            take_profit = price * (1 - TAKE_PROFIT_PCT)

        key = f"{exchange}_{symbol}"
        position = Position(
            symbol=symbol,
            exchange=exchange,
            direction=direction,
            entry_price=price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=LEVERAGE,
            margin=margin
        )

        self.positions[key] = position
        self.exchange_stats[exchange]['positions'][key] = position
        self.exchange_stats[exchange]['balance'] -= margin
        self.balance -= margin

    def close_position(self, key: str, price: float, reason: str):
        if key not in self.positions:
            return

        position = self.positions[key]
        pnl = position.get_pnl(price)
        roe = position.get_roe(price)

        # 전체 잔고 업데이트
        self.balance += position.margin + pnl

        # 거래소별 잔고 업데이트
        exchange = position.exchange
        if exchange in self.exchange_stats:
            self.exchange_stats[exchange]['balance'] += position.margin + pnl

        trade = Trade(
            symbol=position.symbol,
            exchange=position.exchange,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            pnl=pnl,
            roe=roe,
            reason=reason
        )

        self.trades.append(trade)
        if exchange in self.exchange_stats:
            self.exchange_stats[exchange]['trades'].append(trade)
            if key in self.exchange_stats[exchange]['positions']:
                del self.exchange_stats[exchange]['positions'][key]

        del self.positions[key]

    def get_total_value(self, prices: Dict[str, float]) -> float:
        total = self.balance
        for key, position in self.positions.items():
            price = prices.get(f"{position.exchange}_{position.symbol}", position.entry_price)
            total += position.margin + position.get_pnl(price)
        return total

    def get_exchange_value(self, exchange: str, prices: Dict[str, float]) -> float:
        """거래소별 총 자산 계산"""
        if exchange not in self.exchange_stats:
            return 0.0

        total = self.exchange_stats[exchange]['balance']
        for key, position in self.exchange_stats[exchange]['positions'].items():
            price = prices.get(key, position.entry_price)
            total += position.margin + position.get_pnl(price)
        return total


# ===== 대시보드 =====
def print_dashboard(account: Account, prices: Dict[str, float]):
    total_value = account.get_total_value(prices)
    total_return = (total_value / account.initial_capital - 1) * 100

    print("\n" + "=" * 100)
    print(f"{'💰 전체 계좌 현황':^100}")
    print("=" * 100)
    print(f"   잔고: ${account.balance:>10,.2f} | 총 자산: ${total_value:>10,.2f} | 수익률: {total_return:>+6.2f}%")
    print(f"   포지션: {len(account.positions)}/{MAX_POSITIONS} | 총 거래: {len(account.trades)}회")

    # 거래소별 통계
    print("\n📊 거래소별 현황")
    print(f"{'거래소':^15} | {'초기자본':^12} | {'현재자산':^12} | {'수익률':^10} | {'거래수':^8}")
    print("-" * 100)

    for exchange in ['bybit_future', 'binance_future']:
        if exchange in account.exchange_stats:
            stats = account.exchange_stats[exchange]
            init_cap = stats['initial_capital']
            current_val = account.get_exchange_value(exchange, prices)
            return_pct = (current_val / init_cap - 1) * 100
            num_trades = len(stats['trades'])

            emoji = "🟢" if return_pct > 0 else "🔴" if return_pct < 0 else "⚪"
            print(
                f"{exchange:^15} | ${init_cap:>10,.2f} | ${current_val:>10,.2f} | {emoji}{return_pct:>+6.2f}% | {num_trades:^8}")

    if account.positions:
        print(f"\n📈 보유 포지션 ({len(account.positions)}개)")
        print(f"{'거래소':^15} | {'심볼':^12} | {'방향':^6} | {'진입가':^12} | {'현재가':^12} | {'ROE':^8} | {'PNL':^12}")
        print("-" * 100)

        for key, pos in account.positions.items():
            price = prices.get(key, pos.entry_price)
            roe = pos.get_roe(price)
            pnl = pos.get_pnl(price)
            emoji = "🟢" if pnl > 0 else "🔴"

            print(f"{pos.exchange:^15} | {pos.symbol:^12} | {pos.direction:^6} | "
                  f"${pos.entry_price:>10,.4f} | ${price:>10,.4f} | "
                  f"{roe:>+6.1f}% | {emoji} ${pnl:>+8.2f}")

    # 최근 거래 3개 표시
    if account.trades:
        recent_trades = account.trades[-3:]
        print(f"\n📝 최근 거래 ({len(recent_trades)}개)")
        print(f"{'거래소':^15} | {'방향':^6} | {'진입가':^12} | {'청산가':^12} | {'ROE':^8} | {'PNL':^12} | {'사유':^15}")
        print("-" * 100)

        for trade in recent_trades:
            emoji = "🟢" if trade.pnl > 0 else "🔴"
            print(f"{trade.exchange:^15} | {trade.direction:^6} | "
                  f"${trade.entry_price:>10,.4f} | ${trade.exit_price:>10,.4f} | "
                  f"{trade.roe:>+6.1f}% | {emoji} ${trade.pnl:>+8.2f} | {trade.reason:^15}")


# ===== 메인 =====
def main():
    print("\n" + "=" * 100)
    print(f"{'🎯 멀티 거래소 페이퍼 트레이딩 시작':^100}")
    print("=" * 100)
    print(f"   심볼: {SYMBOL}")
    print(f"   초기 자본: ${INITIAL_CAPITAL:,.2f}")
    print(f"   레버리지: {LEVERAGE}x")
    print(f"   포지션 크기: {POSITION_SIZE_PCT * 100:.0f}%")
    print(f"   신뢰도 임계값: {CONF_THRESHOLD:.1%}")
    print("=" * 100)

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
    data_collector = MultiExchangeDataCollector()
    feature_engine = MultiExchangeFeatureEngine()
    predictor = PredictionSystem(model_loader, data_collector, feature_engine)
    account = Account(INITIAL_CAPITAL)

    print("\n✅ 시스템 초기화 완료\n")

    try:
        loop_count = 0
        while True:
            loop_count += 1
            current_time = datetime.now()

            # 화면 클리어 (첫 번째 루프 제외)
            if loop_count > 1:
                clear_screen()

            # 헤더 출력
            print("=" * 100)
            print(f"{'🎯 멀티 거래소 페이퍼 트레이딩':^100}")
            print(f"{'심볼: ' + SYMBOL + ' | 레버리지: ' + str(LEVERAGE) + 'x | 신뢰도: ' + f'{CONF_THRESHOLD:.0%}':^100}")
            scan_info = f'[스캔 #{loop_count}] {current_time.strftime("%Y-%m-%d %H:%M:%S")}'
            print(f"{scan_info:^100}")
            print("=" * 100)

            # 가격 수집
            prices = {}
            for exchange in model_loader.models.keys():
                try:
                    # 심볼 형식 변환
                    symbol_for_exchange = SYMBOL.replace('/', '')  # Bybit와 Binance 모두 슬래시 없음

                    # 간단하게 현재 가격만 가져오기
                    if exchange == 'bybit_future':
                        bybit = ccxt.bybit({'options': {'defaultType': 'swap'}})
                        ticker = bybit.fetch_ticker(symbol_for_exchange)
                    elif exchange == 'binance_future':
                        binance = ccxt.binance({'options': {'defaultType': 'future'}})
                        ticker = binance.fetch_ticker(symbol_for_exchange)
                    elif exchange == 'bybit_spot':
                        bybit = ccxt.bybit({'options': {'defaultType': 'spot'}})
                        ticker = bybit.fetch_ticker(symbol_for_exchange)
                    elif exchange == 'binance_spot':
                        binance = ccxt.binance({'options': {'defaultType': 'spot'}})
                        ticker = binance.fetch_ticker(symbol_for_exchange)
                    else:
                        continue

                    price = ticker['last']
                    if price > 0:
                        prices[f"{exchange}_{SYMBOL}"] = price
                except Exception as e:
                    # 에러 무시 (조용한 실패)
                    pass

            # 포지션 관리
            for key in list(account.positions.keys()):
                position = account.positions[key]
                price_key = f"{position.exchange}_{position.symbol}"
                current_price = prices.get(price_key, position.entry_price)

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
            print_dashboard(account, prices)

            # 신호 스캔
            print(f"\n🔍 신호 스캔")
            print(f"{'거래소':^15} | {'심볼':^12} | {'가격':^12} | {'방향':^8} | {'신뢰도':^8} | {'신호':^20}")
            print("-" * 100)

            debug_mode = (loop_count == 1)
            for exchange in model_loader.models.keys():
                result = predictor.predict(SYMBOL, exchange, debug=False)  # 항상 디버그 끄기

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

                # 진입 조건 (가격 검증 추가)
                key = f"{exchange}_{SYMBOL}"
                if (account.can_open_position(key) and
                        confidence >= CONF_THRESHOLD and
                        direction in ["Long", "Short"] and
                        price > 0):  # 가격이 0보다 큰 경우만
                    account.open_position(SYMBOL, exchange, direction, price)

            # 대기 시간 표시
            print(f"\n⏳ 다음 스캔까지 {INTERVAL_SEC}초... (Ctrl+C로 종료)")
            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n프로그램 종료 중...")
        print_dashboard(account, prices)

        # 최종 통계
        if account.trades:
            wins = sum(1 for t in account.trades if t.pnl > 0)
            losses = sum(1 for t in account.trades if t.pnl < 0)
            win_rate = wins / len(account.trades) * 100 if account.trades else 0
            avg_roe = sum(t.roe for t in account.trades) / len(account.trades)

            print("\n" + "=" * 100)
            print(f"{'📊 최종 결과':^100}")
            print("=" * 100)
            print(f"   총 거래: {len(account.trades)}회 (승: {wins}, 패: {losses})")
            print(f"   승률: {win_rate:.1f}%")
            print(f"   평균 ROE: {avg_roe:+.1f}%")
            print(f"   최종 수익률: {((account.get_total_value(prices) / account.initial_capital) - 1) * 100:+.2f}%")
            print("=" * 100)

        print("\n✅ 프로그램 종료")


if __name__ == "__main__":
    main()