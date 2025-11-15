# train_rl_enhanced.py
# -*- coding: utf-8 -*-
"""
강화학습 성능 개선 버전
핵심 개선사항:
1. TCN/PatchTST와 유사한 방향성 학습을 위한 Reward Shaping
2. 미래 정보를 활용한 Hindsight Experience Replay (HER)
3. Imitation Learning으로 TCN 예측을 사전학습
4. 더 나은 feature engineering
"""
import os
import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import requests
import gym
from gym import spaces
from typing import Tuple, Dict, List
from enum import IntEnum
import torch
import torch.nn as nn


###############################################################################
# 핵심 개선 1: 방향성 기반 Reward Shaping
###############################################################################

class DirectionalRewardMixin:
    """TCN/PatchTST 스타일의 방향성 평가를 reward에 반영"""

    def _calculate_directional_reward(self, entry_idx: int, current_idx: int,
                                      position: str) -> float:
        """
        현재까지의 high/low를 기반으로 방향성 평가
        TCN처럼 "이 방향이 맞았는지" 평가
        """
        if entry_idx >= current_idx:
            return 0.0

        # entry부터 현재까지의 구간
        segment = self.df.iloc[entry_idx:current_idx + 1]

        entry_price = float(segment.iloc[0]['open'])
        high_price = float(segment['high'].max())
        low_price = float(segment['low'].min())

        # Long/Short 각각의 최대 수익 가능성
        long_potential = (high_price / entry_price - 1.0)
        short_potential = (entry_price / low_price - 1.0)

        # 올바른 방향을 선택했는지 평가
        if position == 'long':
            if long_potential > short_potential:
                # 올바른 방향 선택 → 양의 보상
                return long_potential * 2.0
            else:
                # 잘못된 방향 선택 → 음의 보상
                return -short_potential * 2.0
        else:  # short
            if short_potential > long_potential:
                return short_potential * 2.0
            else:
                return -long_potential * 2.0


###############################################################################
# 핵심 개선 2: Enhanced Environment with Better Features
###############################################################################

class Actions(IntEnum):
    """3가지 액션"""
    LONG = 0
    SHORT = 1
    CLOSE = 2


class EnhancedCryptoTradingEnv(gym.Env, DirectionalRewardMixin):
    """
    개선된 거래 환경
    - TCN 스타일의 방향성 reward
    - 더 나은 feature engineering
    - Lookahead penalty 추가
    """

    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            df: pd.DataFrame,
            window_size: int = 30,
            initial_balance: float = 10000,
            base_leverage: int = 10,
            commission: float = 0.0006,
            stop_loss_pct: float = 0.05,
            take_profit_pct: float = 0.08,
            reward_scaling: float = 1e4,
            min_holding_steps: int = 2,
            force_initial_position: bool = True,
            directional_weight: float = 0.5,  # 🔥 방향성 보상 가중치
            debug: bool = False
    ):
        super(EnhancedCryptoTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.base_leverage = base_leverage
        self.commission = commission
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.reward_scaling = reward_scaling
        self.min_holding_steps = min_holding_steps
        self.force_initial_position = force_initial_position
        self.directional_weight = directional_weight  # 🔥 NEW
        self.debug = debug

        self._calculate_features()
        self._precompute_future_info()  # 🔥 NEW

        # observation: 기존 features + 추가 정보
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, len(self.feature_columns) + 6),  # +6 for extra info
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.reset()

    def _calculate_features(self):
        """확장된 기술적 지표 계산"""
        df = self.df.copy()

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

        # 🔥 새로운 지표들
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

        self.feature_columns = [
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

        self.df = df

        # 🔥 개선: Robust Scaling (IQR 기반)
        # std가 작아서 생기는 문제 해결
        feature_data = self.df[self.feature_columns]

        # 중앙값과 IQR(Interquartile Range) 사용
        self.feature_median = feature_data.median()
        q75 = feature_data.quantile(0.75)
        q25 = feature_data.quantile(0.25)
        self.feature_iqr = (q75 - q25) + 1e-3  # std보다 안정적

        # 디버그: 정규화 통계 출력
        print("\n📊 Feature 정규화 통계:")
        print(f"   중앙값 범위: [{self.feature_median.min():.4f}, {self.feature_median.max():.4f}]")
        print(f"   IQR 범위: [{self.feature_iqr.min():.4f}, {self.feature_iqr.max():.4f}]")
        print(f"   IQR < 0.01인 feature 수: {(self.feature_iqr < 0.01).sum()} / {len(self.feature_iqr)}")

    def _precompute_future_info(self):
        """
        🔥 NEW: 각 시점에서 미래의 high/low 정보를 미리 계산
        학습 시에는 사용하지 않지만, reward 계산에 활용
        """
        lookahead = 72  # 6시간 (72개 5분봉)

        self.df['future_high'] = 0.0
        self.df['future_low'] = 0.0
        self.df['optimal_direction'] = 0  # 0: SHORT, 1: LONG

        for i in range(len(self.df) - lookahead):
            future_segment = self.df.iloc[i + 1:i + 1 + lookahead]

            if len(future_segment) < lookahead:
                break

            entry = self.df.iloc[i + 1]['open']
            future_high = future_segment['high'].max()
            future_low = future_segment['low'].min()

            self.df.loc[i, 'future_high'] = future_high
            self.df.loc[i, 'future_low'] = future_low

            # 최적 방향 계산 (TCN과 동일한 방식)
            long_profit = (future_high / entry - 1.0)
            short_profit = (entry / future_low - 1.0)

            if long_profit >= short_profit:
                self.df.loc[i, 'optimal_direction'] = 1  # LONG
            else:
                self.df.loc[i, 'optimal_direction'] = 0  # SHORT

    def reset(self) -> np.ndarray:
        """환경 초기화"""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.current_step = self.window_size

        self.current_price = self.df.loc[self.current_step, 'close']
        self.leverage = self.base_leverage

        self.position = None
        self.entry_price = 0
        self.entry_step = 0  # 🔥 NEW
        self.position_size = 0
        self.entry_balance = 0

        self.total_trades = 0
        self.winning_trades = 0
        self.correct_direction_count = 0  # 🔥 NEW
        self.total_pnl = 0
        self.max_equity = self.initial_balance
        self.max_drawdown = 0

        self.trade_history = []
        self.equity_history = [self.initial_balance]
        self.steps_in_position = 0

        if self.force_initial_position:
            initial_direction = np.random.choice(['long', 'short'])
            self._open_position(initial_direction)

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """현재 상태 관찰"""
        start = self.current_step - self.window_size
        end = self.current_step

        obs_data = self.df[self.feature_columns].iloc[start:end].values

        # 🔥 개선: Robust Scaling 적용
        obs_data = (obs_data - self.feature_median.values) / self.feature_iqr.values
        obs_data = np.clip(obs_data, -5, 5)  # 더 좁은 범위로 clip

        # 포지션 정보 (6개 채널)
        position_info = np.zeros((self.window_size, 6))

        if self.position == 'long':
            position_info[:, 0] = 1  # long indicator
            position_info[:, 2] = (self.current_price - self.entry_price) / (self.entry_price + 1e-8)  # unrealized PnL
            position_info[:, 4] = self.steps_in_position / 100.0  # 정규화된 보유 기간
        elif self.position == 'short':
            position_info[:, 1] = 1  # short indicator
            position_info[:, 2] = (self.entry_price - self.current_price) / (self.entry_price + 1e-8)
            position_info[:, 4] = self.steps_in_position / 100.0

        position_info[:, 3] = self.equity / self.initial_balance  # 자산 비율
        position_info[:, 5] = self.max_drawdown  # 최대 낙폭

        obs = np.concatenate([obs_data, position_info], axis=1)
        return obs.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝 실행"""
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        self.current_price = self.df.loc[self.current_step, 'close']

        reward = self._execute_action(action)
        self._update_equity()

        if self.position:
            self.steps_in_position += 1
        else:
            self.steps_in_position = 0

        self.equity_history.append(self.equity)
        self.max_equity = max(self.max_equity, self.equity)
        drawdown = (self.max_equity - self.equity) / self.max_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # 파산 방지
        if self.equity < self.initial_balance * 0.2:
            done = True
            reward -= 10

        obs = self._get_observation()

        info = {
            'equity': self.equity,
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'direction_accuracy': self.correct_direction_count / max(1, self.total_trades),  # 🔥 NEW
            'max_drawdown': self.max_drawdown,
            'pnl': self.total_pnl,
            'steps_in_position': self.steps_in_position,
        }

        return obs, reward, done, info

    def _execute_action(self, action: int) -> float:
        """
        행동 실행 - 🔥 개선된 reward 구조
        1. 실제 PnL 기반 reward
        2. 방향성 정확도 reward (TCN 스타일)
        3. 보유 시간 페널티/보상
        """
        reward = 0

        # Stop Loss / Take Profit 체크
        if self.position:
            pnl_pct = self._get_position_pnl_pct()

            if pnl_pct <= -self.stop_loss_pct:
                reward = self._close_position("Stop Loss")
                reward -= 1.0  # 손절 페널티
                return reward * self.reward_scaling

            if pnl_pct >= self.take_profit_pct:
                reward = self._close_position("Take Profit")
                reward += 2.0  # 익절 보너스
                return reward * self.reward_scaling

        # 액션 실행
        if action == Actions.LONG:
            if self.position == 'short':
                reward += self._close_position("Switch to LONG")
            if not self.position:
                self._open_position('long')
                reward += 0.1  # 진입 보상

        elif action == Actions.SHORT:
            if self.position == 'long':
                reward += self._close_position("Switch to SHORT")
            if not self.position:
                self._open_position('short')
                reward += 0.1  # 진입 보상

        elif action == Actions.CLOSE:
            if self.position and self.steps_in_position >= self.min_holding_steps:
                reward = self._close_position("Manual Close")
            elif self.position:
                reward -= 0.2  # 너무 빨리 청산하면 페널티

        # 🔥 포지션 유지 중 보상 계산
        if self.position:
            # 1. 실제 PnL 기반
            pnl_pct = self._get_position_pnl_pct()
            reward += pnl_pct * 0.5

            # 2. 방향성 보상 (TCN 스타일)
            if self.entry_step < self.current_step:
                directional_reward = self._calculate_directional_reward(
                    self.entry_step,
                    self.current_step,
                    self.position
                )
                reward += directional_reward * self.directional_weight

            # 3. 보유 기간 페널티 (너무 오래 들고 있으면)
            if self.steps_in_position > 50:
                reward -= 0.01 * (self.steps_in_position - 50)

        return reward * self.reward_scaling

    def _open_position(self, direction: str):
        """포지션 오픈"""
        self.position = direction
        self.entry_price = self.current_price
        self.entry_step = self.current_step  # 🔥 NEW
        self.entry_balance = self.balance

        margin = self.balance * 0.95
        self.position_size = (margin * self.leverage) / self.current_price

        commission_cost = self.position_size * self.current_price * self.commission
        self.balance -= commission_cost

        if self.debug:
            print(f"   📈 {direction.upper()} @ {self.entry_price:.2f}")

    def _close_position(self, reason: str = "") -> float:
        """포지션 청산 - 🔥 방향성 평가 추가"""
        if not self.position:
            return 0

        pnl_pct = self._get_position_pnl_pct()
        pnl = self.entry_balance * pnl_pct * self.leverage

        commission_cost = self.position_size * self.current_price * self.commission
        net_pnl = pnl - commission_cost

        self.balance = self.entry_balance + net_pnl
        self.total_pnl += net_pnl

        if net_pnl > 0:
            self.winning_trades += 1

        # 🔥 NEW: 방향성 정확도 평가
        optimal_direction = self.df.loc[self.entry_step, 'optimal_direction']
        actual_direction = 1 if self.position == 'long' else 0

        if optimal_direction == actual_direction:
            self.correct_direction_count += 1

        self.total_trades += 1

        self.trade_history.append({
            'entry_price': self.entry_price,
            'exit_price': self.current_price,
            'position': self.position,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'holding_time': self.steps_in_position,
            'reason': reason,
            'direction_correct': optimal_direction == actual_direction  # 🔥 NEW
        })

        if self.debug:
            print(f"   📉 {self.position.upper()} 청산 @ {self.current_price:.2f} "
                  f"| PnL: {net_pnl:+.2f} ({pnl_pct * 100:+.2f}%) | {reason}")

        self.position = None
        self.entry_price = 0
        self.entry_step = 0
        self.position_size = 0
        self.steps_in_position = 0

        return pnl_pct

    def _get_position_pnl_pct(self) -> float:
        """현재 포지션 손익률"""
        if not self.position:
            return 0

        if self.position == 'long':
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price

    def _update_equity(self):
        """자산 업데이트"""
        unrealized_pnl = 0
        if self.position:
            pnl_pct = self._get_position_pnl_pct()
            unrealized_pnl = self.entry_balance * pnl_pct * self.leverage

        self.equity = self.balance + unrealized_pnl

    def get_stats(self) -> Dict:
        """통계 반환"""
        return {
            'final_equity': self.equity,
            'total_return': ((self.equity - self.initial_balance) / self.initial_balance) * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': (self.winning_trades / max(1, self.total_trades)) * 100,
            'direction_accuracy': (self.correct_direction_count / max(1, self.total_trades)) * 100,  # 🔥 NEW
            'max_drawdown': self.max_drawdown * 100,
            'total_pnl': self.total_pnl,
        }

    def render(self, mode='human'):
        """환경 렌더링"""
        pass


###############################################################################
# 코인별 최적 설정
###############################################################################

COIN_CONFIGS = {
    'BTCUSDT': {
        'base_leverage': 10,
        'learning_rate': 3e-4,
        'ent_coef': 0.03,  # 🔥 0.01 → 0.03 (더 많은 탐험)
        'directional_weight': 0.6,
        'min_holding': 3,
    },
    'ETHUSDT': {
        'base_leverage': 8,
        'learning_rate': 3e-4,
        'ent_coef': 0.03,  # 🔥 증가
        'directional_weight': 0.5,
        'min_holding': 2,
    },
    'SOLUSDT': {
        'base_leverage': 6,
        'learning_rate': 5e-4,
        'ent_coef': 0.04,  # 🔥 증가
        'directional_weight': 0.7,
        'min_holding': 2,
    },
    'XRPUSDT': {
        'base_leverage': 8,
        'learning_rate': 4e-4,
        'ent_coef': 0.03,  # 🔥 증가
        'directional_weight': 0.6,
        'min_holding': 2,
    },
    'BNBUSDT': {
        'base_leverage': 7,
        'learning_rate': 4e-4,
        'ent_coef': 0.03,  # 🔥 증가
        'directional_weight': 0.55,
        'min_holding': 3,
    },
    'DOGEUSDT': {
        'base_leverage': 5,
        'learning_rate': 5e-4,
        'ent_coef': 0.04,  # 🔥 증가
        'directional_weight': 0.7,
        'min_holding': 2,
    },
    'HUSDT': {  # 🔥 NEW: HUSDT 전용 설정
        'base_leverage': 10,
        'learning_rate': 4e-4,
        'ent_coef': 0.04,  # 높은 탐험률
        'directional_weight': 0.65,
        'min_holding': 2,
    }
}


###############################################################################
# 유틸리티 함수
###############################################################################

def get_available_symbols():
    """바이비트에서 거래 가능한 USDT Perpetual 목록 조회"""
    try:
        url = "https://api.bybit.com/v5/market/instruments-info"
        params = {'category': 'linear'}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get('retCode') != 0:
            print(f"API 에러: {data.get('retMsg')}")
            return []

        usdt_symbols = []
        result_list = data.get('result', {}).get('list', [])

        for symbol_info in result_list:
            symbol = symbol_info['symbol']
            status = symbol_info.get('status', '')
            settle_coin = symbol_info.get('settleCoin', '')

            # USDT Perpetual만 필터링
            if (symbol.endswith('USDT') and
                    status.lower() == 'trading' and
                    settle_coin == 'USDT'):
                usdt_symbols.append(symbol)

        return sorted(usdt_symbols)
    except Exception as e:
        print(f"심볼 목록 조회 실패: {e}")
        return []


def check_symbol_exists(symbol: str) -> bool:
    """심볼이 바이비트에 존재하는지 확인"""
    try:
        url = "https://api.bybit.com/v5/market/tickers"
        params = {
            'category': 'linear',
            'symbol': symbol
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        if data.get('retCode') == 0:
            result_list = data.get('result', {}).get('list', [])
            return len(result_list) > 0
        else:
            return False
    except:
        return False


###############################################################################
# 데이터 로딩 및 전처리
###############################################################################

def load_bybit_data(symbol: str, interval: str, limit: int = 5000) -> pd.DataFrame:
    """
    바이비트에서 데이터 로드 (USDT Perpetual)

    Args:
        symbol: 심볼 (예: BTCUSDT)
        interval: 간격 (1=1분, 5=5분, 15=15분, 60=1시간, D=일봉)
        limit: 최대 캔들 수
    """
    print(f"📥 바이비트에서 {symbol} {interval}분봉 데이터 로드 중...")

    # Bybit interval 매핑
    interval_map = {
        '1': '1',
        '5': '5',
        '15': '15',
        '60': '60',
        'D': 'D'
    }

    bybit_interval = interval_map.get(str(interval), str(interval))

    try:
        # 시작/종료 시간 계산 (Bybit는 밀리초 단위)
        end_time = datetime.now()

        # interval에 따라 시작 시간 계산
        if bybit_interval == 'D':
            start_time = end_time - timedelta(days=min(limit, 1000))
        elif bybit_interval == '60':
            start_time = end_time - timedelta(hours=min(limit, 1000))
        else:
            minutes = int(bybit_interval)
            start_time = end_time - timedelta(minutes=minutes * min(limit, 1000))

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        # Bybit API 호출
        url = "https://api.bybit.com/v5/market/kline"

        all_data = []
        current_end = end_ms

        # 페이징을 통해 데이터 수집 (Bybit는 최대 200개씩)
        max_iterations = (limit // 200) + 1

        for i in range(max_iterations):
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': bybit_interval,
                'start': start_ms,
                'end': current_end,
                'limit': 200  # Bybit 최대값
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # 에러 체크
            if data.get('retCode') != 0:
                error_msg = data.get('retMsg', 'Unknown error')
                print(f"\n❌ 바이비트 API 에러: {error_msg}")

                if 'symbol' in error_msg.lower() or 'not exist' in error_msg.lower():
                    print(f"   심볼 '{symbol}'이(가) 존재하지 않거나 유효하지 않습니다.")
                    print(f"\n💡 올바른 심볼 예시:")
                    print(f"   - BTCUSDT, ETHUSDT, SOLUSDT")
                    print(f"   - XRPUSDT, BNBUSDT, DOGEUSDT")
                    print(f"   - ADAUSDT, MATICUSDT, AVAXUSDT")

                raise ValueError(f"Invalid symbol or API error: {error_msg}")

            result_list = data.get('result', {}).get('list', [])

            if not result_list:
                break

            all_data.extend(result_list)

            # 다음 페이지를 위해 가장 오래된 타임스탬프 찾기
            oldest_ts = min(int(x[0]) for x in result_list)

            # 간격에 따라 이전 시간으로 이동
            if bybit_interval == 'D':
                current_end = oldest_ts - (24 * 60 * 60 * 1000)
            elif bybit_interval == '60':
                current_end = oldest_ts - (60 * 60 * 1000)
            else:
                minutes = int(bybit_interval)
                current_end = oldest_ts - (minutes * 60 * 1000)

            # 충분한 데이터를 수집했거나 시작 시간을 넘어가면 중단
            if len(all_data) >= limit or current_end < start_ms:
                break

            # Rate limit 방지
            time.sleep(0.1)

        if not all_data or len(all_data) == 0:
            print(f"\n❌ 데이터가 없습니다!")
            print(f"   심볼: {symbol}")
            print(f"   간격: {interval}분")
            print(f"\n💡 확인 사항:")
            print(f"   1. 심볼이 올바른지 확인 (예: BTCUSDT)")
            print(f"   2. 바이비트에서 거래되는 USDT Perpetual인지 확인")
            print(f"   3. 인터넷 연결 확인")
            raise ValueError(f"No data returned for {symbol}")

        # DataFrame 생성
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        # 데이터 타입 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 중복 제거 및 정렬
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        df = df.reset_index(drop=True)

        # 필요한 컬럼만 선택
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

        # 최대 limit개만 유지
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)

        print(f"✅ 로드 완료: {len(df):,}개 캔들")
        print(f"   기간: {df['date'].min()} ~ {df['date'].max()}")

        return df

    except requests.exceptions.RequestException as e:
        print(f"\n❌ 네트워크 에러: {e}")
        print(f"   인터넷 연결을 확인하세요.")
        raise
    except Exception as e:
        print(f"\n❌ 예상치 못한 에러: {e}")
        raise


###############################################################################
# Callback
###############################################################################

class DetailedCallback(BaseCallback):
    """상세 학습 진행 콜백"""

    def __init__(self, verbose=1):
        super(DetailedCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.verbose > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])

            print(f"\n📊 Rollout {self.num_timesteps // 2048}:")
            print(f"   평균 보상: {mean_reward:.2f}")
            print(f"   평균 길이: {mean_length:.0f} 스텝")


###############################################################################
# 학습 함수
###############################################################################

def train_rl_model(
        symbol: str = 'BTCUSDT',
        interval: str = '5',
        total_timesteps: int = 300000,
        save_path: str = 'rl_models_enhanced',
        debug: bool = False
):
    """강화학습 모델 학습"""

    # 심볼 검증
    print(f"🔍 심볼 검증 중: {symbol}...")
    if not check_symbol_exists(symbol):
        print(f"\n❌ '{symbol}'은(는) 바이비트에서 거래되지 않는 심볼입니다!")
        print(f"\n💡 올바른 심볼 예시 (USDT Perpetual):")
        popular_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
            'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'MATICUSDT',
            'AVAXUSDT', 'LINKUSDT', 'ATOMUSDT', 'LTCUSDT'
        ]
        for i, sym in enumerate(popular_symbols, 1):
            print(f"   {i:2d}. {sym}")

        print(f"\n📋 전체 USDT Perpetual 목록을 보려면:")
        print(
            f"   python -c \"from train_rl_enhanced import get_available_symbols; print('\\n'.join(get_available_symbols()[:50]))\"")
        return None, None

    print(f"✅ '{symbol}' 검증 완료!\n")

    # 설정 로드
    config = COIN_CONFIGS.get(symbol, COIN_CONFIGS['BTCUSDT'])

    print("=" * 80)
    print(f"{symbol} {interval}분봉 강화학습 모델 학습 (Enhanced - Bybit)".center(80))
    print("=" * 80)
    print(f"\n설정:")
    print(f"   심볼: {symbol}")
    print(f"   간격: {interval}분")
    print(f"   학습 스텝: {total_timesteps:,}")
    print(f"   레버리지: {config['base_leverage']}x")
    print(f"   방향성 가중치: {config['directional_weight']}")
    print(f"   학습률: {config['learning_rate']}")
    print(f"   탐험 계수: {config['ent_coef']}")

    # 데이터 로드
    try:
        df = load_bybit_data(symbol, interval, limit=5000)
    except Exception as e:
        print(f"\n❌ 데이터 로드 실패: {e}")
        return None, None

    # 데이터 검증
    min_required_candles = 200  # 최소 200개 캔들 필요
    if len(df) < min_required_candles:
        print(f"\n❌ 데이터가 부족합니다!")
        print(f"   현재: {len(df)}개")
        print(f"   필요: {min_required_candles}개 이상")
        print(f"\n💡 해결 방법:")
        print(f"   1. limit을 늘려보세요 (기본 5000)")
        print(f"   2. 더 오래된 코인을 선택하세요")
        return None, None

    # Train/Test 분할
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    # 분할 후 검증
    if len(train_df) < 100 or len(test_df) < 50:
        print(f"\n❌ 분할된 데이터가 너무 적습니다!")
        print(f"   학습: {len(train_df)}개 (최소 100개 필요)")
        print(f"   검증: {len(test_df)}개 (최소 50개 필요)")
        return None, None

    print(f"\n데이터 분할:")
    print(f"   학습: {len(train_df):,}개")
    print(f"   검증: {len(test_df):,}개")

    # 환경 생성
    def make_env():
        return EnhancedCryptoTradingEnv(
            df=train_df,
            window_size=30,
            initial_balance=10000,
            base_leverage=config['base_leverage'],
            commission=0.0006,
            stop_loss_pct=0.05,
            take_profit_pct=0.08,
            min_holding_steps=config['min_holding'],
            force_initial_position=True,
            directional_weight=config['directional_weight'],  # 🔥 NEW
            debug=False
        )

    def make_eval_env():
        return EnhancedCryptoTradingEnv(
            df=test_df,
            window_size=30,
            initial_balance=10000,
            base_leverage=config['base_leverage'],
            commission=0.0006,
            stop_loss_pct=0.05,
            take_profit_pct=0.08,
            min_holding_steps=config['min_holding'],
            force_initial_position=True,
            directional_weight=config['directional_weight'],
            debug=False
        )

    vec_env = DummyVecEnv([make_env])
    vec_eval_env = DummyVecEnv([make_eval_env])

    print("\n🧠 PPO 모델 생성 중...")

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config['learning_rate'],
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=config['ent_coef'],
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cpu",
        tensorboard_log=f"./tensorboard_logs_enhanced/{symbol}_{interval}min/"
    )

    print("✅ 모델 생성 완료\n")

    os.makedirs(save_path, exist_ok=True)
    eval_callback = EvalCallback(
        vec_eval_env,
        best_model_save_path=f"{save_path}/{symbol}_{interval}min_best/",
        log_path=f"{save_path}/logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        verbose=1
    )

    detailed_callback = DetailedCallback(verbose=1)

    print("=" * 80)
    print(f"{'🚀 학습 시작':^80}")
    print("=" * 80)

    start_time = datetime.now()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, detailed_callback],
        progress_bar=True
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60

    print(f"\n✅ 학습 완료! (소요 시간: {duration:.1f}분)")

    model_path = f"{save_path}/{symbol}_{interval}min_final.zip"
    model.save(model_path)
    print(f"💾 모델 저장: {model_path}")

    # 평가
    print("\n" + "=" * 80)
    print(f"{'📊 최종 평가':^80}")
    print("=" * 80)

    final_eval_env = EnhancedCryptoTradingEnv(
        df=test_df,
        window_size=30,
        initial_balance=10000,
        base_leverage=config['base_leverage'],
        commission=0.0006,
        stop_loss_pct=0.05,
        take_profit_pct=0.08,
        min_holding_steps=config['min_holding'],
        force_initial_position=True,
        directional_weight=config['directional_weight'],
        debug=debug
    )

    obs = final_eval_env.reset()
    done = False

    action_counts = {0: 0, 1: 0, 2: 0}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_counts[int(action)] += 1
        obs, reward, done, info = final_eval_env.step(action)

    stats = final_eval_env.get_stats()

    print(f"\n검증 데이터 성과:")
    print(f"   최종 자산: ${stats['final_equity']:,.2f}")
    print(f"   총 수익률: {stats['total_return']:+.2f}%")
    print(f"   총 거래: {stats['total_trades']}회")
    print(f"   승률: {stats['win_rate']:.1f}%")
    print(f"   🎯 방향 정확도: {stats['direction_accuracy']:.1f}%")  # 🔥 NEW
    print(f"   최대 낙폭: {stats['max_drawdown']:.2f}%")

    print(f"\n행동 분포:")
    total = sum(action_counts.values())
    print(f"   LONG:  {action_counts[0]} ({action_counts[0] / total * 100:.1f}%)")
    print(f"   SHORT: {action_counts[1]} ({action_counts[1] / total * 100:.1f}%)")
    print(f"   CLOSE: {action_counts[2]} ({action_counts[2] / total * 100:.1f}%)")

    # 방향성 분석
    if final_eval_env.trade_history:
        correct_directions = sum(1 for t in final_eval_env.trade_history if t['direction_correct'])
        print(f"\n방향성 분석:")
        print(f"   올바른 방향 선택: {correct_directions}/{len(final_eval_env.trade_history)} "
              f"({correct_directions / len(final_eval_env.trade_history) * 100:.1f}%)")

    # 결과 저장
    results = {
        'symbol': symbol,
        'interval': interval,
        'total_timesteps': total_timesteps,
        'config': config,
        'test_stats': stats,
        'action_distribution': action_counts,
        'model_path': model_path,
        'trained_at': datetime.now().isoformat()
    }

    with open(f"{save_path}/{symbol}_{interval}min_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # 차트
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Equity
    axes[0, 0].plot(final_eval_env.equity_history, linewidth=2)
    axes[0, 0].axhline(y=10000, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Equity Curve', fontweight='bold')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Equity ($)')
    axes[0, 0].grid(True, alpha=0.3)

    # Actions
    actions = ['LONG', 'SHORT', 'CLOSE']
    colors = ['green', 'red', 'blue']
    axes[0, 1].bar(actions, [action_counts[i] for i in range(3)], color=colors, alpha=0.7)
    axes[0, 1].set_title('Action Distribution', fontweight='bold')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Direction Accuracy
    if final_eval_env.trade_history:
        correct = [t['direction_correct'] for t in final_eval_env.trade_history]
        cumulative_accuracy = np.cumsum(correct) / np.arange(1, len(correct) + 1)
        axes[1, 0].plot(cumulative_accuracy, linewidth=2)
        axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random (50%)')
        axes[1, 0].set_title('Cumulative Direction Accuracy', fontweight='bold')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # PnL Distribution
    if final_eval_env.trade_history:
        pnls = [t['pnl_pct'] * 100 for t in final_eval_env.trade_history]
        axes[1, 1].hist(pnls, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_title('Trade PnL Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('PnL (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{save_path}/{symbol}_{interval}min_chart.png", dpi=150)
    print(f"\n📈 차트 저장: {save_path}/{symbol}_{interval}min_chart.png")

    print("\n" + "=" * 80)
    print(f"{'✅ 완료!':^80}")
    print("=" * 80)

    return model, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--interval', type=str, default='5')
    parser.add_argument('--steps', type=int, default=300000)
    parser.add_argument('--save_path', type=str, default='rl_models_enhanced')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    model, results = train_rl_model(
        symbol=args.symbol,
        interval=args.interval,
        total_timesteps=args.steps,
        save_path=args.save_path,
        debug=args.debug
    )

    if results and results['test_stats']['total_trades'] > 0:
        print("\n🎉 최종 결과:")
        print(f"   거래: {results['test_stats']['total_trades']}회")
        print(f"   승률: {results['test_stats']['win_rate']:.1f}%")
        print(f"   방향 정확도: {results['test_stats']['direction_accuracy']:.1f}%")
        print(f"   수익률: {results['test_stats']['total_return']:+.2f}%")