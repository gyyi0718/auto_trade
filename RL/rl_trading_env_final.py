# rl_trading_env_final.py
# -*- coding: utf-8 -*-
"""
강화학습 기반 암호화폐 트레이딩 환경 (최종 수정 버전)
- 버그 수정: reset에서 current_price 설정
- 초기 랜덤 포지션으로 거래 강제
- 더 단순한 로직
"""
import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Tuple, Dict
from enum import IntEnum


class Actions(IntEnum):
    """행동 정의 - 단순화"""
    LONG = 0  # 롱 포지션 (BUY + HOLD LONG)
    SHORT = 1  # 숏 포지션 (SELL + HOLD SHORT)
    CLOSE = 2  # 청산


class CryptoTradingEnv(gym.Env):
    """
    암호화폐 거래 환경 (최종 수정 버전)
    - 3가지 액션만: LONG, SHORT, CLOSE
    - 초기 랜덤 포지션으로 시작
    - 명확한 거래 로직
    """

    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            df: pd.DataFrame,
            window_size: int = 30,
            initial_balance: float = 10000,
            leverage: int = 10,
            commission: float = 0.0006,
            stop_loss_pct: float = 0.05,
            take_profit_pct: float = 0.08,
            reward_scaling: float = 1e4,
            min_holding_steps: int = 3,
            force_initial_position: bool = True,
            debug: bool = False
    ):
        super(CryptoTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.commission = commission
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.reward_scaling = reward_scaling
        self.min_holding_steps = min_holding_steps
        self.force_initial_position = force_initial_position
        self.debug = debug

        self._calculate_features()

        # 🔥 3가지 액션만!
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, len(self.feature_columns) + 3),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.reset()

    def _calculate_features(self):
        """기술적 지표 계산"""
        df = self.df.copy()

        df['returns'] = df['close'].pct_change()
        df['volume_norm'] = np.log1p(df['volume'])

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'] / 100

        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_ratio_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['sma_ratio_50'] = (df['close'] - df['sma_50']) / df['sma_50']

        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        df['price_change_20'] = df['close'].pct_change(20)
        df['volume_change'] = df['volume'].pct_change()

        df = df.fillna(0).replace([np.inf, -np.inf], 0)

        self.feature_columns = [
            'returns', 'volume_norm', 'rsi', 'bb_position',
            'sma_ratio_20', 'sma_ratio_50',
            'macd', 'macd_signal', 'macd_hist',
            'atr_pct', 'price_change_5', 'price_change_10', 'price_change_20',
            'volume_change'
        ]

        self.df = df
        self.feature_mean = self.df[self.feature_columns].mean()
        self.feature_std = self.df[self.feature_columns].std() + 1e-8

    def reset(self) -> np.ndarray:
        """환경 초기화"""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.current_step = self.window_size

        # 🔥 버그 수정: current_price 설정!
        self.current_price = self.df.loc[self.current_step, 'close']

        self.position = None
        self.entry_price = 0
        self.position_size = 0
        self.entry_balance = 0

        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        self.max_equity = self.initial_balance
        self.max_drawdown = 0

        self.trade_history = []
        self.equity_history = [self.initial_balance]

        self.steps_in_position = 0

        # 🔥 초기 랜덤 포지션으로 거래 강제!
        if self.force_initial_position:
            initial_direction = np.random.choice(['long', 'short'])
            self._open_position(initial_direction)
            if self.debug:
                print(f"\n🎲 초기 포지션: {initial_direction.upper()} @ {self.current_price:.2f}")

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """현재 상태 관찰"""
        start = self.current_step - self.window_size
        end = self.current_step

        obs_data = self.df[self.feature_columns].iloc[start:end].values
        obs_data = (obs_data - self.feature_mean.values) / self.feature_std.values
        obs_data = np.clip(obs_data, -10, 10)

        position_info = np.zeros((self.window_size, 3))
        if self.position == 'long':
            position_info[:, 0] = 1
            position_info[:, 2] = (self.current_price - self.entry_price) / (self.entry_price + 1e-8)
        elif self.position == 'short':
            position_info[:, 1] = 1
            position_info[:, 2] = (self.entry_price - self.current_price) / (self.entry_price + 1e-8)

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

        if self.equity < self.initial_balance * 0.2:
            done = True
            reward -= 5

        obs = self._get_observation()

        info = {
            'equity': self.equity,
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'max_drawdown': self.max_drawdown,
            'pnl': self.total_pnl,
            'steps_in_position': self.steps_in_position
        }

        return obs, reward, done, info

    def _execute_action(self, action: int) -> float:
        """
        행동 실행 (단순화된 로직)

        LONG (0): 롱 포지션 원함
        SHORT (1): 숏 포지션 원함
        CLOSE (2): 청산 원함
        """
        reward = 0

        # 🔥 Stop Loss / Take Profit 체크 (자동 청산)
        if self.position:
            pnl_pct = self._get_position_pnl_pct()

            if pnl_pct <= -self.stop_loss_pct:
                reward = self._close_position("Stop Loss")
                reward -= 0.3
                if self.debug:
                    print(f"Step {self.current_step}: 💔 Stop Loss ({pnl_pct * 100:.2f}%)")
                return reward * self.reward_scaling

            if pnl_pct >= self.take_profit_pct:
                reward = self._close_position("Take Profit")
                reward += 1.5
                if self.debug:
                    print(f"Step {self.current_step}: 💰 Take Profit ({pnl_pct * 100:.2f}%)")
                return reward * self.reward_scaling

        # 🔥 액션 처리 (단순 로직)
        if action == Actions.LONG:
            if self.position == 'long':
                # 이미 롱 → 유지
                pnl = self._get_position_pnl_pct() * self.leverage
                reward = pnl * 0.1
            elif self.position == 'short':
                # 숏 → 롱 전환
                if self.steps_in_position >= self.min_holding_steps:
                    reward = self._close_position("Switch")
                    reward += self._open_position('long')
                    if self.debug:
                        print(f"Step {self.current_step}: 🔄 SHORT → LONG")
                else:
                    # 너무 빨리 전환 시도 → 유지
                    pnl = self._get_position_pnl_pct() * self.leverage
                    reward = pnl * 0.1 - 0.01
            else:
                # 포지션 없음 → 롱 진입
                reward = self._open_position('long')
                if self.debug:
                    print(f"Step {self.current_step}: 📈 OPEN LONG @ {self.current_price:.2f}")

        elif action == Actions.SHORT:
            if self.position == 'short':
                # 이미 숏 → 유지
                pnl = self._get_position_pnl_pct() * self.leverage
                reward = pnl * 0.1
            elif self.position == 'long':
                # 롱 → 숏 전환
                if self.steps_in_position >= self.min_holding_steps:
                    reward = self._close_position("Switch")
                    reward += self._open_position('short')
                    if self.debug:
                        print(f"Step {self.current_step}: 🔄 LONG → SHORT")
                else:
                    # 너무 빨리 전환 시도 → 유지
                    pnl = self._get_position_pnl_pct() * self.leverage
                    reward = pnl * 0.1 - 0.01
            else:
                # 포지션 없음 → 숏 진입
                reward = self._open_position('short')
                if self.debug:
                    print(f"Step {self.current_step}: 📉 OPEN SHORT @ {self.current_price:.2f}")

        elif action == Actions.CLOSE:
            if self.position and self.steps_in_position >= self.min_holding_steps:
                # 청산
                pos_type = self.position.upper()  # 🔥 미리 저장
                reward = self._close_position("Manual")
                if self.debug:
                    print(f"Step {self.current_step}: ✂️  CLOSE {pos_type}")
            elif self.position:
                # 너무 빨리 청산 시도 → 유지
                pnl = self._get_position_pnl_pct() * self.leverage
                reward = pnl * 0.1 - 0.01
            else:
                # 포지션 없는데 CLOSE → 작은 페널티
                reward = -0.01

        return reward * self.reward_scaling

    def _open_position(self, direction: str) -> float:
        """포지션 진입"""
        self.position = direction
        self.entry_price = self.current_price
        self.entry_balance = self.balance
        self.position_size = (self.balance * self.leverage) / self.current_price
        self.steps_in_position = 0

        fee = self.balance * self.commission
        self.balance -= fee

        return -self.commission

    def _close_position(self, reason: str) -> float:
        """포지션 청산"""
        if not self.position:
            return 0

        if self.position == 'long':
            pnl = (self.current_price - self.entry_price) * self.position_size
        else:
            pnl = (self.entry_price - self.current_price) * self.position_size

        fee = (self.position_size * self.current_price) * self.commission
        net_pnl = pnl - fee

        self.balance += net_pnl

        self.total_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1
        self.total_pnl += net_pnl

        self.trade_history.append({
            'entry_price': self.entry_price,
            'exit_price': self.current_price,
            'direction': self.position,
            'pnl': net_pnl,
            'pnl_pct': net_pnl / self.entry_balance,
            'reason': reason,
            'holding_time': self.steps_in_position
        })

        if self.debug:
            print(f"  → Trade #{self.total_trades}: {self.position.upper()} | "
                  f"PNL: ${net_pnl:.2f} ({net_pnl / self.entry_balance * 100:+.2f}%) | {reason}")

        self.position = None
        self.entry_price = 0
        self.position_size = 0

        reward = net_pnl / self.entry_balance

        if net_pnl > 0 and self.steps_in_position < 20:
            reward *= 1.2

        return reward

    def _get_position_pnl_pct(self) -> float:
        """포지션 손익률"""
        if not self.position:
            return 0

        if self.position == 'long':
            return (self.current_price - self.entry_price) / (self.entry_price + 1e-8)
        else:
            return (self.entry_price - self.current_price) / (self.entry_price + 1e-8)

    def _update_equity(self):
        """Equity 업데이트"""
        if self.position:
            unrealized_pnl = self._get_position_pnl_pct() * self.entry_balance * self.leverage
            self.equity = self.balance + unrealized_pnl
        else:
            self.equity = self.balance

    def get_stats(self) -> Dict:
        """통계 반환"""
        if not self.equity_history:
            return {}

        returns = np.diff(self.equity_history) / (np.array(self.equity_history[:-1]) + 1e-8)
        returns = returns[~np.isnan(returns)]

        if len(returns) > 0:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0

        return {
            'final_equity': self.equity,
            'total_return': (self.equity / self.initial_balance - 1) * 100,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades) * 100,
            'max_drawdown': self.max_drawdown * 100,
            'sharpe_ratio': sharpe,
            'total_pnl': self.total_pnl
        }
