# paper_trading_improved.py
# -*- coding: utf-8 -*-
"""
개선 버전 모델용 백테스트/페이퍼 트레이딩
- AdaptiveCryptoTradingEnv 사용
- 동적 파라미터 추적
"""
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import requests
import gym
from gym import spaces
from typing import Tuple, Dict
from enum import IntEnum
import warnings

warnings.filterwarnings('ignore')


###############################################################################
# 환경 정의 (train_rl_improved.py와 동일)
###############################################################################

class Actions(IntEnum):
    """3가지 액션"""
    LONG = 0
    SHORT = 1
    CLOSE = 2


class AdaptiveCryptoTradingEnv(gym.Env):
    """알트코인 최적화 거래 환경"""

    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            df: pd.DataFrame,
            window_size: int = 30,
            initial_balance: float = 10000,
            base_leverage: int = 10,
            commission: float = 0.0006,
            stop_loss_multiplier: float = 2.0,
            take_profit_multiplier: float = 3.0,
            reward_scaling: float = 1e4,
            min_holding_steps: int = 2,
            force_initial_position: bool = True,
            use_dynamic_leverage: bool = True,
            use_adaptive_sltp: bool = True,
            prevent_overtrading: bool = True,  # 🔥 새로 추가
            switch_penalty: float = 0.5,  # 🔥 스위칭 페널티
            debug: bool = False
    ):
        super(AdaptiveCryptoTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.base_leverage = base_leverage
        self.commission = commission
        self.stop_loss_multiplier = stop_loss_multiplier
        self.take_profit_multiplier = take_profit_multiplier
        self.reward_scaling = reward_scaling
        self.min_holding_steps = min_holding_steps
        self.force_initial_position = force_initial_position
        self.use_dynamic_leverage = use_dynamic_leverage
        self.use_adaptive_sltp = use_adaptive_sltp
        self.prevent_overtrading = prevent_overtrading  # 🔥
        self.switch_penalty = switch_penalty  # 🔥
        self.debug = debug

        self._calculate_features()
        self._calculate_market_regime()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, len(self.feature_columns) + 4),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.reset()

    def _calculate_features(self):
        """기술적 지표 계산"""
        df = self.df.copy()

        df['returns'] = df['close'].pct_change()
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

        # 변동성 지표
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volatility_norm'] = df['volatility'] / df['volatility'].rolling(window=100).mean()

        # Price changes
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
            'volume_change', 'volatility_norm'
        ]

        self.df = df
        self.feature_mean = self.df[self.feature_columns].mean()
        self.feature_std = self.df[self.feature_columns].std() + 1e-8

    def _calculate_market_regime(self):
        """시장 상태 분석"""
        self.avg_volatility = self.df['atr_pct'].mean()
        self.volatility_std = self.df['atr_pct'].std()

        if self.avg_volatility < 0.01:
            self.market_regime = 'low_vol'
        elif self.avg_volatility < 0.03:
            self.market_regime = 'medium_vol'
        else:
            self.market_regime = 'high_vol'

    def _get_dynamic_leverage(self, current_volatility: float) -> int:
        """변동성 기반 동적 레버리지"""
        if not self.use_dynamic_leverage:
            return self.base_leverage

        volatility_ratio = current_volatility / (self.avg_volatility + 1e-8)

        if volatility_ratio > 1.5:
            leverage = max(3, self.base_leverage // 3)
        elif volatility_ratio > 1.2:
            leverage = max(5, self.base_leverage // 2)
        else:
            leverage = self.base_leverage

        return int(leverage)

    def _get_adaptive_sltp(self, current_atr_pct: float) -> Tuple[float, float]:
        """ATR 기반 동적 Stop Loss / Take Profit"""
        if not self.use_adaptive_sltp:
            return 0.05, 0.08

        stop_loss = current_atr_pct * self.stop_loss_multiplier
        take_profit = current_atr_pct * self.take_profit_multiplier

        stop_loss = np.clip(stop_loss, 0.02, 0.10)
        take_profit = np.clip(take_profit, 0.03, 0.15)

        return float(stop_loss), float(take_profit)

    def reset(self) -> np.ndarray:
        """환경 초기화"""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.current_step = self.window_size

        self.current_price = self.df.loc[self.current_step, 'close']
        current_atr = self.df.loc[self.current_step, 'atr_pct']

        self.leverage = self._get_dynamic_leverage(current_atr)
        self.stop_loss_pct, self.take_profit_pct = self._get_adaptive_sltp(current_atr)

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

        if self.force_initial_position:
            initial_direction = np.random.choice(['long', 'short'])
            self._open_position(initial_direction)

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """현재 상태 관찰"""
        start = self.current_step - self.window_size
        end = self.current_step

        obs_data = self.df[self.feature_columns].iloc[start:end].values
        obs_data = (obs_data - self.feature_mean.values) / self.feature_std.values
        obs_data = np.clip(obs_data, -10, 10)

        position_info = np.zeros((self.window_size, 4))

        current_vol_norm = self.df.loc[self.current_step, 'volatility_norm']
        position_info[:, 3] = current_vol_norm

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
        current_atr = self.df.loc[self.current_step, 'atr_pct']

        if self.use_dynamic_leverage:
            self.leverage = self._get_dynamic_leverage(current_atr)
        if self.use_adaptive_sltp:
            self.stop_loss_pct, self.take_profit_pct = self._get_adaptive_sltp(current_atr)

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
            'steps_in_position': self.steps_in_position,
            'current_leverage': self.leverage,
            'current_sl': self.stop_loss_pct,
            'current_tp': self.take_profit_pct
        }

        return obs, reward, done, info

    def _execute_action(self, action: int) -> float:
        """행동 실행"""
        reward = 0

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
                    print(f"Step {self.current_step}: 🎯 Take Profit ({pnl_pct * 100:.2f}%)")
                return reward * self.reward_scaling

        # 🔥 오버트레이딩 방지
        if self.prevent_overtrading and self.position:
            # 최소 보유 시간 미충족 시 스위칭 강하게 페널티
            if self.steps_in_position < self.min_holding_steps:
                if (action == Actions.LONG and self.position == 'short') or \
                        (action == Actions.SHORT and self.position == 'long'):
                    reward -= self.switch_penalty
                    if self.debug:
                        print(f"Step {self.current_step}: ⚠️  Too early switching penalty")
                    return reward * self.reward_scaling

        if action == Actions.LONG:
            if self.position == 'short':
                # 🔥 스위칭에 추가 페널티
                if self.prevent_overtrading:
                    reward -= 0.2
                reward += self._close_position("Switch to LONG")
            if not self.position:
                self._open_position('long')
                reward += 0.05

        elif action == Actions.SHORT:
            if self.position == 'long':
                # 🔥 스위칭에 추가 페널티
                if self.prevent_overtrading:
                    reward -= 0.2
                reward += self._close_position("Switch to SHORT")
            if not self.position:
                self._open_position('short')
                reward += 0.05

        elif action == Actions.CLOSE:
            if self.position and self.steps_in_position >= self.min_holding_steps:
                reward = self._close_position("Manual Close")
            elif self.position:
                reward -= 0.1

        if self.position:
            pnl_pct = self._get_position_pnl_pct()
            reward += pnl_pct * 0.2

        return reward * self.reward_scaling

    def _open_position(self, direction: str):
        """포지션 오픈"""
        self.position = direction
        self.entry_price = self.current_price
        self.entry_balance = self.balance

        margin = self.balance * 0.95
        self.position_size = (margin * self.leverage) / self.current_price

        commission_cost = self.position_size * self.current_price * self.commission
        self.balance -= commission_cost

        if self.debug:
            print(f"   📈 {direction.upper()} 진입 @ {self.entry_price:.2f} "
                  f"(Leverage: {self.leverage}x, Size: {self.position_size:.4f})")

    def _close_position(self, reason: str = "") -> float:
        """포지션 청산"""
        if not self.position:
            return 0

        # 🔥 버그 수정: 리셋 전에 값 저장
        entry_price_saved = self.entry_price
        exit_price_saved = self.current_price
        position_saved = self.position
        leverage_saved = self.leverage

        pnl_pct = self._get_position_pnl_pct()
        pnl = self.entry_balance * pnl_pct * self.leverage

        commission_cost = self.position_size * self.current_price * self.commission
        net_pnl = pnl - commission_cost

        self.balance = self.entry_balance + net_pnl
        self.total_pnl += net_pnl

        if net_pnl > 0:
            self.winning_trades += 1

        self.total_trades += 1

        # 🔥 저장된 값 사용
        self.trade_history.append({
            'entry_price': entry_price_saved,
            'exit_price': exit_price_saved,
            'position': position_saved,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'holding_time': self.steps_in_position,
            'reason': reason,
            'leverage': leverage_saved,
            'timestamp': self.df.loc[self.current_step, 'timestamp']
        })

        if self.debug:
            print(f"   📉 {position_saved.upper()} 청산 @ {exit_price_saved:.2f} "
                  f"| PnL: {net_pnl:+.2f} ({pnl_pct * 100:+.2f}%) | {reason}")

        self.position = None
        self.entry_price = 0
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
            'max_drawdown': self.max_drawdown * 100,
            'total_pnl': self.total_pnl,
            'market_regime': self.market_regime,
            'avg_volatility': self.avg_volatility
        }


###############################################################################
# 데이터 로드
###############################################################################

def fetch_bybit_data(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """Bybit에서 데이터 가져오기"""
    url = "https://api.bybit.com/v5/market/kline"

    all_data = []
    end_time = None

    while len(all_data) < limit:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": 200
        }

        if end_time:
            params["end"] = end_time

        response = requests.get(url, params=params)
        data = response.json()

        if data['retCode'] != 0:
            print(f"❌ API 에러: {data['retMsg']}")
            break

        result = data['result']['list']
        if not result:
            break

        all_data.extend(result)
        end_time = int(result[-1][0]) - 1

        if len(result) < 200:
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    return df


###############################################################################
# 백테스트
###############################################################################

def backtest_recent_data(
        model_path: str,
        symbol: str = 'BTCUSDT',
        interval: str = '5',
        limit: int = 1000,
        initial_balance: float = 10000,
        min_holding_steps: int = 3,  # 🔥 새로 추가
        base_leverage: int = 5,  # 🔥 새로 추가
        prevent_overtrading: bool = True,  # 🔥 새로 추가
        debug: bool = False
):
    """최근 데이터로 백테스트"""
    print("\n" + "=" * 80)
    print(f"{'📊 백테스트 (개선 버전 모델)':^80}")
    print("=" * 80)

    print(f"🔄 모델 로드: {model_path}")
    model = PPO.load(model_path, device='cpu')

    print(f"📥 {symbol} {interval}분봉 최신 데이터 다운로드 중...")
    df = fetch_bybit_data(symbol, interval, limit=limit)

    if len(df) < 100:
        print(f"❌ 데이터가 부족합니다: {len(df)}개")
        return None

    print(f"✅ {len(df)}개 캔들 로드")
    print(f"   기간: {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")

    # 환경 생성 (파라미터 사용)
    env = AdaptiveCryptoTradingEnv(
        df=df,
        window_size=30,
        initial_balance=initial_balance,
        base_leverage=base_leverage,  # 🔥 파라미터 사용
        commission=0.0006,
        stop_loss_multiplier=2.5,
        take_profit_multiplier=3.5,
        min_holding_steps=min_holding_steps,  # 🔥 파라미터 사용
        force_initial_position=True,
        use_dynamic_leverage=True,
        use_adaptive_sltp=True,
        prevent_overtrading=prevent_overtrading,  # 🔥 파라미터 사용
        switch_penalty=0.5,
        debug=debug
    )

    print("🚀 백테스트 실행 중...")
    if debug:
        print("\n" + "-" * 80)

    obs = env.reset()
    done = False

    action_counts = {0: 0, 1: 0, 2: 0}
    leverage_history = []
    sltp_history = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_counts[int(action)] += 1
        obs, reward, done, info = env.step(action)

        leverage_history.append(info['current_leverage'])
        sltp_history.append((info['current_sl'], info['current_tp']))

    stats = env.get_stats()

    if debug:
        print("-" * 80 + "\n")

    print("\n" + "=" * 80)
    print(f"{'📊 백테스트 결과':^80}")
    print("=" * 80)

    print(f"\n성과:")
    print(f"   초기 자산: ${initial_balance:,.2f}")
    print(f"   최종 자산: ${stats['final_equity']:,.2f}")
    print(f"   총 수익률: {stats['total_return']:+.2f}%")
    print(f"   총 손익: ${stats['total_pnl']:+,.2f}")

    print(f"\n거래 통계:")
    print(f"   총 거래: {stats['total_trades']}회")
    print(f"   승리: {stats['winning_trades']}회")
    print(f"   패배: {stats['total_trades'] - stats['winning_trades']}회")
    print(f"   승률: {stats['win_rate']:.1f}%")
    print(f"   최대 낙폭: {stats['max_drawdown']:.2f}%")

    # 🔥 오버트레이딩 분석
    if env.trade_history:
        switch_trades = sum(1 for t in env.trade_history if 'Switch' in t['reason'])
        switch_pct = switch_trades / stats['total_trades'] * 100
        print(f"\n⚠️  오버트레이딩 분석:")
        print(f"   스위칭 거래: {switch_trades}회 ({switch_pct:.1f}%)")
        if switch_pct > 70:
            print(f"   ❌ 과도한 스위칭 발생! --min-holding 값을 높이세요")
        elif switch_pct > 50:
            print(f"   ⚠️  스위칭이 많습니다. --min-holding 조정 권장")
        else:
            print(f"   ✅ 스위칭 비율 정상")

    print(f"\n시장 상태:")
    print(f"   변동성: {stats['market_regime'].upper()}")
    print(f"   평균 ATR: {stats['avg_volatility'] * 100:.2f}%")

    print(f"\n동적 파라미터:")
    avg_leverage = np.mean(leverage_history)
    avg_sl = np.mean([s[0] for s in sltp_history]) * 100
    avg_tp = np.mean([s[1] for s in sltp_history]) * 100
    print(f"   평균 레버리지: {avg_leverage:.1f}x")
    print(f"   평균 Stop Loss: {avg_sl:.2f}%")
    print(f"   평균 Take Profit: {avg_tp:.2f}%")

    print(f"\n행동 분포:")
    total_actions = sum(action_counts.values())
    print(f"   LONG:  {action_counts[0]:4d} ({action_counts[0] / total_actions * 100:5.1f}%)")
    print(f"   SHORT: {action_counts[1]:4d} ({action_counts[1] / total_actions * 100:5.1f}%)")
    print(f"   CLOSE: {action_counts[2]:4d} ({action_counts[2] / total_actions * 100:5.1f}%)")

    # 거래 내역
    if env.trade_history:
        print(f"\n거래 상세:")
        avg_holding = np.mean([t['holding_time'] for t in env.trade_history])
        avg_pnl_pct = np.mean([t['pnl_pct'] * 100 for t in env.trade_history])
        winning_pnl = np.mean([t['pnl_pct'] * 100 for t in env.trade_history if t['pnl'] > 0] or [0])
        losing_pnl = np.mean([t['pnl_pct'] * 100 for t in env.trade_history if t['pnl'] < 0] or [0])

        print(f"   평균 보유: {avg_holding:.1f} 스텝 ({avg_holding * int(interval):.0f}분)")
        print(f"   평균 손익: {avg_pnl_pct:+.2f}%")
        print(f"   평균 수익 (승리): {winning_pnl:+.2f}%")
        print(f"   평균 손실 (패배): {losing_pnl:+.2f}%")

        if winning_pnl != 0 and losing_pnl != 0:
            profit_factor = abs(winning_pnl) / abs(losing_pnl)
            print(f"   손익비: {profit_factor:.2f}")

        # 최근 10개 거래
        print(f"\n최근 거래 ({min(10, len(env.trade_history))}개):")
        for i, trade in enumerate(env.trade_history[-10:], 1):
            emoji = "✅" if trade['pnl'] > 0 else "❌"
            print(f"   {emoji} {trade['position'].upper():5s} | "
                  f"진입: ${trade['entry_price']:8.2f} → "
                  f"청산: ${trade['exit_price']:8.2f} | "
                  f"손익: {trade['pnl_pct'] * 100:+6.2f}% | "
                  f"{trade['reason']:15s} | "
                  f"{trade['leverage']}x")

    # 차트 생성
    print("\n📈 차트 생성 중...")
    create_backtest_chart(env, action_counts, leverage_history, sltp_history, symbol, interval)

    # 결과 저장
    results = {
        'model_path': model_path,
        'symbol': symbol,
        'interval': interval,
        'backtest_period': {
            'start': str(df['timestamp'].iloc[0]),
            'end': str(df['timestamp'].iloc[-1]),
            'candles': len(df)
        },
        'stats': stats,
        'action_distribution': action_counts,
        'avg_leverage': float(avg_leverage),
        'avg_stop_loss': float(avg_sl),
        'avg_take_profit': float(avg_tp),
        'trades': env.trade_history,
        'tested_at': datetime.now().isoformat()
    }

    output_file = f'backtest_improved_{symbol}_{interval}min_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾 결과 저장: {output_file}")

    print("\n" + "=" * 80)
    print(f"{'✅ 백테스트 완료!':^80}")
    print("=" * 80)

    return results


def create_backtest_chart(env, action_counts, leverage_history, sltp_history, symbol, interval):
    """백테스트 결과 차트"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Equity Curve (큼)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(env.equity_history, linewidth=2, color='blue', label='Equity')
    ax1.axhline(y=env.initial_balance, color='red', linestyle='--', alpha=0.5, label='Initial')
    ax1.fill_between(range(len(env.equity_history)), env.initial_balance, env.equity_history,
                     where=[e >= env.initial_balance for e in env.equity_history],
                     color='green', alpha=0.1)
    ax1.fill_between(range(len(env.equity_history)), env.initial_balance, env.equity_history,
                     where=[e < env.initial_balance for e in env.equity_history],
                     color='red', alpha=0.1)
    ax1.set_title('Equity Curve', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Equity ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Action Distribution
    ax2 = fig.add_subplot(gs[0, 2])
    actions = ['LONG', 'SHORT', 'CLOSE']
    colors = ['green', 'red', 'blue']
    ax2.bar(actions, [action_counts[i] for i in range(3)], color=colors, alpha=0.7)
    ax2.set_title('Action Distribution', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate([action_counts[i] for i in range(3)]):
        ax2.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

    # 3. Dynamic Leverage
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(leverage_history, linewidth=1, alpha=0.7, color='purple')
    ax3.set_title('Dynamic Leverage', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Leverage')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=np.mean(leverage_history), color='orange', linestyle='--',
                alpha=0.5, label=f'Avg: {np.mean(leverage_history):.1f}x')
    ax3.legend()

    # 4. Stop Loss / Take Profit
    ax4 = fig.add_subplot(gs[1, 1])
    sl_values = [s[0] * 100 for s in sltp_history]
    tp_values = [s[1] * 100 for s in sltp_history]
    ax4.plot(sl_values, label='Stop Loss %', linewidth=1, alpha=0.7, color='red')
    ax4.plot(tp_values, label='Take Profit %', linewidth=1, alpha=0.7, color='green')
    ax4.set_title('Adaptive SL/TP', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Percentage (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Trade PnL Distribution
    ax5 = fig.add_subplot(gs[1, 2])
    if env.trade_history:
        pnls = [t['pnl_pct'] * 100 for t in env.trade_history]
        colors_pnl = ['green' if p > 0 else 'red' for p in pnls]
        ax5.bar(range(len(pnls)), pnls, color=colors_pnl, alpha=0.7)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.set_title('Trade PnL (%)', fontweight='bold', fontsize=12)
        ax5.set_xlabel('Trade #')
        ax5.set_ylabel('PnL (%)')
        ax5.grid(True, alpha=0.3, axis='y')

    # 6. Drawdown
    ax6 = fig.add_subplot(gs[2, 0])
    equity_array = np.array(env.equity_history)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (running_max - equity_array) / running_max * 100
    ax6.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
    ax6.plot(drawdown, color='red', linewidth=1)
    ax6.set_title('Drawdown', fontweight='bold', fontsize=12)
    ax6.set_xlabel('Steps')
    ax6.set_ylabel('Drawdown (%)')
    ax6.grid(True, alpha=0.3)
    ax6.invert_yaxis()

    # 7. Win/Loss Ratio
    ax7 = fig.add_subplot(gs[2, 1])
    if env.trade_history:
        wins = sum(1 for t in env.trade_history if t['pnl'] > 0)
        losses = len(env.trade_history) - wins
        ax7.pie([wins, losses], labels=['Win', 'Loss'], colors=['lightgreen', 'lightcoral'],
                autopct='%1.1f%%', startangle=90)
        ax7.set_title('Win/Loss Ratio', fontweight='bold', fontsize=12)

    # 8. Statistics Summary
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    stats = env.get_stats()
    summary_text = f"""
SUMMARY

Final Equity: ${stats['final_equity']:,.2f}
Total Return: {stats['total_return']:+.2f}%
Total Trades: {stats['total_trades']}
Win Rate: {stats['win_rate']:.1f}%
Max DD: {stats['max_drawdown']:.2f}%

Market: {stats['market_regime'].upper()}
Avg ATR: {stats['avg_volatility'] * 100:.2f}%
    """
    ax8.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'Backtest Results: {symbol} ({interval}min)', fontsize=16, fontweight='bold')

    output_file = f'backtest_improved_{symbol}_{interval}min_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   저장: {output_file}")
    plt.close()


###############################################################################
# 메인
###############################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='개선 버전 모델 백테스트')
    parser.add_argument('--model', type=str, required=True, help='모델 경로 (.zip)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='거래 심볼')
    parser.add_argument('--interval', type=str, default='5', help='캔들 간격 (분)')
    parser.add_argument('--limit', type=int, default=1000, help='데이터 개수')
    parser.add_argument('--balance', type=float, default=10000, help='초기 자산')
    parser.add_argument('--min-holding', type=int, default=3, help='최소 보유 스텝 (오버트레이딩 방지)')
    parser.add_argument('--leverage', type=int, default=5, help='기본 레버리지')
    parser.add_argument('--no-prevent-overtrading', action='store_true', help='오버트레이딩 방지 비활성화')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')

    args = parser.parse_args()

    # 환경 설정 출력
    print(f"\n⚙️  백테스트 설정:")
    print(f"   최소 보유: {args.min_holding} 스텝 ({args.min_holding * int(args.interval)}분)")
    print(f"   레버리지: {args.leverage}x")
    print(f"   오버트레이딩 방지: {'OFF' if args.no_prevent_overtrading else 'ON'}")

    results = backtest_recent_data(
        model_path=args.model,
        symbol=args.symbol,
        interval=args.interval,
        limit=args.limit,
        initial_balance=args.balance,
        min_holding_steps=args.min_holding,  # 🔥 파라미터 전달
        base_leverage=args.leverage,  # 🔥 파라미터 전달
        prevent_overtrading=not args.no_prevent_overtrading,  # 🔥 파라미터 전달
        debug=args.debug
    )