# train_rl_improved.py
# -*- coding: utf-8 -*-
"""
강화학습 모델 학습 (알트코인 최적화 버전)
- 코인별 동적 파라미터 적용
- 변동성 기반 SL/TP
- 적응형 레버리지
"""
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import requests
import gym
from gym import spaces
from typing import Tuple, Dict
from enum import IntEnum


###############################################################################
# 환경 정의
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
            stop_loss_multiplier: float = 2.0,  # ATR 배수
            take_profit_multiplier: float = 3.0,  # ATR 배수
            reward_scaling: float = 1e4,
            min_holding_steps: int = 2,  # 더 짧게
            force_initial_position: bool = True,
            use_dynamic_leverage: bool = True,  # 🔥 동적 레버리지
            use_adaptive_sltp: bool = True,  # 🔥 적응형 SL/TP
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
        self.debug = debug

        self._calculate_features()
        self._calculate_market_regime()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, len(self.feature_columns) + 4),  # +1 for volatility
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.reset()

    def _calculate_features(self):
        """기술적 지표 계산 (기존과 동일)"""
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

        # ATR (중요!)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        # 🔥 변동성 지표 추가
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
            'volume_change', 'volatility_norm'  # 🔥 추가
        ]

        self.df = df
        self.feature_mean = self.df[self.feature_columns].mean()
        self.feature_std = self.df[self.feature_columns].std() + 1e-8

    def _calculate_market_regime(self):
        """🔥 시장 상태 분석 (변동성 기반)"""
        self.avg_volatility = self.df['atr_pct'].mean()
        self.volatility_std = self.df['atr_pct'].std()

        # 변동성 구간별 분류
        if self.avg_volatility < 0.01:
            self.market_regime = 'low_vol'
        elif self.avg_volatility < 0.03:
            self.market_regime = 'medium_vol'
        else:
            self.market_regime = 'high_vol'

        if self.debug:
            print(f"📊 시장 상태: {self.market_regime.upper()} (ATR: {self.avg_volatility * 100:.2f}%)")

    def _get_dynamic_leverage(self, current_volatility: float) -> int:
        """🔥 변동성 기반 동적 레버리지"""
        if not self.use_dynamic_leverage:
            return self.base_leverage

        # 변동성이 높을수록 레버리지 낮춤
        volatility_ratio = current_volatility / (self.avg_volatility + 1e-8)

        if volatility_ratio > 1.5:  # 고변동성
            leverage = max(3, self.base_leverage // 3)
        elif volatility_ratio > 1.2:  # 중변동성
            leverage = max(5, self.base_leverage // 2)
        else:  # 저변동성
            leverage = self.base_leverage

        return int(leverage)

    def _get_adaptive_sltp(self, current_atr_pct: float) -> Tuple[float, float]:
        """🔥 ATR 기반 동적 Stop Loss / Take Profit"""
        if not self.use_adaptive_sltp:
            # 기본 고정값
            return 0.05, 0.08

        # ATR의 배수로 설정
        stop_loss = current_atr_pct * self.stop_loss_multiplier
        take_profit = current_atr_pct * self.take_profit_multiplier

        # 최소/최대 제한
        stop_loss = np.clip(stop_loss, 0.02, 0.10)  # 2%~10%
        take_profit = np.clip(take_profit, 0.03, 0.15)  # 3%~15%

        return float(stop_loss), float(take_profit)

    def reset(self) -> np.ndarray:
        """환경 초기화"""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.current_step = self.window_size

        self.current_price = self.df.loc[self.current_step, 'close']
        current_atr = self.df.loc[self.current_step, 'atr_pct']
        current_vol = self.df.loc[self.current_step, 'volatility_norm']

        # 🔥 동적 파라미터 설정
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
            if self.debug:
                print(f"\n🎲 초기 포지션: {initial_direction.upper()} @ {self.current_price:.2f}")
                print(f"   레버리지: {self.leverage}x")
                print(f"   SL: {self.stop_loss_pct * 100:.2f}% / TP: {self.take_profit_pct * 100:.2f}%")

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """현재 상태 관찰 (변동성 정보 추가)"""
        start = self.current_step - self.window_size
        end = self.current_step

        obs_data = self.df[self.feature_columns].iloc[start:end].values
        obs_data = (obs_data - self.feature_mean.values) / self.feature_std.values
        obs_data = np.clip(obs_data, -10, 10)

        position_info = np.zeros((self.window_size, 4))  # +1 for current volatility

        current_vol_norm = self.df.loc[self.current_step, 'volatility_norm']
        position_info[:, 3] = current_vol_norm  # 변동성 정보

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

        # 🔥 동적 파라미터 업데이트 (매 스텝마다)
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
        """행동 실행 (기존과 동일하지만 동적 SL/TP 사용)"""
        reward = 0

        # Stop Loss / Take Profit (동적으로 계산된 값 사용)
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

        # 액션 실행
        if action == Actions.LONG:
            if self.position == 'short':
                reward += self._close_position("Switch to LONG")
            if not self.position:
                self._open_position('long')
                reward += 0.05

        elif action == Actions.SHORT:
            if self.position == 'long':
                reward += self._close_position("Switch to SHORT")
            if not self.position:
                self._open_position('short')
                reward += 0.05

        elif action == Actions.CLOSE:
            if self.position and self.steps_in_position >= self.min_holding_steps:
                reward = self._close_position("Manual Close")
            elif self.position:
                reward -= 0.1

        # 포지션 유지 보상
        if self.position:
            pnl_pct = self._get_position_pnl_pct()
            reward += pnl_pct * 0.2

        return reward * self.reward_scaling

    def _open_position(self, direction: str):
        """포지션 오픈 (동적 레버리지 사용)"""
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

        pnl_pct = self._get_position_pnl_pct()
        pnl = self.entry_balance * pnl_pct * self.leverage

        commission_cost = self.position_size * self.current_price * self.commission
        net_pnl = pnl - commission_cost

        self.balance = self.entry_balance + net_pnl
        self.total_pnl += net_pnl

        if net_pnl > 0:
            self.winning_trades += 1

        self.total_trades += 1

        self.trade_history.append({
            'entry_price': self.entry_price,
            'exit_price': self.current_price,
            'position': self.position,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'holding_time': self.steps_in_position,
            'reason': reason
        })

        if self.debug:
            print(f"   📉 {self.position.upper()} 청산 @ {self.current_price:.2f} "
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
# 콜백
###############################################################################

class DetailedCallback(BaseCallback):
    """상세 학습 로그"""

    def __init__(self, verbose=0):
        super(DetailedCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            self.episode_rewards.append(info.get('equity', 0))
            self.episode_lengths.append(info.get('total_trades', 0))

            if len(self.episode_rewards) % 10 == 0:
                avg_equity = np.mean(self.episode_rewards[-10:])
                avg_trades = np.mean(self.episode_lengths[-10:])
                print(f"   Episodes: {len(self.episode_rewards)} | "
                      f"Avg Equity: ${avg_equity:.2f} | "
                      f"Avg Trades: {avg_trades:.1f}")

        return True


###############################################################################
# 🔥 코인별 최적 설정
###############################################################################

COIN_CONFIGS = {
    'BTCUSDT': {
        'stop_loss_mult': 2.0,
        'take_profit_mult': 3.0,
        'base_leverage': 10,
        'min_holding': 3,
        'learning_rate': 3e-4,
        'ent_coef': 0.05,
    },
    'ETHUSDT': {
        'stop_loss_mult': 2.5,
        'take_profit_mult': 3.5,
        'base_leverage': 8,
        'min_holding': 2,
        'learning_rate': 3e-4,
        'ent_coef': 0.07,
    },
    'SOLUSDT': {
        'stop_loss_mult': 3.0,  # 더 큰 변동성
        'take_profit_mult': 4.0,
        'base_leverage': 5,  # 낮은 레버리지
        'min_holding': 2,
        'learning_rate': 5e-4,  # 더 빠른 학습
        'ent_coef': 0.1,  # 더 많은 탐험
    },
    'DEFAULT': {  # 기타 알트코인
        'stop_loss_mult': 3.0,
        'take_profit_mult': 4.0,
        'base_leverage': 5,
        'min_holding': 2,
        'learning_rate': 5e-4,
        'ent_coef': 0.1,
    }
}


def get_coin_config(symbol: str) -> Dict:
    """코인별 설정 가져오기"""
    return COIN_CONFIGS.get(symbol, COIN_CONFIGS['DEFAULT'])


###############################################################################
# 학습
###############################################################################

def train_rl_model(
        symbol: str = 'BTCUSDT',
        interval: str = '5',
        total_timesteps: int = 200000,
        save_path: str = 'rl_models_improved',
        debug: bool = False
):
    """강화학습 모델 학습 (개선 버전)"""
    print("\n" + "=" * 80)
    print(f"{'🚀 강화학습 모델 학습 (알트코인 최적화 버전!)':^80}")
    print("=" * 80)

    # 🔥 코인별 설정 로드
    config = get_coin_config(symbol)

    print(f"\n설정:")
    print(f"   심볼: {symbol}")
    print(f"   간격: {interval}분")
    print(f"   총 스텝: {total_timesteps:,}")
    print(f"   🔥 동적 레버리지: {config['base_leverage']}x (변동성 기반 조정)")
    print(f"   🔥 적응형 SL/TP: ATR × {config['stop_loss_mult']}/{config['take_profit_mult']}")
    print(f"   🔥 학습률: {config['learning_rate']}")

    df = fetch_bybit_data(symbol, interval, limit=20000)

    if len(df) < 1000:
        print(f"❌ 데이터가 너무 적습니다: {len(df)}개")
        return None, None

    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)

    print(f"학습 데이터: {len(train_df)}개")
    print(f"검증 데이터: {len(test_df)}개\n")

    # 환경 생성
    def make_train_env():
        return AdaptiveCryptoTradingEnv(
            df=train_df,
            window_size=30,
            initial_balance=10000,
            base_leverage=config['base_leverage'],
            commission=0.0006,
            stop_loss_multiplier=config['stop_loss_mult'],
            take_profit_multiplier=config['take_profit_mult'],
            min_holding_steps=config['min_holding'],
            force_initial_position=True,
            use_dynamic_leverage=True,
            use_adaptive_sltp=True,
            debug=False
        )

    def make_eval_env():
        return AdaptiveCryptoTradingEnv(
            df=test_df,
            window_size=30,
            initial_balance=10000,
            base_leverage=config['base_leverage'],
            commission=0.0006,
            stop_loss_multiplier=config['stop_loss_mult'],
            take_profit_multiplier=config['take_profit_mult'],
            min_holding_steps=config['min_holding'],
            force_initial_position=True,
            use_dynamic_leverage=True,
            use_adaptive_sltp=True,
            debug=debug
        )

    vec_env = DummyVecEnv([make_train_env])
    vec_eval_env = DummyVecEnv([make_eval_env])

    print("🧠 PPO 모델 생성 중...")

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
        ent_coef=config['ent_coef'],  # 🔥 코인별 탐험 계수
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cpu",
        tensorboard_log=f"./tensorboard_logs_improved/{symbol}_{interval}min/"
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

    if debug:
        print("\n🐛 디버그 모드 - 거래 로그:\n" + "-" * 80)

    final_eval_env = AdaptiveCryptoTradingEnv(
        df=test_df,
        window_size=30,
        initial_balance=10000,
        base_leverage=config['base_leverage'],
        commission=0.0006,
        stop_loss_multiplier=config['stop_loss_mult'],
        take_profit_multiplier=config['take_profit_mult'],
        min_holding_steps=config['min_holding'],
        force_initial_position=True,
        use_dynamic_leverage=True,
        use_adaptive_sltp=True,
        debug=debug
    )

    obs = final_eval_env.reset()
    done = False

    action_counts = {0: 0, 1: 0, 2: 0}
    leverage_history = []
    sltp_history = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_counts[int(action)] += 1
        obs, reward, done, info = final_eval_env.step(action)

        # 동적 파라미터 추적
        leverage_history.append(info['current_leverage'])
        sltp_history.append((info['current_sl'], info['current_tp']))

    stats = final_eval_env.get_stats()

    print(f"\n검증 데이터 성과:")
    print(f"   최종 자산: ${stats['final_equity']:,.2f}")
    print(f"   총 수익률: {stats['total_return']:+.2f}%")
    print(f"   총 거래: {stats['total_trades']}회 ⭐")
    print(f"   승률: {stats['win_rate']:.1f}%")
    print(f"   최대 낙폭: {stats['max_drawdown']:.2f}%")
    print(f"   시장 상태: {stats['market_regime'].upper()}")

    print(f"\n동적 파라미터 통계:")
    avg_leverage = np.mean(leverage_history)
    avg_sl = np.mean([s[0] for s in sltp_history]) * 100
    avg_tp = np.mean([s[1] for s in sltp_history]) * 100
    print(f"   평균 레버리지: {avg_leverage:.1f}x")
    print(f"   평균 Stop Loss: {avg_sl:.2f}%")
    print(f"   평균 Take Profit: {avg_tp:.2f}%")

    print(f"\n행동 분포:")
    total = sum(action_counts.values())
    print(f"   LONG:  {action_counts[0]} ({action_counts[0] / total * 100:.1f}%)")
    print(f"   SHORT: {action_counts[1]} ({action_counts[1] / total * 100:.1f}%)")
    print(f"   CLOSE: {action_counts[2]} ({action_counts[2] / total * 100:.1f}%)")

    # 상태 평가
    print(f"\n상태 평가:")
    if stats['total_trades'] >= 10:
        print(f"   ✅ 거래가 발생했습니다! ({stats['total_trades']}회)")
        if stats['win_rate'] > 55:
            print(f"   ✅ 우수한 승률입니다!")
        if stats['total_return'] > 10:
            print(f"   ✅ 우수한 수익률입니다!")
    elif stats['total_trades'] >= 5:
        print(f"   ⚠️  거래가 발생했지만 적습니다 ({stats['total_trades']}회)")
    else:
        print(f"   ❌ 거래가 {stats['total_trades']}회만 발생했습니다")

    # 거래 내역
    if final_eval_env.trade_history:
        print(f"\n거래 상세:")
        avg_holding = np.mean([t['holding_time'] for t in final_eval_env.trade_history])
        avg_pnl_pct = np.mean([t['pnl_pct'] * 100 for t in final_eval_env.trade_history])

        print(f"   평균 보유: {avg_holding:.1f} 스텝")
        print(f"   평균 손익: {avg_pnl_pct:+.2f}%")

    # 결과 저장
    results = {
        'symbol': symbol,
        'interval': interval,
        'total_timesteps': total_timesteps,
        'config': config,
        'test_stats': stats,
        'action_distribution': action_counts,
        'avg_leverage': float(avg_leverage),
        'avg_stop_loss': float(avg_sl),
        'avg_take_profit': float(avg_tp),
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

    # Dynamic Leverage
    axes[1, 0].plot(leverage_history, linewidth=1, alpha=0.7)
    axes[1, 0].set_title('Dynamic Leverage Over Time', fontweight='bold')
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('Leverage')
    axes[1, 0].grid(True, alpha=0.3)

    # Dynamic SL/TP
    sl_values = [s[0] * 100 for s in sltp_history]
    tp_values = [s[1] * 100 for s in sltp_history]
    axes[1, 1].plot(sl_values, label='Stop Loss %', linewidth=1, alpha=0.7, color='red')
    axes[1, 1].plot(tp_values, label='Take Profit %', linewidth=1, alpha=0.7, color='green')
    axes[1, 1].set_title('Adaptive SL/TP Over Time', fontweight='bold')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

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
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--save_path', type=str, default='rl_models_improved')
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
        print("\n🎉 성공!")
        print(f"   거래: {results['test_stats']['total_trades']}회")
        print(f"   승률: {results['test_stats']['win_rate']:.1f}%")
        print(f"   수익률: {results['test_stats']['total_return']:+.2f}%")
        print(f"   평균 레버리지: {results['avg_leverage']:.1f}x")