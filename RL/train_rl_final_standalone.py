# train_rl_final_standalone.py
# -*- coding: utf-8 -*-
"""
강화학습 모델 학습 (최종 독립 버전)
- 환경을 직접 포함
- 거래 보장
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
# 환경 정의 (직접 포함)
###############################################################################

class Actions(IntEnum):
    """3가지 액션만"""
    LONG = 0  # 롱 포지션
    SHORT = 1  # 숏 포지션
    CLOSE = 2  # 청산


class CryptoTradingEnv(gym.Env):
    """암호화폐 거래 환경 - 최종 버전"""

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

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, len(self.feature_columns) + 3),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 🔥 3개만!

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

        # 🔥 초기 랜덤 포지션!
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
        """행동 실행"""
        reward = 0

        # Stop Loss / Take Profit
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

        # 액션 처리
        if action == Actions.LONG:
            if self.position == 'long':
                pnl = self._get_position_pnl_pct() * self.leverage
                reward = pnl * 0.1
            elif self.position == 'short':
                if self.steps_in_position >= self.min_holding_steps:
                    reward = self._close_position("Switch")
                    reward += self._open_position('long')
                    if self.debug:
                        print(f"Step {self.current_step}: 🔄 SHORT → LONG")
                else:
                    pnl = self._get_position_pnl_pct() * self.leverage
                    reward = pnl * 0.1 - 0.01
            else:
                reward = self._open_position('long')
                if self.debug:
                    print(f"Step {self.current_step}: 📈 OPEN LONG @ {self.current_price:.2f}")

        elif action == Actions.SHORT:
            if self.position == 'short':
                pnl = self._get_position_pnl_pct() * self.leverage
                reward = pnl * 0.1
            elif self.position == 'long':
                if self.steps_in_position >= self.min_holding_steps:
                    reward = self._close_position("Switch")
                    reward += self._open_position('short')
                    if self.debug:
                        print(f"Step {self.current_step}: 🔄 LONG → SHORT")
                else:
                    pnl = self._get_position_pnl_pct() * self.leverage
                    reward = pnl * 0.1 - 0.01
            else:
                reward = self._open_position('short')
                if self.debug:
                    print(f"Step {self.current_step}: 📉 OPEN SHORT @ {self.current_price:.2f}")

        elif action == Actions.CLOSE:
            if self.position and self.steps_in_position >= self.min_holding_steps:
                pos_type = self.position.upper()
                reward = self._close_position("Manual")
                if self.debug:
                    print(f"Step {self.current_step}: ✂️ CLOSE {pos_type}")
            elif self.position:
                pnl = self._get_position_pnl_pct() * self.leverage
                reward = pnl * 0.1 - 0.01
            else:
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


###############################################################################
# 학습 코드
###############################################################################

class DetailedCallback(BaseCallback):
    """학습 중 상세 메트릭 로깅"""

    def __init__(self, verbose=0):
        super(DetailedCallback, self).__init__(verbose)
        self.episode_trades = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        if 'equity' in self.locals['infos'][0]:
            info = self.locals['infos'][0]

            self.logger.record('train/equity', info['equity'])
            self.logger.record('train/steps_in_position', info.get('steps_in_position', 0))

            if info['total_trades'] > 0:
                self.logger.record('train/win_rate', info['win_rate'])
                self.logger.record('train/total_trades', info['total_trades'])

        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            if 'total_trades' in self.locals['infos'][0]:
                trades = self.locals['infos'][0]['total_trades']
                self.episode_trades.append(trades)
                self.logger.record('episode/total_trades', trades)

                if len(self.episode_trades) >= 10:
                    avg_trades = np.mean(self.episode_trades[-10:])
                    self.logger.record('episode/avg_trades_last_10', avg_trades)

        return True


def fetch_bybit_data(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """바이비트에서 데이터 가져오기"""
    url = "https://api.bybit.com/v5/market/kline"

    all_data = []
    end_time = None

    print(f"📥 {symbol} {interval}분봉 데이터 다운로드 중...")

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

        if data.get("retCode") != 0:
            print(f"❌ 오류: {data.get('retMsg')}")
            break

        candles = data["result"]["list"]
        if not candles:
            break

        all_data.extend(candles)
        end_time = int(candles[-1][0]) - 1

        if len(all_data) % 1000 == 0:
            print(f"   다운로드: {len(all_data)}/{limit}")

        if len(candles) < 200:
            break

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"✅ 다운로드 완료: {len(df)}개 캔들\n")

    return df


def train_rl_model(
        symbol: str = "BTCUSDT",
        interval: str = "5",
        total_timesteps: int = 200000,
        save_path: str = "rl_models_standalone",
        debug: bool = False
):
    """강화학습 모델 학습"""
    print("\n" + "=" * 80)
    print(f"{'🚀 강화학습 모델 학습 (독립 버전 - 거래 보장!)':^80}")
    print("=" * 80)
    print(f"\n설정:")
    print(f"   심볼: {symbol}")
    print(f"   간격: {interval}분")
    print(f"   총 스텝: {total_timesteps:,}")
    print(f"   🔥 액션: 3가지 (LONG=0, SHORT=1, CLOSE=2)")
    print(f"   🔥 초기 포지션: 랜덤 강제 진입")

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
        return CryptoTradingEnv(
            df=train_df,
            window_size=30,
            initial_balance=10000,
            leverage=10,
            commission=0.0006,
            stop_loss_pct=0.05,
            take_profit_pct=0.08,
            min_holding_steps=3,
            force_initial_position=True,
            debug=False
        )

    def make_eval_env():
        return CryptoTradingEnv(
            df=test_df,
            window_size=30,
            initial_balance=10000,
            leverage=10,
            commission=0.0006,
            stop_loss_pct=0.05,
            take_profit_pct=0.08,
            min_holding_steps=3,
            force_initial_position=True,
            debug=debug
        )

    vec_env = DummyVecEnv([make_train_env])
    vec_eval_env = DummyVecEnv([make_eval_env])

    print("🧠 PPO 모델 생성 중...")

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cpu",
        tensorboard_log=f"./tensorboard_logs_standalone/{symbol}_{interval}min/"
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

    # 🔥 새로운 평가 환경 생성 (확실하게!)
    final_eval_env = CryptoTradingEnv(
        df=test_df,
        window_size=30,
        initial_balance=10000,
        leverage=10,
        commission=0.0006,
        stop_loss_pct=0.05,
        take_profit_pct=0.08,
        min_holding_steps=3,
        force_initial_position=True,
        debug=debug
    )

    obs = final_eval_env.reset()
    done = False

    action_counts = {0: 0, 1: 0, 2: 0}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_counts[int(action)] += 1  # 🔥 int로 변환!
        obs, reward, done, info = final_eval_env.step(action)

    stats = final_eval_env.get_stats()

    print(f"\n검증 데이터 성과:")
    print(f"   최종 자산: ${stats['final_equity']:,.2f}")
    print(f"   총 수익률: {stats['total_return']:+.2f}%")
    print(f"   총 거래: {stats['total_trades']}회 ⭐")
    print(f"   승률: {stats['win_rate']:.1f}%")
    print(f"   최대 낙폭: {stats['max_drawdown']:.2f}%")

    print(f"\n행동 분포 (3가지):")
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
        'test_stats': stats,
        'action_distribution': action_counts,
        'model_path': model_path,
        'trained_at': datetime.now().isoformat()
    }

    with open(f"{save_path}/{symbol}_{interval}min_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # 차트
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Equity
    ax1.plot(final_eval_env.equity_history, linewidth=2)
    ax1.axhline(y=10000, color='r', linestyle='--', alpha=0.5)
    ax1.set_title('Equity Curve', fontweight='bold')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True, alpha=0.3)

    # Actions
    actions = ['LONG', 'SHORT', 'CLOSE']
    colors = ['green', 'red', 'blue']
    ax2.bar(actions, [action_counts[i] for i in range(3)], color=colors, alpha=0.7)
    ax2.set_title('Action Distribution', fontweight='bold')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3, axis='y')

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
    parser.add_argument('--save_path', type=str, default='rl_models_standalone')
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
