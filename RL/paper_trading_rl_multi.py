# paper_trading_rl_multi.py
# -*- coding: utf-8 -*-
"""
강화학습 모델 기반 멀티 심볼 실시간 페이퍼 트레이딩 시스템
- 여러 심볼 동시 거래
- 환경설정 파일(JSON) 기반
- 심볼별 독립적인 모델 및 설정
"""
import os
import time
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
CONFIG_FILE = "trading_config.json"


# ===== 액션 정의 =====
class Actions(IntEnum):
    """강화학습 액션"""
    LONG = 0
    SHORT = 1
    CLOSE = 2


# ===== Bybit API =====
class BybitAPI:
    """Bybit Public API 클라이언트"""

    def __init__(self, testnet: bool = False):
        self.base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"

    def get_ticker(self, symbol: str) -> Dict:
        """현재 가격 조회"""
        url = f"{self.base_url}/v5/market/tickers"
        params = {"category": "linear", "symbol": symbol}
        try:
            response = requests.get(url, params=params, timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """캔들 데이터 조회"""
        url = f"{self.base_url}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if data.get("retCode") != 0 or not data.get("result", {}).get("list"):
                return pd.DataFrame()

            df = pd.DataFrame(data["result"]["list"], columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])

            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
        except Exception as e:
            return pd.DataFrame()


# ===== 데이터 전처리 =====
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 계산"""
    df = df.copy()

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

    return df


def get_observation(df: pd.DataFrame, position: Optional[str], entry_price: float,
                    current_price: float, window_size: int = 30) -> np.ndarray:
    """모델 입력용 observation 생성"""
    feature_columns = [
        'returns', 'volume_norm', 'rsi', 'bb_position',
        'sma_ratio_20', 'sma_ratio_50',
        'macd', 'macd_signal', 'macd_hist',
        'atr_pct', 'price_change_5', 'price_change_10', 'price_change_20',
        'volume_change'
    ]

    if len(df) < window_size:
        padding = np.zeros((window_size - len(df), len(feature_columns)))
        obs_data = np.vstack([padding, df[feature_columns].values])
    else:
        obs_data = df[feature_columns].iloc[-window_size:].values

    feature_mean = obs_data.mean(axis=0)
    feature_std = obs_data.std(axis=0) + 1e-8
    obs_data = (obs_data - feature_mean) / feature_std
    obs_data = np.clip(obs_data, -10, 10)

    position_info = np.zeros((window_size, 3))
    if position == 'long':
        position_info[:, 0] = 1
        position_info[:, 2] = (current_price - entry_price) / (entry_price + 1e-8)
    elif position == 'short':
        position_info[:, 1] = 1
        position_info[:, 2] = (entry_price - current_price) / (entry_price + 1e-8)

    obs = np.concatenate([obs_data, position_info], axis=1)
    return obs.astype(np.float32)


# ===== 포지션 관리 =====
@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    direction: str
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    leverage: int
    margin: float
    liquidation_price: float

    def get_pnl(self, current_price: float) -> float:
        if self.direction == "long":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def get_roe(self, current_price: float) -> float:
        if self.direction == "long":
            price_change_pct = (current_price / self.entry_price - 1) * 100
        else:
            price_change_pct = (1 - current_price / self.entry_price) * 100
        return price_change_pct * self.leverage

    def should_close(self, current_price: float) -> Tuple[bool, str]:
        if self.direction == "long" and current_price <= self.liquidation_price:
            return True, "Liquidation"
        if self.direction == "short" and current_price >= self.liquidation_price:
            return True, "Liquidation"

        if self.direction == "long" and current_price <= self.stop_loss:
            return True, "Stop Loss"
        if self.direction == "short" and current_price >= self.stop_loss:
            return True, "Stop Loss"

        if self.direction == "long" and current_price >= self.take_profit:
            return True, "Take Profit"
        if self.direction == "short" and current_price <= self.take_profit:
            return True, "Take Profit"

        return False, ""


@dataclass
class Trade:
    """완료된 거래 기록"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    roe: float
    reason: str
    leverage: int


class SymbolAccount:
    """심볼별 계좌 관리"""

    def __init__(self, symbol: str, allocated_capital: float, config: Dict):
        self.symbol = symbol
        self.allocated_capital = allocated_capital
        self.balance = allocated_capital
        self.config = config
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []

    def can_open_position(self) -> bool:
        return self.position is None and self.balance > 0

    def open_position(self, direction: str, price: float):
        if not self.can_open_position():
            return

        position_size_pct = self.config.get("position_size_pct", 0.20)
        leverage = self.config.get("leverage", 10)
        stop_loss_pct = self.config.get("stop_loss_pct", 0.05)
        take_profit_pct = self.config.get("take_profit_pct", 0.08)
        liquidation_buffer = self.config.get("liquidation_buffer", 0.8)

        # 포지션 크기 계산 (레버리지 적용)
        position_value = self.balance * position_size_pct  # 증거금으로 사용할 금액
        margin = position_value  # 실제 사용하는 증거금
        leveraged_value = position_value * leverage  # 레버리지 적용 거래 규모
        quantity = leveraged_value / price  # 실제 수량

        if direction == "long":
            liquidation_price = price * (1 - liquidation_buffer / leverage)
            stop_loss = price * (1 - stop_loss_pct)
            take_profit = price * (1 + take_profit_pct)
        else:
            liquidation_price = price * (1 + liquidation_buffer / leverage)
            stop_loss = price * (1 + stop_loss_pct)
            take_profit = price * (1 - take_profit_pct)

        # 잔고에서 증거금 차감 ⭐ 중요!
        self.balance -= margin

        self.position = Position(
            symbol=self.symbol,
            direction=direction,
            entry_price=price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
            margin=margin,
            liquidation_price=liquidation_price
        )

        # 간소화된 진입 메시지 (한 줄)
        print(f"🔥 [{self.symbol}] {direction.upper()} 진입 | 가격: ${price:,.2f} | 증거금: ${margin:,.0f} | {leverage}x")

    def close_position(self, price: float, reason: str):
        if not self.position:
            return

        pos = self.position
        pnl = pos.get_pnl(price)
        roe = pos.get_roe(price)

        commission = pos.quantity * price * 0.0006 * 2
        pnl -= commission

        self.balance += pnl

        trade = Trade(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            entry_time=pos.entry_time,
            exit_time=datetime.now(),
            pnl=pnl,
            roe=roe,
            reason=reason,
            leverage=pos.leverage
        )
        self.trades.append(trade)

        # 간소화된 청산 메시지 (한 줄)
        pnl_emoji = "🟢" if pnl > 0 else "🔴"
        print(f"{pnl_emoji} [{self.symbol}] {reason} | 손익: ${pnl:+,.0f} ({roe:+.1f}%) | 잔고: ${self.balance:,.0f}")

        self.position = None


class MultiSymbolManager:
    """멀티 심볼 매니저"""

    def __init__(self, config: Dict):
        self.config = config
        self.api = BybitAPI(config['trading']['use_testnet'])

        # 총 자본 계산
        total_capital = config['trading']['initial_capital']
        enabled_symbols = [s for s in config['symbols'] if s['enabled']]

        if not enabled_symbols:
            raise ValueError("활성화된 심볼이 없습니다!")

        # 각 심볼에 동일하게 자본 분배
        capital_per_symbol = total_capital / len(enabled_symbols)

        # 심볼별 계좌 생성
        self.accounts: Dict[str, SymbolAccount] = {}
        self.models: Dict[str, PPO] = {}
        self.df_cache: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, float] = {}

        for sym_config in enabled_symbols:
            symbol = sym_config['symbol']

            # 계좌 생성
            self.accounts[symbol] = SymbolAccount(
                symbol=symbol,
                allocated_capital=capital_per_symbol,
                config=sym_config
            )

            # 모델 로드
            try:
                model_path = sym_config['model_path']
                self.models[symbol] = PPO.load(model_path)
                print(f"✅ [{symbol}] 모델 로드: {model_path}")
            except Exception as e:
                print(f"❌ [{symbol}] 모델 로드 실패: {e}")
                raise

            self.df_cache[symbol] = pd.DataFrame()
            self.last_update[symbol] = 0

    def get_total_equity(self, prices: Dict[str, float]) -> float:
        """전체 평가액 계산"""
        total = 0
        for symbol, account in self.accounts.items():
            total += account.balance
            if account.position:
                current_price = prices.get(symbol, account.position.entry_price)
                total += account.position.get_pnl(current_price)
        return total

    def get_total_trades(self) -> List[Trade]:
        """전체 거래 내역"""
        all_trades = []
        for account in self.accounts.values():
            all_trades.extend(account.trades)
        return sorted(all_trades, key=lambda t: t.exit_time)


def clear_screen():
    """터미널 화면 지우기"""
    os.system('cls' if os.name == 'nt' else 'clear')


# ===== 대시보드 =====
def print_dashboard(manager: MultiSymbolManager, prices: Dict[str, float],
                    predictions: Dict[str, Tuple[int, np.ndarray]], loop_count: int = 0):
    """멀티 심볼 대시보드"""
    clear_screen()  # 화면 지우기

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    scan_interval = manager.config['trading']['scan_interval_sec']

    print("\n" + "=" * 120)
    print(f"{'📊 멀티 심볼 실시간 페이퍼 트레이딩':^120}")
    print(
        f"{'스캔 #' + str(loop_count) + ' | ' + current_time + ' | 다음 스캔: ' + str(scan_interval) + '초 후 (Ctrl+C: 종료)':^120}")
    print("=" * 120)

    # 전체 계좌 정보 - 개선된 표시
    initial_capital = manager.config['trading']['initial_capital']

    # 각 계좌의 잔고와 포지션 가치 합산
    total_cash = 0
    total_margin = 0
    total_position_pnl = 0

    for symbol, account in manager.accounts.items():
        total_cash += account.balance
        if account.position:
            total_margin += account.position.margin
            price = prices.get(symbol, 0)
            total_position_pnl += account.position.get_pnl(price)

    total_position_value = total_margin + total_position_pnl
    total_equity = total_cash + total_position_value
    profit_pct = (total_equity / initial_capital - 1) * 100
    profit_emoji = "📈" if profit_pct >= 0 else "📉"

    print(f"\n💰 전체 계좌")
    print(f"   현금 잔고:       ${total_cash:>10,.2f}")
    if total_margin > 0:
        print(f"   포지션 증거금:   ${total_margin:>10,.2f}")
        pnl_color = "🟢" if total_position_pnl >= 0 else "🔴"
        print(f"   포지션 손익:     {pnl_color} ${total_position_pnl:>+10,.2f}")
        print(f"   포지션 평가액:   ${total_position_value:>10,.2f}")
    print(f"   " + "-" * 40)
    print(f"   총 자산:         ${total_equity:>10,.2f}  {profit_emoji} ({profit_pct:+.2f}%)")

    # 심볼별 상태
    print(f"\n📊 심볼별 상태")
    print(f"{'심볼':^12} | {'가격':^12} | {'포지션':^8} | {'손익':^14} | {'ROE':^8} | {'액션':^8} | {'확률':^20}")
    print("-" * 120)

    for symbol in manager.accounts.keys():
        account = manager.accounts[symbol]
        price = prices.get(symbol, 0)
        action, action_probs = predictions.get(symbol, (None, np.array([0.33, 0.33, 0.34])))

        if account.position:
            pos = account.position
            pnl = pos.get_pnl(price)
            roe = pos.get_roe(price)
            pos_emoji = "📈" if pos.direction == "long" else "📉"
            pnl_emoji = "🟢" if pnl > 0 else "🔴"
            pos_str = f"{pos_emoji}  {pos.direction.upper()}"
            pnl_str = f"{pnl_emoji}  ${pnl:+,.2f}"
            roe_str = f"{roe:+.1f}%"
        else:
            pos_str = "⚪ NONE"
            pnl_str = "-"
            roe_str = "-"

        if action is not None:
            action_names = ["LONG", "SHORT", "CLOSE"]
            action_str = action_names[action]
            prob_str = f"L:{action_probs[0] * 100:.0f}% S:{action_probs[1] * 100:.0f}% C:{action_probs[2] * 100:.0f}%"
        else:
            action_str = "-"
            prob_str = "-"

        print(
            f"{symbol:^12} | ${price:>10,.2f} | {pos_str:^8} | {pnl_str:^14} | {roe_str:^8} | {action_str:^8} | {prob_str:^20}")

    # 거래 통계
    all_trades = manager.get_total_trades()
    if all_trades:
        wins = [t for t in all_trades if t.pnl > 0]
        win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0
        total_pnl = sum(t.pnl for t in all_trades)
        avg_roe = np.mean([t.roe for t in all_trades])

        print(f"\n📈 거래 통계")
        print(f"   총 거래:  {len(all_trades):>3}회 | 승률: {win_rate:>5.1f}% ({len(wins)}승 {len(all_trades) - len(wins)}패)")
        print(f"   총 손익:  ${total_pnl:>+12,.2f} | 평균 ROE: {avg_roe:>+6.1f}%")

        # 최근 5개 거래 내역
        if len(all_trades) > 0:
            print(f"\n📋 최근 거래 (최대 5개)")
            recent_trades = all_trades[-5:]  # 최근 5개
            for t in reversed(recent_trades):
                pnl_emoji = "🟢" if t.pnl > 0 else "🔴"
                time_str = t.exit_time.strftime("%H:%M:%S")
                print(
                    f"   {time_str} | {t.symbol:10s} | {t.direction.upper():5s} | {pnl_emoji} ${t.pnl:>+8,.0f} ({t.roe:>+6.1f}%) | {t.reason}")

    print("\n" + "=" * 120)


# ===== 메인 루프 =====
def main():
    """메인 트레이딩 루프"""
    # 설정 로드
    print(f"\n{'=' * 120}")
    print(f"{'🚀 멀티 심볼 강화학습 페이퍼 트레이딩':^120}")
    print(f"{'=' * 120}")

    if not os.path.exists(CONFIG_FILE):
        print(f"❌ 설정 파일이 없습니다: {CONFIG_FILE}")
        print(f"   trading_config.json 파일을 생성하세요.")
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)

    enabled_symbols = [s['symbol'] for s in config['symbols'] if s['enabled']]
    print(f"\n설정:")
    print(f"   초기 자본:  ${config['trading']['initial_capital']:,.2f}")
    print(f"   활성 심볼:  {len(enabled_symbols)}개 - {', '.join(enabled_symbols)}")
    print(f"   스캔 주기:  {config['trading']['scan_interval_sec']}초")

    # 매니저 생성
    print(f"\n🤖 모델 로드 중...")
    try:
        manager = MultiSymbolManager(config)
        print(f"✅ 모든 모델 로드 완료")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return

    scan_interval = config['trading']['scan_interval_sec']

    try:
        loop_count = 0
        while True:
            loop_count += 1
            current_time = datetime.now()

            # 가격 조회
            prices = {}
            for symbol in manager.accounts.keys():
                ticker = manager.api.get_ticker(symbol)
                if ticker.get("retCode") == 0 and ticker.get("result", {}).get("list"):
                    prices[symbol] = float(ticker["result"]["list"][0]["lastPrice"])
                else:
                    prices[symbol] = prices.get(symbol, 0)

            # 캔들 데이터 업데이트
            for symbol in manager.accounts.keys():
                if time.time() - manager.last_update[symbol] > 60 or manager.df_cache[symbol].empty:
                    sym_config = next(s for s in config['symbols'] if s['symbol'] == symbol)
                    df = manager.api.get_klines(symbol, str(sym_config['interval_minutes']), limit=200)
                    if not df.empty:
                        manager.df_cache[symbol] = calculate_features(df)
                        manager.last_update[symbol] = time.time()

            # 포지션 청산 체크
            for symbol, account in manager.accounts.items():
                if account.position:
                    current_price = prices.get(symbol, account.position.entry_price)
                    should_close, reason = account.position.should_close(current_price)
                    if should_close:
                        account.close_position(current_price, reason)

            # 모델 예측 및 거래
            predictions = {}
            for symbol, account in manager.accounts.items():
                if manager.df_cache[symbol].empty:
                    continue

                current_price = prices.get(symbol, 0)
                position = None
                entry_price = 0
                if account.position:
                    position = account.position.direction
                    entry_price = account.position.entry_price

                sym_config = next(s for s in config['symbols'] if s['symbol'] == symbol)
                window_size = config['technical_indicators']['window_size']

                obs = get_observation(manager.df_cache[symbol], position, entry_price,
                                      current_price, window_size)

                model = manager.models[symbol]
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

                try:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    with torch.no_grad():
                        distribution = model.policy.get_distribution(obs_tensor)
                        action_probs = distribution.distribution.probs.cpu().numpy()[0]
                except:
                    action_probs = np.array([0.33, 0.33, 0.34])

                predictions[symbol] = (action, action_probs)

                # 액션 실행
                if action == Actions.LONG and account.can_open_position():
                    account.open_position("long", current_price)
                elif action == Actions.SHORT and account.can_open_position():
                    account.open_position("short", current_price)
                elif action == Actions.CLOSE and account.position:
                    account.close_position(current_price, "RL Model Signal")

            # 대시보드 출력 (loop_count 전달)
            print_dashboard(manager, prices, predictions, loop_count)

            time.sleep(scan_interval)

    except KeyboardInterrupt:
        print("\n\n프로그램 종료 중...")

        # 미결제 포지션 청산
        for symbol, account in manager.accounts.items():
            if account.position:
                ticker = manager.api.get_ticker(symbol)
                if ticker.get("retCode") == 0 and ticker.get("result", {}).get("list"):
                    current_price = float(ticker["result"]["list"][0]["lastPrice"])
                    account.close_position(current_price, "Manual Close")

        # 거래 내역 저장
        all_trades = manager.get_total_trades()
        if all_trades:
            data = []
            for trade in all_trades:
                trade_dict = asdict(trade)
                trade_dict['entry_time'] = trade_dict['entry_time'].isoformat()
                trade_dict['exit_time'] = trade_dict['exit_time'].isoformat()
                data.append(trade_dict)

            trade_log_file = config['trading']['trade_log_file']
            with open(trade_log_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n💾 거래 내역 저장: {trade_log_file}")

        # 최종 통계
        if all_trades:
            print("\n" + "=" * 120)
            print(f"{'📊 최종 결과':^120}")
            print("=" * 120)

            initial_capital = config['trading']['initial_capital']
            total_equity = manager.get_total_equity(prices)
            final_return = (total_equity / initial_capital - 1) * 100

            wins = [t for t in all_trades if t.pnl > 0]
            win_rate = len(wins) / len(all_trades) * 100
            total_pnl = sum(t.pnl for t in all_trades)
            avg_roe = np.mean([t.roe for t in all_trades])

            print(f"   최종 평가액: ${total_equity:,.2f}")
            print(f"   총 수익률:   {final_return:+.2f}%")
            print(f"   총 거래:     {len(all_trades)}회")
            print(f"   승률:        {win_rate:.1f}%")
            print(f"   총 손익:     ${total_pnl:+,.2f}")
            print(f"   평균 ROE:    {avg_roe:+.1f}%")

            # 심볼별 통계
            print(f"\n   심볼별 성과:")
            for symbol, account in manager.accounts.items():
                if account.trades:
                    sym_pnl = sum(t.pnl for t in account.trades)
                    sym_wins = [t for t in account.trades if t.pnl > 0]
                    sym_win_rate = len(sym_wins) / len(account.trades) * 100
                    print(
                        f"      {symbol:10s}: {len(account.trades):3}회 | 승률: {sym_win_rate:5.1f}% | 손익: ${sym_pnl:+,.2f}")

            print("=" * 120)

        print("\n✅ 프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()