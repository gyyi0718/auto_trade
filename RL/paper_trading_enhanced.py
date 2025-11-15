# paper_trading_rl_enhanced.py
# -*- coding: utf-8 -*-
"""
강화학습 모델 기반 실시간 페이퍼 트레이딩 시스템 (Enhanced 버전)
- train_rl_enhanced.py로 학습한 모델 사용
- Bybit USDT Perpetual 실시간 데이터
- 3가지 액션: LONG, SHORT, CLOSE
- 24개 향상된 features 지원
"""
import os
import time
import json
import warnings
from datetime import datetime, timedelta
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
MODEL_PATH = "rl_models_enhanced/BTCUSDT_5min_final"  # 학습된 모델 경로
SYMBOL = "BTCUSDT"
INTERVAL_MINUTES = 5  # 캔들 간격 (분)
SCAN_INTERVAL_SEC = 10  # 스캔 주기 (초)

# 페이퍼 트레이딩 설정
INITIAL_CAPITAL = 1000.0  # 초기 자본 (USDT)
BASE_LEVERAGE = 10  # 기본 레버리지 배율
POSITION_SIZE_PCT = 0.20  # 포지션 크기 (잔고의 20%)
COMMISSION = 0.0006  # 수수료 (0.06%)
STOP_LOSS_PCT = 0.05  # 손절 (5%)
TAKE_PROFIT_PCT = 0.08  # 익절 (8%)
LIQUIDATION_BUFFER = 0.8  # 청산 버퍼

TRADE_LOG_FILE = "trades_rl_enhanced.json"
USE_TESTNET = False

# 기술적 지표 계산용 상수
WINDOW_SIZE = 30  # 모델 입력 윈도우 크기


# ===== 액션 정의 =====
class Actions(IntEnum):
    """강화학습 액션"""
    LONG = 0  # 롱 포지션
    SHORT = 1  # 숏 포지션
    CLOSE = 2  # 청산


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
            print(f"❌ Klines 조회 오류: {e}")
            return pd.DataFrame()


# ===== 데이터 전처리 (train_rl_enhanced.py와 동일) =====
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    기술적 지표 계산 - train_rl_enhanced.py와 동일한 24개 features
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


def get_observation(df: pd.DataFrame, position: Optional[str], entry_price: float,
                    current_price: float, steps_in_position: int = 0,
                    equity_ratio: float = 1.0, max_drawdown: float = 0.0) -> np.ndarray:
    """
    모델 입력용 observation 생성 - train_rl_enhanced.py와 동일
    """
    # 24개 features (train_rl_enhanced.py와 동일)
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

    # 최근 WINDOW_SIZE개 데이터 추출
    if len(df) < WINDOW_SIZE:
        # 데이터가 부족하면 패딩
        padding = np.zeros((WINDOW_SIZE - len(df), len(feature_columns)))
        obs_data = np.vstack([padding, df[feature_columns].values])
    else:
        obs_data = df[feature_columns].iloc[-WINDOW_SIZE:].values

    # 정규화
    feature_mean = obs_data.mean(axis=0)
    feature_std = obs_data.std(axis=0) + 1e-8
    obs_data = (obs_data - feature_mean) / feature_std
    obs_data = np.clip(obs_data, -10, 10)

    # 포지션 정보 추가 (6개 채널 - train_rl_enhanced.py와 동일)
    position_info = np.zeros((WINDOW_SIZE, 6))

    if position == 'long':
        position_info[:, 0] = 1  # long indicator
        position_info[:, 2] = (current_price - entry_price) / (entry_price + 1e-8)  # unrealized PnL
        position_info[:, 4] = steps_in_position / 100.0  # 정규화된 보유 기간
    elif position == 'short':
        position_info[:, 1] = 1  # short indicator
        position_info[:, 2] = (entry_price - current_price) / (entry_price + 1e-8)
        position_info[:, 4] = steps_in_position / 100.0

    position_info[:, 3] = equity_ratio  # 자산 비율
    position_info[:, 5] = max_drawdown  # 최대 낙폭

    obs = np.concatenate([obs_data, position_info], axis=1)
    return obs.astype(np.float32)


# ===== 포지션 관리 =====
@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    leverage: int
    margin: float
    liquidation_price: float
    entry_step: int = 0  # 진입 스텝 추가

    def get_pnl(self, current_price: float) -> float:
        """손익 계산"""
        if self.direction == "long":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def get_roe(self, current_price: float) -> float:
        """ROE 계산 (레버리지 반영)"""
        if self.direction == "long":
            price_change_pct = (current_price / self.entry_price - 1) * 100
        else:
            price_change_pct = (1 - current_price / self.entry_price) * 100
        return price_change_pct * self.leverage

    def should_close(self, current_price: float) -> Tuple[bool, str]:
        """포지션 청산 여부 체크"""
        roe = self.get_roe(current_price)

        # 청산가 체크
        if self.direction == "long":
            if current_price <= self.liquidation_price:
                return True, "Liquidation"
        else:
            if current_price >= self.liquidation_price:
                return True, "Liquidation"

        # Stop Loss
        if self.direction == "long":
            if current_price <= self.stop_loss:
                return True, f"Stop Loss (ROE: {roe:.2f}%)"
        else:
            if current_price >= self.stop_loss:
                return True, f"Stop Loss (ROE: {roe:.2f}%)"

        # Take Profit
        if self.direction == "long":
            if current_price >= self.take_profit:
                return True, f"Take Profit (ROE: {roe:.2f}%)"
        else:
            if current_price <= self.take_profit:
                return True, f"Take Profit (ROE: {roe:.2f}%)"

        return False, ""


@dataclass
class Trade:
    """거래 기록"""
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


class Account:
    """계좌 관리"""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.max_equity = initial_capital
        self.max_drawdown = 0.0
        self.current_step = 0  # 현재 스텝 추가

    def can_open_position(self) -> bool:
        """포지션 오픈 가능 여부"""
        return self.position is None and self.balance > 0

    def open_position(self, symbol: str, direction: str, current_price: float):
        """포지션 오픈"""
        if not self.can_open_position():
            print(f"⚠️  포지션 오픈 불가 (이미 보유 중)")
            return

        # 포지션 크기 계산
        margin = self.balance * POSITION_SIZE_PCT
        quantity = (margin * BASE_LEVERAGE) / current_price

        # 수수료 차감
        commission = quantity * current_price * COMMISSION
        self.balance -= commission

        # 증거금 차감 (중요!)
        self.balance -= margin

        # Stop Loss / Take Profit 계산
        if direction == "long":
            stop_loss = current_price * (1 - STOP_LOSS_PCT)
            take_profit = current_price * (1 + TAKE_PROFIT_PCT)
            liquidation_price = current_price * (1 - (1 / BASE_LEVERAGE) * LIQUIDATION_BUFFER)
        else:
            stop_loss = current_price * (1 + STOP_LOSS_PCT)
            take_profit = current_price * (1 - TAKE_PROFIT_PCT)
            liquidation_price = current_price * (1 + (1 / BASE_LEVERAGE) * LIQUIDATION_BUFFER)

        self.position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=BASE_LEVERAGE,
            margin=margin,
            liquidation_price=liquidation_price,
            entry_step=self.current_step
        )

        print(f"\n{'=' * 80}")
        emoji = "📈" if direction == "long" else "📉"
        print(f"{emoji} 포지션 오픈: {direction.upper()}")
        print(f"   진입가:    ${current_price:,.4f}")
        print(f"   수량:      {quantity:,.4f}")
        print(f"   증거금:    ${margin:,.2f}")
        print(f"   레버리지:  {BASE_LEVERAGE}x")
        print(f"   손절가:    ${stop_loss:,.4f}")
        print(f"   익절가:    ${take_profit:,.4f}")
        print(f"   청산가:    ${liquidation_price:,.4f}")
        print(f"{'=' * 80}\n")

    def close_position(self, current_price: float, reason: str = "Manual"):
        """포지션 청산"""
        if not self.position:
            return

        pos = self.position
        pnl = pos.get_pnl(current_price)
        roe = pos.get_roe(current_price)

        # 수수료 차감
        commission = pos.quantity * current_price * COMMISSION
        net_pnl = pnl - commission

        self.balance += pos.margin + net_pnl

        # 거래 기록
        trade = Trade(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=current_price,
            quantity=pos.quantity,
            entry_time=pos.entry_time,
            exit_time=datetime.now(),
            pnl=net_pnl,
            roe=roe,
            reason=reason,
            leverage=pos.leverage
        )
        self.trades.append(trade)

        # 통계 업데이트
        current_equity = self.balance
        self.max_equity = max(self.max_equity, current_equity)
        drawdown = (self.max_equity - current_equity) / self.max_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)

        print(f"\n{'=' * 80}")
        emoji = "🟢" if net_pnl > 0 else "🔴"
        print(f"{emoji} 포지션 청산: {pos.direction.upper()}")
        print(f"   진입가:    ${pos.entry_price:,.4f}")
        print(f"   청산가:    ${current_price:,.4f}")
        print(f"   손익:      ${net_pnl:+,.2f}")
        print(f"   ROE:       {roe:+.2f}%")
        print(f"   사유:      {reason}")
        print(f"   잔고:      ${self.balance:,.2f}")
        print(f"{'=' * 80}\n")

        self.position = None

    def get_stats(self) -> Dict:
        """거래 통계"""
        if not self.trades:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "avg_roe": 0,
                "max_pnl": 0,
                "min_pnl": 0
            }

        wins = sum(1 for t in self.trades if t.pnl > 0)
        losses = len(self.trades) - wins
        avg_pnl = np.mean([t.pnl for t in self.trades])
        avg_roe = np.mean([t.roe for t in self.trades])
        max_pnl = max([t.pnl for t in self.trades])
        min_pnl = min([t.pnl for t in self.trades])

        return {
            "total_trades": len(self.trades),
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / len(self.trades)) * 100,
            "avg_pnl": avg_pnl,
            "avg_roe": avg_roe,
            "max_pnl": max_pnl,
            "min_pnl": min_pnl
        }


# ===== UI =====
def clear_screen():
    """화면 지우기"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_dashboard(account: Account, current_price: float, action: int,
                    action_probs: np.ndarray, loop_count: int = 0, scan_interval: int = 10):
    """대시보드 출력"""
    clear_screen()

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("\n" + "=" * 100)
    print(f"{'📊 실시간 페이퍼 트레이딩 (Enhanced RL)':^100}")
    print(
        f"{'스캔 #' + str(loop_count) + ' | ' + current_time + ' | 다음: ' + str(scan_interval) + '초 후 (Ctrl+C: 종료)':^100}")
    print("=" * 100)

    global cash_balance
    global position_value
    # 💰 계좌 정보
    cash_balance = account.balance
    margin_used = 0
    position_value = 0
    position_pnl = 0

    if account.position:
        margin_used = account.position.margin
        position_pnl = account.position.get_pnl(current_price)
        position_value = margin_used + position_pnl

    total_equity = cash_balance + position_value
    profit_pct = (total_equity / account.initial_capital - 1) * 100
    profit_emoji = "📈" if profit_pct >= 0 else "📉"

    print(f"\n💰 계좌 정보")
    print(f"   현금 잔고:       ${cash_balance:>10,.2f}")
    if account.position:
        print(f"   포지션 증거금:   ${margin_used:>10,.2f}")
        pnl_color = "🟢" if position_pnl >= 0 else "🔴"
        print(f"   포지션 손익:     {pnl_color} ${position_pnl:>+10,.2f}")
        print(f"   포지션 평가액:   ${position_value:>10,.2f}")
    print(f"   " + "-" * 40)
    print(f"   총 자산:         ${total_equity:>10,.2f}  {profit_emoji} ({profit_pct:+.2f}%)")
    print(f"   현재가:          ${current_price:>10,.4f}")

    # 📍 포지션 정보
    if account.position:
        pos = account.position
        pnl = pos.get_pnl(current_price)
        roe = pos.get_roe(current_price)
        holding_time = (datetime.now() - pos.entry_time).total_seconds() / 60  # 분

        emoji = "📈" if pos.direction == "long" else "📉"
        pnl_emoji = "🟢" if pnl > 0 else "🔴"

        print(f"\n📍 보유 포지션")
        print(f"   방향:      {emoji} {pos.direction.upper()}")
        print(f"   진입가:    ${pos.entry_price:,.4f}")
        print(f"   현재가:    ${current_price:,.4f}")
        print(f"   수량:      {pos.quantity:,.4f}")
        print(f"   증거금:    ${pos.margin:,.2f}")
        print(f"   레버리지:  {pos.leverage}x")
        print(f"   손익:      {pnl_emoji} ${pnl:+,.2f} (ROE: {roe:+.2f}%)")
        print(f"   보유시간:  {holding_time:.1f}분")
        print(f"   청산가:    ${pos.liquidation_price:,.4f}")
        print(f"   손절가:    ${pos.stop_loss:,.4f}")
        print(f"   익절가:    ${pos.take_profit:,.4f}")
    else:
        print(f"\n📍 보유 포지션: 없음")

    # 🤖 모델 예측
    print(f"\n🤖 모델 예측 (Enhanced RL)")
    action_names = ["LONG", "SHORT", "CLOSE"]
    print(f"   추천 액션:  {action_names[int(action)]}")
    print(f"   확률 분포:")
    for i, name in enumerate(action_names):
        bar_length = int(action_probs[i] * 30)
        bar = "█" * bar_length
        print(f"      {name:5s}: {bar:30s} {action_probs[i] * 100:5.1f}%")

    # 📊 거래 통계
    stats = account.get_stats()
    if stats["total_trades"] > 0:
        print(f"\n📊 거래 통계")
        print(f"   총 거래:    {stats['total_trades']:>3}회")
        print(f"   승률:       {stats['win_rate']:>6.1f}% ({stats['wins']}승 {stats['losses']}패)")
        print(f"   평균 손익:  ${stats['avg_pnl']:>+12,.2f}")
        print(f"   평균 ROE:   {stats['avg_roe']:>+6.1f}%")
        print(f"   최대 수익:  ${stats['max_pnl']:>12,.2f}")
        print(f"   최대 손실:  ${stats['min_pnl']:>12,.2f}")
        print(f"   Max DD:     {account.max_drawdown * 100:>6.2f}%")

    print("\n" + "=" * 100)


def main():
    """메인 트레이딩 루프"""
    print("\n" + "=" * 100)
    print(f"{'🚀 강화학습 기반 실시간 페이퍼 트레이딩 (Enhanced)':^100}")
    print("=" * 100)
    print(f"\n설정:")
    print(f"   심볼:       {SYMBOL}")
    print(f"   초기 자본:  ${INITIAL_CAPITAL:,.2f}")
    print(f"   레버리지:   {BASE_LEVERAGE}x")
    print(f"   포지션 크기: {POSITION_SIZE_PCT * 100:.0f}% (거래당 ${INITIAL_CAPITAL * POSITION_SIZE_PCT:,.2f})")
    print(f"   모델:       {MODEL_PATH}")
    print(f"   스캔 주기:  {SCAN_INTERVAL_SEC}초")
    print(f"   Features:   24개 (Enhanced)")

    # API 초기화
    api = BybitAPI(USE_TESTNET)

    # 모델 로드
    print(f"\n🤖 모델 로드 중...")
    try:
        model = PPO.load(MODEL_PATH)
        print(f"✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print(f"\n💡 확인 사항:")
        print(f"   1. 모델 경로가 올바른지 확인: {MODEL_PATH}")
        print(f"   2. train_rl_enhanced.py로 학습한 모델인지 확인")
        print(f"   3. 파일이 존재하는지 확인")
        return

    # 계좌 생성
    account = Account(INITIAL_CAPITAL)

    # 캔들 데이터 캐시
    df_cache = pd.DataFrame()
    last_kline_update = 0

    print(f"\n✅ 초기화 완료! {SCAN_INTERVAL_SEC}초 후 트레이딩 시작...\n")
    time.sleep(3)

    try:
        loop_count = 0
        while True:
            loop_count += 1
            account.current_step = loop_count
            current_time = datetime.now()

            # 현재 가격 조회
            ticker = api.get_ticker(SYMBOL)
            if ticker.get("retCode") != 0 or not ticker.get("result", {}).get("list"):
                print(f"❌ 가격 조회 실패")
                time.sleep(SCAN_INTERVAL_SEC)
                continue

            current_price = float(ticker["result"]["list"][0]["lastPrice"])

            # 캔들 데이터 업데이트 (매 분마다 또는 최초)
            if time.time() - last_kline_update > 60 or df_cache.empty:
                df = api.get_klines(SYMBOL, str(INTERVAL_MINUTES), limit=200)
                if not df.empty:
                    df_cache = calculate_features(df)
                    last_kline_update = time.time()

            if df_cache.empty:
                print(f"❌ 데이터 부족, 대기 중...")
                time.sleep(SCAN_INTERVAL_SEC)
                continue

            # 포지션 청산 체크
            if account.position:
                should_close, reason = account.position.should_close(current_price)
                if should_close:
                    account.close_position(current_price, reason)

            # Observation 생성
            position = None
            entry_price = 0
            steps_in_position = 0
            if account.position:
                position = account.position.direction
                entry_price = account.position.entry_price
                steps_in_position = loop_count - account.position.entry_step

            equity_ratio = (
                               cash_balance + position_value if account.position else account.balance) / account.initial_capital

            obs = get_observation(
                df_cache,
                position,
                entry_price,
                current_price,
                steps_in_position,
                equity_ratio,
                account.max_drawdown
            )

            # 모델 예측
            action, _states = model.predict(obs, deterministic=True)
            action = int(action)

            # 액션 확률 계산
            try:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    distribution = model.policy.get_distribution(obs_tensor)
                    action_probs = distribution.distribution.probs.cpu().numpy()[0]
            except Exception as e:
                action_probs = np.array([0.33, 0.33, 0.34])

            # 대시보드 출력
            print_dashboard(account, current_price, action, action_probs, loop_count, SCAN_INTERVAL_SEC)

            # 🎯 추천 액션 실행
            if action == Actions.LONG:
                # SHORT 보유 중이면 먼저 청산
                if account.position and account.position.direction == "short":
                    print(f"\n🔄 방향 전환: SHORT → LONG")
                    account.close_position(current_price, "Direction Change")
                    time.sleep(1)  # 청산 후 잠시 대기

                # LONG 진입
                if account.can_open_position():
                    account.open_position(SYMBOL, "long", current_price)

            elif action == Actions.SHORT:
                # LONG 보유 중이면 먼저 청산
                if account.position and account.position.direction == "long":
                    print(f"\n🔄 방향 전환: LONG → SHORT")
                    account.close_position(current_price, "Direction Change")
                    time.sleep(1)  # 청산 후 잠시 대기

                # SHORT 진입
                if account.can_open_position():
                    account.open_position(SYMBOL, "short", current_price)

            elif action == Actions.CLOSE:
                # 포지션 있으면 청산
                if account.position:
                    account.close_position(current_price, "RL Model Signal")

            time.sleep(SCAN_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n프로그램 종료 중...")

        # 미결제 포지션 청산
        if account.position:
            ticker = api.get_ticker(SYMBOL)
            if ticker.get("retCode") == 0 and ticker.get("result", {}).get("list"):
                current_price = float(ticker["result"]["list"][0]["lastPrice"])
                account.close_position(current_price, "Manual Close")

        # 거래 내역 저장
        if account.trades:
            data = [asdict(t) for t in account.trades]
            # datetime을 문자열로 변환
            for trade in data:
                trade['entry_time'] = trade['entry_time'].isoformat()
                trade['exit_time'] = trade['exit_time'].isoformat()

            with open(TRADE_LOG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n💾 거래 내역 저장: {TRADE_LOG_FILE}")

        # 최종 통계
        stats = account.get_stats()
        print("\n" + "=" * 100)
        print(f"{'📊 최종 결과':^100}")
        print("=" * 100)

        final_balance = account.balance
        if account.position:
            ticker = api.get_ticker(SYMBOL)
            if ticker.get("retCode") == 0 and ticker.get("result", {}).get("list"):
                current_price = float(ticker["result"]["list"][0]["lastPrice"])
                final_balance += account.position.margin + account.position.get_pnl(current_price)

        final_return = (final_balance / account.initial_capital - 1) * 100

        print(f"   초기 자본:  ${account.initial_capital:,.2f}")
        print(f"   최종 잔고:  ${final_balance:,.2f}")
        print(f"   총 수익률:  {final_return:+.2f}%")

        if stats["total_trades"] > 0:
            print(f"   총 거래:    {stats['total_trades']}회")
            print(f"   승률:       {stats['win_rate']:.1f}%")
            print(f"   평균 ROE:   {stats['avg_roe']:+.1f}%")
            print(f"   Max DD:     {account.max_drawdown * 100:.2f}%")

        print("=" * 100)
        print("\n✅ 프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()