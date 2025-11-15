# paper_trading_rl.py
# -*- coding: utf-8 -*-
"""
강화학습 모델 기반 실시간 페이퍼 트레이딩 시스템
- PPO 모델을 사용한 자동매매 시뮬레이션
- Bybit API로 실시간 가격 수신
- 3가지 액션: LONG, SHORT, CLOSE
- 레버리지 거래 시뮬레이션
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
MODEL_PATH = "btc_best_model.zip"  # 학습된 모델 경로
SYMBOL = "BTCUSDT"
INTERVAL_MINUTES = 5  # 캔들 간격 (분)
SCAN_INTERVAL_SEC = 10  # 스캔 주기 (초)

# 페이퍼 트레이딩 설정
INITIAL_CAPITAL = 1000.0  # 초기 자본 (USDT)
LEVERAGE = 50  # 레버리지 배율
POSITION_SIZE_PCT = 0.50  # 포지션 크기 (잔고의 20%) ⭐ 중요!
COMMISSION = 0.0006  # 수수료 (0.06%)
STOP_LOSS_PCT = 0.05  # 손절 (5%)
TAKE_PROFIT_PCT = 0.08  # 익절 (8%)
LIQUIDATION_BUFFER = 0.8  # 청산 버퍼

TRADE_LOG_FILE = "trades_rl.json"
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


# ===== 데이터 전처리 =====
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 계산"""
    df = df.copy()

    # 기본 지표
    df['returns'] = df['close'].pct_change()
    df['volume_norm'] = np.log1p(df['volume'])

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'] / 100

    # 볼린저 밴드
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # 이동평균
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_ratio_20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['sma_ratio_50'] = (df['close'] - df['sma_50']) / df['sma_50']

    # MACD
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['ema_12_ratio'] = (df['close'] - df['ema_12']) / df['ema_12']
    df['ema_26_ratio'] = (df['close'] - df['ema_26']) / df['ema_26']
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

    # 가격 변화율
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    df['price_change_20'] = df['close'].pct_change(20)
    df['volume_change'] = df['volume'].pct_change()

    df = df.fillna(0).replace([np.inf, -np.inf], 0)

    return df


def get_observation(df: pd.DataFrame, position: Optional[str], entry_price: float,
                    current_price: float) -> np.ndarray:
    """모델 입력용 observation 생성"""
    feature_columns = [
        'returns', 'volume_norm', 'rsi', 'bb_position',
        'sma_ratio_20', 'sma_ratio_50',
        'macd', 'macd_signal', 'macd_hist',
        'atr_pct', 'price_change_5', 'price_change_10', 'price_change_20',
        'volume_change', 'ema_12_ratio', 'ema_26_ratio'
    ]

    # 최근 WINDOW_SIZE개 데이터 추출
    if len(df) < WINDOW_SIZE:
        # 데이터가 부족하면 패딩
        padding = np.zeros((WINDOW_SIZE - len(df), len(feature_columns)))
        obs_data = np.vstack([padding, df[feature_columns].values])
    else:
        obs_data = df[feature_columns].iloc[-WINDOW_SIZE:].values

    # 정규화 (간단한 방법)
    feature_mean = obs_data.mean(axis=0)
    feature_std = obs_data.std(axis=0) + 1e-8
    obs_data = (obs_data - feature_mean) / feature_std
    obs_data = np.clip(obs_data, -10, 10)

    # 포지션 정보 추가
    position_info = np.zeros((WINDOW_SIZE, 3))
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
    direction: str  # "long" or "short"
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    leverage: int
    margin: float
    liquidation_price: float

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
        """청산 여부 판단"""
        # 강제 청산
        if self.direction == "long" and current_price <= self.liquidation_price:
            return True, "Liquidation"
        if self.direction == "short" and current_price >= self.liquidation_price:
            return True, "Liquidation"

        # 손절
        if self.direction == "long" and current_price <= self.stop_loss:
            return True, "Stop Loss"
        if self.direction == "short" and current_price >= self.stop_loss:
            return True, "Stop Loss"

        # 익절
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


class Account:
    """계좌 관리"""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []

    def can_open_position(self) -> bool:
        """포지션 진입 가능 여부"""
        return self.position is None and self.balance > 0

    def open_position(self, symbol: str, direction: str, price: float):
        """포지션 진입"""
        if not self.can_open_position():
            return

        # 포지션 크기 계산 (레버리지 적용)
        position_value = self.balance * POSITION_SIZE_PCT  # 예: $10,000 * 0.20 = $2,000
        margin = position_value  # 실제 사용하는 증거금 = $2,000
        leveraged_value = position_value * LEVERAGE  # 레버리지 적용 = $2,000 * 10 = $20,000
        quantity = leveraged_value / price  # 실제 수량 = $20,000 / 가격

        # 청산가 계산
        if direction == "long":
            liquidation_price = price * (1 - LIQUIDATION_BUFFER / LEVERAGE)
            stop_loss = price * (1 - STOP_LOSS_PCT)
            take_profit = price * (1 + TAKE_PROFIT_PCT)
        else:
            liquidation_price = price * (1 + LIQUIDATION_BUFFER / LEVERAGE)
            stop_loss = price * (1 + STOP_LOSS_PCT)
            take_profit = price * (1 - TAKE_PROFIT_PCT)

        # 잔고에서 증거금 차감
        self.balance -= margin

        self.position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=LEVERAGE,
            margin=margin,
            liquidation_price=liquidation_price
        )

        # 간소화된 진입 메시지 (한 줄)
        print(f"🔥 {direction.upper()} 진입 | 가격: ${price:,.2f} | 증거금: ${margin:,.0f} | {LEVERAGE}x")

    def close_position(self, price: float, reason: str):
        """포지션 청산"""
        if not self.position:
            return

        pos = self.position
        pnl = pos.get_pnl(price)
        roe = pos.get_roe(price)

        # 수수료 차감
        commission_cost = pos.quantity * price * COMMISSION * 2  # 진입 + 청산
        pnl -= commission_cost

        # 잔고 업데이트 (증거금 반환 + 손익)
        self.balance += pos.margin + pnl

        # 거래 기록
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
        print(f"{pnl_emoji} {reason} | 손익: ${pnl:+,.0f} ({roe:+.1f}%) | 잔고: ${self.balance:,.0f}")

        self.position = None

    def get_stats(self) -> Dict:
        """통계 계산"""
        if not self.trades:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl": 0,
                "avg_roe": 0,
                "max_pnl": 0,
                "min_pnl": 0
            }

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.trades) * 100,
            "total_pnl": sum(t.pnl for t in self.trades),
            "avg_pnl": np.mean([t.pnl for t in self.trades]),
            "avg_roe": np.mean([t.roe for t in self.trades]),
            "max_pnl": max(t.pnl for t in self.trades),
            "min_pnl": min(t.pnl for t in self.trades)
        }


# ===== 메인 트레이딩 로직 =====
def clear_screen():
    """터미널 화면 지우기"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_dashboard(account: Account, current_price: float, action: str,
                    action_probs: np.ndarray, loop_count: int = 0, scan_interval: int = 10):
    """대시보드 출력 - 개선된 잔고 표시"""
    clear_screen()

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("\n" + "=" * 100)
    print(f"{'📊 실시간 페이퍼 트레이딩':^100}")
    print(
        f"{'스캔 #' + str(loop_count) + ' | ' + current_time + ' | 다음: ' + str(scan_interval) + '초 후 (Ctrl+C: 종료)':^100}")
    print("=" * 100)

    # 💰 계좌 정보 - 명확한 구분
    cash_balance = account.balance  # 현금 잔고
    margin_used = 0  # 증거금
    position_value = 0  # 포지션 평가액
    position_pnl = 0  # 포지션 손익

    if account.position:
        margin_used = account.position.margin
        position_pnl = account.position.get_pnl(current_price)
        position_value = margin_used + position_pnl

    total_equity = cash_balance + position_value  # 총 자산
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

        emoji = "📈" if pos.direction == "long" else "📉"
        pnl_emoji = "🟢" if pnl > 0 else "🔴"

        print(f"\n📍 보유 포지션")
        print(f"   방향:      {emoji} {pos.direction.upper()}")
        print(f"   진입가:    ${pos.entry_price:,.4f}")
        print(f"   현재가:    ${current_price:,.4f}")
        print(f"   수량:      {pos.quantity:,.4f}")
        print(f"   증거금:    ${pos.margin:,.2f}")
        print(f"   손익:      {pnl_emoji} ${pnl:+,.2f} (ROE: {roe:+.2f}%)")
        print(f"   청산가:    ${pos.liquidation_price:,.4f}")
        print(f"   손절가:    ${pos.stop_loss:,.4f}")
        print(f"   익절가:    ${pos.take_profit:,.4f}")
    else:
        print(f"\n📍 보유 포지션: 없음")

    # 🤖 모델 예측
    print(f"\n🤖 모델 예측")
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

    print("\n" + "=" * 100)


def main():
    """메인 트레이딩 루프"""
    print("\n" + "=" * 100)
    print(f"{'🚀 강화학습 기반 실시간 페이퍼 트레이딩':^100}")
    print("=" * 100)
    print(f"\n설정:")
    print(f"   심볼:       {SYMBOL}")
    print(f"   초기 자본:  ${INITIAL_CAPITAL:,.2f}")
    print(f"   레버리지:   {LEVERAGE}x")
    print(f"   포지션 크기: {POSITION_SIZE_PCT * 100:.0f}% (거래당 ${INITIAL_CAPITAL * POSITION_SIZE_PCT:,.2f})")
    print(f"   모델:       {MODEL_PATH}")
    print(f"   스캔 주기:  {SCAN_INTERVAL_SEC}초")

    # API 초기화
    api = BybitAPI(USE_TESTNET)

    # 모델 로드
    print(f"\n🤖 모델 로드 중...")
    try:
        model = PPO.load(MODEL_PATH)
        print(f"✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 계좌 생성
    account = Account(INITIAL_CAPITAL)

    # 캔들 데이터 캐시
    df_cache = pd.DataFrame()
    last_kline_update = 0

    try:
        loop_count = 0
        while True:
            loop_count += 1
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
                print(f"📊 캔들 데이터 업데이트 중...")
                df = api.get_klines(SYMBOL, str(INTERVAL_MINUTES), limit=200)
                if not df.empty:
                    df_cache = calculate_features(df)
                    last_kline_update = time.time()
                else:
                    print(f"⚠️  캔들 데이터 조회 실패, 캐시 사용")

            if df_cache.empty:
                print(f"❌ 데이터 부족")
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
            if account.position:
                position = account.position.direction
                entry_price = account.position.entry_price

            obs = get_observation(df_cache, position, entry_price, current_price)

            # 모델 예측
            action, _states = model.predict(obs, deterministic=True)
            action = int(action)

            # 액션 확률 계산 (정책 네트워크 출력)
            try:
                import torch
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    distribution = model.policy.get_distribution(obs_tensor)
                    action_probs = distribution.distribution.probs.cpu().numpy()[0]
            except Exception as e:
                action_probs = np.array([0.33, 0.33, 0.34])

            # 대시보드 출력
            print_dashboard(account, current_price, action, action_probs, loop_count, SCAN_INTERVAL_SEC)

            # 액션 실행
            if action == Actions.LONG:
                if account.can_open_position():
                    account.open_position(SYMBOL, "long", current_price)
            elif action == Actions.SHORT:
                if account.can_open_position():
                    account.open_position(SYMBOL, "short", current_price)
            elif action == Actions.CLOSE:
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
        if stats["total_trades"] > 0:
            print("\n" + "=" * 100)
            print(f"{'📊 최종 결과':^100}")
            print("=" * 100)
            final_balance = account.balance
            final_return = (final_balance / account.initial_capital - 1) * 100
            print(f"   최종 잔고:  ${final_balance:,.2f}")
            print(f"   총 수익률:  {final_return:+.2f}%")
            print(f"   총 거래:    {stats['total_trades']}회")
            print(f"   승률:       {stats['win_rate']:.1f}%")
            print(f"   평균 ROE:   {stats['avg_roe']:+.1f}%")
            print("=" * 100)

        print("\n✅ 프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()