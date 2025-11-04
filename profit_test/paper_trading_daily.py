# paper_trading_daily.py
# -*- coding: utf-8 -*-
"""
TCN_Simple 모델 기반 페이퍼 트레이딩 시스템 (일일 데이터용)
- train_tcn_daily.py로 학습한 모델 사용
- 실시간 신호 기반 자동 매매 시뮬레이션
- 레버리지 거래 시뮬레이션
- 포지션 관리 및 손익 계산
"""
import os
import time
import json
import warnings
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import requests
import certifi

warnings.filterwarnings("ignore")
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# ===== CONFIG =====
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,DOGEUSDT").split(",")
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "60"))  # 1분마다 체크
TCN_CKPT = os.getenv("TCN_CKPT", "D:/ygy_work/coin/tcn/models_daily_v2/daily_simple_best.ckpt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.55"))  # 신뢰도 임계값
USE_TESTNET = os.getenv("USE_TESTNET", "0") == "1"

# 페이퍼 트레이딩 설정
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000"))
POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", "0.1"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.03"))
MAX_HOLD_HOURS = int(os.getenv("MAX_HOLD_HOURS", "24"))  # 일일 데이터이므로 시간 단위
LIQUIDATION_BUFFER = float(os.getenv("LIQUIDATION_BUFFER", "0.8"))
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "trades_daily.json")


# ===== 데이터 클래스 =====
@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    direction: str  # "Long" or "Short"
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
        if self.direction == "Long":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def get_pnl_pct(self, current_price: float) -> float:
        """손익률 계산"""
        pnl = self.get_pnl(current_price)
        return (pnl / self.margin) * 100

    def get_roe(self, current_price: float) -> float:
        """ROE 계산"""
        if self.direction == "Long":
            price_change_pct = (current_price / self.entry_price - 1) * 100
        else:
            price_change_pct = (1 - current_price / self.entry_price) * 100
        return price_change_pct * self.leverage

    def get_liquidation_distance(self, current_price: float) -> float:
        """청산가까지 거리 (%)"""
        if self.direction == "Long":
            return (current_price - self.liquidation_price) / current_price * 100
        else:
            return (self.liquidation_price - current_price) / current_price * 100

    def should_close(self, current_price: float, current_time: datetime) -> tuple[bool, str]:
        """청산 여부 판단"""
        # 강제 청산
        if self.direction == "Long" and current_price <= self.liquidation_price:
            return True, "Liquidation"
        if self.direction == "Short" and current_price >= self.liquidation_price:
            return True, "Liquidation"

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
        hold_hours = (current_time - self.entry_time).total_seconds() / 3600
        if hold_hours >= MAX_HOLD_HOURS:
            return True, "Time Limit"

        return False, ""


@dataclass
class Trade:
    """거래 기록"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int
    margin: float
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float
    roe: float
    exit_reason: str


class Account:
    """계좌 관리"""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.total_pnl = 0.0

    def get_available_balance(self) -> float:
        """사용 가능한 잔고"""
        used_margin = sum(p.margin for p in self.positions.values())
        return self.balance - used_margin

    def get_total_value(self, prices: Dict[str, float]) -> float:
        """총 자산"""
        unrealized_pnl = sum(
            p.get_pnl(prices.get(p.symbol, p.entry_price))
            for p in self.positions.values()
        )
        return self.balance + unrealized_pnl

    def can_open_position(self, symbol: str) -> bool:
        """포지션 진입 가능 여부"""
        if symbol in self.positions:
            return False
        if len(self.positions) >= MAX_POSITIONS:
            return False
        margin_needed = self.initial_capital * POSITION_SIZE_PCT
        if self.get_available_balance() < margin_needed:
            return False
        return True

    def open_position(self, symbol: str, direction: str, price: float):
        """포지션 진입"""
        margin = self.initial_capital * POSITION_SIZE_PCT
        position_value = margin * LEVERAGE
        quantity = position_value / price

        if direction == "Long":
            stop_loss = price * (1 - STOP_LOSS_PCT)
            take_profit = price * (1 + TAKE_PROFIT_PCT)
            liquidation_price = price * (1 - (1 / LEVERAGE) * LIQUIDATION_BUFFER)
        else:
            stop_loss = price * (1 + STOP_LOSS_PCT)
            take_profit = price * (1 - TAKE_PROFIT_PCT)
            liquidation_price = price * (1 + (1 / LEVERAGE) * LIQUIDATION_BUFFER)

        position = Position(
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

        self.positions[symbol] = position
        print(f"\n{'=' * 90}")
        print(f"🔔 포지션 진입: {symbol}")
        print(f"   방향: {direction}")
        print(f"   레버리지: {LEVERAGE}x")
        print(f"   진입가: ${price:,.4f}")
        print(f"   수량: {quantity:.6f}")
        print(f"   증거금: ${margin:,.2f}")
        print(f"   포지션 크기: ${position_value:,.2f}")
        print(f"   손절가: ${stop_loss:,.4f} (-{STOP_LOSS_PCT * 100:.1f}%)")
        print(f"   익절가: ${take_profit:,.4f} (+{TAKE_PROFIT_PCT * 100:.1f}%)")
        print(f"   청산가: ${liquidation_price:,.4f}")
        print(f"   사용 가능 잔고: ${self.get_available_balance():,.2f}")
        print(f"{'=' * 90}\n")

    def close_position(self, symbol: str, price: float, reason: str):
        """포지션 청산"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        pnl = position.get_pnl(price)
        pnl_pct = position.get_pnl_pct(price)
        roe = position.get_roe(price)

        if reason == "Liquidation":
            pnl = -position.margin
            pnl_pct = -100
            roe = -100

        trade = Trade(
            symbol=symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=position.quantity,
            leverage=position.leverage,
            margin=position.margin,
            entry_time=position.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            exit_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pnl=pnl,
            pnl_pct=pnl_pct,
            roe=roe,
            exit_reason=reason
        )

        self.trades.append(trade)
        self.balance += pnl
        self.total_pnl += pnl

        emoji = "💀" if reason == "Liquidation" else ("🟢" if pnl > 0 else "🔴")

        print(f"\n{'=' * 90}")
        print(f"{emoji} 포지션 청산: {symbol}")
        print(f"   이유: {reason}")
        print(f"   레버리지: {position.leverage}x")
        print(f"   진입가: ${position.entry_price:,.4f}")
        print(f"   청산가: ${price:,.4f}")
        print(f"   증거금: ${position.margin:,.2f}")
        print(f"   손익: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        print(f"   ROE: {roe:+.2f}%")
        print(f"   보유 시간: {(datetime.now() - position.entry_time).total_seconds() / 3600:.1f}시간")
        print(f"   누적 손익: ${self.total_pnl:+,.2f} ({(self.total_pnl / self.initial_capital) * 100:+.2f}%)")
        print(f"   현재 잔고: ${self.balance:,.2f}")
        print(f"{'=' * 90}\n")

        del self.positions[symbol]

    def get_stats(self) -> dict:
        """통계 계산"""
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "max_pnl": 0.0,
                "min_pnl": 0.0,
            }

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        liquidations = [t for t in self.trades if t.exit_reason == "Liquidation"]

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "liquidations": len(liquidations),
            "win_rate": len(wins) / len(self.trades) * 100 if self.trades else 0,
            "total_pnl": self.total_pnl,
            "total_return_pct": (self.total_pnl / self.initial_capital) * 100,
            "avg_pnl": self.total_pnl / len(self.trades),
            "avg_roe": np.mean([t.roe for t in self.trades]),
            "max_pnl": max(t.pnl for t in self.trades),
            "min_pnl": min(t.pnl for t in self.trades),
            "max_roe": max(t.roe for t in self.trades),
            "min_roe": min(t.roe for t in self.trades),
            "avg_win": np.mean([t.pnl for t in wins]) if wins else 0.0,
            "avg_loss": np.mean([t.pnl for t in losses]) if losses else 0.0,
        }


# ===== TCN 모델 정의 (TCN_Simple) =====
class Chomp1d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):
        return x[:, :, :-self.c].contiguous() if self.c > 0 else x


def wconv(i, o, k, d):
    pad = (k - 1) * d
    return nn.utils.weight_norm(nn.Conv1d(i, o, k, padding=pad, dilation=d))


class Block(nn.Module):
    def __init__(self, i, o, k, d, drop):
        super().__init__()
        self.c1 = wconv(i, o, k, d)
        self.h1 = Chomp1d((k - 1) * d)
        self.r1 = nn.ReLU()
        self.dr1 = nn.Dropout(drop)
        self.c2 = wconv(o, o, k, d)
        self.h2 = Chomp1d((k - 1) * d)
        self.r2 = nn.ReLU()
        self.dr2 = nn.Dropout(drop)
        self.ds = nn.Conv1d(i, o, 1) if i != o else None
        self.r = nn.ReLU()

    def forward(self, x):
        y = self.dr1(self.r1(self.h1(self.c1(x))))
        y = self.dr2(self.r2(self.h2(self.c2(y))))
        res = x if self.ds is None else self.ds(x)
        return self.r(y + res)


class TCN_Simple(nn.Module):
    """train_tcn_daily.py와 동일한 모델 구조"""
    def __init__(self, in_f, hidden=128, levels=6, k=3, drop=0.2):
        super().__init__()
        L = []
        ch = in_f
        for i in range(levels):
            L.append(Block(ch, hidden, k, 2 ** i, drop))
            ch = hidden
        self.tcn = nn.Sequential(*L)
        self.head_side = nn.Linear(hidden, 2)  # Binary: Long(1) or Short(0)

    def forward(self, X):
        X = X.transpose(1, 2)
        H = self.tcn(X)[:, :, -1]
        return self.head_side(H)


# ===== 모델 로드 =====
print(f"[INIT] 모델 로드 중: {TCN_CKPT}")
try:
    checkpoint = torch.load(TCN_CKPT, map_location="cpu")
    FEAT_COLS = checkpoint['feat_cols']
    META = checkpoint['meta']
    SEQ_LEN = META['seq_len']
    SCALER_MU = checkpoint['scaler_mu']
    SCALER_SD = checkpoint['scaler_sd']

    MODEL = TCN_Simple(in_f=len(FEAT_COLS), hidden=128, levels=6, k=3, drop=0.2).eval()
    MODEL.load_state_dict(checkpoint['model'])
    print(f"   ✓ 모델 로드 완료 (seq_len={SEQ_LEN}, features={len(FEAT_COLS)})")
    print(f"   ✓ Feature columns: {FEAT_COLS}")
except Exception as e:
    print(f"   ✗ 모델 로드 실패: {e}")
    exit(1)


# ===== BYBIT API =====
class BybitPublic:
    def __init__(self, testnet: bool = False):
        self.base = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self.session = requests.Session()
        self.session.verify = certifi.where()

    def get_kline(self, symbol: str, interval: str, limit: int):
        """캔들스틱 데이터 가져오기"""
        url = f"{self.base}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        try:
            r = self.session.get(url, params=params, timeout=10)
            data = r.json()
            return ((data.get("result") or {}).get("list") or [])
        except Exception as e:
            print(f"   ✗ API 오류 ({symbol}): {e}")
            return []

    def get_ticker(self, symbol: str):
        """현재 가격 가져오기"""
        url = f"{self.base}/v5/market/tickers"
        params = {"category": "linear", "symbol": symbol}
        try:
            r = self.session.get(url, params=params, timeout=5)
            data = r.json()
            rows = ((data.get("result") or {}).get("list") or [])
            return rows[0] if rows else {}
        except Exception as e:
            print(f"   ✗ API 오류 ({symbol}): {e}")
            return {}


API = BybitPublic(testnet=USE_TESTNET)


# ===== 데이터 가져오기 =====
def get_recent_data(symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
    """
    최근 N일 데이터 가져오기 (일봉 데이터)
    일봉 = "D"
    """
    lst = API.get_kline(symbol, "D", days)  # 일봉
    if not lst:
        return None

    rows = lst[::-1]
    df = pd.DataFrame([{
        "date": pd.to_datetime(int(z[0]), unit="ms", utc=True),
        "open": float(z[1]),
        "high": float(z[2]),
        "low": float(z[3]),
        "close": float(z[4]),
        "volume": float(z[5]),
    } for z in rows])

    return df


# ===== 피처 생성 (train_tcn_daily.py와 동일) =====
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 생성"""
    g = df.copy().sort_values("date")

    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g["logc"].diff().fillna(0.0)

    def roll_std(s, w):
        return s.rolling(w, min_periods=max(2, w // 3)).std()

    for w in (5, 10, 20, 60):
        g[f"rv{w}"] = roll_std(g["ret1"], w)

    def mom(s, w):
        ema = s.ewm(span=w, adjust=False).mean()
        return s / ema - 1.0

    for w in (5, 10, 20, 60):
        g[f"mom{w}"] = mom(g["close"], w)

    for w in (10, 20, 60):
        mu = g["volume"].rolling(w, min_periods=max(2, w // 3)).mean()
        sd = g["volume"].rolling(w, min_periods=max(2, w // 3)).std()
        sd = sd.replace(0, np.nan)
        g[f"vz{w}"] = (g["volume"] - mu) / sd.fillna(1.0)

    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14, min_periods=5).mean()

    return g


# ===== 예측 함수 =====
@torch.no_grad()
def predict(symbol: str) -> dict:
    """실시간 예측"""
    # 충분한 데이터 가져오기 (SEQ_LEN + 여유분)
    df = get_recent_data(symbol, days=SEQ_LEN + 30)
    if df is None or len(df) < SEQ_LEN:
        return {
            "symbol": symbol,
            "error": "데이터 부족",
            "direction": None,
            "confidence": 0.0
        }

    df_feat = make_features(df)
    if len(df_feat) < SEQ_LEN:
        return {
            "symbol": symbol,
            "error": "피처 생성 후 데이터 부족",
            "direction": None,
            "confidence": 0.0
        }

    # 마지막 SEQ_LEN개 데이터 사용
    X = df_feat[FEAT_COLS].tail(SEQ_LEN).to_numpy(np.float32)
    X = (X - SCALER_MU) / SCALER_SD
    X_tensor = torch.from_numpy(X[None, ...])

    # 예측
    logits = MODEL(X_tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = int(logits.argmax(dim=1).item())

    # 0 = Short, 1 = Long
    direction = "Long" if pred_class == 1 else "Short"
    confidence = float(probs[pred_class])

    # 현재 가격
    ticker = API.get_ticker(symbol)
    current_price = float(ticker.get("lastPrice", 0))

    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "current_price": current_price,
        "probs": probs.tolist(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ===== 화면 출력 =====
def print_dashboard(account: Account, prices: Dict[str, float]):
    """대시보드 출력"""
    os.system('clear' if os.name == 'posix' else 'cls')

    print("\n" + "=" * 110)
    print(f"{'🎯 페이퍼 트레이딩 시스템 (Daily TCN_Simple, 레버리지 ' + str(LEVERAGE) + 'x)':^110}")
    print("=" * 110)

    # 계좌 정보
    total_value = account.get_total_value(prices)
    unrealized_pnl = total_value - account.balance
    total_return = (total_value / account.initial_capital - 1) * 100

    print(f"\n💰 계좌 현황")
    print(f"   초기 자본:     ${account.initial_capital:>12,.2f}")
    print(f"   현재 잔고:     ${account.balance:>12,.2f}")
    print(f"   사용 가능:     ${account.get_available_balance():>12,.2f}")
    print(f"   평가 손익:     ${unrealized_pnl:>+12,.2f}")
    print(f"   총 자산:       ${total_value:>12,.2f}  ({total_return:>+6.2f}%)")
    print(f"   실현 손익:     ${account.total_pnl:>+12,.2f}  ({(account.total_pnl / account.initial_capital) * 100:>+6.2f}%)")

    # 포지션
    if account.positions:
        print(f"\n📍 보유 포지션 ({len(account.positions)}/{MAX_POSITIONS})")
        print(f"{'심볼':^12} | {'방향':^8} | {'진입가':^12} | {'현재가':^12} | {'손익(ROE)':^22} | {'청산가':^12} | {'보유':^10}")
        print("-" * 110)

        for symbol, pos in account.positions.items():
            current_price = prices.get(symbol, pos.entry_price)
            pnl = pos.get_pnl(current_price)
            roe = pos.get_roe(current_price)
            hold_hours = (datetime.now() - pos.entry_time).total_seconds() / 3600
            liq_dist = pos.get_liquidation_distance(current_price)

            emoji = "📈" if pos.direction == "Long" else "📉"
            pnl_emoji = "🟢" if pnl > 0 else "🔴"
            liq_warning = "⚠️" if liq_dist < 3 else ""

            print(f"{symbol:^12} | {emoji} {pos.direction:^6} | ${pos.entry_price:>10,.2f} | "
                  f"${current_price:>10,.2f} | {pnl_emoji} ${pnl:>+8,.2f} ({roe:>+6.1f}%) | "
                  f"${pos.liquidation_price:>10,.2f}{liq_warning} | {hold_hours:>8.1f}h")
    else:
        print(f"\n📍 보유 포지션: 없음")

    # 통계
    stats = account.get_stats()
    if stats["total_trades"] > 0:
        print(f"\n📊 거래 통계")
        print(f"   총 거래:       {stats['total_trades']:>3}회")
        print(f"   승률:          {stats['win_rate']:>6.1f}% ({stats['wins']}승 {stats['losses']}패)")
        if stats['liquidations'] > 0:
            print(f"   강제 청산:     {stats['liquidations']:>3}회 💀")
        print(f"   평균 손익:     ${stats['avg_pnl']:>+12,.2f}")
        print(f"   평균 ROE:      {stats['avg_roe']:>+6.1f}%")
        print(f"   최대 수익:     ${stats['max_pnl']:>12,.2f}  (ROE: {stats['max_roe']:>+6.1f}%)")
        print(f"   최대 손실:     ${stats['min_pnl']:>12,.2f}  (ROE: {stats['min_roe']:>+6.1f}%)")
        if stats['wins'] > 0 and stats['losses'] > 0:
            rr = abs(stats['avg_win'] / stats['avg_loss'])
            print(f"   Risk/Reward:   {rr:>6.2f}")

    print("\n" + "=" * 110)


def save_trades(account: Account):
    """거래 내역 저장"""
    if not account.trades:
        return

    data = [asdict(t) for t in account.trades]
    with open(TRADE_LOG_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n💾 거래 내역 저장: {TRADE_LOG_FILE}")


# ===== 메인 루프 =====
def main():
    print("\n" + "=" * 110)
    print(f"{'🎯 페이퍼 트레이딩 시작 (Daily TCN_Simple)':^110}")
    print(f"{'초기 자본: $' + f'{INITIAL_CAPITAL:,.2f}':^110}")
    print(f"{'레버리지: ' + f'{LEVERAGE}x (포지션 크기: {POSITION_SIZE_PCT * 100:.0f}%)':^110}")
    print(f"{'신뢰도 임계값: ' + f'{CONF_THRESHOLD:.0%}':^110}")
    print("=" * 110)

    account = Account(INITIAL_CAPITAL)

    try:
        loop_count = 0
        while True:
            loop_count += 1
            current_time = datetime.now()

            # 현재 가격 가져오기
            prices = {}
            for symbol in SYMBOLS:
                symbol = symbol.strip()
                ticker = API.get_ticker(symbol)
                prices[symbol] = float(ticker.get("lastPrice", 0))

            # 포지션 관리
            for symbol in list(account.positions.keys()):
                position = account.positions[symbol]
                current_price = prices.get(symbol, position.entry_price)

                # 청산 조건 확인
                should_close, reason = position.should_close(current_price, current_time)
                if should_close:
                    account.close_position(symbol, current_price, reason)
                else:
                    # 반대 신호로 청산
                    result = predict(symbol)
                    if result.get("confidence", 0) >= CONF_THRESHOLD:
                        signal_dir = result["direction"]
                        if (position.direction == "Long" and signal_dir == "Short") or \
                                (position.direction == "Short" and signal_dir == "Long"):
                            account.close_position(symbol, current_price, "Reverse Signal")

            # 대시보드 출력
            print_dashboard(account, prices)

            # 신호 스캔
            print(f"\n🔍 신호 스캔")
            print(f"{'심볼':^12} | {'가격':^12} | {'방향':^10} | {'신뢰도':^10} | {'확률':^20} | {'신호':^20}")
            print("-" * 100)

            for symbol in SYMBOLS:
                symbol = symbol.strip()
                result = predict(symbol)

                if "error" in result:
                    print(f"{symbol:^12} | {'N/A':^12} | {'오류':^10} | {'N/A':^10} | {'N/A':^20} | ❌ {result['error']}")
                    continue

                direction = result["direction"]
                confidence = result["confidence"]
                price = result["current_price"]
                probs = result.get("probs", [0, 0])

                dir_icon = "📈" if direction == "Long" else "📉"

                if confidence < CONF_THRESHOLD:
                    signal = f"⚪ 신호 약함"
                elif direction == "Long":
                    signal = f"🟢 매수 신호"
                else:
                    signal = f"🔴 매도 신호"

                prob_str = f"S:{probs[0]:.2f}/L:{probs[1]:.2f}"

                print(f"{symbol:^12} | ${price:>10,.2f} | {dir_icon} {direction:^8} | "
                      f"{confidence:>8.1%} | {prob_str:^20} | {signal}")

                # 진입 조건
                if account.can_open_position(symbol) and confidence >= CONF_THRESHOLD:
                    account.open_position(symbol, direction, price)

            print(f"\n[스캔 #{loop_count}] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"다음 스캔까지 {INTERVAL_SEC}초... (Ctrl+C로 종료)")

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n프로그램 종료 중...")
        print_dashboard(account, prices)
        save_trades(account)

        # 최종 통계
        stats = account.get_stats()
        if stats["total_trades"] > 0:
            print("\n" + "=" * 110)
            print(f"{'📊 최종 결과':^110}")
            print("=" * 110)
            final_balance = account.balance
            final_return = (final_balance / account.initial_capital - 1) * 100
            print(f"   최종 잔고:     ${final_balance:,.2f}")
            print(f"   총 수익률:     {final_return:+.2f}%")
            print(f"   총 거래:       {stats['total_trades']}회")
            print(f"   승률:          {stats['win_rate']:.1f}%")
            print(f"   평균 ROE:      {stats['avg_roe']:+.1f}%")
            if stats['liquidations'] > 0:
                print(f"   강제 청산:     {stats['liquidations']}회 💀")
            print("=" * 110)

        print("\n✅ 프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()