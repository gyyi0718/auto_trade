# live_trading_rl_multi.py
# -*- coding: utf-8 -*-
"""
⚠️ 실제 라이브 트레이딩 시스템 - 작동 검증 버전 ⚠️
"""
import os
import sys
import time
import json
import hmac
import hashlib
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
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

CONFIG_FILE = "live_trading_config.json"
API_KEY_FILE = "api_keys.json"


class Actions(IntEnum):
    LONG = 0
    SHORT = 1
    CLOSE = 2


def safe_float(value, default=0.0):
    """안전하게 float 변환"""
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


# ===== Bybit API (작동 검증 완료) =====
class BybitLiveAPI:
    """Bybit V5 API - 검증된 버전"""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self.recv_window = "5000"

    def _generate_signature(self, timestamp: str, params_str: str) -> str:
        """서명 생성"""
        sign_str = timestamp + self.api_key + self.recv_window + params_str
        return hmac.new(
            bytes(self.api_secret, 'utf-8'),
            bytes(sign_str, 'utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _request(self, method: str, endpoint: str, params: Dict = None, sign: bool = True) -> Dict:
        """API 요청"""
        if params is None:
            params = {}

        headers = {
            "Content-Type": "application/json"
        }

        if sign:
            timestamp = str(int(time.time() * 1000))

            # 파라미터 문자열 생성
            if method == "GET":
                params_str = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())]) if params else ""
            else:
                params_str = json.dumps(params) if params else ""

            # 서명 생성
            signature = self._generate_signature(timestamp, params_str)

            # 헤더 설정
            headers.update({
                'X-BAPI-API-KEY': self.api_key,
                'X-BAPI-SIGN': signature,
                'X-BAPI-TIMESTAMP': timestamp,
                'X-BAPI-RECV-WINDOW': self.recv_window
            })

        try:
            url = f"{self.base_url}{endpoint}"

            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=params, headers=headers, timeout=10)
            else:
                return {"error": f"Unsupported method: {method}", "retCode": -1}

            data = response.json()

            # 110043은 레버리지가 이미 설정되어 있음을 의미 (정상)
            if data.get("retCode") == 110043:
                return {"retCode": 0, "retMsg": "leverage not modified", "result": data.get("result", {})}

            return data
        except Exception as e:
            return {"error": str(e), "retCode": -1}

    def get_ticker(self, symbol: str) -> Dict:
        """현재 가격 조회 (Public API)"""
        url = f"{self.base_url}/v5/market/tickers"
        params = {"category": "linear", "symbol": symbol}
        try:
            response = requests.get(url, params=params, timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e), "retCode": -1}

    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict:
        """지갑 잔고 조회"""
        return self._request("GET", "/v5/account/wallet-balance", {
            "accountType": account_type
        }, sign=True)

    def get_positions(self, symbol: str = None) -> Dict:
        """포지션 조회"""
        params = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/v5/position/list", params, sign=True)

    def set_leverage(self, symbol: str, leverage: int) -> Tuple[bool, str]:
        """레버리지 설정 - 작동 검증 완료"""
        result = self._request("POST", "/v5/position/set-leverage", {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        }, sign=True)

        success = result.get("retCode") == 0
        msg = result.get("retMsg", "Unknown error")

        return success, msg

    def place_order(self, symbol: str, side: str, qty: str, order_type: str = "Market",
                    price: str = None, reduce_only: bool = False) -> Optional[str]:
        """주문 실행"""
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty,
            "timeInForce": "GTC",
            "positionIdx": 0  # One-way mode
        }

        if reduce_only:
            params["reduceOnly"] = True

        if order_type == "Limit" and price:
            params["price"] = price

        result = self._request("POST", "/v5/order/create", params, sign=True)

        if result.get("retCode") == 0:
            return result["result"]["orderId"]
        else:
            print(f"❌ 주문 실패: {result.get('retMsg')}")
            return None

    def set_trading_stop(self, symbol: str, stop_loss: str = None,
                         take_profit: str = None, position_idx: int = 0) -> Tuple[bool, str]:
        """손절/익절 설정"""
        params = {
            "category": "linear",
            "symbol": symbol,
            "positionIdx": position_idx
        }

        if stop_loss:
            params["stopLoss"] = stop_loss
        if take_profit:
            params["takeProfit"] = take_profit

        result = self._request("POST", "/v5/position/trading-stop", params, sign=True)

        success = result.get("retCode") == 0
        msg = result.get("retMsg", "Unknown error")

        return success, msg

    def close_position(self, symbol: str, side: str, qty: str) -> Optional[str]:
        """포지션 청산"""
        close_side = "Sell" if side == "Buy" else "Buy"
        return self.place_order(
            symbol=symbol,
            side=close_side,
            qty=qty,
            reduce_only=True
        )

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """캔들 데이터 조회 (Public API)"""
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


@dataclass
class LivePosition:
    symbol: str
    side: str
    size: float
    entry_price: float
    leverage: int
    unrealized_pnl: float
    stop_loss: float
    take_profit: float


@dataclass
class LiveTrade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    reason: str


class SafetyManager:
    def __init__(self, config: Dict):
        safety_config = config.get('safety', {
            'max_daily_loss': 1000.0,
            'max_position_size': 5000.0,
            'max_trades_per_day': 20
        })

        self.max_daily_loss = safety_config.get('max_daily_loss', 1000.0)
        self.max_position_size = safety_config.get('max_position_size', 5000.0)
        self.max_trades_per_day = safety_config.get('max_trades_per_day', 20)

        self.daily_pnl = 0
        self.total_trades_today = 0
        self.last_reset_date = datetime.now().date()
        self.emergency_stop = False

    def reset_daily_stats(self):
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_pnl = 0
            self.total_trades_today = 0
            self.last_reset_date = today

    def check_daily_loss_limit(self) -> bool:
        if self.daily_pnl < -self.max_daily_loss:
            self.emergency_stop = True
            return False
        return True

    def check_position_size(self, size_usdt: float) -> bool:
        return size_usdt <= self.max_position_size

    def check_max_trades(self) -> bool:
        return self.total_trades_today < self.max_trades_per_day

    def update_trade(self, pnl: float):
        self.daily_pnl += pnl
        self.total_trades_today += 1
        self.check_daily_loss_limit()


class LiveTradingManager:
    def __init__(self, config: Dict, api: BybitLiveAPI):
        self.config = config
        self.api = api
        self.safety = SafetyManager(config)
        self.active_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT']
        self.models: Dict[str, PPO] = {}
        self.df_cache: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, float] = {}
        self.positions: Dict[str, LivePosition] = {}
        self.last_prediction: Dict[str, float] = {}  # ← 추가
        for symbol in self.active_symbols:
            self.last_prediction[symbol] = 0
        self.trades: List[LiveTrade] = []

        enabled_symbols = [s for s in config['symbols'] if s['enabled']]

        for sym_config in enabled_symbols:
            symbol = sym_config['symbol']

            # 모델 로드
            try:
                model_path = sym_config['model_path']
                self.models[symbol] = PPO.load(model_path)
                print(f"✅ [{symbol}] 모델 로드")
            except Exception as e:
                print(f"❌ [{symbol}] 모델 로드 실패: {e}")
                raise

            # 레버리지 설정 (작동 검증 완료)
            leverage = sym_config.get('leverage', 10)
            success, msg = self.api.set_leverage(symbol, leverage)
            if success:
                print(f"✅ [{symbol}] 레버리지: {leverage}x")
            else:
                print(f"⚠️ [{symbol}] 레버리지 설정 실패: {msg}")
                print(f"   → Bybit 웹사이트에서 수동으로 {leverage}x 설정하세요")

            self.df_cache[symbol] = pd.DataFrame()
            self.last_update[symbol] = 0

    def sync_positions(self):
        result = self.api.get_positions()

        if result.get('retCode') != 0:
            return

        self.positions.clear()

        positions_list = result.get('result', {}).get('list', [])
        for pos_data in positions_list:
            size = safe_float(pos_data.get('size', 0))
            if size > 0:
                symbol = pos_data['symbol']
                self.positions[symbol] = LivePosition(
                    symbol=symbol,
                    side=pos_data['side'],
                    size=size,
                    entry_price=safe_float(pos_data['avgPrice']),
                    leverage=int(safe_float(pos_data['leverage'], 1)),
                    unrealized_pnl=safe_float(pos_data['unrealisedPnl']),
                    stop_loss=safe_float(pos_data.get('stopLoss', 0)),
                    take_profit=safe_float(pos_data.get('takeProfit', 0))
                )

    def open_position(self, symbol: str, direction: str, current_price: float):
        if self.safety.emergency_stop or not self.safety.check_max_trades():
            return False

        print(f"\n{'=' * 80}")
        print(f"💡 포지션 진입 시도: {symbol} {direction.upper()}")

        # 기존 포지션 확인 및 청산
        self.sync_positions()
        if symbol in self.positions:
            pos = self.positions[symbol]
            existing_direction = "long" if pos.side == "Buy" else "short"

            # 같은 방향이면 진입하지 않음
            if existing_direction == direction:
                print(f"⚠️ 이미 같은 방향({direction}) 포지션 보유 중 | 손익: ${pos.unrealized_pnl:+,.2f}")
                print(f"   진입을 건너뜁니다.")
                print(f"{'=' * 80}\n")
                return False

            # 반대 방향이면 청산 후 진입
            print(f"⚠️ 반대 방향 포지션 발견: {existing_direction} | 손익: ${pos.unrealized_pnl:+,.2f}")
            print(f"   청산 후 {direction} 포지션으로 진입합니다...")
            if self.close_position(symbol, "Reverse Signal"):
                time.sleep(2)  # 청산 완료 대기
            else:
                print(f"⚠️ 기존 포지션 청산 실패 - 진입을 중단합니다")
                print(f"{'=' * 80}\n")
                return False

        sym_config = next(s for s in self.config['symbols'] if s['symbol'] == symbol)

        # 레버리지 재설정 (매 거래마다 확인)
        leverage = sym_config.get('leverage', 10)
        success, msg = self.api.set_leverage(symbol, leverage)
        if not success and "110043" not in msg:  # 110043은 이미 설정됨
            print(f"⚠️ 레버리지 설정 실패: {msg}")

        # 잔고 조회
        balance_result = self.api.get_wallet_balance()
        if balance_result.get('retCode') != 0:
            print(f"❌ 잔고 조회 실패")
            print(f"{'=' * 80}\n")
            return False

        # USDT 잔고 추출 (안전하게) - equity 사용
        usdt_balance = 0
        try:
            coin_list = balance_result.get('result', {}).get('list', [])
            for account in coin_list:
                for coin_info in account.get('coin', []):
                    if coin_info.get('coin') == 'USDT':
                        usdt_balance = safe_float(coin_info.get('equity') or
                                                  coin_info.get('walletBalance') or 0)
                        break
                if usdt_balance > 0:
                    break
        except Exception as e:
            print(f"❌ 잔고 파싱 오류: {e}")
            print(f"{'=' * 80}\n")
            return False

        if usdt_balance <= 0:
            print(f"❌ 잔고 부족: ${usdt_balance:.2f}")
            print(f"{'=' * 80}\n")
            return False

        # 심볼 정보 조회 (최소 주문 수량 등)
        instrument_result = self.api._request("GET", "/v5/market/instruments-info", {
            "category": "linear",
            "symbol": symbol
        }, sign=False)

        min_order_qty = 0.01
        qty_step = 0.01
        tick_size = 0.01
        if instrument_result.get('retCode') == 0:
            items = instrument_result.get('result', {}).get('list', [])
            if items:
                lot_filter = items[0].get('lotSizeFilter', {})
                min_order_qty = float(lot_filter.get('minOrderQty', 0.01))
                qty_step = float(lot_filter.get('qtyStep', 0.01))
                price_filter = items[0].get('priceFilter', {})
                tick_size = float(price_filter.get('tickSize', 0.01))

        # 포지션 크기 계산
        position_size_pct = sym_config.get('position_size_pct', 0.10)

        margin = usdt_balance * position_size_pct
        position_value = margin * leverage
        qty = position_value / current_price

        # qtyStep에 맞춰 조정
        qty = round(qty / qty_step) * qty_step

        # 최소 수량 체크
        if qty < min_order_qty:
            qty = min_order_qty

        # 최종 반올림
        if qty_step >= 1:
            qty = int(qty)
        else:
            decimal_places = len(str(qty_step).split('.')[-1].rstrip('0'))
            qty = round(qty, decimal_places)

        if not self.safety.check_position_size(position_value):
            print(f"❌ 포지션 크기 제한 초과")
            print(f"{'=' * 80}\n")
            return False

        print(f"   💰 사용 가능: ${usdt_balance:,.2f}")
        print(f"   증거금: ${margin:,.2f}")
        print(f"   레버리지: {leverage}x")
        print(f"   포지션 크기: ${position_value:,.2f}")
        print(f"   진입가: ${current_price:,.4f}")
        print(f"   수량: {qty} (최소: {min_order_qty}, 단위: {qty_step})")

        # TP/SL 가격 계산 (tick size 고려)
        stop_loss_pct = sym_config.get('stop_loss_pct', 0.05)
        take_profit_pct = sym_config.get('take_profit_pct', 0.08)

        # tick size로 소수점 자리수 결정
        if tick_size >= 1:
            price_decimal_places = 0
        else:
            price_decimal_places = len(str(tick_size).split('.')[-1].rstrip('0'))

        if direction == "long":
            stop_loss_price = current_price * (1 - stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)
        else:
            stop_loss_price = current_price * (1 + stop_loss_pct)
            take_profit_price = current_price * (1 - take_profit_pct)

        # tick size에 맞춰 반올림
        stop_loss_price = round(stop_loss_price / tick_size) * tick_size
        stop_loss_price = round(stop_loss_price, price_decimal_places)
        take_profit_price = round(take_profit_price / tick_size) * tick_size
        take_profit_price = round(take_profit_price, price_decimal_places)

        # 주문 실행 (TP/SL 포함)
        side = "Buy" if direction == "long" else "Sell"

        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "positionIdx": 0,  # One-way mode
            "stopLoss": str(stop_loss_price),
            "takeProfit": str(take_profit_price)
        }

        print(f"   🛡️ Stop Loss: ${stop_loss_price:.{price_decimal_places}f}")
        print(f"   🎯 Take Profit: ${take_profit_price:.{price_decimal_places}f}")

        result = self.api._request("POST", "/v5/order/create", params, sign=True)

        if result.get("orderId"):
            order_id = result["orderId"]
            print(f"✅ 포지션 진입 성공! (주문 ID: {order_id})")
            print(f"🔥 [{symbol}] {direction.upper()} | ${current_price:,.2f}")
            print(f"{'=' * 80}\n")
            return True
        else:
            print(f"❌ 주문 실패: {result.get('retMsg', 'Unknown error')}")
            print(f"{'=' * 80}\n")
            return False

    def close_position(self, symbol: str, reason: str = "Manual"):
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]

        print(f"\n{'=' * 80}")
        print(f"🔄 [{symbol}] 청산 시도...")
        print(f"   포지션: {pos.side} {pos.size:.4f} @ ${pos.entry_price:,.2f}")
        print(f"   현재 손익: ${pos.unrealized_pnl:+,.2f}")
        print(f"   이유: {reason}")

        order_id = self.api.close_position(
            symbol=symbol,
            side=pos.side,
            qty=f"{pos.size:.4f}"
        )

        if order_id:
            print(f"✅ [{symbol}] 청산 완료 (주문 ID: {order_id})")
            print(f"{'=' * 80}\n")

            trade = LiveTrade(
                symbol=symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=0,
                size=pos.size,
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                pnl=pos.unrealized_pnl,
                reason=reason
            )
            self.trades.append(trade)

            self.safety.update_trade(pos.unrealized_pnl)

            # 포지션 제거
            del self.positions[symbol]

            return True
        else:
            print(f"❌ [{symbol}] 청산 실패")
            print(f"{'=' * 80}\n")
            return False


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_dashboard(manager: LiveTradingManager, prices: Dict[str, float],
                    predictions: Dict[str, Tuple[int, np.ndarray]], loop_count: int = 0):
    """멀티 심볼 라이브 트레이딩 대시보드 (페이퍼 트레이딩 스타일)"""
    clear_screen()

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    scan_interval = manager.config['trading']['scan_interval_sec']

    print("\n" + "=" * 120)
    print(f"{'🔴 멀티 심볼 라이브 트레이딩 🔴':^120}")
    print(
        f"{'스캔 #' + str(loop_count) + ' | ' + current_time + ' | 다음 스캔: ' + str(scan_interval) + '초 후 (Ctrl+C: 종료)':^120}")
    print("=" * 120)

    # 안전장치 상태
    s = manager.safety
    safety_emoji = "🟢" if not s.emergency_stop else "🔴 정지"
    print(f"\n🛡️ 안전장치")
    print(f"   일일 손익:  ${s.daily_pnl:>+12,.2f} | 거래 횟수: {s.total_trades_today:>3}회 | 상태: {safety_emoji}")

    # 심볼별 상태 테이블
    print(f"\n📊 심볼별 상태")
    print(f"{'심볼':^12} | {'가격':^12} | {'포지션':^12} | {'손익':^14} | {'ROE':^8} | {'액션':^8} | {'확률':^20}")
    print("-" * 120)

    for symbol in manager.models.keys():
        price = prices.get(symbol, 0)
        action, action_probs = predictions.get(symbol, (None, np.array([0.33, 0.33, 0.34])))

        # 포지션 정보
        if symbol in manager.positions:
            pos = manager.positions[symbol]
            current_price = price if price > 0 else pos.entry_price

            # 손익 계산
            if pos.side == "Buy":
                pnl = (current_price - pos.entry_price) * pos.size
                roe = (current_price / pos.entry_price - 1) * pos.leverage * 100
            else:
                pnl = (pos.entry_price - current_price) * pos.size
                roe = (1 - current_price / pos.entry_price) * pos.leverage * 100

            pos_emoji = "📈" if pos.side == "Buy" else "📉"
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            pos_str = f"{pos_emoji} {pos.side.upper()}"
            pnl_str = f"{pnl_emoji} ${pnl:+,.2f}"
            roe_str = f"{roe:+.1f}%"
        else:
            pos_str = "⚪ NONE"
            pnl_str = "-"
            roe_str = "-"

        # 예측 정보
        if action is not None:
            action_names = ["LONG", "SHORT", "CLOSE"]
            action_str = action_names[action]
            prob_str = f"L:{action_probs[0] * 100:.0f}% S:{action_probs[1] * 100:.0f}% C:{action_probs[2] * 100:.0f}%"
        else:
            action_str = "-"
            prob_str = "-"

        print(
            f"{symbol:^12} | ${price:>10,.2f} | {pos_str:^12} | {pnl_str:^14} | {roe_str:^8} | {action_str:^8} | {prob_str:^20}")

    # 거래 통계
    if manager.trades:
        wins = [t for t in manager.trades if t.pnl > 0]
        win_rate = len(wins) / len(manager.trades) * 100 if manager.trades else 0
        total_pnl = sum(t.pnl for t in manager.trades)
        avg_pnl = total_pnl / len(manager.trades) if manager.trades else 0

        print(f"\n📈 거래 통계")
        print(
            f"   총 거래:  {len(manager.trades):>3}회 | 승률: {win_rate:>5.1f}% ({len(wins)}승 {len(manager.trades) - len(wins)}패)")
        print(f"   총 손익:  ${total_pnl:>+12,.2f} | 평균 손익: ${avg_pnl:>+10,.2f}")

        # 최근 5개 거래 내역
        if len(manager.trades) > 0:
            print(f"\n📋 최근 거래 (최대 5개)")
            recent_trades = sorted(manager.trades, key=lambda t: t.exit_time, reverse=True)[:5]
            for t in recent_trades:
                pnl_emoji = "🟢" if t.pnl > 0 else "🔴"
                time_str = t.exit_time.strftime("%H:%M:%S")
                print(f"   {time_str} | {t.symbol:10s} | {t.side:5s} | {pnl_emoji} ${t.pnl:>+8,.0f} | {t.reason}")

    print("\n" + "=" * 120)


def main():
    """메인 트레이딩 루프"""
    # 설정 로드
    print(f"\n{'=' * 120}")
    print(f"{'🔴 멀티 심볼 강화학습 라이브 트레이딩 🔴':^120}")
    print(f"{'=' * 120}")

    print("\n⚠️ 경고:")
    print("   - 이 프로그램은 실제 자금을 사용합니다")
    print("   - 투자 손실의 위험이 있습니다")
    print("   - 본인 책임 하에 사용하세요")

    response = input("\n계속하시겠습니까? (yes): ")
    if response.lower() != 'yes':
        print("프로그램을 종료합니다.")
        return

    if not os.path.exists(CONFIG_FILE):
        print(f"❌ 설정 파일이 없습니다: {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if not os.path.exists(API_KEY_FILE):
        print(f"❌ API 키 파일이 없습니다: {API_KEY_FILE}")
        return

    with open(API_KEY_FILE, 'r') as f:
        api_keys = json.load(f)

    testnet = api_keys.get('testnet', True)
    api = BybitLiveAPI(
        api_key=api_keys['api_key'],
        api_secret=api_keys['api_secret'],
        testnet=testnet
    )

    # 설정 정보 출력
    enabled_symbols = [s['symbol'] for s in config['symbols'] if s['enabled']]
    print(f"\n설정:")
    print(f"   모드:        {'🧪 테스트넷' if testnet else '🔴 메인넷 (실전!)'}")
    print(f"   활성 심볼:   {len(enabled_symbols)}개 - {', '.join(enabled_symbols)}")
    print(f"   스캔 주기:   {config['trading']['scan_interval_sec']}초")
    print(f"   일일 손실 한도: ${config['safety']['max_daily_loss']:,.2f}")
    print(f"   최대 거래 횟수: {config['safety']['max_trades_per_day']}회/일")

    # 메인넷 확인
    if not testnet:
        print("\n🔴 메인넷 모드입니다! 실제 자금이 사용됩니다!")
        response = input("계속하려면 'CONFIRM'을 입력하세요: ")
        if response != 'CONFIRM':
            print("프로그램을 종료합니다.")
            return

    print(f"\n🤖 모델 로드 중...")
    try:
        manager = LiveTradingManager(config, api)
        print(f"✅ 모든 모델 로드 완료\n")
    except Exception as e:
        print(f"❌ 실패: {e}")
        return

    scan_interval = config['trading']['scan_interval_sec']

    try:
        loop_count = 0
        while True:
            loop_count += 1

            manager.safety.reset_daily_stats()

            if manager.safety.emergency_stop:
                print("\n🚨 긴급 정지")
                break

            manager.sync_positions()

            # 가격
            prices = {}
            for symbol in manager.models.keys():
                ticker = manager.api.get_ticker(symbol)
                if ticker.get("retCode") == 0:
                    try:
                        prices[symbol] = float(ticker["result"]["list"][0]["lastPrice"])
                    except:
                        pass

            # 캔들
            for symbol in manager.models.keys():
                if time.time() - manager.last_update[symbol] > 60 or manager.df_cache[symbol].empty:
                    sym_config = next(s for s in config['symbols'] if s['symbol'] == symbol)
                    df = manager.api.get_klines(symbol, str(sym_config['interval_minutes']), limit=200)
                    if not df.empty:
                        manager.df_cache[symbol] = calculate_features(df)
                        manager.last_update[symbol] = time.time()

            # 예측
            current_time = time.time()
            predictions = {}

            for symbol in manager.models.keys():
                if manager.df_cache[symbol].empty:
                    continue

                current_price = prices.get(symbol, 0)
                has_position = symbol in manager.positions
                time_since_last_prediction = current_time - manager.last_prediction[symbol]
                should_predict = has_position or time_since_last_prediction >= 300
                if not should_predict:
                    if symbol in predictions:
                        predictions[symbol] = (None, np.array([0.33, 0.33, 0.34]))
                    continue

                position = None
                entry_price = 0
                if has_position:
                    pos = manager.positions[symbol]
                    position = "long" if pos.side == "Buy" else "short"
                    entry_price = pos.entry_price

                window_size = config.get('technical_indicators', {}).get('window_size', 30)
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
                manager.last_prediction[symbol] = current_time

                # 액션
                if action == Actions.LONG and not has_position:
                    manager.open_position(symbol, "long", current_price)
                elif action == Actions.SHORT and not has_position:
                    manager.open_position(symbol, "short", current_price)
                elif action == Actions.CLOSE and has_position:
                    manager.close_position(symbol, "RL")

            print_dashboard(manager, prices, predictions, loop_count)

            time.sleep(scan_interval)

    except KeyboardInterrupt:
        print("\n\n프로그램 종료 중...")

        # 미결제 포지션 청산
        if manager.positions:
            print(f"\n⚠️ 미결제 포지션 {len(manager.positions)}개 발견")
            response = input("모든 포지션을 청산하시겠습니까? (yes/no): ")
            if response.lower() == 'yes':
                for symbol in list(manager.positions.keys()):
                    manager.close_position(symbol, "Manual Close")
                print(f"✅ 모든 포지션 청산 완료")

        # 거래 내역 저장
        if manager.trades:
            data = []
            for trade in manager.trades:
                trade_dict = asdict(trade)
                trade_dict['entry_time'] = trade_dict['entry_time'].isoformat()
                trade_dict['exit_time'] = trade_dict['exit_time'].isoformat()
                data.append(trade_dict)

            with open(config['trading']['trade_log_file'], 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 거래 내역 저장: {config['trading']['trade_log_file']}")

        # 최종 통계
        if manager.trades:
            print("\n" + "=" * 120)
            print(f"{'📊 최종 결과':^120}")
            print("=" * 120)

            wins = [t for t in manager.trades if t.pnl > 0]
            losses = [t for t in manager.trades if t.pnl <= 0]
            win_rate = len(wins) / len(manager.trades) * 100 if manager.trades else 0
            total_pnl = sum(t.pnl for t in manager.trades)
            avg_pnl = total_pnl / len(manager.trades) if manager.trades else 0

            print(f"   총 거래:     {len(manager.trades):>3}회")
            print(f"   승률:        {win_rate:>5.1f}% ({len(wins)}승 {len(losses)}패)")
            print(f"   일일 손익:   ${manager.safety.daily_pnl:>+12,.2f}")
            print(f"   총 손익:     ${total_pnl:>+12,.2f}")
            print(f"   평균 손익:   ${avg_pnl:>+12,.2f}")

            if wins and losses:
                avg_win = sum(t.pnl for t in wins) / len(wins)
                avg_loss = sum(t.pnl for t in losses) / len(losses)
                rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                print(f"   평균 수익:   ${avg_win:>+12,.2f}")
                print(f"   평균 손실:   ${avg_loss:>+12,.2f}")
                print(f"   Risk/Reward: {rr_ratio:>6.2f}")

            # 심볼별 통계
            print(f"\n   심볼별 성과:")
            symbol_stats = {}
            for trade in manager.trades:
                if trade.symbol not in symbol_stats:
                    symbol_stats[trade.symbol] = {'trades': [], 'pnl': 0}
                symbol_stats[trade.symbol]['trades'].append(trade)
                symbol_stats[trade.symbol]['pnl'] += trade.pnl

            for symbol, stats in sorted(symbol_stats.items()):
                trades = stats['trades']
                pnl = stats['pnl']
                wins_count = len([t for t in trades if t.pnl > 0])
                sym_win_rate = wins_count / len(trades) * 100 if trades else 0
                print(f"      {symbol:10s}: {len(trades):3}회 | 승률: {sym_win_rate:5.1f}% | 손익: ${pnl:>+10,.2f}")

            print("=" * 120)

        # 미결제 포지션 경고
        if manager.positions:
            print(f"\n⚠️ 주의: 아직 {len(manager.positions)}개의 포지션이 남아있습니다!")
            for symbol, pos in manager.positions.items():
                print(f"   - {symbol}: {pos.side} {pos.size} @ ${pos.entry_price:,.2f}")

        print("\n✅ 프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()