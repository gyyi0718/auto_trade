# live_trading_final.py
# -*- coding: utf-8 -*-
"""
🚨 실거래 자동 트레이딩 시스템 - 회귀 TCN 모델 기반
⚠️  WARNING: 실제 자금을 사용합니다! 소액으로 먼저 테스트하세요!

사용 전 체크리스트:
1. Testnet에서 충분히 테스트했는가? ✓
2. API 키 권한 확인 (거래 권한만) ✓
3. 최대 손실 한도 설정 확인 ✓
4. 모델 성능 검증 완료 ✓
"""
import os, time, hmac, hashlib, json, warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import requests, certifi

warnings.filterwarnings("ignore")
os.environ["SSL_CERT_FILE"] = certifi.where()

# ===== 필수 설정 =====
API_KEY = "Dlp4eJD6YFmO99T8vC"
API_SECRET = "YYYB5tMw2TWvfVF5wqi6lQRHqEIiDSpDJF1U"

USE_TESTNET = os.getenv("USE_TESTNET", "0") == "1"  # 기본: Testnet

if not API_KEY or not API_SECRET:
    print("\n" + "=" * 100)
    print("❌ API 키가 설정되지 않았습니다!")
    print("=" * 100)
    print("\n설정 방법 (PowerShell):")
    print("  $env:BYBIT_API_KEY = 'your_api_key'")
    print("  $env:BYBIT_API_SECRET = 'your_api_secret'")
    print("  $env:USE_TESTNET = '1'  # Testnet 사용 (기본값)")
    print("  $env:USE_TESTNET = '0'  # 실거래 (주의!)\n")
    exit(1)

# 모델 경로
MODEL_PATHS = {
    "LSKUSDT": "D:/ygy_work/coin/multimodel/models_regression/lsk/best_model.pth",
    "EVAAUSDT": "D:/ygy_work/coin/multimodel/models_regression/evaa/best_model.pth",
    "BTCUSDT": "D:/ygy_work/coin/multimodel/models_regression/best_regression_model.pth",
}

SYMBOLS = os.getenv("SYMBOLS", "EVAAUSDT").split(",")
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "5"))

# 회귀 모델 임계값
LONG_THRESHOLD = float(os.getenv("LONG_THRESHOLD", "5.0"))
SHORT_THRESHOLD = float(os.getenv("SHORT_THRESHOLD", "5.0"))
STRONG_THRESHOLD = float(os.getenv("STRONG_THRESHOLD", "15.0"))

# 리스크 관리 - 소지금 비율 기반
POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", "30"))  # 소지금의 10% (기본값)
STRONG_POSITION_SIZE_PCT = float(os.getenv("STRONG_POSITION_SIZE_PCT", "30"))  # 강한 신호일 때 30%
MIN_MARGIN = float(os.getenv("MIN_MARGIN", "10"))  # 최소 증거금 USDT
MAX_MARGIN = float(os.getenv("MAX_MARGIN", "500"))  # 최대 증거금 USDT (안전장치)
LEVERAGE = int(os.getenv("LEVERAGE", "20"))  # 레버리지
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))  # 최대 포지션
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.03"))  # 3%
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.05"))  # 5%
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "100"))  # 일일 최대 손실
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "60"))
MIN_HOLD_MINUTES = int(os.getenv("MIN_HOLD_MINUTES", "5"))

TRADE_LOG_FILE = "live_trades.json"
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


# ===== 데이터 클래스 =====
@dataclass
class Position:
    symbol: str
    direction: str  # "Buy" or "Sell"
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    leverage: int
    margin: float
    predicted_bps: float
    confidence: str
    order_id: str = ""

    def get_unrealized_pnl(self, current_price: float) -> float:
        if self.direction == "Buy":
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size

    def get_roe(self, current_price: float) -> float:
        pnl = self.get_unrealized_pnl(current_price)
        return (pnl / self.margin) * 100 if self.margin > 0 else 0


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: str
    exit_time: str
    pnl: float
    roe: float
    exit_reason: str
    predicted_bps: float
    actual_bps: float


# ===== Bybit API =====
class BybitAPI:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self.recv_window = "20000"  # 20초 (네트워크 지연 고려)
        self._instrument_cache = {}  # 심볼 정보 캐시

        # 서버 시간 동기화
        self.time_offset = 0
        self._sync_time()

    def _sync_time(self):
        """서버 시간과 로컬 시간 동기화"""
        try:
            response = requests.get(f"{self.base_url}/v3/public/time", timeout=5)
            server_time = response.json()['result']['timeSecond']
            server_time_ms = int(server_time) * 1000
            local_time_ms = int(time.time() * 1000)
            self.time_offset = server_time_ms - local_time_ms

            if abs(self.time_offset) > 1000:  # 1초 이상 차이
                print(
                    f"⚠️  시간 동기화: 로컬 시간이 서버보다 {self.time_offset / 1000:.1f}초 {'느림' if self.time_offset > 0 else '빠름'}")
        except Exception as e:
            print(f"⚠️  시간 동기화 실패 (계속 진행): {e}")
            self.time_offset = 0

    def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False) -> dict:
        """API 요청 (Bybit V5)"""
        if params is None:
            params = {}

        headers = {
            "Content-Type": "application/json"
        }

        url = f"{self.base_url}{endpoint}"
        json_body = ""

        if signed:
            # 서버 시간과 동기화된 타임스탬프 사용
            timestamp = str(int(time.time() * 1000) + self.time_offset)

            # Bybit V5 서명 방식
            if method == "GET":
                # GET: query string으로 파라미터 전달
                query_string = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())]) if params else ""
                sign_payload = timestamp + self.api_key + self.recv_window + query_string
            else:
                # POST: JSON body로 파라미터 전달
                json_body = json.dumps(params, separators=(',', ':')) if params else ""
                sign_payload = timestamp + self.api_key + self.recv_window + json_body

            # 서명 생성
            signature = hmac.new(
                bytes(self.api_secret, 'utf-8'),
                bytes(sign_payload, 'utf-8'),
                hashlib.sha256
            ).hexdigest()

            # 헤더 설정
            headers.update({
                'X-BAPI-API-KEY': self.api_key,
                'X-BAPI-SIGN': signature,
                'X-BAPI-TIMESTAMP': timestamp,
                'X-BAPI-RECV-WINDOW': self.recv_window
            })

        try:
            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=10)
            else:
                # POST: 서명에 사용한 json_body를 그대로 전송
                if signed:
                    response = requests.post(url, data=json_body, headers=headers, timeout=10)
                else:
                    response = requests.post(url, json=params, headers=headers, timeout=10)

            response.raise_for_status()
            return response.json()
        except Exception as e:
            if DEBUG_MODE:
                print(f"API 요청 실패: {e}")
            return {"retCode": -1, "retMsg": str(e)}

    def get_ticker(self, symbol: str) -> dict:
        """현재가 조회"""
        return self._request("GET", "/v5/market/tickers",
                             {"category": "linear", "symbol": symbol})

    def get_klines(self, symbol: str, interval: str = "5", limit: int = 200) -> pd.DataFrame:
        """캔들 데이터 조회"""
        result = self._request("GET", "/v5/market/kline",
                               {"category": "linear", "symbol": symbol,
                                "interval": interval, "limit": limit})

        if result.get("retCode") == 0 and result.get("result", {}).get("list"):
            df = pd.DataFrame(result["result"]["list"],
                              columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
            df = df.astype({"open": float, "high": float, "low": float,
                            "close": float, "volume": float, "turnover": float})
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
        return pd.DataFrame()

    def get_balance(self) -> float:
        """
        현재 USDT 잔고 조회 (사용 가능한 금액)
        Returns:
            float: 사용 가능한 USDT 잔고
        """
        try:
            result = self._request("GET", "/v5/account/wallet-balance",
                                   {"accountType": "CONTRACT"}, signed=True)

            if result.get("retCode") == 0:
                coins = result.get("result", {}).get("list", [{}])[0].get("coin", [])
                for coin in coins:
                    if coin.get("coin") == "USDT":
                        # availableToWithdraw: 출금 가능 금액 (포지션에 사용중인 금액 제외)
                        available_balance = float(coin.get("availableToWithdraw", 0))
                        return available_balance

            if DEBUG_MODE:
                print(f"⚠️  잔고 조회 실패: {result.get('retMsg')}")
            return 0.0

        except Exception as e:
            if DEBUG_MODE:
                print(f"❌ 잔고 조회 오류: {e}")
            return 0.0

    def get_instrument_info(self, symbol: str) -> dict:
        """심볼 정보 조회 (최소/최대 주문 수량, 가격 단위 등)"""
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        result = self._request("GET", "/v5/market/instruments-info",
                               {"category": "linear", "symbol": symbol})

        if result.get("retCode") == 0 and result.get("result", {}).get("list"):
            info = result["result"]["list"][0]
            lot_size_filter = info.get("lotSizeFilter", {})
            price_filter = info.get("priceFilter", {})

            instrument_info = {
                "min_qty": float(lot_size_filter.get("minOrderQty", 0)),
                "max_qty": float(lot_size_filter.get("maxOrderQty", 0)),
                "qty_step": float(lot_size_filter.get("qtyStep", 0.01)),
                "tick_size": float(price_filter.get("tickSize", 0.01))
            }

            self._instrument_cache[symbol] = instrument_info
            return instrument_info

        return {"min_qty": 0.01, "max_qty": 10000, "qty_step": 0.01, "tick_size": 0.01}

    def get_positions(self) -> List[dict]:
        """현재 보유 포지션 조회"""
        result = self._request("GET", "/v5/position/list",
                               {"category": "linear", "settleCoin": "USDT"}, signed=True)

        if result.get("retCode") == 0:
            return result.get("result", {}).get("list", [])
        return []

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """레버리지 설정"""
        result = self._request("POST", "/v5/position/set-leverage",
                               {"category": "linear", "symbol": symbol,
                                "buyLeverage": str(leverage), "sellLeverage": str(leverage)},
                               signed=True)
        return result.get("retCode") == 0

    def place_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> dict:
        """시장가 주문"""
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "GTC",
            "positionIdx": 0  # 단방향 모드
        }

        if reduce_only:
            params["reduceOnly"] = True

        return self._request("POST", "/v5/order/create", params, signed=True)

    def cancel_all_orders(self, symbol: str) -> dict:
        """모든 주문 취소"""
        return self._request("POST", "/v5/order/cancel-all",
                             {"category": "linear", "symbol": symbol}, signed=True)


# ===== TCN 모델 =====
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                        stride=1, dilation=dilation, padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RegressionTCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super(RegressionTCN, self).__init__()
        self.tcn = TCN(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        y = self.tcn(x)
        y = y[:, :, -1]
        return self.linear(y)


# ===== 피처 생성 =====
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 생성"""
    df = df.copy()

    # 가격 변화율
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # 이동평균
    for period in [5, 10, 20, 50]:
        df[f"sma_{period}"] = df["close"].rolling(period).mean()
        df[f"ema_{period}"] = df["close"].ewm(span=period).mean()

    # 볼린저 밴드
    df["bb_middle"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ATR (Average True Range)
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["atr"] = true_range.rolling(14).mean()

    # 거래량 지표
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    # 모멘텀
    df["momentum"] = df["close"] - df["close"].shift(10)
    df["roc"] = (df["close"] - df["close"].shift(10)) / df["close"].shift(10) * 100

    # Stochastic
    low_min = df["low"].rolling(14).min()
    high_max = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # 가격 위치
    df["price_position"] = (df["close"] - df["low"].rolling(20).min()) / \
                           (df["high"].rolling(20).max() - df["low"].rolling(20).min())

    return df.dropna()


# ===== 모델 로딩 및 예측 =====
MODELS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(symbol: str):
    """모델 로딩"""
    if symbol in MODELS:
        return MODELS[symbol]

    model_path = MODEL_PATHS.get(symbol)
    if not model_path or not os.path.exists(model_path):
        return None

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

        # 하이퍼파라미터
        input_size = checkpoint.get("input_size", 30)
        num_channels = checkpoint.get("num_channels", [64, 128, 256])

        model = RegressionTCN(input_size, num_channels).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        MODELS[symbol] = {
            "model": model,
            "input_size": input_size,
            "feature_cols": checkpoint.get("feature_cols", [])
        }

        return MODELS[symbol]
    except Exception as e:
        print(f"❌ 모델 로딩 실패 ({symbol}): {e}")
        return None


def predict(symbol: str) -> dict:
    """예측"""
    try:
        # 모델 로딩
        model_data = load_model(symbol)
        if not model_data:
            return {"error": "모델 로딩 실패"}

        model = model_data["model"]
        feature_cols = model_data["feature_cols"]

        # 데이터 조회
        df = API.get_klines(symbol, interval="5", limit=200)
        if df.empty:
            return {"error": "데이터 조회 실패"}

        # 피처 생성
        df = calculate_features(df)
        if len(df) < 50:
            return {"error": "데이터 부족"}

        # 최신 50개 시퀀스
        X = df[feature_cols].values[-50:]
        X = torch.FloatTensor(X).unsqueeze(0).transpose(1, 2).to(DEVICE)

        # 예측
        with torch.no_grad():
            pred_bps = model(X).item()

        # 신호 생성
        if pred_bps > STRONG_THRESHOLD:
            signal = "LONG"
            confidence = "strong"
        elif pred_bps > LONG_THRESHOLD:
            signal = "LONG"
            confidence = "normal"
        elif pred_bps < -STRONG_THRESHOLD:
            signal = "SHORT"
            confidence = "strong"
        elif pred_bps < -SHORT_THRESHOLD:
            signal = "SHORT"
            confidence = "normal"
        else:
            signal = "HOLD"
            confidence = "weak"

        return {
            "signal": signal,
            "predicted_bps": pred_bps,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": f"예측 오류: {e}"}


# ===== 트레이딩 매니저 =====
class TradingManager:
    def __init__(self, api: BybitAPI):
        self.api = api
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.load_trades()

    def load_trades(self):
        """거래 기록 로딩"""
        if os.path.exists(TRADE_LOG_FILE):
            try:
                with open(TRADE_LOG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.trades = [Trade(**t) for t in data]
            except:
                pass

    def save_trade(self, trade: Trade):
        """거래 기록 저장"""
        self.trades.append(trade)
        with open(TRADE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([asdict(t) for t in self.trades], f, indent=2, ensure_ascii=False)

    def sync_positions(self):
        """실제 포지션과 동기화"""
        try:
            positions = self.api.get_positions()

            active_symbols = set()
            for pos_data in positions:
                size = float(pos_data.get("size", 0))
                if size == 0:
                    continue

                symbol = pos_data["symbol"]
                active_symbols.add(symbol)

                if symbol not in self.positions:
                    # 외부에서 진입한 포지션 추가
                    side = pos_data.get("side", "")
                    entry_price = float(pos_data.get("avgPrice", 0))

                    self.positions[symbol] = Position(
                        symbol=symbol,
                        direction=side,
                        entry_price=entry_price,
                        size=size,
                        entry_time=datetime.now(),
                        stop_loss=0,
                        take_profit=0,
                        leverage=int(pos_data.get("leverage", LEVERAGE)),
                        margin=float(pos_data.get("positionIM", 0)),
                        predicted_bps=0,
                        confidence="unknown"
                    )

            # 청산된 포지션 제거
            for symbol in list(self.positions.keys()):
                if symbol not in active_symbols:
                    del self.positions[symbol]

        except Exception as e:
            if DEBUG_MODE:
                print(f"포지션 동기화 오류: {e}")

    def can_open_position(self) -> bool:
        """새 포지션 진입 가능 여부"""
        if len(self.positions) >= MAX_POSITIONS:
            return False
        if self.daily_pnl <= -MAX_DAILY_LOSS:
            print("⚠️  일일 손실 한도 도달")
            return False
        return True

    def calculate_position_size(self, symbol: str, price: float, confidence: str) -> tuple:
        """
        소지금 비율 기반 포지션 사이즈 계산
        Returns:
            (증거금, 수량) tuple
        """
        try:
            # 현재 잔고 조회
            balance = self.api.get_balance()

            if balance <= 0:
                print(f"  ⚠️  잔고 부족 또는 조회 실패: ${balance:.2f}")
                return 0, 0

            # 신호 강도에 따른 비율 선택
            position_pct = STRONG_POSITION_SIZE_PCT if confidence == "strong" else POSITION_SIZE_PCT

            # 증거금 계산 (소지금의 X%)
            margin = balance * (position_pct / 100)

            # 최소/최대 제한 적용
            margin = max(MIN_MARGIN, min(margin, MAX_MARGIN))

            # 포지션 크기 계산 (레버리지 적용)
            position_value = margin * LEVERAGE
            qty = position_value / price

            # 심볼별 최소/최대 수량 제한
            instrument = self.api.get_instrument_info(symbol)
            qty_step = instrument["qty_step"]
            min_qty = instrument["min_qty"]
            max_qty = instrument["max_qty"]

            # 수량 반올림 (거래소 규칙에 맞춤)
            qty = round(qty / qty_step) * qty_step
            qty = max(min_qty, min(qty, max_qty))

            # 실제 사용될 증거금 재계산
            actual_margin = (qty * price) / LEVERAGE

            if DEBUG_MODE:
                print(f"  💰 잔고: ${balance:.2f} | 사용 비율: {position_pct}%")
                print(f"  📊 증거금: ${actual_margin:.2f} | 수량: {qty:.4f}")

            return actual_margin, qty

        except Exception as e:
            print(f"  ❌ 포지션 사이즈 계산 오류: {e}")
            return 0, 0

    def open_position(self, symbol: str, signal: str, price: float, predicted_bps: float, confidence: str):
        """포지션 진입"""
        try:
            # 포지션 사이즈 계산 (비율 기반)
            margin, qty = self.calculate_position_size(symbol, price, confidence)

            if margin <= 0 or qty <= 0:
                print(f"  ⚠️  {symbol}: 포지션 사이즈 계산 실패")
                return False

            # 레버리지 설정
            self.api.set_leverage(symbol, LEVERAGE)

            # 주문 실행
            side = "Buy" if signal == "LONG" else "Sell"
            result = self.api.place_order(symbol, side, qty)

            if result.get("retCode") != 0:
                print(f"  ❌ {symbol} 주문 실패: {result.get('retMsg')}")
                return False

            # 손절/익절 설정
            if signal == "LONG":
                stop_loss = price * (1 - STOP_LOSS_PCT)
                take_profit = price * (1 + TAKE_PROFIT_PCT)
            else:
                stop_loss = price * (1 + STOP_LOSS_PCT)
                take_profit = price * (1 - TAKE_PROFIT_PCT)

            # 포지션 저장
            self.positions[symbol] = Position(
                symbol=symbol,
                direction=side,
                entry_price=price,
                size=qty,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=LEVERAGE,
                margin=margin,
                predicted_bps=predicted_bps,
                confidence=confidence,
                order_id=result.get("result", {}).get("orderId", "")
            )

            emoji = "🟢🔥" if confidence == "strong" else ("🟢" if signal == "LONG" else "🔴")
            print(f"  {emoji} 포지션 진입: {symbol} {side}")
            print(f"     가격: ${price:,.4f} | 수량: {qty:.4f}")
            print(f"     증거금: ${margin:.2f} ({confidence})")
            print(f"     예상 수익: {predicted_bps:+.2f} bps")
            return True

        except Exception as e:
            print(f"  ❌ 포지션 진입 오류: {e}")
            return False

    def close_position(self, symbol: str, reason: str):
        """포지션 청산"""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # 현재가 조회
        ticker = self.api.get_ticker(symbol)
        if ticker.get("retCode") != 0:
            return

        current_price = float(ticker["result"]["list"][0]["lastPrice"])

        # 청산 주문 (reduce_only=True)
        side = "Sell" if pos.direction == "Buy" else "Buy"
        result = self.api.place_order(symbol, side, pos.size, reduce_only=True)

        if result.get("retCode") == 0:
            pnl = pos.get_unrealized_pnl(current_price)
            roe = pos.get_roe(current_price)

            if pos.direction == "Buy":
                actual_bps = (current_price / pos.entry_price - 1) * 10000
            else:
                actual_bps = (1 - current_price / pos.entry_price) * 10000

            trade = Trade(
                symbol=symbol,
                direction=pos.direction,
                entry_price=pos.entry_price,
                exit_price=current_price,
                size=pos.size,
                entry_time=pos.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                exit_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                pnl=pnl,
                roe=roe,
                exit_reason=reason,
                predicted_bps=pos.predicted_bps,
                actual_bps=actual_bps
            )

            self.save_trade(trade)
            self.daily_pnl += pnl
            del self.positions[symbol]

            emoji = "💀" if reason == "Liquidation" else ("🔴" if pnl < 0 else "🟢")
            print(f"  {emoji} 포지션 청산: {reason}")
            print(f"     손익: ${pnl:+,.2f} ({roe:+.2f}% ROE)")
        else:
            print(f"  ❌ 청산 실패: {result.get('retMsg')}")

    def manage_positions(self):
        """포지션 관리 (손절/익절/시간)"""
        current_time = datetime.now()

        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]

            # 현재가 조회
            ticker = self.api.get_ticker(symbol)
            if ticker.get("retCode") != 0:
                continue

            current_price = float(ticker["result"]["list"][0]["lastPrice"])
            hold_minutes = (current_time - pos.entry_time).total_seconds() / 60

            # 최소 보유 시간 체크
            if hold_minutes < MIN_HOLD_MINUTES:
                continue

            # 손절/익절 체크
            if pos.direction == "Buy":
                if current_price <= pos.stop_loss:
                    self.close_position(symbol, "Stop Loss")
                elif current_price >= pos.take_profit:
                    self.close_position(symbol, "Take Profit")
                elif hold_minutes >= MAX_HOLD_MINUTES:
                    self.close_position(symbol, "Max Hold Time")
            else:
                if current_price >= pos.stop_loss:
                    self.close_position(symbol, "Stop Loss")
                elif current_price <= pos.take_profit:
                    self.close_position(symbol, "Take Profit")
                elif hold_minutes >= MAX_HOLD_MINUTES:
                    self.close_position(symbol, "Max Hold Time")

    def get_stats(self):
        """통계"""
        if not self.trades:
            return {"total": 0, "wins": 0, "win_rate": 0, "total_pnl": 0, "avg_roe": 0}

        wins = [t for t in self.trades if t.pnl > 0]
        total_pnl = sum(t.pnl for t in self.trades)

        return {
            "total": len(self.trades),
            "wins": len(wins),
            "win_rate": len(wins) / len(self.trades) * 100,
            "total_pnl": total_pnl,
            "avg_roe": sum(t.roe for t in self.trades) / len(self.trades)
        }


# ===== 메인 =====
def main():
    print("\n" + "=" * 100)
    print(f"{'🚨 실거래 시스템 시작':^100}")
    print("=" * 100)
    print(f"모드: {'🔧 Testnet' if USE_TESTNET else '💰 실거래 (LIVE!)'}")
    print(f"레버리지: {LEVERAGE}x | 최대 포지션: {MAX_POSITIONS}")
    print(f"포지션 크기: 소지금의 {POSITION_SIZE_PCT}% (강한 신호: {STRONG_POSITION_SIZE_PCT}%)")
    print(f"임계값: LONG > {LONG_THRESHOLD} bps, SHORT < -{SHORT_THRESHOLD} bps")
    print(f"일일 최대 손실: ${MAX_DAILY_LOSS}")
    print("=" * 100)

    # 초기 잔고 확인
    initial_balance = API.get_balance()
    print(f"\n💰 현재 USDT 잔고: ${initial_balance:,.2f}")
    print(f"   일반 신호 시 사용 금액: ${initial_balance * POSITION_SIZE_PCT / 100:,.2f}")
    print(f"   강한 신호 시 사용 금액: ${initial_balance * STRONG_POSITION_SIZE_PCT / 100:,.2f}\n")

    manager = TradingManager(API)
    iteration = 0

    try:
        while True:
            iteration += 1
            clear_screen()

            print(f"\n{'=' * 100}")
            print(f"{'🤖 실시간 자동 트레이딩':^100}")
            print(f"{'=' * 100}")
            print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 반복: {iteration}")

            # 현재 잔고 표시
            current_balance = API.get_balance()
            print(f"💰 잔고: ${current_balance:,.2f} | 일일 PnL: ${manager.daily_pnl:+,.2f}")
            print(f"{'=' * 100}\n")

            # 포지션 동기화
            manager.sync_positions()

            # 포지션 관리
            manager.manage_positions()

            # 포지션 현황
            print(f"📊 포지션: {len(manager.positions)}/{MAX_POSITIONS}")
            if manager.positions:
                print(f"{'─' * 100}")
                for symbol, pos in manager.positions.items():
                    ticker = API.get_ticker(symbol)
                    if ticker.get("retCode") == 0:
                        price = float(ticker["result"]["list"][0]["lastPrice"])
                        pnl = pos.get_unrealized_pnl(price)
                        roe = pos.get_roe(price)
                        hold_min = (datetime.now() - pos.entry_time).total_seconds() / 60

                        dir_emoji = "🟢" if pos.direction == "Buy" else "🔴"
                        pnl_emoji = "📈" if pnl > 0 else "📉"

                        print(f"{dir_emoji} {symbol} | ${pos.entry_price:,.4f} → ${price:,.4f}")
                        print(
                            f"   {pnl_emoji} ${pnl:+,.2f} ({roe:+.2f}% ROE) | ⏱️  {hold_min:.1f}분 | 증거금: ${pos.margin:.2f}\n")

            # 신호 확인
            print(f"\n🔍 신호 모니터링:")
            print(f"{'─' * 100}")

            for symbol in SYMBOLS:
                symbol = symbol.strip()

                if symbol in manager.positions:
                    print(f"  ⏸️  {symbol}: 포지션 보유중")
                    continue

                pred = predict(symbol)

                if pred.get("error"):
                    print(f"  ⚠️  {symbol}: {pred['error']}")
                    continue

                sig = pred["signal"]
                bps = pred["predicted_bps"]
                conf = pred["confidence"]

                if sig == "LONG":
                    emoji = "🟢🔥" if conf == "strong" else "🟢"
                elif sig == "SHORT":
                    emoji = "🔴🔥" if conf == "strong" else "🔴"
                else:
                    emoji = "⚪"

                print(f"  {emoji} {symbol}: {sig} ({conf}) | {bps:+.2f} bps")

                # 진입 시도
                if sig != "HOLD" and manager.can_open_position():
                    ticker = API.get_ticker(symbol)
                    if ticker.get("retCode") == 0:
                        price = float(ticker["result"]["list"][0]["lastPrice"])
                        manager.open_position(symbol, sig, price, bps, conf)

            # 통계
            stats = manager.get_stats()
            print(f"\n{'─' * 100}")
            print(f"📈 통계: 거래 {stats['total']} | 승률 {stats['win_rate']:.1f}% | 총PnL ${stats['total_pnl']:+,.2f}")

            print(f"\n{'=' * 100}")
            print(f"다음 업데이트: {INTERVAL_SEC}초 후...")

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        clear_screen()
        print("\n프로그램 종료\n")
        stats = manager.get_stats()
        final_balance = API.get_balance()
        print(f"{'=' * 100}")
        print(f"{'📊 최종 통계':^100}")
        print(f"{'=' * 100}")
        print(f"최종 잔고: ${final_balance:,.2f}")
        print(f"총 거래: {stats['total']} | 승률: {stats['win_rate']:.1f}%")
        print(f"총 손익: ${stats['total_pnl']:+,.2f} | 평균 ROE: {stats['avg_roe']:+.2f}%")
        print(f"{'=' * 100}\n")


if __name__ == "__main__":
    # API 초기화
    API = BybitAPI(API_KEY, API_SECRET, testnet=USE_TESTNET)
    main()