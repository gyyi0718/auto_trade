# bybit_live_trading.py
# -*- coding: utf-8 -*-
"""
TCN 모델 기반 바이비트 실거래 시스템 (심볼별 모델 사용)
- 각 심볼마다 별도의 딥러닝 모델 로드
- 실시간 신호 기반 자동 매매
- 레버리지 거래
- 포지션 관리 및 손익 계산
- 거래 내역 저장

⚠️ 주의사항:
1. 실제 자금이 거래되므로 신중하게 사용하세요
2. API Key와 Secret을 안전하게 보관하세요
3. 처음에는 소액으로 테스트하세요
4. 레버리지를 낮게 설정하여 리스크를 관리하세요
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
from pybit.unified_trading import HTTP

warnings.filterwarnings("ignore")
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# ===== CONFIG =====
# Bybit API 인증 정보 (환경변수로 설정 권장)
API_KEY = "Rk7Vjkn1UqiPNVTwEN"
API_SECRET = "2bJSHW3XieOTU9SB2iDdkTlnmFuEA2EG3Rxf"
USE_TESTNET = os.getenv("USE_TESTNET", "0") == "1"  # 기본값: 테스트넷

# 거래 설정
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "5"))

# ✅ 심볼별 모델 경로 설정
MODEL_PATHS = {
    "BTCUSDT": "D:/ygy_work/coin/multimodel/models_5min_btc/5min_2class_best.ckpt",
    "ETHUSDT": "D:/ygy_work/coin/multimodel/models_5min_eth/5min_2class_best.ckpt",
    "SOLUSDT": "D:/ygy_work/coin/multimodel/models_5min_sol/5min_2class_best.ckpt",
    "DOGEUSDT": "D:/ygy_work/coin/multimodel/models_5min_doge/5min_2class_best.ckpt",
    "BNBUSDT": "D:/ygy_work/coin/multimodel/models_5min_bnb/5min_2class_best.ckpt",
    "XRPUSDT": "D:/ygy_work/coin/multimodel/models_5min_xrp/5min_2class_best.ckpt",
    "SAPIENUSDT": "D:/ygy_work/coin/multimodel/models_5min_sapien_v2/model_v2_best.pt",
    "FLMUSDT": "D:/ygy_work/coin/multimodel/models_5min_flm_v2/model_v2_best.pt",
    "TRUMPUSDT": "D:/ygy_work/coin/multimodel/models_5min_trump/5min_2class_best.ckpt",
    "JELLYJELLYUSDT": "D:/ygy_work/coin/multimodel/models_minutes_jellyjelly/5min_2class_best.ckpt",
    "ARCUSDT": "D:/ygy_work/coin/multimodel/models_5min_arc/5min_2class_best.ckpt",
    "DASHUSDT": "D:/ygy_work/coin/multimodel/models_5min_dash/5min_2class_best.ckpt",
    "MMTUSDT": "D:/ygy_work/coin/multimodel/models_5min_mmt/5min_2class_best.ckpt",
    "AIAUSDT": "D:/ygy_work/coin/multimodel/models_5min_aia/5min_2class_best.ckpt",
    "GIGGLEUSDT": "D:/ygy_work/coin/multimodel/models_5min_giggle/5min_2class_best.ckpt",
    "XNOUSDT": "D:/ygy_work/coin/multimodel/models_5min_xno/5min_2class_best.ckpt",
    "SOONUSDT": "D:/ygy_work/coin/multimodel/models_5min_soon/5min_2class_best.ckpt"

}

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.55"))

# 거래 설정
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "10"))  # 포지션 크기 (USDT)
LEVERAGE = int(os.getenv("LEVERAGE", "20"))  # 레버리지 배율
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))  # 최대 동시 포지션
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))  # 손절 (2%)
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.03"))  # 익절 (3%)
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "30"))  # 최대 보유 시간
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "live_trades.json")

# 안전 설정
DRY_RUN = os.getenv("DRY_RUN", "1") == "1"  # 기본값: 드라이런 (실제 주문 X)
MIN_BALANCE_USDT = float(os.getenv("MIN_BALANCE_USDT", "50"))  # 최소 잔고


# ===== Bybit API 클래스 =====
class BybitAPI:
    """Bybit API 래퍼"""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.testnet = testnet
        self.session = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret,
        )
        print(f"🔗 Bybit {'테스트넷' if testnet else '실거래'} 연결")

    def get_ticker(self, symbol: str) -> dict:
        """현재 가격 조회"""
        try:
            response = self.session.get_tickers(
                category="linear",
                symbol=symbol
            )
            return response
        except Exception as e:
            return {"error": str(e)}

    def get_klines(self, symbol: str, interval: str = "5", limit: int = 200) -> pd.DataFrame:
        """캔들 데이터 조회"""
        try:
            response = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )

            if response.get("retCode") != 0:
                return pd.DataFrame()

            data = response["result"]["list"]
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])

            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            df = df.sort_values("timestamp").reset_index(drop=True)
            return df

        except Exception as e:
            print(f"❌ 캔들 조회 오류 ({symbol}): {e}")
            return pd.DataFrame()

    def get_balance(self) -> float:
        """USDT 잔고 조회"""
        try:
            response = self.session.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )

            if response.get("retCode") != 0:
                return 0.0

            balance = float(response["result"]["list"][0]["coin"][0]["walletBalance"])
            return balance

        except Exception as e:
            print(f"❌ 잔고 조회 오류: {e}")
            return 0.0

    def get_positions(self) -> List[dict]:
        """열린 포지션 조회"""
        try:
            response = self.session.get_positions(
                category="linear",
                settleCoin="USDT"
            )

            if response.get("retCode") != 0:
                return []

            positions = []
            for pos in response["result"]["list"]:
                if float(pos["size"]) > 0:  # 포지션이 있는 경우만
                    positions.append({
                        "symbol": pos["symbol"],
                        "side": pos["side"],  # Buy or Sell
                        "size": float(pos["size"]),
                        "avgPrice": float(pos["avgPrice"]),
                        "leverage": float(pos["leverage"]),
                        "unrealisedPnl": float(pos["unrealisedPnl"]),
                        "positionValue": float(pos["positionValue"])
                    })

            return positions

        except Exception as e:
            print(f"❌ 포지션 조회 오류: {e}")
            return []

    def set_leverage(self, symbol: str, leverage: int):
        """레버리지 설정"""
        try:
            response = self.session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            return response.get("retCode") == 0
        except Exception as e:
            print(f"❌ 레버리지 설정 오류 ({symbol}): {e}")
            return False

    def place_order(self, symbol: str, side: str, qty: float,
                    stop_loss: float = None, take_profit: float = None) -> dict:
        """시장가 주문"""
        try:
            # 주문 파라미터
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,  # "Buy" or "Sell"
                "orderType": "Market",
                "qty": str(qty),
                "timeInForce": "GTC",
            }

            # 손절/익절 설정
            if stop_loss:
                params["stopLoss"] = str(stop_loss)
            if take_profit:
                params["takeProfit"] = str(take_profit)

            response = self.session.place_order(**params)
            return response

        except Exception as e:
            print(f"❌ 주문 오류 ({symbol}): {e}")
            return {"error": str(e)}

    def close_position(self, symbol: str, side: str, qty: float) -> dict:
        """포지션 청산"""
        # 청산은 반대 방향 주문
        close_side = "Sell" if side == "Buy" else "Buy"
        return self.place_order(symbol, close_side, qty)

    def cancel_all_orders(self, symbol: str = None):
        """모든 오픈 주문 취소"""
        try:
            params = {"category": "linear"}
            if symbol:
                params["symbol"] = symbol

            response = self.session.cancel_all_orders(**params)
            return response
        except Exception as e:
            print(f"❌ 주문 취소 오류: {e}")
            return {"error": str(e)}


# ===== TCN 모델 정의 (기존 코드와 동일) =====
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(self, input_size=5, num_channels=[64, 64, 64, 64], kernel_size=3,
                 dropout=0.2, num_classes=3):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        y = self.tcn(x)
        y = y[:, :, -1]
        return self.fc(y)


# ===== 모델 로딩 =====
def load_models() -> Dict[str, dict]:
    """심볼별 모델 및 메타데이터 로드"""
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for symbol, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"⚠️  모델 파일 없음: {symbol} ({path})")
            continue

        try:
            # 체크포인트 로드
            checkpoint = torch.load(path, map_location=device)

            # 체크포인트 구조 파악
            if "model" in checkpoint:
                # 학습 코드 형식: {"model": state_dict, "feat_cols": ..., "meta": ..., "scaler_mu": ..., "scaler_sd": ...}
                state_dict = checkpoint["model"]
                meta = checkpoint.get("meta", {})
                num_classes = meta.get("num_classes", 2)
                scaler_mu = checkpoint.get("scaler_mu", None)
                scaler_sd = checkpoint.get("scaler_sd", None)
                feat_cols = checkpoint.get("feat_cols", None)

                print(f"   {symbol}: 학습 메타데이터 발견")
                print(f"      - 클래스 수: {num_classes}")
                print(f"      - 스케일러: {'있음' if scaler_mu is not None else '없음'}")
                if feat_cols:
                    print(f"      - 특성: {feat_cols}")

            elif "state_dict" in checkpoint:
                # Lightning 형식: {"state_dict": ...}
                state_dict = {k.replace("model.", ""): v
                              for k, v in checkpoint["state_dict"].items()}
                num_classes = 2
                scaler_mu = None
                scaler_sd = None
                feat_cols = None
            else:
                # 순수 state_dict
                state_dict = checkpoint
                num_classes = 2
                scaler_mu = None
                scaler_sd = None
                feat_cols = None

            # 모델 생성
            model = TCNModel(
                input_size=5,
                num_channels=[64, 64, 64, 64],
                kernel_size=3,
                dropout=0.2,
                num_classes=num_classes
            ).to(device)

            # 가중치 로드
            model.load_state_dict(state_dict)
            model.eval()

            # 모델 및 메타데이터 저장
            models[symbol] = {
                "model": model,
                "num_classes": num_classes,
                "scaler_mu": scaler_mu,
                "scaler_sd": scaler_sd,
                "feat_cols": feat_cols,
                "meta": meta if "model" in checkpoint else {}
            }

            print(f"✅ 모델 로드 완료: {symbol}")

        except Exception as e:
            print(f"❌ 모델 로드 실패: {symbol}")
            print(f"   오류: {e}")
            import traceback
            traceback.print_exc()

    return models


# ===== 데이터 전처리 및 예측 =====
def prepare_features(df: pd.DataFrame, lookback: int = 60,
                     scaler_mu: np.ndarray = None, scaler_sd: np.ndarray = None) -> Optional[torch.Tensor]:
    """특성 준비 및 정규화"""
    if len(df) < lookback:
        return None

    df = df.copy()

    # 기술적 지표 계산
    df['returns'] = df['close'].pct_change()
    df['log_volume'] = np.log1p(df['volume'])

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # 볼린저 밴드
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # 이동평균
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # 5가지 특성 선택
    feature_df = pd.DataFrame({
        'returns': df['returns'],
        'log_volume': df['log_volume'],
        'rsi': df['rsi'] / 100,
        'bb_position': df['bb_position'],
        'sma_ratio': (df['close'] - df['sma_20']) / df['sma_20']
    })

    feature_df = feature_df.fillna(0).replace([np.inf, -np.inf], 0)
    feature_array = feature_df.values[-lookback:]

    # 정규화 (학습 시 사용한 스케일러 우선 사용)
    if scaler_mu is not None and scaler_sd is not None:
        # 학습 시 스케일러 사용
        feature_array = (feature_array - scaler_mu) / scaler_sd
    else:
        # 자체 정규화
        mean = feature_array.mean(axis=0)
        std = feature_array.std(axis=0) + 1e-8
        feature_array = (feature_array - mean) / std

    # 클리핑
    feature_array = np.clip(feature_array, -10.0, 10.0)

    # Tensor 변환
    tensor = torch.FloatTensor(feature_array).T.unsqueeze(0)
    return tensor


def predict(api: BybitAPI, symbol: str, model_info: dict, device: torch.device) -> dict:
    """예측 실행"""
    try:
        # 모델 및 메타데이터 추출
        model = model_info["model"]
        num_classes = model_info["num_classes"]
        scaler_mu = model_info.get("scaler_mu")
        scaler_sd = model_info.get("scaler_sd")

        # 데이터 가져오기
        df = api.get_klines(symbol, interval="5", limit=200)
        if df.empty:
            return {"error": "데이터 없음"}

        # 특성 준비 (스케일러 활용)
        features = prepare_features(df, lookback=60,
                                    scaler_mu=scaler_mu,
                                    scaler_sd=scaler_sd)
        if features is None:
            return {"error": "특성 준비 실패"}

        # 예측
        features = features.to(device)
        with torch.no_grad():
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        # 현재 가격
        current_price = float(df['close'].iloc[-1])

        # 방향 결정 (2-class 또는 3-class)
        if num_classes == 2:
            # 2-class: 0=Short, 1=Long
            direction = "Long" if pred_class == 1 else "Short"
        else:
            # 3-class: 0=Short, 1=Flat, 2=Long
            if pred_class == 2:
                direction = "Long"
            elif pred_class == 0:
                direction = "Short"
            else:
                direction = "Flat"

        return {
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "current_price": current_price,
            "pred_class": pred_class,
            "num_classes": num_classes
        }

    except Exception as e:
        return {"error": str(e)}


# ===== 거래 관리 =====
@dataclass
class Trade:
    """거래 기록"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float
    exit_reason: str


class TradingManager:
    """거래 관리자"""

    def __init__(self, api: BybitAPI):
        self.api = api
        self.trades: List[Trade] = []
        self.active_positions: Dict[str, dict] = {}
        self.position_entry_times: Dict[str, datetime] = {}

    def can_open_position(self, symbol: str) -> tuple[bool, str]:
        """포지션 진입 가능 여부"""
        # 이미 포지션이 있는 경우
        if symbol in self.active_positions:
            return False, "이미 포지션 보유 중"

        # 최대 포지션 수 확인
        if len(self.active_positions) >= MAX_POSITIONS:
            return False, f"최대 포지션 수 초과 ({MAX_POSITIONS})"

        # 잔고 확인
        balance = self.api.get_balance()
        if balance < MIN_BALANCE_USDT:
            return False, f"잔고 부족 (${balance:.2f})"

        if balance < POSITION_SIZE_USDT:
            return False, f"포지션 크기보다 잔고 부족"

        return True, "OK"

    def open_position(self, symbol: str, direction: str, price: float,
                      confidence: float) -> bool:
        """포지션 진입"""
        try:
            # 레버리지 설정
            self.api.set_leverage(symbol, LEVERAGE)

            # 수량 계산
            qty = POSITION_SIZE_USDT * LEVERAGE / price
            qty = round(qty, 3)  # 소수점 3자리

            # 손절/익절 계산
            if direction == "Long":
                side = "Buy"
                stop_loss = price * (1 - STOP_LOSS_PCT)
                take_profit = price * (1 + TAKE_PROFIT_PCT)
            else:  # Short
                side = "Sell"
                stop_loss = price * (1 + STOP_LOSS_PCT)
                take_profit = price * (1 - TAKE_PROFIT_PCT)

            print(f"\n{'=' * 60}")
            print(f"🎯 포지션 진입 시도")
            print(f"   심볼: {symbol}")
            print(f"   방향: {direction} ({side})")
            print(f"   가격: ${price:,.4f}")
            print(f"   수량: {qty}")
            print(f"   레버리지: {LEVERAGE}x")
            print(f"   신뢰도: {confidence:.1%}")
            print(f"   손절: ${stop_loss:,.4f} (-{STOP_LOSS_PCT * 100}%)")
            print(f"   익절: ${take_profit:,.4f} (+{TAKE_PROFIT_PCT * 100}%)")

            if DRY_RUN:
                print(f"   ⚠️  DRY RUN: 실제 주문 미실행")
                order_id = f"DRYRUN_{symbol}_{int(time.time())}"
            else:
                # 실제 주문
                result = self.api.place_order(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

                if result.get("retCode") != 0:
                    print(f"   ❌ 주문 실패: {result.get('retMsg', '알 수 없음')}")
                    return False

                order_id = result["result"]["orderId"]
                print(f"   ✅ 주문 성공: {order_id}")

            # 포지션 기록
            self.active_positions[symbol] = {
                "direction": direction,
                "side": side,
                "entry_price": price,
                "quantity": qty,
                "leverage": LEVERAGE,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "order_id": order_id,
                "confidence": confidence
            }
            self.position_entry_times[symbol] = datetime.now()

            print(f"{'=' * 60}\n")
            return True

        except Exception as e:
            print(f"❌ 포지션 진입 오류: {e}")
            return False

    def close_position(self, symbol: str, reason: str = "Manual"):
        """포지션 청산"""
        if symbol not in self.active_positions:
            return

        try:
            pos = self.active_positions[symbol]

            # 현재 가격 조회
            ticker = self.api.get_ticker(symbol)
            if ticker.get("retCode") != 0:
                print(f"❌ 가격 조회 실패: {symbol}")
                return

            current_price = float(ticker["result"]["list"][0]["lastPrice"])

            print(f"\n{'=' * 60}")
            print(f"🔔 포지션 청산")
            print(f"   심볼: {symbol}")
            print(f"   방향: {pos['direction']}")
            print(f"   진입가: ${pos['entry_price']:,.4f}")
            print(f"   현재가: ${current_price:,.4f}")
            print(f"   사유: {reason}")

            if not DRY_RUN:
                # 실제 청산
                result = self.api.close_position(
                    symbol=symbol,
                    side=pos['side'],
                    qty=pos['quantity']
                )

                if result.get("retCode") != 0:
                    print(f"   ❌ 청산 실패: {result.get('retMsg', '알 수 없음')}")
                    return

                print(f"   ✅ 청산 성공")
            else:
                print(f"   ⚠️  DRY RUN: 실제 청산 미실행")

            # 손익 계산
            if pos['direction'] == "Long":
                pnl = (current_price - pos['entry_price']) * pos['quantity']
            else:
                pnl = (pos['entry_price'] - current_price) * pos['quantity']

            margin = POSITION_SIZE_USDT
            pnl_pct = (pnl / margin) * 100

            print(f"   손익: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
            print(f"{'=' * 60}\n")

            # 거래 기록
            entry_time = self.position_entry_times.get(symbol, datetime.now())
            trade = Trade(
                symbol=symbol,
                direction=pos['direction'],
                entry_price=pos['entry_price'],
                exit_price=current_price,
                quantity=pos['quantity'],
                leverage=pos['leverage'],
                entry_time=entry_time.isoformat(),
                exit_time=datetime.now().isoformat(),
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason=reason
            )
            self.trades.append(trade)

            # 포지션 제거
            del self.active_positions[symbol]
            if symbol in self.position_entry_times:
                del self.position_entry_times[symbol]

        except Exception as e:
            print(f"❌ 포지션 청산 오류: {e}")

    def check_positions(self):
        """포지션 관리 (손절/익절/시간 체크)"""
        for symbol in list(self.active_positions.keys()):
            try:
                pos = self.active_positions[symbol]

                # 현재 가격 조회
                ticker = self.api.get_ticker(symbol)
                if ticker.get("retCode") != 0:
                    continue

                current_price = float(ticker["result"]["list"][0]["lastPrice"])

                # 시간 체크
                entry_time = self.position_entry_times.get(symbol, datetime.now())
                hold_minutes = (datetime.now() - entry_time).total_seconds() / 60

                # 청산 조건 확인
                should_close = False
                reason = ""

                # 손절/익절
                if pos['direction'] == "Long":
                    if current_price <= pos['stop_loss']:
                        should_close = True
                        reason = "Stop Loss"
                    elif current_price >= pos['take_profit']:
                        should_close = True
                        reason = "Take Profit"
                else:  # Short
                    if current_price >= pos['stop_loss']:
                        should_close = True
                        reason = "Stop Loss"
                    elif current_price <= pos['take_profit']:
                        should_close = True
                        reason = "Take Profit"

                # 시간 초과
                if hold_minutes >= MAX_HOLD_MINUTES:
                    should_close = True
                    reason = "Time Limit"

                if should_close:
                    self.close_position(symbol, reason)

            except Exception as e:
                print(f"❌ 포지션 체크 오류 ({symbol}): {e}")

    def save_trades(self):
        """거래 내역 저장"""
        if not self.trades:
            return

        data = [asdict(t) for t in self.trades]
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 거래 내역 저장: {TRADE_LOG_FILE}")

    def get_stats(self) -> dict:
        """거래 통계"""
        if not self.trades:
            return {}

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.trades) * 100 if self.trades else 0,
            "total_pnl": sum(t.pnl for t in self.trades),
            "avg_pnl": np.mean([t.pnl for t in self.trades]),
            "avg_pnl_pct": np.mean([t.pnl_pct for t in self.trades]),
            "max_win": max([t.pnl for t in wins]) if wins else 0,
            "max_loss": min([t.pnl for t in losses]) if losses else 0,
        }


# ===== 대시보드 =====
def print_dashboard(manager: TradingManager, api: BybitAPI):
    """대시보드 출력"""
    balance = api.get_balance()

    print("\n" + "=" * 100)
    print(f"{'💰 계좌 현황':^100}")
    print("=" * 100)
    print(f"잔고: ${balance:,.2f} USDT")

    # 포지션 현황
    if manager.active_positions:
        print(f"\n📍 보유 포지션 ({len(manager.active_positions)}개)")
        print(f"{'심볼':^12} | {'방향':^8} | {'진입가':^12} | {'현재가':^12} | "
              f"{'손익':^15} | {'보유시간':^10}")
        print("-" * 100)

        for symbol, pos in manager.active_positions.items():
            ticker = api.get_ticker(symbol)
            if ticker.get("retCode") == 0:
                current_price = float(ticker["result"]["list"][0]["lastPrice"])

                if pos['direction'] == "Long":
                    pnl = (current_price - pos['entry_price']) * pos['quantity']
                else:
                    pnl = (pos['entry_price'] - current_price) * pos['quantity']

                pnl_pct = (pnl / POSITION_SIZE_USDT) * 100

                entry_time = manager.position_entry_times.get(symbol, datetime.now())
                hold_min = (datetime.now() - entry_time).total_seconds() / 60

                emoji = "📈" if pos['direction'] == "Long" else "📉"
                pnl_emoji = "🟢" if pnl > 0 else "🔴"

                print(f"{symbol:^12} | {emoji} {pos['direction']:^6} | "
                      f"${pos['entry_price']:>10,.4f} | ${current_price:>10,.4f} | "
                      f"{pnl_emoji} ${pnl:>+7,.2f} ({pnl_pct:>+5.1f}%) | {hold_min:>8.1f}분")
    else:
        print(f"\n📍 보유 포지션: 없음")

    # 거래 통계
    stats = manager.get_stats()
    if stats:
        print(f"\n📊 거래 통계")
        print(f"   총 거래: {stats['total_trades']}회")
        print(f"   승률: {stats['win_rate']:.1f}% ({stats['wins']}승 {stats['losses']}패)")
        print(f"   총 손익: ${stats['total_pnl']:+,.2f}")
        print(f"   평균 손익: ${stats['avg_pnl']:+,.2f} ({stats['avg_pnl_pct']:+.2f}%)")
        if stats['wins'] > 0:
            print(f"   최대 수익: ${stats['max_win']:,.2f}")
        if stats['losses'] > 0:
            print(f"   최대 손실: ${stats['max_loss']:,.2f}")

    print("\n" + "=" * 100)


# ===== 메인 함수 =====
def main():
    print("\n" + "=" * 100)
    print(f"{'🚀 바이비트 실거래 시스템':^100}")
    print(f"{'⚠️  주의: 실제 자금이 거래됩니다':^100}")
    print("=" * 100)
    print(f"\n설정:")
    print(f"   모드: {'🧪 DRY RUN (테스트)' if DRY_RUN else '💸 LIVE (실거래)'}")
    print(f"   네트워크: {'테스트넷' if USE_TESTNET else '실거래'}")
    print(f"   레버리지: {LEVERAGE}x")
    print(f"   포지션 크기: ${POSITION_SIZE_USDT} USDT")
    print(f"   최대 포지션: {MAX_POSITIONS}개")
    print(f"   신뢰도 임계값: {CONF_THRESHOLD:.0%}")
    print(f"   스캔 주기: {INTERVAL_SEC}초")

    if not DRY_RUN and not USE_TESTNET:
        print(f"\n{'⚠️  경고: 실제 거래 모드입니다!':^100}")
        confirm = input("\n계속하시겠습니까? (yes 입력): ")
        if confirm.lower() != "yes":
            print("종료합니다.")
            return

    # API 연결
    if API_KEY == "YOUR_API_KEY":
        print("\n❌ API Key를 설정해주세요!")
        print("환경변수 BYBIT_API_KEY와 BYBIT_API_SECRET을 설정하거나")
        print("코드 상단의 API_KEY, API_SECRET을 수정하세요.")
        return

    api = BybitAPI(API_KEY, API_SECRET, USE_TESTNET)

    # 모델 로드
    print(f"\n📦 모델 로딩 중...")
    models = load_models()

    if not models:
        print("\n❌ 모델을 로드할 수 없습니다.")
        return

    print(f"✅ {len(models)}개 모델 로드 완료")

    # 거래 관리자
    manager = TradingManager(api)

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  연산 장치: {device}")

    print("\n" + "=" * 100)
    print("거래 시작...")
    print("=" * 100)

    try:
        loop_count = 0
        while True:
            loop_count += 1
            current_time = datetime.now()

            # 포지션 관리
            manager.check_positions()

            # 대시보드
            print_dashboard(manager, api)

            # 신호 스캔
            print(f"\n🔍 신호 스캔 ({len(models)}개 심볼)")
            print(f"{'심볼':^12} | {'가격':^12} | {'방향':^8} | {'신뢰도':^8} | {'신호':^20}")
            print("-" * 80)

            for symbol, model_info in models.items():
                # 예측
                result = predict(api, symbol, model_info, device)

                if "error" in result:
                    print(f"{symbol:^12} | {'N/A':^12} | {'오류':^8} | {'N/A':^8} | "
                          f"❌ {result['error']}")
                    continue

                direction = result["direction"]
                confidence = result["confidence"]
                price = result["current_price"]

                # 이모지
                dir_icon = {"Long": "📈", "Short": "📉", "Flat": "➖"}.get(direction, "❓")

                # 신호 판단
                if confidence < CONF_THRESHOLD:
                    signal = f"⚠️  신호 약함 ({confidence:.1%})"
                elif direction == "Long":
                    signal = f"🟢 매수 신호 ({confidence:.1%})"
                elif direction == "Short":
                    signal = f"🔴 매도 신호 ({confidence:.1%})"
                else:  # Flat
                    signal = f"⚪ 관망 ({confidence:.1%})"

                print(f"{symbol:^12} | ${price:>10,.4f} | {dir_icon} {direction:^6} | "
                      f"{confidence:>6.1%} | {signal}")

                # 진입 조건 (Long 또는 Short만)
                if confidence >= CONF_THRESHOLD and direction in ["Long", "Short"]:
                    can_open, msg = manager.can_open_position(symbol)
                    if can_open:
                        manager.open_position(symbol, direction, price, confidence)
                    elif loop_count == 1:  # 첫 스캔에만 메시지 출력
                        print(f"      ⚠️  진입 불가: {msg}")

            print(f"\n[스캔 #{loop_count}] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"다음 스캔까지 {INTERVAL_SEC}초... (Ctrl+C로 종료)")

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n프로그램 종료 중...")

        # 모든 포지션 청산
        if manager.active_positions:
            print("\n🔄 열린 포지션 청산 중...")
            for symbol in list(manager.active_positions.keys()):
                manager.close_position(symbol, "Manual Close")

        # 최종 대시보드
        print_dashboard(manager, api)

        # 거래 내역 저장
        manager.save_trades()

        # 최종 통계
        stats = manager.get_stats()
        if stats:
            balance = api.get_balance()
            print(f"\n{'📊 최종 결과':^100}")
            print("=" * 100)
            print(f"   최종 잔고: ${balance:,.2f} USDT")
            print(f"   총 거래: {stats['total_trades']}회")
            print(f"   승률: {stats['win_rate']:.1f}%")
            print(f"   총 손익: ${stats['total_pnl']:+,.2f}")
            print("=" * 100)

        print("\n✅ 프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()