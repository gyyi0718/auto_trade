# live_trading.py
# -*- coding: utf-8 -*-
"""
TCN 모델 기반 실전 자동 트레이딩 시스템 (심볼별 모델 사용)
⚠️  WARNING: 실제 자금을 사용합니다. 신중하게 사용하세요!
"""
import os
import time
import hmac
import hashlib
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
# API 인증 (필수!)
API_KEY = os.getenv("BINANCE_API_KEY", "aFWo0NJ58y7WiXB11R7n2vNv9QYVh1P7YCy0i90GKMfnxPzY9KFTtdISsMutezB6")
API_SECRET = os.getenv("BINANCE_API_SECRET", "0evwKyLeeJg8djHhCRfbluEnowjCVjMCbGPgB0by0S5tkYi72GU5RhnEBJTkOw6t")

USE_TESTNET = os.getenv("USE_TESTNET", "0") == "1"  # 기본값: Testnet

if not API_KEY or not API_SECRET:
    print("❌ ERROR: BYBIT_API_KEY 및 BYBIT_API_SECRET 환경변수를 설정하세요!")
    print("\n설정 방법 (Windows PowerShell):")
    print("  $env:BYBIT_API_KEY='your_api_key'")
    print("  $env:BYBIT_API_SECRET='your_api_secret'")
    print("  $env:USE_TESTNET='1'  # Testnet (기본값)")
    print("  $env:USE_TESTNET='0'  # Mainnet (실전)")
    print("\n설정 방법 (Linux/Mac):")
    print("  export BYBIT_API_KEY='your_api_key'")
    print("  export BYBIT_API_SECRET='your_api_secret'")
    print("  export USE_TESTNET=1  # Testnet (기본값)")
    print("  export USE_TESTNET=0  # Mainnet (실전)")
    print("\n포지션 모드 설정:")
    print("  $env:POSITION_MODE='hedge'     # Hedge Mode (양방향)")
    print("  $env:POSITION_MODE='one-way'   # One-Way Mode (단방향, 기본값)")
    print("\n⚠️  중요:")
    print("  - Testnet API Key: https://testnet.binance.com 에서 발급")
    print("  - Mainnet API Key: https://www.binance.com 에서 발급")
    print("  - Testnet과 Mainnet API Key는 서로 다릅니다!")
    print("  - API 권한: Contract Trading, Account Transfer 필수")
    print("  - 포지션 모드는 Binance 웹사이트 설정과 일치해야 합니다!")
    exit(1)

# 거래 설정
SYMBOLS_ENV = os.getenv("SYMBOLS", "").strip()

INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "2"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.7"))  # 실전은 더 높게

# ✅ 심볼별 모델 경로 설정 (딕셔너리 형태)
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
    "GIGGLEUSDT": "D:/ygy_work/coin/multimodel/models_5min_giggle_binance/5min_2class_best.ckpt"

}
if SYMBOLS_ENV:
    # 환경변수가 설정되어 있으면 사용
    SYMBOLS = [s.strip() for s in SYMBOLS_ENV.split(",") if s.strip()]
else:
    # 환경변수가 없으면 MODEL_PATHS의 모든 심볼 사용
    SYMBOLS = list(MODEL_PATHS.keys())
# 리스크 관리 (매우 중요!)
MARGIN_PER_POSITION = float(os.getenv("MARGIN_PER_POSITION", "200"))  # 포지션당 증거금
LEVERAGE = int(os.getenv("LEVERAGE", "20"))  # 레버리지 (실전은 낮게!)
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))  # 최대 동시 포지션
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))  # 손절 2%
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.03"))  # 익절 3%
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "500"))  # 일일 최대 손실
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "30"))  # 최대 보유 시간
MIN_HOLD_MINUTES = int(os.getenv("MIN_HOLD_MINUTES", "5"))  # 최소 보유 시간 (반대 신호 청산 방지)
LIQUIDATION_BUFFER = float(os.getenv("LIQUIDATION_BUFFER", "0.8"))  # 청산 버퍼 (80%)

# 포지션 모드
POSITION_MODE = os.getenv("POSITION_MODE", "one-way").lower()  # "one-way" 또는 "hedge"

# 로그
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "live_trades.json")
ORDER_LOG_FILE = os.getenv("ORDER_LOG_FILE", "orders.json")
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"


# ===== 데이터 클래스 =====
@dataclass
class Position:
    """포지션 정보 (개선된 버전)"""
    symbol: str
    direction: str  # "Long" or "Short"
    entry_price: float
    size: float  # 수량
    entry_time: datetime
    stop_loss: float
    take_profit: float
    leverage: int
    margin: float  # 실제 사용한 증거금
    liquidation_price: float  # 청산가
    unrealized_pnl: float = 0.0  # 평가 손익
    position_value: float = 0.0  # 포지션 가치

    def get_pnl(self, current_price: float) -> float:
        """손익 계산 (레버리지 적용)"""
        if self.direction == "Long":
            return (current_price - self.entry_price) * self.size
        else:  # Short
            return (self.entry_price - current_price) * self.size

    def get_pnl_pct(self, current_price: float) -> float:
        """손익률 계산 (증거금 기준)"""
        pnl = self.get_pnl(current_price)
        return (pnl / self.margin) * 100 if self.margin > 0 else 0

    def get_roe(self, current_price: float) -> float:
        """ROE 계산 (레버리지 반영)"""
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
        # 강제 청산 (레버리지 고려)
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
        hold_minutes = (current_time - self.entry_time).total_seconds() / 60
        if hold_minutes >= MAX_HOLD_MINUTES:
            return True, "Time Limit"

        return False, ""


@dataclass
class Trade:
    """거래 기록"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    leverage: int
    margin: float
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float
    roe: float
    exit_reason: str


# ===== TCN 모델 정의 =====
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


class TCN_MT(nn.Module):
    def __init__(self, in_f, hidden=128, levels=6, k=3, drop=0.1):
        super().__init__()
        L = []
        ch = in_f
        for i in range(levels):
            L.append(Block(ch, hidden, k, 2 ** i, drop))
            ch = hidden
        self.tcn = nn.Sequential(*L)
        self.head_cls = nn.Linear(hidden, 3)
        self.head_ttt = nn.Linear(hidden, 1)

    def forward(self, X):
        X = X.transpose(1, 2)
        H = self.tcn(X)[:, :, -1]
        return self.head_cls(H), self.head_ttt(H)


# ===== 모델 로드 =====
print("\n🤖 심볼별 모델 로드 중...")
MODELS = {}
MODEL_CONFIGS = {}

for symbol in SYMBOLS:
    symbol = symbol.strip()
    if symbol not in MODEL_PATHS:
        print(f"⚠️  {symbol}: 모델 경로 미설정")
        continue

    model_path = MODEL_PATHS[symbol]
    if not os.path.exists(model_path):
        print(f"❌ {symbol}: 모델 파일 없음 - {model_path}")
        continue

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        feat_cols = checkpoint['feat_cols']
        meta = checkpoint.get('meta', {})
        seq_len = meta.get('seq_len', 60)
        scaler_mu = checkpoint.get('scaler_mu')
        scaler_sd = checkpoint.get('scaler_sd')

        model_dict = checkpoint['model']

        # 모델 구조 감지
        max_layer = max(
            [int(k.split('.')[1]) for k in model_dict.keys() if k.startswith('tcn.') and len(k.split('.')) > 2],
            default=0)
        levels = max_layer + 1
        hidden = model_dict['tcn.0.c1.weight_v'].shape[0] if 'tcn.0.c1.weight_v' in model_dict else 32
        k = model_dict['tcn.0.c1.weight_v'].shape[2] if 'tcn.0.c1.weight_v' in model_dict else 3
        drop = meta.get('dropout', 0.2)

        # 모델 타입 및 클래스 수 감지
        if 'head.weight' in model_dict:
            # Single-task 모델 - 클래스 수 자동 감지
            num_classes = model_dict['head.weight'].shape[0]


            class TCN_SingleTask(nn.Module):
                def __init__(self, in_f, hidden, levels, k, drop, num_classes):
                    super().__init__()
                    L = []
                    ch = in_f
                    for i in range(levels):
                        L.append(Block(ch, hidden, k, 2 ** i, drop))
                        ch = hidden
                    self.tcn = nn.Sequential(*L)
                    self.head = nn.Linear(hidden, num_classes)

                def forward(self, X):
                    X = X.transpose(1, 2)
                    H = self.tcn(X)[:, :, -1]
                    return self.head(H)


            model = TCN_SingleTask(len(feat_cols), hidden, levels, k, drop, num_classes)
        elif 'head_cls.weight' in model_dict:
            # Multi-task 모델 - 클래스 수 자동 감지
            num_classes = model_dict['head_cls.weight'].shape[0]
            model = TCN_MT(len(feat_cols), hidden, levels, k, drop)
            # TCN_MT는 기본적으로 3-class이므로 재정의
            model.head_cls = nn.Linear(hidden, num_classes)
        else:
            # 기본값
            num_classes = 3
            model = TCN_MT(len(feat_cols), hidden, levels, k, drop)

        model.load_state_dict(model_dict)
        model.eval()

        MODELS[symbol] = model
        MODEL_CONFIGS[symbol] = {
            'feat_cols': feat_cols,
            'seq_len': seq_len,
            'scaler_mu': scaler_mu,
            'scaler_sd': scaler_sd,
            'is_single_task': 'head.weight' in model_dict,
            'num_classes': num_classes
        }

        class_type = f"{num_classes}-class"
        task_type = 'Single-task' if MODEL_CONFIGS[symbol]['is_single_task'] else 'Multi-task'
        print(f"✅ {symbol}: {task_type} {class_type} 모델 로드 완료 "
              f"(levels={levels}, hidden={hidden}, k={k})")

    except Exception as e:
        print(f"❌ {symbol}: 모델 로드 실패 - {e}")

if not MODELS:
    print("\n❌ ERROR: 로드된 모델이 없습니다!")
    print("확인 사항:")
    print("   1. MODEL_PATHS에 심볼별 모델 경로가 올바르게 설정되어 있는지 확인")
    print("   2. 모델 파일이 실제로 존재하는지 확인")
    print("   3. SYMBOLS 환경변수에 설정된 심볼과 MODEL_PATHS의 키가 일치하는지 확인")
    print(f"\n현재 설정:")
    print(f"   SYMBOLS 환경변수: {SYMBOLS_ENV if SYMBOLS_ENV else '(미설정 - 모든 모델 사용)'}")
    print(f"   시도한 심볼: {', '.join(SYMBOLS)}")
    print(f"   MODEL_PATHS에 있는 심볼: {', '.join(MODEL_PATHS.keys())}")
    exit(1)

print(f"\n✅ 총 {len(MODELS)}개 모델 로드 완료")
print(f"   사용 가능 심볼: {', '.join(MODELS.keys())}")

# 환경변수에 설정된 심볼 중 로드 실패한 것들 안내
failed_symbols = [s for s in SYMBOLS if s not in MODELS]
if failed_symbols:
    print(f"\n⚠️  로드 실패한 심볼 ({len(failed_symbols)}개): {', '.join(failed_symbols)}")


# ===== Binance API =====
class BinanceAPI:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._instrument_cache = {}  # 심볼 정보 캐시

        if testnet:
            self.base_url = "https://api-testnet.binance.com"
        else:
            self.base_url = "https://api.binance.com"

        self.recv_window = "5000"

    def _generate_signature(self, timestamp: str, params_str: str) -> str:
        """서명 생성"""
        # signature = HMAC_SHA256(timestamp + api_key + recv_window + params_str)
        sign_str = timestamp + self.api_key + self.recv_window + params_str

        if DEBUG_MODE:
            print(f"[DEBUG] Sign string: {sign_str}")

        signature = hmac.new(
            bytes(self.api_secret, 'utf-8'),
            bytes(sign_str, 'utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature

    def _request(self, method: str, endpoint: str, params: dict = None, sign: bool = False) -> dict:
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
                # GET: 쿼리 파라미터를 알파벳 순으로 정렬
                params_str = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())]) if params else ""
            else:
                # POST: JSON body
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

            if DEBUG_MODE:
                print(f"\n[DEBUG] {method} {endpoint}")
                print(f"[DEBUG] Timestamp: {timestamp}")
                print(f"[DEBUG] Params: {params}")
                print(f"[DEBUG] Params str: {params_str}")
                print(f"[DEBUG] Signature: {signature}")

        try:
            url = f"{self.base_url}{endpoint}"

            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")

            if DEBUG_MODE:
                print(f"[DEBUG] Status: {response.status_code}")
                print(f"[DEBUG] Response: {response.text[:500]}")

            data = response.json()
            return data

        except Exception as e:
            return {"retCode": -1, "retMsg": str(e)}

    def get_ticker(self, symbol: str) -> dict:
        """현재가 조회"""
        return self._request("GET", "/v5/market/tickers", {"category": "linear", "symbol": symbol})

    def get_klines(self, symbol: str, interval: str = "5", limit: int = 200) -> pd.DataFrame:
        """캔들 데이터 조회"""
        result = self._request("GET", "/v5/market/kline", {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        })

        if result.get("retCode") != 0:
            return pd.DataFrame()

        data = result["result"]["list"]
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df = df.astype({
            "timestamp": "int64",
            "open": "float",
            "high": "float",
            "low": "float",
            "close": "float",
            "volume": "float"
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.sort_values("timestamp").reset_index(drop=True)

    def get_balance(self) -> float:
        """USDT 잔고 조회"""
        result = self._request("GET", "/v5/account/wallet-balance", {
            "accountType": "UNIFIED"
        }, sign=True)

        if result.get("retCode") != 0:
            if DEBUG_MODE:
                print(f"[DEBUG] Balance API Error: {result}")
            return 0.0

    def get_positions(self, Position=None) -> List:
        """포지션 조회"""
        from datetime import datetime

        result = self._request("GET", "/fapi/v2/positionRisk", sign=True)

        if isinstance(result, dict) and "code" in result:
            return []

        positions = []
        for item in result:
            position_amt = float(item.get("positionAmt", 0))
            if position_amt == 0:
                continue

            symbol = item.get("symbol")
            direction = "Long" if position_amt > 0 else "Short"
            size = abs(position_amt)
            entry_price = float(item.get("entryPrice", 0))
            leverage = int(item.get("leverage", LEVERAGE))
            unrealized_pnl = float(item.get("unRealizedProfit", 0))
            liquidation_price = float(item.get("liquidationPrice", 0))

            # 증거금 계산
            position_value = size * entry_price
            margin = position_value / leverage if leverage > 0 else position_value

            # TP/SL 계산
            if direction == "Long":
                stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
            else:
                stop_loss = entry_price * (1 + STOP_LOSS_PCT)
                take_profit = entry_price * (1 - TAKE_PROFIT_PCT)

            if Position:
                positions.append(Position(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    size=size,
                    entry_time=datetime.now(),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=leverage,
                    margin=margin,
                    liquidation_price=liquidation_price,
                    unrealized_pnl=unrealized_pnl,
                    position_value=position_value
                ))

        return positions

    def get_instrument_info(self, symbol: str) -> dict:
        """심볼 정보 조회"""
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        result = self._request("GET", "/fapi/v1/exchangeInfo")

        if isinstance(result, dict) and "code" in result:
            default_info = {
                "minOrderQty": 0.001,
                "qtyStep": 0.001,
                "minNotional": 5.0
            }
            self._instrument_cache[symbol] = default_info
            return default_info

        for symbol_info in result.get("symbols", []):
            if symbol_info.get("symbol") == symbol:
                filters = {f["filterType"]: f for f in symbol_info.get("filters", [])}

                lot_size = filters.get("LOT_SIZE", {})
                min_notional = filters.get("MIN_NOTIONAL", {})

                info = {
                    "minOrderQty": float(lot_size.get("minQty", 0.001)),
                    "qtyStep": float(lot_size.get("stepSize", 0.001)),
                    "maxOrderQty": float(lot_size.get("maxQty", 1000000)),
                    "minNotional": float(min_notional.get("notional", 5.0))
                }

                self._instrument_cache[symbol] = info

                if DEBUG_MODE:
                    print(f"ℹ️  {symbol} 정보: minQty={info['minOrderQty']}, "
                          f"qtyStep={info['qtyStep']}, minNotional=${info['minNotional']}")

                return info

        default_info = {
            "minOrderQty": 0.001,
            "qtyStep": 0.001,
            "minNotional": 5.0
        }
        self._instrument_cache[symbol] = default_info
        return default_info

    def set_leverage(self, symbol: str, leverage: int) -> tuple[bool, str]:
        """레버리지 설정"""
        result = self._request("POST", "/fapi/v1/leverage", {
            "symbol": symbol,
            "leverage": leverage
        }, sign=True)

        if isinstance(result, dict) and "code" in result and result["code"] != 0:
            return False, result.get("msg", "Unknown error")

        return True, "Success"

    def place_order(self, symbol: str, side: str, qty: float, order_type: str = "MARKET",
                    price: Optional[float] = None, reduce_only: bool = False) -> Optional[str]:
        """주문 생성"""
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": qty
        }

        if reduce_only:
            params["reduceOnly"] = "true"

        if order_type.upper() == "LIMIT" and price:
            params["price"] = price
            params["timeInForce"] = "GTC"

        result = self._request("POST", "/fapi/v1/order", params, sign=True)

        if isinstance(result, dict):
            if "code" in result and result["code"] != 0:
                ret_code = result.get("code")
                ret_msg = result.get("msg", "알 수 없는 오류")
                print(f"❌ 주문 실패: {symbol} | {side} | {qty}")
                print(f"   에러 코드: {ret_code} | 메시지: {ret_msg}")

                if "insufficient" in ret_msg.lower() or "balance" in ret_msg.lower():
                    print(f"   💡 잔고 부족")
                elif "qty" in ret_msg.lower() or "size" in ret_msg.lower():
                    print(f"   💡 수량 오류")
                elif "leverage" in ret_msg.lower():
                    print(f"   💡 레버리지 오류")

                return None
            else:
                order_id = str(result.get("orderId"))
                if DEBUG_MODE:
                    print(f"✅ 주문 성공: {symbol} | {side} | {qty} | Order ID: {order_id}")
                return order_id

        return None

    def set_trading_stop(self, symbol: str, side: str, stop_loss: float = None,
                         take_profit: float = None) -> tuple[bool, str]:
        """TP/SL 설정 (Binance는 별도 주문으로 처리)"""
        # Binance에서는 TP/SL을 별도 주문으로 생성해야 함
        # 여기서는 간단히 성공으로 처리 (실제로는 STOP_MARKET 주문 생성 필요)
        return True, "Success (TP/SL은 수동 설정 필요)"

        try:
            coins = result["result"]["list"][0]["coin"]
            usdt = next((c for c in coins if c["coin"] == "USDT"), None)
            return float(usdt["walletBalance"]) if usdt else 0.0
        except (KeyError, IndexError, TypeError) as e:
            if DEBUG_MODE:
                print(f"[DEBUG] Balance parsing error: {e}")
            return 0.0

    def get_positions(self) -> List[Position]:
        """포지션 조회 (개선된 버전)"""
        result = self._request("GET", "/v5/position/list", {"category": "linear", "settleCoin": "USDT"}, sign=True)

        if result.get("retCode") != 0:
            return []

        positions = []
        try:
            for item in result["result"]["list"]:
                size = float(item.get("size", 0))
                if size == 0:
                    continue

                side = item.get("side")
                direction = "Long" if side == "Buy" else "Short"
                entry_price = float(item.get("avgPrice", 0))
                symbol = item.get("symbol")
                leverage = int(item.get("leverage", LEVERAGE))
                unrealized_pnl = float(item.get("unrealisedPnl", 0))

                # 증거금 계산
                position_value = size * entry_price
                margin = position_value / leverage

                # 청산가 계산
                liq_price_str = item.get("liqPrice", "0")
                if liq_price_str and liq_price_str != "":
                    liquidation_price = float(liq_price_str)
                else:
                    # API에서 청산가를 제공하지 않으면 직접 계산
                    if direction == "Long":
                        liquidation_price = entry_price * (1 - (1 / leverage) * LIQUIDATION_BUFFER)
                    else:
                        liquidation_price = entry_price * (1 + (1 / leverage) * LIQUIDATION_BUFFER)

                # 손절/익절 가격 계산
                if direction == "Long":
                    stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                    take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
                else:
                    stop_loss = entry_price * (1 + STOP_LOSS_PCT)
                    take_profit = entry_price * (1 - TAKE_PROFIT_PCT)

                # 포지션 생성 시간 (API에서 제공하지 않으면 현재 시간 사용)
                created_time_str = item.get("createdTime", "")
                if created_time_str:
                    try:
                        # 밀리초 timestamp를 초 단위로 변환
                        created_timestamp = int(created_time_str)
                        if created_timestamp > 9999999999:  # 밀리초 단위
                            created_timestamp = created_timestamp / 1000
                        entry_time = datetime.fromtimestamp(created_timestamp)

                        if DEBUG_MODE:
                            print(f"[DEBUG] {symbol} createdTime: {created_time_str} -> {entry_time}")
                    except (ValueError, OSError) as e:
                        if DEBUG_MODE:
                            print(f"[DEBUG] {symbol} createdTime 파싱 실패: {e}")
                        entry_time = datetime.now()
                else:
                    entry_time = datetime.now()

                positions.append(Position(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    size=size,
                    entry_time=entry_time,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=leverage,
                    margin=margin,
                    liquidation_price=liquidation_price,
                    unrealized_pnl=unrealized_pnl,
                    position_value=position_value
                ))

        except Exception as e:
            print(f"❌ 포지션 파싱 오류: {e}")

        return positions

    def get_instrument_info(self, symbol: str) -> dict:
        """심볼 정보 조회 (최소 수량, 수량 단위 등) - 캐시 사용"""
        # 캐시 확인
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        result = self._request("GET", "/v5/market/instruments-info", {
            "category": "linear",
            "symbol": symbol
        })

        if result.get("retCode") != 0 or not result.get("result", {}).get("list"):
            # 기본값 반환
            default_info = {
                "minOrderQty": 0.01,
                "qtyStep": 0.01,
                "minNotionalValue": 5.0
            }
            self._instrument_cache[symbol] = default_info
            return default_info

        info = result["result"]["list"][0]
        lot_size_filter = info.get("lotSizeFilter", {})

        instrument_info = {
            "minOrderQty": float(lot_size_filter.get("minOrderQty", 0.01)),
            "qtyStep": float(lot_size_filter.get("qtyStep", 0.01)),
            "maxOrderQty": float(lot_size_filter.get("maxOrderQty", 1000000)),
            "minNotionalValue": float(lot_size_filter.get("minNotionalValue", 5.0))
        }

        # 캐시 저장
        self._instrument_cache[symbol] = instrument_info

        if DEBUG_MODE:
            print(f"ℹ️  {symbol} 정보: minQty={instrument_info['minOrderQty']}, "
                  f"qtyStep={instrument_info['qtyStep']}, "
                  f"minNotional=${instrument_info['minNotionalValue']}")

        return instrument_info

    def set_trading_stop(self, symbol: str, side: str, stop_loss: float = None,
                         take_profit: float = None) -> tuple[bool, str]:
        """TP/SL 설정"""
        params = {
            "category": "linear",
            "symbol": symbol,
            "positionIdx": 0  # One-way mode
        }

        if POSITION_MODE == "hedge":
            params["positionIdx"] = 1 if side == "Buy" else 2

        if stop_loss:
            params["stopLoss"] = str(stop_loss)

        if take_profit:
            params["takeProfit"] = str(take_profit)

        if not stop_loss and not take_profit:
            return False, "TP 또는 SL이 필요합니다"

        result = self._request("POST", "/v5/position/trading-stop", params, sign=True)

        success = result.get("retCode") == 0
        msg = result.get("retMsg", "알 수 없는 오류")

        return success, msg

    def set_leverage(self, symbol: str, leverage: int) -> tuple[bool, str]:
        """레버리지 설정"""
        result = self._request("POST", "/v5/position/set-leverage", {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        }, sign=True)

        success = result.get("retCode") == 0
        msg = result.get("retMsg", "알 수 없는 오류")

        return success, msg

    def place_order(self, symbol: str, side: str, qty: float, order_type: str = "Market",
                    price: Optional[float] = None, reduce_only: bool = False) -> Optional[str]:
        """주문 생성"""
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": "GTC"  # Good Till Cancel
        }

        if reduce_only:
            params["reduceOnly"] = True  # boolean 타입

        if order_type == "Limit" and price:
            params["price"] = str(price)

        if POSITION_MODE == "hedge":
            params["positionIdx"] = 1 if side == "Buy" else 2
        else:
            params["positionIdx"] = 0  # One-way mode

        result = self._request("POST", "/v5/order/create", params, sign=True)

        if result.get("retCode") == 0:
            order_id = result["result"]["orderId"]
            if DEBUG_MODE:
                print(f"✅ 주문 성공: {symbol} | {side} | {qty} | Order ID: {order_id}")
            return order_id
        else:
            ret_code = result.get("retCode")
            ret_msg = result.get("retMsg", "알 수 없는 오류")
            print(f"❌ 주문 실패: {symbol} | {side} | {qty}")
            print(f"   에러 코드: {ret_code} | 메시지: {ret_msg}")

            # 일반적인 에러 해석
            if "insufficient" in ret_msg.lower() or "balance" in ret_msg.lower():
                print(f"   💡 잔고 부족 - 사용 가능한 증거금을 확인하세요")
            elif "qty" in ret_msg.lower() or "size" in ret_msg.lower():
                print(f"   💡 수량 오류 - 최소/최대 수량 또는 수량 단위를 확인하세요")
            elif "leverage" in ret_msg.lower():
                print(f"   💡 레버리지 오류 - 허용된 레버리지 범위를 확인하세요")
            elif "risk limit" in ret_msg.lower():
                print(f"   💡 리스크 한도 초과 - 포지션 크기를 줄이거나 레버리지를 낮추세요")
            elif "sign" in ret_msg.lower():
                print(f"   💡 API 서명 오류 - API Key와 Secret을 확인하세요")
            elif "reduce" in ret_msg.lower() or "position" in ret_msg.lower():
                print(f"   💡 포지션 오류 - 청산할 포지션이 없거나 수량이 맞지 않습니다")

            return None


API = BinanceAPI(API_KEY, API_SECRET, USE_TESTNET)


# ===== 포지션 관리자 =====
class PositionManager:
    """포지션 관리 클래스"""

    def __init__(self):
        self.positions_entry_time: Dict[str, datetime] = {}  # 심볼별 진입 시간 추적
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.daily_reset_time = datetime.now().date()

    def reset_daily_stats(self):
        """일일 통계 초기화"""
        current_date = datetime.now().date()
        if current_date != self.daily_reset_time:
            self.daily_pnl = 0.0
            self.daily_reset_time = current_date

    def can_open_position(self, symbol: str, direction: str, current_positions: List[Position]) -> bool:
        """포지션 진입 가능 여부 (같은 방향이면 유지, 반대 방향이면 청산 후 진입)"""
        # 같은 심볼의 기존 포지션 확인
        existing = next((p for p in current_positions if p.symbol == symbol), None)

        if existing:
            # 같은 방향이면 진입하지 않음 (기존 포지션 유지)
            if existing.direction == direction:
                if DEBUG_MODE:
                    print(f"ℹ️  {symbol}: 같은 방향 포지션 이미 보유 중 ({direction}) - 유지")
                return False
            # 반대 방향이면 진입 가능 (메인 루프에서 청산 처리)
            return True

        # 최대 포지션 수 확인
        if len(current_positions) >= MAX_POSITIONS:
            return False

        # 일일 손실 한도 확인
        if abs(self.daily_pnl) >= MAX_DAILY_LOSS:
            print(f"⚠️  일일 손실 한도 도달: ${self.daily_pnl:.2f}")
            return False

        return True

    def open_position(self, symbol: str, direction: str, price: float) -> bool:
        """포지션 진입"""
        try:
            # 레버리지 설정 (실패 시 경고만 출력하고 계속 진행)
            success, msg = API.set_leverage(symbol, LEVERAGE)
            if not success:
                # 이미 설정되어 있거나 레버리지 변경이 필요없는 경우
                if "leverage not modified" in msg.lower() or "same" in msg.lower():
                    if DEBUG_MODE:
                        print(f"ℹ️  {symbol}: 레버리지 이미 설정됨")
                else:
                    print(f"⚠️  {symbol}: 레버리지 설정 실패 - {msg} (기존 레버리지 사용)")

            # 심볼 정보 조회
            instrument_info = API.get_instrument_info(symbol)
            min_qty = instrument_info["minOrderQty"]
            qty_step = instrument_info["qtyStep"]
            min_notional = instrument_info["minNotionalValue"]

            # 포지션 크기 계산
            position_value = MARGIN_PER_POSITION * LEVERAGE
            qty = position_value / price

            # 최소 주문 금액 확인
            notional_value = qty * price
            if notional_value < min_notional:
                print(f"❌ {symbol}: 주문 금액 부족 (${notional_value:.2f} < ${min_notional:.2f})")
                return False

            # 최소 수량 확인
            if qty < min_qty:
                print(f"❌ {symbol}: 수량 부족 ({qty:.4f} < {min_qty:.4f})")
                return False

            # ✅ 개선: qty를 step 단위로 올림 (반올림이 아닌 올림)
            # 이렇게 하면 반올림 후 최소 수량보다 작아지는 문제 방지
            import math
            steps = math.ceil(qty / qty_step)
            qty = steps * qty_step

            # ✅ 부동소수점 정밀도 문제 해결
            # qty_step의 소수점 자리수 계산
            step_str = f"{qty_step:.10f}".rstrip('0')
            if '.' in step_str:
                decimals = len(step_str.split('.')[1])
            else:
                decimals = 0

            # 해당 자리수로 반올림
            qty = round(qty, decimals)

            # 최종 qty 검증
            if qty < min_qty:
                print(f"❌ {symbol}: 반올림 후 수량 부족 ({qty:.4f} < {min_qty:.4f})")
                return False

            # 최종 주문 금액 재확인
            final_notional = qty * price
            if final_notional < min_notional:
                print(f"❌ {symbol}: 최종 주문 금액 부족 (${final_notional:.2f} < ${min_notional:.2f})")
                return False

            if DEBUG_MODE:
                print(f"\n[주문 상세] {symbol}")
                print(f"  포지션 크기: ${position_value:.2f}")
                print(f"  계산된 수량: {position_value / price:.8f}")
                print(f"  qtyStep: {qty_step}")
                print(f"  반올림 후: {qty}")
                print(f"  최종 금액: ${final_notional:.2f}")

            # TP/SL 가격 계산
            if direction == "Long":
                stop_loss = price * (1 - STOP_LOSS_PCT)
                take_profit = price * (1 + TAKE_PROFIT_PCT)
            else:  # Short
                stop_loss = price * (1 + STOP_LOSS_PCT)
                take_profit = price * (1 - TAKE_PROFIT_PCT)

            # 주문 생성
            side = "Buy" if direction == "Long" else "Sell"
            order_id = API.place_order(symbol, side, qty)

            if order_id:
                # 포지션 진입 성공 - TP/SL 설정
                time.sleep(0.5)  # 주문 체결 대기

                success, msg = API.set_trading_stop(symbol, side, stop_loss, take_profit)
                if success:
                    if DEBUG_MODE:
                        print(f"   ✅ TP/SL 설정 완료: SL=${stop_loss:,.4f}, TP=${take_profit:,.4f}")
                else:
                    print(f"   ⚠️  TP/SL 설정 실패: {msg}")
                    print(f"   수동 설정: SL=${stop_loss:,.4f}, TP=${take_profit:,.4f}")

                self.positions_entry_time[symbol] = datetime.now()

                print(f"🟢 포지션 진입: {symbol} | {direction} | ${price:,.4f} | 수량: {qty} | "
                      f"레버리지: {LEVERAGE}x | 증거금: ${MARGIN_PER_POSITION}")
                print(f"   📍 SL: ${stop_loss:,.4f} ({-STOP_LOSS_PCT * 100:.1f}%) | "
                      f"TP: ${take_profit:,.4f} ({TAKE_PROFIT_PCT * 100:.1f}%)")

                return True

        except Exception as e:
            print(f"❌ {symbol}: 포지션 진입 실패 - {e}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()

        return False

    def close_position(self, position: Position, reason: str) -> bool:
        """포지션 청산"""
        try:
            # 반대 방향으로 주문 (청산)
            side = "Sell" if position.direction == "Long" else "Buy"

            # 정확한 포지션 수량 사용
            qty = position.size

            if DEBUG_MODE:
                print(f"🔄 청산 시도: {position.symbol} | {side} | qty={qty} | reason={reason}")

            order_id = API.place_order(position.symbol, side, qty, reduce_only=True)

            if order_id:
                exit_time = datetime.now()
                entry_time_str = self.positions_entry_time.get(position.symbol, position.entry_time)
                if isinstance(entry_time_str, datetime):
                    entry_time_str = entry_time_str.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    entry_time_str = str(entry_time_str)

                # 현재가로 손익 계산
                ticker = API.get_ticker(position.symbol)
                if ticker.get("retCode") == 0 and ticker["result"]["list"]:
                    current_price = float(ticker["result"]["list"][0]["lastPrice"])
                else:
                    current_price = position.entry_price  # 조회 실패 시 진입가 사용

                pnl = position.get_pnl(current_price)
                pnl_pct = position.get_pnl_pct(current_price)
                roe = position.get_roe(current_price)

                # 거래 기록 저장
                trade = Trade(
                    symbol=position.symbol,
                    direction=position.direction,
                    entry_price=position.entry_price,
                    exit_price=current_price,
                    size=position.size,
                    leverage=position.leverage,
                    margin=position.margin,
                    entry_time=entry_time_str,
                    exit_time=exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    roe=roe,
                    exit_reason=reason
                )

                self.trades.append(trade)
                self.daily_pnl += pnl

                # 진입 시간 삭제
                if position.symbol in self.positions_entry_time:
                    del self.positions_entry_time[position.symbol]

                emoji = "🟢" if pnl > 0 else "🔴"
                print(f"{emoji} 포지션 청산: {position.symbol} | {position.direction} | "
                      f"진입: ${position.entry_price:,.4f} → 청산: ${current_price:,.4f} | "
                      f"손익: ${pnl:+,.2f} ({roe:+.1f}%) | 사유: {reason}")

                return True
            else:
                print(f"⚠️  {position.symbol}: 청산 주문 실패 - 수동으로 청산하세요")

        except Exception as e:
            print(f"❌ {position.symbol}: 포지션 청산 실패 - {e}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()

        return False

    def save_trades(self):
        """거래 내역 저장"""
        if not self.trades:
            return

        data = [asdict(t) for t in self.trades]
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n💾 거래 내역 저장: {TRADE_LOG_FILE}")

    def get_stats(self) -> dict:
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
                "min_pnl": 0,
                "max_roe": 0,
                "min_roe": 0,
                "liquidations": 0
            }

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        liquidations = [t for t in self.trades if t.exit_reason == "Liquidation"]

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(self.trades) * 100) if self.trades else 0,
            "avg_pnl": sum(t.pnl for t in self.trades) / len(self.trades),
            "avg_roe": sum(t.roe for t in self.trades) / len(self.trades),
            "max_pnl": max(t.pnl for t in self.trades),
            "min_pnl": min(t.pnl for t in self.trades),
            "max_roe": max(t.roe for t in self.trades),
            "min_roe": min(t.roe for t in self.trades),
            "avg_win": sum(t.pnl for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t.pnl for t in losses) / len(losses) if losses else 0,
            "liquidations": len(liquidations)
        }


manager = PositionManager()


# ===== 예측 함수 =====
def predict(symbol: str, debug: bool = False) -> dict:
    """
    심볼별 모델로 예측 수행

    Returns:
        dict: {
            "direction": "Long" | "Short" | "Flat",
            "confidence": 0.0~1.0,
            "current_price": float,
            "probabilities": [p_flat, p_long, p_short]
        }
    """
    if symbol not in MODELS:
        return {"error": f"모델 없음: {symbol}"}

    try:
        model = MODELS[symbol]
        config = MODEL_CONFIGS[symbol]

        # 데이터 가져오기
        df = API.get_klines(symbol, interval="5", limit=config['seq_len'] + 50)
        if len(df) < config['seq_len']:
            return {"error": f"데이터 부족 ({len(df)}/{config['seq_len']})"}

        # 피처 계산
        df = calculate_features(df)

        # 최근 seq_len 구간 추출
        recent = df[config['feat_cols']].iloc[-config['seq_len']:].copy()

        # 정규화
        if config['scaler_mu'] is not None and config['scaler_sd'] is not None:
            recent = (recent - config['scaler_mu']) / (config['scaler_sd'] + 1e-8)

        # 텐서 변환
        X = torch.FloatTensor(recent.values).unsqueeze(0)  # (1, seq_len, n_features)

        # 예측
        with torch.no_grad():
            if config['is_single_task']:
                logits = model(X)
            else:
                logits, _ = model(X)

            probs = torch.softmax(logits, dim=1).squeeze().numpy()

        # 결과 해석 (클래스 수에 따라 다르게 처리)
        num_classes = config['num_classes']

        if num_classes == 2:
            # 2-class 모델: [Long, Short]
            direction_map = {0: "Short", 1: "Long"}
            pred_class = int(probs.argmax())
            direction = direction_map[pred_class]
            confidence = float(probs[pred_class])

            # 3-class 형식으로 변환 (호환성을 위해)
            probs_3class = np.array([0.0, probs[0], probs[1]])  # [Flat=0, Long, Short]

        elif num_classes == 3:
            # 3-class 모델: [Flat, Long, Short]
            direction_map = {0: "Short", 1: "Flat", 2: "Long"}
            pred_class = int(probs.argmax())
            direction = direction_map[pred_class]
            confidence = float(probs[pred_class])
            probs_3class = probs

        else:
            return {"error": f"지원하지 않는 클래스 수: {num_classes}"}

        current_price = float(df['close'].iloc[-1])

        if debug:
            print(f"\n[DEBUG] {symbol} 예측")
            print(f"   모델 타입: {num_classes}-class")
            print(f"   데이터: {len(df)}개 캔들")
            print(f"   현재가: ${current_price:,.4f}")
            if num_classes == 2:
                print(f"   확률: Long={probs[0]:.1%}, Short={probs[1]:.1%}")
            else:
                print(f"   확률: Flat={probs[0]:.1%}, Long={probs[1]:.1%}, Short={probs[2]:.1%}")
            print(f"   결과: {direction} ({confidence:.1%})")

        return {
            "direction": direction,
            "confidence": confidence,
            "current_price": current_price,
            "probabilities": probs_3class.tolist(),
            "num_classes": num_classes
        }

    except Exception as e:
        return {"error": str(e)}


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 계산 (학습 코드와 동일)"""
    g = df.copy()

    # ===== 기본 수익률 =====
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g["logc"].diff().fillna(0.0)

    # ===== 변동성 =====
    for w in (8, 20, 40, 120):
        g[f"rv{w}"] = g["ret1"].rolling(w, min_periods=max(2, w // 3)).std().fillna(0.0)

    # ===== 모멘텀 =====
    for w in (8, 20, 40, 120):
        ema = g["close"].ewm(span=w, adjust=False).mean()
        g[f"mom{w}"] = (g["close"] / ema - 1.0).fillna(0.0)

    # ===== 볼륨 분석 =====
    # 1. 거래량 Z-score
    for w in (20, 40, 120):
        mu = g["volume"].rolling(w, min_periods=max(2, w // 3)).mean()
        sd = g["volume"].rolling(w, min_periods=max(2, w // 3)).std().replace(0, 1.0)
        g[f"vz{w}"] = ((g["volume"] - mu) / sd).fillna(0.0)

    # 2. 거래량 급등
    g["vol_spike"] = (g["volume"] / g["volume"].shift(1) - 1.0).fillna(0.0)

    # 3. 거래량 가속도
    g["vol_accel"] = g["vol_spike"].diff().fillna(0.0)

    # 4. 거래량-가격 상관관계
    for w in [8, 20]:
        g[f"vol_price_corr{w}"] = g["ret1"].rolling(w).corr(g["vol_spike"]).fillna(0.0)

    # ===== ATR =====
    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14, min_periods=5).mean().fillna(0.0)

    # ===== 가격 패턴 =====
    g["hl_spread"] = ((g["high"] - g["low"]) / g["close"]).fillna(0.0)
    g["close_position"] = ((g["close"] - g["low"]) / (g["high"] - g["low"] + 1e-10)).fillna(0.5)
    g["body_size"] = ((g["close"] - g["open"]).abs() / g["open"]).fillna(0.0)
    g["upper_shadow"] = ((g["high"] - g[["open", "close"]].max(axis=1)) / g["close"]).fillna(0.0)
    g["lower_shadow"] = ((g[["open", "close"]].min(axis=1) - g["low"]) / g["close"]).fillna(0.0)

    # 갭
    prev_close_val = g["close"].shift(1)
    g["gap"] = ((g["open"] - prev_close_val) / prev_close_val).fillna(0.0)

    # ===== 추세 분석 =====
    # 1. 수익률 (다양한 기간)
    for w in [2, 4, 8, 12, 24]:
        g[f"ret{w}"] = g["logc"].diff(w).fillna(0.0)

    # 2. 모멘텀 가속도
    g["mom_accel"] = g["mom8"].diff().fillna(0.0)

    # 3. 추세 강도
    for w in [4, 8, 12]:
        g[f"trend_strength{w}"] = g[f"ret{w}"].abs()

    # 4. 가격이 이동평균 위/아래
    for w in [20, 40]:
        ma = g["close"].rolling(w, min_periods=w // 2).mean()
        g[f"above_ma{w}"] = ((g["close"] > ma).astype(float) - 0.5) * 2
        g[f"above_ma{w}"] = g[f"above_ma{w}"].fillna(0.0)

    # ===== RSI =====
    for w in [14, 28]:
        delta = g["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(w, min_periods=w // 2).mean()
        avg_loss = loss.rolling(w, min_periods=w // 2).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        g[f"rsi{w}"] = (100 - (100 / (1 + rs)) - 50) / 50
        g[f"rsi{w}"] = g[f"rsi{w}"].fillna(0.0)

    # ===== MACD =====
    ema12 = g["close"].ewm(span=12, adjust=False).mean()
    ema26 = g["close"].ewm(span=26, adjust=False).mean()
    g["macd"] = ((ema12 - ema26) / g["close"]).fillna(0.0)
    g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean().fillna(0.0)
    g["macd_hist"] = (g["macd"] - g["macd_signal"]).fillna(0.0)

    # ===== 볼린저 밴드 =====
    for w in [20]:
        ma = g["close"].rolling(w, min_periods=w // 2).mean()
        std = g["close"].rolling(w, min_periods=w // 2).std()
        g[f"bb_upper{w}"] = ((ma + 2 * std - g["close"]) / g["close"]).fillna(0.0)
        g[f"bb_lower{w}"] = ((g["close"] - (ma - 2 * std)) / g["close"]).fillna(0.0)
        g[f"bb_width{w}"] = ((4 * std) / ma).fillna(0.0)

    # ===== 시간 패턴 =====
    hod = g["timestamp"].dt.hour
    g["hod_sin"] = np.sin(2 * np.pi * hod / 24.0)
    g["hod_cos"] = np.cos(2 * np.pi * hod / 24.0)

    for h in [0, 6, 12, 18]:
        g[f"hour_{h}"] = (hod == h).astype(float)

    dow = g["timestamp"].dt.dayofweek
    g["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    g["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    g["is_weekend"] = (dow >= 5).astype(float)

    # ===== 최근 극값 =====
    for w in [8, 24, 48]:
        recent_high = g["high"].rolling(w, min_periods=max(2, w // 3)).max()
        recent_low = g["low"].rolling(w, min_periods=max(2, w // 3)).min()
        g[f"price_vs_high{w}"] = ((g["close"] - recent_high) / recent_high).fillna(0.0)
        g[f"price_vs_low{w}"] = ((g["close"] - recent_low) / recent_low).fillna(0.0)

    return g.dropna()


# ===== 대시보드 =====
def print_dashboard(balance: float, positions: List[Position], prices: Dict[str, float]):
    """대시보드 출력 (개선된 버전)"""
    os.system('cls' if os.name == 'nt' else 'clear')

    print("\n" + "=" * 140)
    print(f"{'📊 실전 트레이딩 대시보드':^140}")
    print("=" * 140)

    # 계좌 정보
    used_margin = sum(p.margin for p in positions)
    unrealized_pnl = sum(p.unrealized_pnl for p in positions)
    total_value = balance + unrealized_pnl
    available = balance - used_margin

    print(f"\n💰 계좌 정보")
    print(f"   총 잔고:       ${balance:>12,.2f}")
    print(f"   사용 증거금:   ${used_margin:>12,.2f}")
    print(f"   가용 잔고:     ${available:>12,.2f}")
    print(f"   평가 손익:     ${unrealized_pnl:>+12,.2f}")
    print(f"   총 자산:       ${total_value:>12,.2f}")
    print(f"   일일 실현손익: ${manager.daily_pnl:>+12,.2f}")

    # 포지션
    if positions:
        print(f"\n📍 보유 포지션 ({len(positions)}/{MAX_POSITIONS})")
        print(
            f"{'심볼':^12} | {'방향':^8} | {'진입가':^12} | {'현재가':^12} | {'손익(ROE)':^22} | "
            f"{'SL':^12} | {'TP':^12} | {'청산가':^12} | {'보유':^8}")
        print("-" * 140)

        for pos in positions:
            current_price = prices.get(pos.symbol, pos.entry_price)
            pnl = pos.get_pnl(current_price)
            roe = pos.get_roe(current_price)
            liq_dist = pos.get_liquidation_distance(current_price)

            # 보유 시간 계산
            if pos.symbol in manager.positions_entry_time:
                hold_min = (datetime.now() - manager.positions_entry_time[pos.symbol]).total_seconds() / 60
            else:
                hold_min = (datetime.now() - pos.entry_time).total_seconds() / 60

            emoji = "📈" if pos.direction == "Long" else "📉"
            pnl_emoji = "🟢" if pnl > 0 else "🔴"
            liq_warning = "⚠️" if liq_dist < 3 else ""

            print(f"{pos.symbol:^12} | {emoji} {pos.direction:^6} | ${pos.entry_price:>10,.4f} | "
                  f"${current_price:>10,.4f} | {pnl_emoji} ${pnl:>+8,.2f} ({roe:>+6.1f}%) | "
                  f"${pos.stop_loss:>10,.4f} | ${pos.take_profit:>10,.4f} | "
                  f"${pos.liquidation_price:>10,.4f}{liq_warning} | {hold_min:>6.1f}분")
    else:
        print(f"\n📍 보유 포지션: 없음")

    # 통계
    stats = manager.get_stats()
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

    print("\n" + "=" * 140)


def test_api_connection():
    """API 연결 테스트"""
    print("\n🔍 API 연결 테스트 중...")

    # Public API 테스트
    print("   테스트 심볼: BTCUSDT")
    ticker = API.get_ticker("BTCUSDT")
    if ticker.get("retCode") == 0 and ticker["result"]["list"]:
        price = float(ticker["result"]["list"][0]["lastPrice"])
        print(f"   ✓ Ticker 조회 성공: ${price:,.2f}")
    else:
        print(f"   ✗ Ticker 조회 실패: {ticker.get('retMsg')}")
        return False

    df = API.get_klines("BTCUSDT", interval="5", limit=10)
    if not df.empty:
        print(f"   ✓ Klines 조회 성공: {len(df)}개 캔들")
        print(f"   최신 가격: ${df['close'].iloc[-1]:,.2f}")
    else:
        print(f"   ✗ Klines 조회 실패")
        return False

    # Private API 테스트
    balance = API.get_balance()
    if balance >= 0:
        print(f"   ✓ 잔고 조회 성공: ${balance:,.2f} USDT")
        if balance == 0:
            print(f"   ⚠️  잔고가 0입니다. 입금 후 거래를 시작하세요.")
    else:
        print(f"   ✗ 잔고 조회 실패")
        print(f"\n   확인 사항:")
        print(f"      1. API 권한: Contract Trading, Account Transfer")
        print(f"      2. API 모드: {'Mainnet' if not USE_TESTNET else 'Testnet'}")
        print(f"      3. IP 화이트리스트 설정 확인")
        return False

    # 심볼 정보 사전 로드
    print("\n📋 심볼 정보 로드 중...")
    loaded_count = 0
    for symbol in MODELS.keys():
        try:
            info = API.get_instrument_info(symbol)
            loaded_count += 1
            if DEBUG_MODE:
                print(f"   ✓ {symbol}: minQty={info['minOrderQty']}, qtyStep={info['qtyStep']}")
        except Exception as e:
            print(f"   ⚠️  {symbol}: 정보 로드 실패 - {e}")

    print(f"   ✓ {loaded_count}/{len(MODELS)}개 심볼 정보 로드 완료")

    print("\n✅ API 연결 정상\n")
    return True


# ===== 메인 루프 =====
def main():
    # 경고 메시지
    mode = "🧪 TESTNET" if USE_TESTNET else "🔴 LIVE"
    print("\n" + "=" * 110)
    print(f"{'⚠️  실전 자동 트레이딩 시스템  ⚠️':^110}")
    print(f"{mode:^110}")
    print("=" * 110)
    print("\n⚠️  경고:")
    print("   - 이 프로그램은 실제 자금을 사용합니다")
    print("   - 투자 손실의 위험이 있습니다")
    print("   - 본인 책임 하에 사용하세요")
    print("\nAPI 설정:")
    print(f"   - API Key: {API_KEY[:8]}...{API_KEY[-4:]} (길이: {len(API_KEY)})")
    print(f"   - API Secret: {'*' * 8}...{'*' * 4} (길이: {len(API_SECRET)})")
    print(f"   - 모드: {'Testnet' if USE_TESTNET else 'Mainnet (실전!)'}")
    print(f"   - 베이스 URL: {API.base_url}")

    # API 키 유효성 간단 체크
    if len(API_KEY) < 10 or len(API_SECRET) < 10:
        print("\n❌ ERROR: API Key 또는 Secret이 너무 짧습니다!")
        print("   환경변수가 제대로 설정되었는지 확인하세요.")
        print("\n현재 값:")
        print(f"   BYBIT_API_KEY 길이: {len(API_KEY)}")
        print(f"   BYBIT_API_SECRET 길이: {len(API_SECRET)}")
        return
    print("\n거래 설정:")
    print(f"   - 포지션 모드: {POSITION_MODE.upper()}")
    print(f"   - 레버리지: {LEVERAGE}x")
    print(f"   - 증거금/포지션: ${MARGIN_PER_POSITION}")
    print(f"   - 포지션 크기: ${MARGIN_PER_POSITION * LEVERAGE} (증거금 × 레버리지)")
    print(f"   - 최대 포지션: {MAX_POSITIONS}개")
    print(f"   - 일일 최대 손실: ${MAX_DAILY_LOSS}")
    print(f"   - 신뢰도 임계값: {CONF_THRESHOLD:.0%}")
    print(f"   - 사용 모델: {len(MODELS)}개 심볼별 모델")

    # 확인
    print("\n계속하시겠습니까? (y/n): ", end='')
    if input().lower() != 'y':
        print("프로그램을 종료합니다.")
        return

    # API 연결 테스트
    if not test_api_connection():
        print("\n❌ API 연결 실패. 프로그램을 종료합니다.")
        print("\n확인 사항:")
        print("   1. API Key와 Secret이 올바른지 확인")
        print("   2. API 권한 설정 확인 (Contract, Position, Wallet)")
        print("   3. IP 화이트리스트 설정 확인")
        print("   4. Testnet/Mainnet 모드 확인")
        return

    print("\n시작합니다...\n")

    try:
        loop_count = 0
        while True:
            loop_count += 1
            current_time = datetime.now()

            # 일일 통계 리셋
            manager.reset_daily_stats()

            # 계좌 정보
            balance = API.get_balance()
            positions = API.get_positions()

            # 현재 가격 가져오기 (모델이 있는 심볼만)
            prices = {}
            for symbol in MODELS.keys():
                ticker = API.get_ticker(symbol)
                if ticker.get("retCode") == 0 and ticker["result"]["list"]:
                    prices[symbol] = float(ticker["result"]["list"][0]["lastPrice"])

            # 포지션 관리
            for position in list(positions):  # 리스트 복사로 순회 중 수정 방지
                current_price = prices.get(position.symbol, position.entry_price)

                # ✅ 수정: entry_time 동기화
                # 1. manager에 기록된 시간이 있으면 사용
                if position.symbol in manager.positions_entry_time:
                    entry_time_to_use = manager.positions_entry_time[position.symbol]
                else:
                    # 2. API에서 가져온 시간 사용
                    entry_time_to_use = position.entry_time
                    # 3. manager에 등록 (다음 루프부터 사용)
                    manager.positions_entry_time[position.symbol] = position.entry_time

                # 보유 시간 계산
                hold_minutes = (current_time - entry_time_to_use).total_seconds() / 60

                # 현재 손익
                pnl = position.get_pnl(current_price)
                roe = position.get_roe(current_price)

                # 청산 조건 디버그
                print(f"\n[체크] {position.symbol} | 보유: {hold_minutes:.1f}분 | 손익: ${pnl:+.2f} ({roe:+.1f}%)")
                print(f"       진입: ${position.entry_price:.4f} | 현재: ${current_price:.4f}")
                print(f"       SL: ${position.stop_loss:.4f} | TP: ${position.take_profit:.4f}")

                # ✅ 청산 조건 직접 체크 (hold_minutes 사용)
                should_close = False
                reason = ""

                # 1. 청산가
                if position.direction == "Long" and current_price <= position.liquidation_price:
                    should_close, reason = True, "Liquidation"
                elif position.direction == "Short" and current_price >= position.liquidation_price:
                    should_close, reason = True, "Liquidation"
                # 2. 손절
                elif position.direction == "Long" and current_price <= position.stop_loss:
                    should_close, reason = True, "Stop Loss"
                elif position.direction == "Short" and current_price >= position.stop_loss:
                    should_close, reason = True, "Stop Loss"
                # 3. 익절
                elif position.direction == "Long" and current_price >= position.take_profit:
                    should_close, reason = True, "Take Profit"
                elif position.direction == "Short" and current_price <= position.take_profit:
                    should_close, reason = True, "Take Profit"
                # 4. 시간 초과
                elif hold_minutes >= MAX_HOLD_MINUTES:
                    should_close, reason = True, "Time Limit"

                if should_close:
                    print(f"       ➡️  청산 조건 충족: {reason}")
                    manager.close_position(position, reason)
                else:
                    print(f"       ✓ 유지")

                # ✅ 반대 신호 청산 제거 - TP/SL/시간으로만 관리

            # 포지션 업데이트 (청산 후)
            new_positions = API.get_positions()

            # Binance 자동 청산 감지
            if len(new_positions) < len(positions):
                for old_pos in positions:
                    if not any(p.symbol == old_pos.symbol for p in new_positions):
                        print(f"\n🔔 {old_pos.symbol} 포지션이 사라졌습니다!")
                        print(f"   💡 Binance가 TP/SL을 자동 실행했을 가능성이 있습니다")
                        print(f"   📊 Binance 웹사이트 > Orders > Closed에서 확인하세요")

            positions = new_positions

            # 대시보드 출력
            print_dashboard(balance, positions, prices)

            # 신호 스캔 테이블
            print(f"\n🔍 신호 스캔 ({len(MODELS)}개 심볼)")
            print(f"{'심볼':^12} | {'가격':^12} | {'방향':^10} | {'신뢰도':^8} | {'신호':^20}")
            print("-" * 80)

            # 신호 스캔 및 진입 (모델이 있는 심볼만)
            debug_mode = (loop_count == 1)  # 첫 스캔만 디버그 모드
            for symbol in MODELS.keys():
                result = predict(symbol, debug=debug_mode)

                if "error" in result:
                    print(f"{symbol:^12} | {'N/A':^12} | {'오류':^10} | {'N/A':^8} | ❌ {result.get('error', '알 수 없음')}")
                    continue

                direction = result["direction"]
                confidence = result["confidence"]
                price = result["current_price"]

                # 기존 포지션 확인
                existing = next((p for p in positions if p.symbol == symbol), None)
                position_status = ""
                if existing:
                    if existing.direction == direction:
                        position_status = f" [보유: {existing.direction}]"
                    else:
                        position_status = f" [전환: {existing.direction}→{direction}]"

                # 방향 이모지
                dir_icon = {"Long": "📈", "Short": "📉", "Flat": "➖"}.get(direction, "❓")

                # 신호 판단
                if confidence < CONF_THRESHOLD:
                    signal = f"⚠️  신호 약함 ({confidence:.1%}){position_status}"
                elif direction == "Long":
                    signal = f"🟢 매수 신호 ({confidence:.1%}){position_status}"
                elif direction == "Short":
                    signal = f"🔴 매도 신호 ({confidence:.1%}){position_status}"
                else:
                    signal = f"⚪ 관망 ({confidence:.1%}){position_status}"

                print(
                    f"{symbol:^12} | ${price:>10,.4f} | {dir_icon} {direction:^8} | {confidence:>6.1%} | {signal:^20}")

                # 진입 조건
                if confidence >= CONF_THRESHOLD and direction in ["Long", "Short"]:
                    # 기존 포지션이 있고 반대 방향이면 청산
                    if existing and existing.direction != direction:
                        print(f"\n⚠️  {symbol}: 반대 방향 신호 감지 ({existing.direction} → {direction})")
                        print(f"   기존 포지션 청산 중...")
                        manager.close_position(existing, "Reverse Signal")
                        time.sleep(1)  # 청산 처리 대기
                        positions = API.get_positions()  # 포지션 재조회

                    # 새로운 포지션 진입 (기존 포지션이 없거나 청산 후)
                    if manager.can_open_position(symbol, direction, positions):
                        if manager.open_position(symbol, direction, price):
                            time.sleep(1)  # 주문 처리 대기
                            positions = API.get_positions()  # 포지션 업데이트

            print(f"\n[스캔 #{loop_count}] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"다음 스캔까지 {INTERVAL_SEC}초... (Ctrl+C로 종료)")

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n프로그램 종료 중...")

        # 최종 상태
        balance = API.get_balance()
        positions = API.get_positions()
        prices = {}
        for symbol in MODELS.keys():
            ticker = API.get_ticker(symbol)
            if ticker.get("retCode") == 0 and ticker["result"]["list"]:
                prices[symbol] = float(ticker["result"]["list"][0]["lastPrice"])

        print_dashboard(balance, positions, prices)
        manager.save_trades()

        # 최종 통계
        stats = manager.get_stats()
        if stats["total_trades"] > 0:
            print("\n" + "=" * 110)
            print(f"{'📊 최종 결과':^110}")
            print("=" * 110)
            print(f"   최종 잔고:     ${balance:,.2f}")
            print(f"   일일 손익:     ${manager.daily_pnl:+,.2f}")
            print(f"   총 거래:       {stats['total_trades']}회")
            print(f"   승률:          {stats['win_rate']:.1f}%")
            print(f"   평균 ROE:      {stats['avg_roe']:+.1f}%")
            if stats['liquidations'] > 0:
                print(f"   강제 청산:     {stats['liquidations']}회 💀")
            print("=" * 110)

        if positions:
            print("\n⚠️  주의: 아직 포지션이 남아있습니다!")
            for pos in positions:
                pnl = pos.get_pnl(prices.get(pos.symbol, pos.entry_price))
                print(f"   - {pos.symbol}: {pos.direction} | ${pnl:+,.2f}")

        print("\n✅ 프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()