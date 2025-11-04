# live_trading.py
# -*- coding: utf-8 -*-
"""
TCN 모델 기반 실전 자동 트레이딩 시스템
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
API_KEY = "Dlp4eJD6YFmO99T8vC"
API_SECRET = "YYYB5tMw2TWvfVF5wqi6lQRHqEIiDSpDJF1U"
USE_TESTNET = os.getenv("USE_TESTNET", "0") == "1"  # 기본값: Testnet

if not API_KEY or not API_SECRET:
    print("❌ ERROR: BYBIT_API_KEY 및 BYBIT_API_SECRET 환경변수를 설정하세요!")
    print("\n설정 방법:")
    print("  export BYBIT_API_KEY='your_api_key'")
    print("  export BYBIT_API_SECRET='your_api_secret'")
    print("  export USE_TESTNET=1  # Testnet (기본값)")
    print("  export USE_TESTNET=0  # Mainnet (실전)")
    print("\n포지션 모드 설정:")
    print("  export POSITION_MODE=hedge     # Hedge Mode (양방향, 기본값)")
    print("  export POSITION_MODE=one-way   # One-Way Mode (단방향)")
    print("\n⚠️  중요:")
    print("  - Testnet API Key: https://testnet.bybit.com 에서 발급")
    print("  - Mainnet API Key: https://www.bybit.com 에서 발급")
    print("  - Testnet과 Mainnet API Key는 서로 다릅니다!")
    print("  - 포지션 모드는 Bybit 웹사이트 설정과 일치해야 합니다!")
    exit(1)

# 거래 설정
SYMBOLS = os.getenv("SYMBOLS", "PIPPINUSDT").split(",")
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "2"))
TCN_CKPT = os.getenv("TCN_CKPT", "D:/ygy_work/coin/multimodel/models_5min/5min_2class_best_v1.ckpt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.8"))  # 실전은 더 높게

# 리스크 관리 (매우 중요!)
MARGIN_PER_POSITION = float(os.getenv("MARGIN_PER_POSITION", "30"))  # 포지션당 증거금
LEVERAGE = int(os.getenv("LEVERAGE", "100"))  # 레버리지 (실전은 낮게!)
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))  # 최대 동시 포지션
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))  # 손절 1.5%
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.03"))  # 익절 2.5%
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "500"))  # 일일 최대 손실
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "30"))

# 포지션 모드
POSITION_MODE = os.getenv("POSITION_MODE", "one-way").lower()  # "one-way" 또는 "hedge"

# 로그
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "live_trades.json")
ORDER_LOG_FILE = os.getenv("ORDER_LOG_FILE", "orders.json")
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"


# ===== 데이터 클래스 =====
@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    side: str  # "Buy" or "Sell"
    size: float
    entry_price: float
    leverage: int
    unrealized_pnl: float
    position_value: float

    def get_direction(self) -> str:
        return "Long" if self.side == "Buy" else "Short"


@dataclass
class Order:
    """주문 정보"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    qty: float
    price: float
    status: str
    timestamp: str


@dataclass
class Trade:
    """거래 기록"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    leverage: int
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float
    exit_reason: str
    margin: float = 0.0  # ✅ 추가!
    roe: float = 0.0  # ✅ 추가!


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
print(f"[INIT] 모델 로드 중: {TCN_CKPT}")
try:
    checkpoint = torch.load(TCN_CKPT, map_location="cpu", weights_only=False)
    FEAT_COLS = checkpoint['feat_cols']
    META = checkpoint.get('meta', {})
    SEQ_LEN = META.get('seq_len', 60)
    SCALER_MU = checkpoint.get('scaler_mu')
    SCALER_SD = checkpoint.get('scaler_sd')

    # 체크포인트에서 실제 모델 구조 자동 감지
    model_dict = checkpoint['model']

    # 1. TCN 레이어 수 감지
    max_layer = 0
    for key in model_dict.keys():
        if key.startswith('tcn.') and '.' in key[4:]:
            try:
                layer_num = int(key.split('.')[1])
                max_layer = max(max_layer, layer_num)
            except:
                pass
    levels = max_layer + 1

    # 2. Hidden channels 감지
    if 'tcn.0.c1.weight_v' in model_dict:
        hidden = model_dict['tcn.0.c1.weight_v'].shape[0]
    else:
        hidden = META.get('hidden', 32)

    # 3. Kernel size 감지
    if 'tcn.0.c1.weight_v' in model_dict:
        k = model_dict['tcn.0.c1.weight_v'].shape[2]
    else:
        k = META.get('k', 3)

    # 4. Dropout
    drop = META.get('dropout', 0.2)

    # 5. Head 타입 감지 (Single-task vs Multi-task)
    has_cls_head = 'head_cls.weight' in model_dict
    has_ttt_head = 'head_ttt.weight' in model_dict
    has_single_head = 'head.weight' in model_dict

    print(f"   📊 감지된 모델 구조:")
    print(f"      - Hidden channels: {hidden}")
    print(f"      - TCN levels: {levels}")
    print(f"      - Kernel size: {k}")
    print(f"      - Dropout: {drop}")
    print(f"      - Head type: {'Multi-task' if (has_cls_head or has_ttt_head) else 'Single-task'}")

    # 적절한 모델 클래스 선택
    if has_single_head:
        # Single-task 모델 (2-class)
        print(f"   ⚠️  경고: Single-task 모델이 감지되었습니다.")
        print(f"   이 모델은 방향만 예측하고 TP는 예측하지 않습니다.")


        # Single-task 모델 정의
        class TCN_SingleTask(nn.Module):
            def __init__(self, in_f, hidden, levels, k, drop):
                super().__init__()
                L = []
                ch = in_f
                for i in range(levels):
                    L.append(Block(ch, hidden, k, 2 ** i, drop))
                    ch = hidden
                self.tcn = nn.Sequential(*L)
                self.head = nn.Linear(hidden, 2)  # Single head for 2-class

            def forward(self, X):
                X = X.transpose(1, 2)
                H = self.tcn(X)[:, :, -1]
                return self.head(H), torch.zeros((X.shape[0], 1))  # Dummy TTT


        MODEL = TCN_SingleTask(in_f=len(FEAT_COLS), hidden=hidden, levels=levels, k=k, drop=drop).eval()
    else:
        # Multi-task 모델
        MODEL = TCN_MT(in_f=len(FEAT_COLS), hidden=hidden, levels=levels, k=k, drop=drop).eval()

    # 모델 로드 (strict=False로 누락된 키 무시)
    MODEL.load_state_dict(checkpoint['model'], strict=False)
    print(f"   ✓ 모델 로드 완료")

    # 🔍 디버그 정보
    print(f"\n[LIVE MODEL INFO]")
    print(f"   체크포인트: {TCN_CKPT}")
    print(f"   SEQ_LEN: {SEQ_LEN}")
    print(f"   FEAT_COLS 개수: {len(FEAT_COLS)}")
    if 'head.weight' in checkpoint['model']:
        hw = checkpoint['model']['head.weight']
        print(f"   head.weight shape: {hw.shape}")
        print(f"   head.weight sum: {hw.sum():.10f}")
        print(f"   head.weight[0] mean: {hw[0].mean():.10f}")
        print(f"   head.weight[1] mean: {hw[1].mean():.10f}")
    print()
    print(f"\n[SCALER INFO]")
    print(f"  SCALER_MU type: {type(SCALER_MU)}")
    print(f"  SCALER_MU shape: {SCALER_MU.shape if hasattr(SCALER_MU, 'shape') else 'NO SHAPE'}")
    print(f"  SCALER_SD type: {type(SCALER_SD)}")
    print(f"  SCALER_SD shape: {SCALER_SD.shape if hasattr(SCALER_SD, 'shape') else 'NO SHAPE'}")

    if hasattr(SCALER_MU, '__len__'):
        print(f"  SCALER_MU[:5]: {SCALER_MU[:5]}")
        print(f"  SCALER_SD[:5]: {SCALER_SD[:5]}")
        print(f"  SCALER_MU mean: {SCALER_MU.mean():.6f}")
        print(f"  SCALER_SD mean: {SCALER_SD.mean():.6f}")
    else:
        print(f"  SCALER_MU: {SCALER_MU}")
        print(f"  SCALER_SD: {SCALER_SD}")
except Exception as e:
    print(f"   ✗ 모델 로드 실패: {e}")
    print(f"\n💡 해결 방법:")
    print(f"   1. check_checkpoint.py로 체크포인트 분석:")
    print(f'      python check_checkpoint.py "{TCN_CKPT}"')
    print(f"   2. 올바른 체크포인트 경로 확인")
    print(f"   3. 다른 체크포인트 파일 시도")
    import traceback

    traceback.print_exc()
    exit(1)

    import traceback

    traceback.print_exc()
    exit(1)


# ===== BYBIT API =====
class BybitAPI:
    """Bybit Private API"""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self.session = requests.Session()
        self.session.verify = certifi.where()
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

    def _request(self, method: str, endpoint: str, params: dict = None) -> dict:
        """API 요청"""
        if params is None:
            params = {}

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
        headers = {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-SIGN': signature,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': self.recv_window,
            'Content-Type': 'application/json'
        }

        url = f"{self.base}{endpoint}"

        if DEBUG_MODE:
            print(f"\n[DEBUG] {method} {endpoint}")
            print(f"[DEBUG] Timestamp: {timestamp}")
            print(f"[DEBUG] Params: {params}")
            print(f"[DEBUG] Params str: {params_str}")
            print(f"[DEBUG] Signature: {signature}")

        try:
            if method == "GET":
                r = self.session.get(url, params=params, headers=headers, timeout=10)
            else:
                r = self.session.post(url, json=params, headers=headers, timeout=10)

            if DEBUG_MODE:
                print(f"[DEBUG] Status: {r.status_code}")
                print(f"[DEBUG] Response: {r.text[:500]}")

            if r.status_code == 401:
                print(f"❌ 인증 실패 (401)")
                print(f"   API Key: {self.api_key[:8]}...{self.api_key[-4:]}")
                print(f"   Endpoint: {endpoint}")
                print(f"   모드: {'Testnet' if 'testnet' in self.base else 'Mainnet'}")
                print(f"   응답: {r.text[:200] if r.text else '(비어있음)'}")
                return {}

            if r.status_code != 200:
                print(f"❌ HTTP Error {r.status_code}: {r.text[:200]}")
                return {}

            data = r.json()

            # retCode 확인
            ret_code = data.get("retCode", -1)

            # 110043: leverage not modified - 이미 레버리지가 설정되어 있음 (정상)
            if ret_code == 110043:
                if DEBUG_MODE:
                    print(f"[DEBUG] 레버리지 이미 설정됨 (정상)")
                return data.get("result", {})

            if ret_code != 0:
                print(f"❌ API Error [{ret_code}]: {data.get('retMsg', 'Unknown error')}")
                if DEBUG_MODE:
                    print(f"[DEBUG] Full response: {data}")
                return {}

            return data.get("result", {})

        except json.JSONDecodeError as e:
            print(f"❌ JSON Decode Error: {e}")
            print(f"   Response: {r.text[:200]}")
            return {}
        except Exception as e:
            print(f"❌ Request Error: {e}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
            return {}

    def get_balance(self) -> float:
        """계좌 잔고 조회"""
        result = self._request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})
        if not result:
            print("⚠️  잔고 조회 실패 - API 인증 정보를 확인하세요")
            return 0.0

        try:
            coins = result.get("list", [{}])[0].get("coin", [])
            for coin in coins:
                if coin.get("coin") == "USDT":
                    balance = float(coin.get("walletBalance", 0))
                    print(f"✓ 잔고 조회 성공: ${balance:,.2f}")
                    return balance
        except Exception as e:
            print(f"❌ 잔고 파싱 오류: {e}")

        return 0.0

    def get_positions(self) -> List[Position]:
        """포지션 조회"""
        result = self._request("GET", "/v5/position/list", {
            "category": "linear",
            "settleCoin": "USDT"
        })
        if not result:
            return []

        positions = []
        for item in result.get("list", []):
            size = float(item.get("size", 0))
            if size == 0:
                continue

            positions.append(Position(
                symbol=item["symbol"],
                side=item["side"],
                size=size,
                entry_price=float(item.get("avgPrice", 0)),
                leverage=int(item.get("leverage", 1)),
                unrealized_pnl=float(item.get("unrealisedPnl", 0)),
                position_value=float(item.get("positionValue", 0))
            ))
        return positions

    def get_instrument_info(self, symbol: str) -> dict:
        """심볼 정보 조회 (최소 주문 수량 등)"""
        url = f"{self.base}/v5/market/instruments-info"
        try:
            r = self.session.get(url, params={"category": "linear", "symbol": symbol}, timeout=5)
            data = r.json()
            if data.get("retCode") == 0:
                items = data.get("result", {}).get("list", [])
                if items:
                    return items[0]
        except:
            pass
        return {}

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """레버리지 설정"""
        result = self._request("POST", "/v5/position/set-leverage", {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        })

        # retCode가 없거나 110043(leverage not modified)이면 성공으로 처리
        # 이미 레버리지가 설정되어 있으면 변경할 필요 없음
        if not result:
            # API 호출 실패
            return False

        # 성공으로 간주 (이미 설정되어 있어도 OK)
        if DEBUG_MODE:
            print(f"[DEBUG] 레버리지 설정 완료 또는 이미 설정됨: {symbol} {leverage}x")
        return True

    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = "Market", price: float = None) -> Optional[str]:
        """주문 생성"""
        # positionIdx 결정
        if POSITION_MODE == "one-way":
            position_idx = 0
        else:  # hedge mode
            position_idx = 1 if side == "Buy" else 2

        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "positionIdx": position_idx
        }

        # Market 주문이 아닐 때만 가격 추가
        if order_type == "Limit" and price:
            params["price"] = str(price)
            params["timeInForce"] = "GTC"

        if DEBUG_MODE:
            print(f"[DEBUG] 주문 생성: {symbol} {side} {qty} (positionIdx={position_idx})")

        result = self._request("POST", "/v5/order/create", params)
        order_id = result.get("orderId")

        if order_id:
            # 주문 로그 저장
            self._log_order({
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "qty": qty,
                "price": price,
                "timestamp": datetime.now().isoformat()
            })
            print(f"✓ 주문 생성 성공: {order_id}")
        else:
            print(f"✗ 주문 생성 실패")

        return order_id

    def close_position(self, symbol: str, side: str, qty: float) -> Optional[str]:
        """포지션 청산"""
        # 포지션 방향의 반대로 주문
        close_side = "Sell" if side == "Buy" else "Buy"
        return self.place_order(symbol, close_side, qty, "Market")

    def set_trading_stop(self, symbol: str, side: str,
                         stop_loss: float = None, take_profit: float = None) -> bool:
        """손절/익절 설정"""
        # positionIdx 결정
        if POSITION_MODE == "one-way":
            position_idx = 0
        else:  # hedge mode
            position_idx = 1 if side == "Buy" else 2

        params = {
            "category": "linear",
            "symbol": symbol,
            "positionIdx": position_idx
        }

        if stop_loss:
            params["stopLoss"] = str(stop_loss)
        if take_profit:
            params["takeProfit"] = str(take_profit)

        if DEBUG_MODE:
            print(f"[DEBUG] 손절/익절 설정: {symbol} SL={stop_loss} TP={take_profit} (positionIdx={position_idx})")

        result = self._request("POST", "/v5/position/trading-stop", params)
        return bool(result)

    def get_ticker(self, symbol: str) -> dict:
        """현재가 조회 (Public)"""
        url = f"{self.base}/v5/market/tickers"
        try:
            r = self.session.get(url, params={"category": "linear", "symbol": symbol}, timeout=5)
            data = r.json()
            rows = ((data.get("result") or {}).get("list") or [])
            return rows[0] if rows else {}
        except:
            return {}

    def get_kline(self, symbol: str, interval: str, limit: int):
        """K라인 조회 (Public)"""
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
        except:
            return []

    def _log_order(self, order: dict):
        """주문 로그 저장"""
        try:
            orders = []
            if os.path.exists(ORDER_LOG_FILE):
                with open(ORDER_LOG_FILE, 'r') as f:
                    orders = json.load(f)

            orders.append(order)

            with open(ORDER_LOG_FILE, 'w') as f:
                json.dump(orders, f, indent=2)
        except:
            pass


# API 초기화
API = BybitAPI(API_KEY, API_SECRET, testnet=USE_TESTNET)


# ===== 트레이딩 매니저 =====
class TradingManager:
    """트레이딩 관리"""

    def __init__(self):
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        self.position_entry_times: Dict[str, datetime] = {}

    def check_daily_loss_limit(self) -> bool:
        """일일 손실 한도 체크"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = today

        if self.daily_pnl <= -MAX_DAILY_LOSS:
            print(f"\n⛔ 일일 손실 한도 도달: ${self.daily_pnl:.2f}")
            return False
        return True

    def can_open_position(self, symbol: str, current_positions: List[Position]) -> bool:
        """포지션 진입 가능 여부"""
        # 최대 포지션 수 체크 (같은 심볼은 제외)
        other_symbols_count = len([p for p in current_positions if p.symbol != symbol])
        if other_symbols_count >= MAX_POSITIONS:
            return False

        # 일일 손실 한도 체크
        if not self.check_daily_loss_limit():
            return False

        return True

    def open_position(self, symbol: str, direction: str, price: float) -> bool:
        """포지션 진입"""
        print(f"\n{'=' * 80}")
        print(f"💡 포지션 진입 시도: {symbol} {direction}")

        # 기존 포지션 확인 및 청산
        existing_positions = API.get_positions()
        for pos in existing_positions:
            if pos.symbol == symbol:
                existing_direction = pos.get_direction()

                # 같은 방향이면 진입하지 않음
                if existing_direction == direction:
                    print(f"⚠️  이미 같은 방향({direction}) 포지션 보유 중 | 손익: ${pos.unrealized_pnl:+,.2f}")
                    print(f"   진입을 건너뜁니다.")
                    print(f"{'=' * 80}\n")
                    return False

                # 반대 방향이면 청산 후 진입
                print(f"⚠️  반대 방향 포지션 발견: {existing_direction} | 손익: ${pos.unrealized_pnl:+,.2f}")
                print(f"   청산 후 {direction} 포지션으로 진입합니다...")

                # 기존 포지션 청산
                close_order_id = API.close_position(pos.symbol, pos.side, pos.size)
                if close_order_id:
                    print(f"✓ 기존 포지션 청산 완료 (주문 ID: {close_order_id})")

                    # 거래 기록
                    ticker = API.get_ticker(symbol)
                    current_price = float(ticker.get("lastPrice", price))

                    pnl = pos.unrealized_pnl
                    self.daily_pnl += pnl

                    # ROE 계산
                    margin = pos.position_value / pos.leverage
                    roe = (pnl / margin) * 100 if margin > 0 else 0

                    entry_time = self.position_entry_times.get(symbol)
                    trade = Trade(
                        symbol=symbol,
                        direction=existing_direction,
                        entry_price=pos.entry_price,
                        exit_price=current_price,
                        size=pos.size,
                        leverage=pos.leverage,
                        margin=margin,
                        entry_time=entry_time.isoformat() if entry_time else "",
                        exit_time=datetime.now().isoformat(),
                        pnl=pnl,
                        pnl_pct=(pnl / margin) * 100 if margin > 0 else 0,
                        roe=roe,
                        exit_reason="Position Reversal"
                    )
                    self.trades.append(trade)
                    self.save_trades()

                    # 진입 시간 삭제
                    if symbol in self.position_entry_times:
                        del self.position_entry_times[symbol]

                    time.sleep(2)  # 청산 완료 대기
                else:
                    print(f"⚠️  기존 포지션 청산 실패 - 진입을 중단합니다")
                    print(f"{'=' * 80}\n")
                    return False

        # 레버리지 설정
        if not API.set_leverage(symbol, LEVERAGE):
            print(f"⚠️  레버리지 설정 실패했지만 계속 진행 (이미 설정되어 있을 수 있음)")

        # 심볼 정보 조회
        instrument = API.get_instrument_info(symbol)
        lot_size_filter = instrument.get("lotSizeFilter", {})
        min_order_qty = float(lot_size_filter.get("minOrderQty", 0.01))
        max_order_qty = float(lot_size_filter.get("maxMktOrderQty", 1000000000))
        qty_step = float(lot_size_filter.get("qtyStep", 0.01))

        # 포지션 크기 계산 (증거금 * 레버리지)
        position_value = MARGIN_PER_POSITION * LEVERAGE
        qty = position_value / price

        # qtyStep에 맞춰 조정
        qty = round(qty / qty_step) * qty_step


        # 최소/최대 수량 체크
        if qty < min_order_qty:
            qty = min_order_qty
        elif qty > max_order_qty:
            qty = max_order_qty


        # 최종 반올림 (소수점 처리)
        if qty_step >= 1:
            qty = int(qty)
        else:
            decimal_places = len(str(qty_step).split('.')[-1].rstrip('0'))
            qty = round(qty, decimal_places)



        print(f"   증거금: ${MARGIN_PER_POSITION:,.2f}")
        print(f"   레버리지: {LEVERAGE}x")
        print(f"   포지션 크기: ${position_value:,.2f}")
        print(f"   계산된 수량: {qty}")
        print(f"   수량 단위: {qty_step}")
        print(f"   진입가: ${price:,.4f}")

        # 주문 생성
        side = "Buy" if direction == "Long" else "Sell"
        order_id = API.place_order(symbol, side, qty, "Market")

        if not order_id:
            print(f"❌ 주문 실패: {symbol}")
            print(f"{'=' * 80}\n")
            return False

        # 손절/익절 설정
        if direction == "Long":
            stop_loss = price * (1 - STOP_LOSS_PCT)
            take_profit = price * (1 + TAKE_PROFIT_PCT)
        else:
            stop_loss = price * (1 + STOP_LOSS_PCT)
            take_profit = price * (1 - TAKE_PROFIT_PCT)

        time.sleep(2)  # 포지션 생성 대기

        # 손절/익절 설정 (실패해도 계속)
        if not API.set_trading_stop(symbol, side, stop_loss, take_profit):
            print(f"⚠️  손절/익절 설정 실패 (수동으로 관리됩니다)")

        # 진입 시간 기록
        self.position_entry_times[symbol] = datetime.now()

        print(f"✅ 포지션 진입 성공!")
        print(f"   주문 ID: {order_id}")
        print(f"   방향: {direction}")
        print(f"   수량: {qty}")
        print(f"   손절가: ${stop_loss:,.4f} ({STOP_LOSS_PCT * 100:.1f}%)")
        print(f"   익절가: ${take_profit:,.4f} ({TAKE_PROFIT_PCT * 100:.1f}%)")
        print(f"   일일 손익: ${self.daily_pnl:+,.2f}")
        print(f"{'=' * 80}\n")

        return True

    def should_close_position(self, position: Position) -> tuple[bool, str]:
        """포지션 청산 여부"""
        symbol = position.symbol

        # 시간 초과
        if symbol in self.position_entry_times:
            hold_time = (datetime.now() - self.position_entry_times[symbol]).total_seconds() / 60
            if hold_time >= MAX_HOLD_MINUTES:
                return True, "Time Limit"

        return False, ""

    def close_position(self, position: Position, reason: str) -> bool:
        """포지션 청산"""
        order_id = API.close_position(position.symbol, position.side, position.size)

        if not order_id:
            print(f"❌ 청산 실패: {position.symbol}")
            return False

        # 현재가
        ticker = API.get_ticker(position.symbol)
        current_price = float(ticker.get("lastPrice", 0))

        # 손익
        pnl = position.unrealized_pnl
        self.daily_pnl += pnl

        # 거래 기록
        direction = position.get_direction()
        entry_time = self.position_entry_times.get(position.symbol)

        trade = Trade(
            symbol=position.symbol,
            direction=direction,
            entry_price=position.entry_price,
            exit_price=current_price,
            size=position.size,
            leverage=position.leverage,
            entry_time=entry_time.isoformat() if entry_time else "",
            exit_time=datetime.now().isoformat(),
            pnl=pnl,
            pnl_pct=(pnl / (position.position_value / LEVERAGE)) * 100,
            exit_reason=reason
        )

        self.trades.append(trade)
        self.save_trades()

        # 진입 시간 삭제
        if position.symbol in self.position_entry_times:
            del self.position_entry_times[position.symbol]

        emoji = "🟢" if pnl > 0 else "🔴"
        print(f"\n{'=' * 80}")
        print(f"{emoji} 포지션 청산: {position.symbol}")
        print(f"   주문 ID: {order_id}")
        print(f"   이유: {reason}")
        print(f"   진입가: ${position.entry_price:,.4f}")
        print(f"   청산가: ${current_price:,.4f}")
        print(f"   손익: ${pnl:+,.2f}")
        print(f"   일일 손익: ${self.daily_pnl:+,.2f}")
        print(f"{'=' * 80}\n")

        return True

    def save_trades(self):
        """거래 내역 저장"""
        if not self.trades:
            return

        data = [asdict(t) for t in self.trades]
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(data, f, indent=2)


manager = TradingManager()


# ===== 데이터 처리 =====
def get_recent_data(symbol: str, minutes: int = 300) -> Optional[pd.DataFrame]:
    """최근 데이터 가져오기"""
    lst = API.get_kline(symbol, "1", minutes)
    if not lst:
        return None

    rows = lst[::-1]
    df = pd.DataFrame([{
        "timestamp": pd.to_datetime(int(z[0]), unit="ms", utc=True),
        "open": float(z[1]),
        "high": float(z[2]),
        "low": float(z[3]),
        "close": float(z[4]),
        "volume": float(z[5]),
    } for z in rows])

    return df


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_trading용 make_features 함수
train_tcn_5minutes.py와 완전히 동일한 피처 생성
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    실시간 트레이딩용 Feature Engineering
    train_tcn_5minutes.py의 make_features()와 동일
    단, 단일 심볼용으로 간소화 (groupby 제거)
    """
    g = df.copy().sort_values("timestamp").reset_index(drop=True)

    # timestamp를 date로 변경 (train과 일치)
    if 'date' not in g.columns:
        g['date'] = pd.to_datetime(g['timestamp'], unit='ms', utc=True)

    # ===== 기본 수익률 =====
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g["logc"].diff().fillna(0.0)

    # ===== 변동성 =====
    for w in (8, 20, 40, 120):
        g[f"rv{w}"] = g["ret1"].rolling(w, min_periods=max(2, w // 3)).std()

    # ===== 모멘텀 =====
    for w in (8, 20, 40, 120):
        g[f"mom{w}"] = g["close"] / g["close"].ewm(span=w, adjust=False).mean() - 1.0

    # ===== 볼륨 분석 =====
    # 1. 거래량 Z-score
    for w in (20, 40, 120):
        mu = g["volume"].rolling(w, min_periods=max(2, w // 3)).mean()
        sd = g["volume"].rolling(w, min_periods=max(2, w // 3)).std().replace(0, np.nan)
        g[f"vz{w}"] = (g["volume"] - mu) / sd.fillna(1.0)

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
    g["atr14"] = tr.rolling(14, min_periods=5).mean()

    # ===== 가격 패턴 =====
    # 1. High-Low 스프레드
    g["hl_spread"] = (g["high"] - g["low"]) / g["close"]

    # 2. Close 위치 (High-Low 범위 내)
    g["close_position"] = (g["close"] - g["low"]) / (g["high"] - g["low"] + 1e-10)

    # 3. 캔들 바디 크기
    g["body_size"] = (g["close"] - g["open"]).abs() / g["open"]

    # 4. 위/아래 꼬리 길이
    g["upper_shadow"] = (g["high"] - g[["open", "close"]].max(axis=1)) / g["close"]
    g["lower_shadow"] = (g[["open", "close"]].min(axis=1) - g["low"]) / g["close"]

    # 5. 갭 (이전 종가 대비 현재 시가)
    g["gap"] = ((g["open"] - g["close"].shift(1)) / g["close"].shift(1)).fillna(0.0)

    # ===== 추세 분석 =====
    # 1. 수익률 (다양한 기간)
    for w in [2, 4, 8, 12, 24]:
        g[f"ret{w}"] = g["logc"].diff(w).fillna(0.0)

    # 2. 모멘텀 가속도
    g["mom_accel"] = g["mom8"].diff().fillna(0.0)

    # 3. 추세 강도 (수익률 절댓값)
    for w in [4, 8, 12]:
        g[f"trend_strength{w}"] = g[f"ret{w}"].abs()

    # 4. 가격이 이동평균 위/아래
    for w in [20, 40]:
        ma = g["close"].rolling(w, min_periods=w // 2).mean()
        g[f"above_ma{w}"] = ((g["close"] > ma).astype(float) - 0.5) * 2  # -1 ~ 1

    # ===== RSI =====
    for w in [14, 28]:
        delta = g["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(w, min_periods=w // 2).mean()
        avg_loss = loss.rolling(w, min_periods=w // 2).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        g[f"rsi{w}"] = 100 - (100 / (1 + rs))
        g[f"rsi{w}"] = (g[f"rsi{w}"] - 50) / 50  # -1 ~ 1 정규화

    # ===== MACD =====
    ema12 = g["close"].ewm(span=12, adjust=False).mean()
    ema26 = g["close"].ewm(span=26, adjust=False).mean()
    g["macd"] = (ema12 - ema26) / g["close"]
    g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
    g["macd_hist"] = g["macd"] - g["macd_signal"]

    # ===== 볼린저 밴드 =====
    for w in [20]:
        ma = g["close"].rolling(w, min_periods=w // 2).mean()
        std = g["close"].rolling(w, min_periods=w // 2).std()
        g[f"bb_upper{w}"] = (ma + 2 * std - g["close"]) / g["close"]
        g[f"bb_lower{w}"] = (g["close"] - (ma - 2 * std)) / g["close"]
        g[f"bb_width{w}"] = (4 * std) / ma

    # ===== 시간 패턴 =====
    hod = g["date"].dt.hour
    g["hod_sin"] = np.sin(2 * np.pi * hod / 24.0)
    g["hod_cos"] = np.cos(2 * np.pi * hod / 24.0)

    # 주요 시간대 더미 변수
    for h in [0, 6, 12, 18]:
        g[f"hour_{h}"] = (hod == h).astype(float)

    # 요일 효과
    dow = g["date"].dt.dayofweek
    g["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    g["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # 주말 여부
    g["is_weekend"] = (dow >= 5).astype(float)

    # ===== 최근 극값 =====
    # 최근 N시간 내 최고가/최저가 대비 현재 위치
    for w in [8, 24, 48]:
        recent_high = g["high"].rolling(w, min_periods=max(2, w // 3)).max()
        recent_low = g["low"].rolling(w, min_periods=max(2, w // 3)).min()
        g[f"price_vs_high{w}"] = (g["close"] - recent_high) / recent_high
        g[f"price_vs_low{w}"] = (g["close"] - recent_low) / recent_low

    # ===== Feature 이상치 클리핑 =====
    feats = [
        # 수익률
        "ret1", "ret2", "ret4", "ret8", "ret12", "ret24",
        # 변동성
        "rv8", "rv20", "rv40", "rv120",
        # 모멘텀
        "mom8", "mom20", "mom40", "mom120", "mom_accel",
        # 거래량
        "vz20", "vz40", "vz120",
        "vol_spike", "vol_accel",
        "vol_price_corr8", "vol_price_corr20",
        # ATR
        "atr14",
        # 가격 패턴
        "hl_spread", "close_position", "body_size",
        "upper_shadow", "lower_shadow", "gap",
        # 추세
        "trend_strength4", "trend_strength8", "trend_strength12",
        "above_ma20", "above_ma40",
        # RSI
        "rsi14", "rsi28",
        # MACD
        "macd", "macd_signal", "macd_hist",
        # 볼린저 밴드
        "bb_upper20", "bb_lower20", "bb_width20",
        # 시간
        "hod_sin", "hod_cos",
        "hour_0", "hour_6", "hour_12", "hour_18",
        "dow_sin", "dow_cos", "is_weekend",
        # 극값 대비 위치
        "price_vs_high8", "price_vs_low8",
        "price_vs_high24", "price_vs_low24",
        "price_vs_high48", "price_vs_low48"
    ]

    for feat in feats:
        if feat not in ['hod_sin', 'hod_cos', 'dow_sin', 'dow_cos']:  # 사인/코사인 제외
            q01 = g[feat].quantile(0.01)
            q99 = g[feat].quantile(0.99)
            g[feat] = g[feat].clip(q01, q99)

    return g


@torch.no_grad()
def predict(symbol: str) -> dict:
    """예측"""
    df = get_recent_data(symbol, SEQ_LEN + 100)
    if df is None or len(df) < SEQ_LEN:
        return {"symbol": symbol, "error": "데이터 부족", "direction": None, "confidence": 0.0}

    df_feat = make_features(df)
    if len(df_feat) < SEQ_LEN:
        return {"symbol": symbol, "error": "피처 생성 실패", "direction": None, "confidence": 0.0}

    X = df_feat[FEAT_COLS].tail(SEQ_LEN).to_numpy(np.float32)
    print(f"\n[LIVE INPUT DEBUG] {symbol}:")
    print(f"  BEFORE norm mean: {X.mean():.6f}, std: {X.std():.6f}")
    X = (X - SCALER_MU) / SCALER_SD
    X = np.clip(X, -10.0, 10.0)  # 🔧 극단값 클리핑 추가
    print(f"  AFTER norm mean: {X.mean():.6f}, std: {X.std():.6f}")
    print(f"  AFTER norm min/max: {X.min():.6f} / {X.max():.6f}")
    X_tensor = torch.from_numpy(X[None, ...])
    logits, _ = MODEL(X_tensor)

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = int(logits.argmax(dim=1).item())

    # 🔍 디버그 출력
    print(f"\n[LIVE DEBUG] {symbol}: pred_class={pred_class}, probs={probs}, logits={logits.detach().numpy()[0]}")

    # 🔧 동적 방향 매핑 (2-클래스 vs 3-클래스 자동 감지)
    if len(probs) == 2:
        direction_map = {0: "Short", 1: "Long"}
    else:
        direction_map = {0: "Short", 1: "Flat", 2: "Long"}

    direction = direction_map[pred_class]
    confidence = float(probs.max())

    # 🔍 디버그 출력 - 매핑 결과
    print(f"[LIVE DEBUG] direction_map={direction_map}, final_direction={direction}, confidence={confidence:.4f}\n")
    print(f"\n[LIVE INPUT DEBUG] {symbol}:")
    print(f"  X shape: {X.shape}")
    print(f"  X mean: {X.mean():.6f}, std: {X.std():.6f}")
    print(f"  X min/max: {X.min():.6f} / {X.max():.6f}")
    print(f"  Last row sample: {X[-1, :3]}")

    ticker = API.get_ticker(symbol)
    current_price = float(ticker.get("lastPrice", 0))

    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "current_price": current_price
    }


# ===== 메인 루프 =====
def test_api_connection():
    """API 연결 테스트"""
    print("\n🔍 API 연결 테스트 중...")

    # 1. Public API 테스트
    print("   1. Public API 테스트...", end=" ")
    ticker = API.get_ticker("BTCUSDT")
    if ticker and ticker.get("lastPrice"):
        print(f"✓ (BTC 가격: ${float(ticker['lastPrice']):,.2f})")
    else:
        print("✗ 실패")
        return False

    # 2. Private API 테스트 (잔고)
    print("   2. Private API 인증 테스트...", end=" ")
    balance = API.get_balance()
    if balance >= 0:
        print(f"✓")
    else:
        print("✗ 실패")
        return False

    # 3. 포지션 조회 테스트
    print("   3. 포지션 조회 테스트...", end=" ")
    positions = API.get_positions()
    print(f"✓ (현재 {len(positions)}개 포지션)")

    print("\n✅ API 연결 성공!\n")
    return True


def print_dashboard(balance: float, positions: List[Position], manager: 'TradingManager', prices: Dict[str, float]):
    """대시보드 출력 (paper_trading.py 스타일)"""
    os.system('clear' if os.name == 'posix' else 'cls')

    mode = "🧪 TESTNET" if USE_TESTNET else "🔴 LIVE"
    print("\n" + "=" * 110)
    print(f"{'⚠️  실전 자동 트레이딩 시스템 (레버리지 ' + str(LEVERAGE) + 'x) ' + mode:^110}")
    print("=" * 110)

    # 계좌 정보
    unrealized_pnl = sum(p.unrealized_pnl for p in positions)
    total_value = balance + unrealized_pnl
    used_margin = len(positions) * MARGIN_PER_POSITION

    print(f"\n💰 계좌 현황")
    print(f"   현재 잔고:     ${balance:>12,.2f}")
    print(f"   사용 증거금:   ${used_margin:>12,.2f}")
    print(f"   사용 가능:     ${balance - used_margin:>12,.2f}")
    print(f"   평가 손익:     ${unrealized_pnl:>+12,.2f}")
    print(f"   총 자산:       ${total_value:>12,.2f}")
    print(f"   일일 손익:     ${manager.daily_pnl:>+12,.2f}")

    # 포지션
    if positions:
        print(f"\n📍 보유 포지션 ({len(positions)}/{MAX_POSITIONS})")
        print(f"{'심볼':^12} | {'방향':^8} | {'진입가':^12} | {'현재가':^12} | {'손익':^22} | {'레버리지':^10} | {'보유':^8}")
        print("-" * 110)

        for pos in positions:
            current_price = prices.get(pos.symbol, pos.entry_price)
            roe = (pos.unrealized_pnl / MARGIN_PER_POSITION) * 100

            # 포지션 진입 시간 계산 (실제 시간은 API에서 가져와야 하지만 여기서는 근사치)
            hold_min = 0  # API에서 실제 진입 시간을 가져와야 함

            emoji = "📈" if pos.get_direction() == "Long" else "📉"
            pnl_emoji = "🟢" if pos.unrealized_pnl > 0 else "🔴"

            print(f"{pos.symbol:^12} | {emoji} {pos.get_direction():^6} | ${pos.entry_price:>10,.4f} | "
                  f"${current_price:>10,.4f} | {pnl_emoji} ${pos.unrealized_pnl:>+8,.2f} ({roe:>+6.1f}%) | "
                  f"{pos.leverage:>8}x | {'N/A':>6}")
    else:
        print(f"\n📍 보유 포지션: 없음")

    # 통계
    if manager.trades:
        wins = sum(1 for t in manager.trades if t.pnl > 0)
        losses = sum(1 for t in manager.trades if t.pnl <= 0)
        total_trades = len(manager.trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = sum(t.pnl for t in manager.trades) / total_trades if total_trades > 0 else 0
        avg_roe = sum(t.pnl_pct for t in manager.trades) / total_trades if total_trades > 0 else 0
        max_pnl = max([t.pnl for t in manager.trades]) if manager.trades else 0
        min_pnl = min([t.pnl for t in manager.trades]) if manager.trades else 0
        max_roe = max([t.pnl_pct for t in manager.trades]) if manager.trades else 0
        min_roe = min([t.pnl_pct for t in manager.trades]) if manager.trades else 0

        print(f"\n📊 거래 통계")
        print(f"   총 거래:       {total_trades:>3}회")
        print(f"   승률:          {win_rate:>6.1f}% ({wins}승 {losses}패)")
        print(f"   평균 손익:     ${avg_pnl:>+12,.2f}")
        print(f"   평균 ROE:      {avg_roe:>+6.1f}%")
        print(f"   최대 수익:     ${max_pnl:>12,.2f}  (ROE: {max_roe:>+6.1f}%)")
        print(f"   최대 손실:     ${min_pnl:>12,.2f}  (ROE: {min_roe:>+6.1f}%)")

        if wins > 0 and losses > 0:
            avg_win = sum(t.pnl for t in manager.trades if t.pnl > 0) / wins
            avg_loss = sum(t.pnl for t in manager.trades if t.pnl <= 0) / losses
            rr = abs(avg_win / avg_loss)
            print(f"   Risk/Reward:   {rr:>6.2f}")

    print("\n" + "=" * 110)


def main():
    # 경고 메시지
    mode = "🧪 TESTNET" if USE_TESTNET else "🔴 LIVE"
    print("\n" + "=" * 100)
    print(f"{'⚠️  실전 자동 트레이딩 시스템  ⚠️':^100}")
    print(f"{mode:^100}")
    print("=" * 100)
    print("\n⚠️  경고:")
    print("   - 이 프로그램은 실제 자금을 사용합니다")
    print("   - 투자 손실의 위험이 있습니다")
    print("   - 본인 책임 하에 사용하세요")
    print("\nAPI 설정:")
    print(f"   - API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    print(f"   - 모드: {'Testnet' if USE_TESTNET else 'Mainnet (실전!)'}")
    print("\n거래 설정:")
    print(f"   - 포지션 모드: {POSITION_MODE.upper()}")
    print(f"   - 레버리지: {LEVERAGE}x")
    print(f"   - 증거금/포지션: ${MARGIN_PER_POSITION}")
    print(f"   - 포지션 크기: ${MARGIN_PER_POSITION * LEVERAGE} (증거금 × 레버리지)")
    print(f"   - 최대 포지션: {MAX_POSITIONS}개")
    print(f"   - 일일 최대 손실: ${MAX_DAILY_LOSS}")
    print(f"   - 신뢰도 임계값: {CONF_THRESHOLD:.0%}")

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

            # 계좌 정보
            balance = API.get_balance()
            positions = API.get_positions()

            # 현재 가격 가져오기
            prices = {}
            for symbol in SYMBOLS:
                symbol = symbol.strip()
                ticker = API.get_ticker(symbol)
                prices[symbol] = float(ticker.get("lastPrice", 0))

            # 포지션 관리
            # live_trading.py의 포지션 관리 부분 (라인 1315 이후)을 수정:

            for position in positions:
                current_price = prices.get(position.symbol, position.entry_price)

                # ✅ 추가: 손절 체크
                if position.get_direction() == "Long":
                    stop_loss = position.entry_price * (1 - STOP_LOSS_PCT)
                    take_profit = position.entry_price * (1 + TAKE_PROFIT_PCT)

                    if current_price <= stop_loss:
                        manager.close_position(position, "Stop Loss")
                        continue
                    if current_price >= take_profit:
                        manager.close_position(position, "Take Profit")
                        continue
                else:  # Short
                    stop_loss = position.entry_price * (1 + STOP_LOSS_PCT)
                    take_profit = position.entry_price * (1 - TAKE_PROFIT_PCT)

                    if current_price >= stop_loss:
                        manager.close_position(position, "Stop Loss")
                        continue
                    if current_price <= take_profit:
                        manager.close_position(position, "Take Profit")
                        continue
            # 대시보드 출력
            print_dashboard(balance, positions, manager, prices)

            # 신호 스캔 테이블
            print(f"\n🔍 신호 스캔")
            print(f"{'심볼':^12} | {'가격':^12} | {'방향':^10} | {'신뢰도':^8} | {'신호':^20}")
            print("-" * 80)

            # 신호 스캔 및 진입
            for symbol in SYMBOLS:
                symbol = symbol.strip()

                result = predict(symbol)

                if "error" in result:
                    print(f"{symbol:^12} | {'N/A':^12} | {'오류':^10} | {'N/A':^8} | ❌ 데이터 부족")
                    continue

                direction = result["direction"]
                confidence = result["confidence"]
                price = result["current_price"]

                # 방향 이모지
                dir_icon = {"Long": "📈", "Short": "📉", "Flat": "➖"}.get(direction, "❓")

                # 신호 판단
                if confidence < CONF_THRESHOLD:
                    signal = f"⚠️  신호 약함 ({confidence:.1%})"
                elif direction == "Long":
                    signal = f"🟢 매수 신호 ({confidence:.1%})"
                elif direction == "Short":
                    signal = f"🔴 매도 신호 ({confidence:.1%})"
                else:
                    signal = f"⚪ 관망 ({confidence:.1%})"

                print(
                    f"{symbol:^12} | ${price:>10,.4f} | {dir_icon} {direction:^8} | {confidence:>6.1%} | {signal:^20}")

                # 진입 조건
                if manager.can_open_position(symbol, positions) and confidence >= CONF_THRESHOLD and direction in [
                    "Long",
                    "Short"]:
                    if manager.open_position(symbol, direction, price):
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
for symbol in SYMBOLS:
    symbol = symbol.strip()
    ticker = API.get_ticker(symbol)
    prices[symbol] = float(ticker.get("lastPrice", 0))

print_dashboard(balance, positions, manager, prices)
manager.save_trades()

# 최종 통계
if manager.trades:
    wins = sum(1 for t in manager.trades if t.pnl > 0)
    losses = sum(1 for t in manager.trades if t.pnl <= 0)
    total_trades = len(manager.trades)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    avg_roe = sum(t.pnl_pct for t in manager.trades) / total_trades if total_trades > 0 else 0

    print("\n" + "=" * 110)
    print(f"{'📊 최종 결과':^110}")
    print("=" * 110)
    print(f"   최종 잔고:     ${balance:,.2f}")
    print(f"   일일 손익:     ${manager.daily_pnl:+,.2f}")
    print(f"   총 거래:       {total_trades}회")
    print(f"   승률:          {win_rate:.1f}%")
    print(f"   평균 ROE:      {avg_roe:+.1f}%")
    print("=" * 110)

if positions:
    print("\n⚠️  주의: 아직 포지션이 남아있습니다!")
    for pos in positions:
        print(f"   - {pos.symbol}: {pos.get_direction()} | ${pos.unrealized_pnl:+,.2f}")

print("\n✅ 프로그램이 종료되었습니다.")

if __name__ == "__main__":
    print(f"현재 코드 설정: {POSITION_MODE}")
    main()