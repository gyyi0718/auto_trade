# live_trading_patchtst_no_time_limit.py
# -*- coding: utf-8 -*-
"""
PatchTST 모델 기반 실전 자동 트레이딩 시스템 (시간 조건 제거 버전)
⚠️  WARNING: 실제 자금을 사용합니다. 신중하게 사용하세요!

🔒 청산 조건: TP/SL만 사용 (시간 초과 조건 완전 제거)
✅ 수정 사항:
   1. 시간 기반 청산 완전 제거 (TP/SL만 사용)
   2. API TP/SL 값 우선 사용 (재계산 안 함)
   3. 진입 시간 파싱 오류 영향 제거
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

USE_TESTNET = os.getenv("USE_TESTNET", "0") == "1"

if not API_KEY or not API_SECRET:
    print("❌ ERROR: BYBIT_API_KEY 및 BYBIT_API_SECRET 환경변수를 설정하세요!")
    exit(1)

# 거래 설정
SYMBOLS_ENV = os.getenv("SYMBOLS", "").strip()

# PatchTST 모델 경로 설정
MODEL_PATHS = {
    "HUSDT": "./models_patchtst_improved/patchtst_best.ckpt",
}
if SYMBOLS_ENV:
    SYMBOLS = [s.strip() for s in SYMBOLS_ENV.split(",") if s.strip()]
else:
    SYMBOLS = list(MODEL_PATHS.keys())

INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "10"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.6"))

# 리스크 관리
MARGIN_PER_POSITION = float(os.getenv("MARGIN_PER_POSITION", "200"))
LEVERAGE = int(os.getenv("LEVERAGE", "20"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.03"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "100"))
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "30"))
LIQUIDATION_BUFFER = float(os.getenv("LIQUIDATION_BUFFER", "0.8"))

# 포지션 모드
POSITION_MODE = os.getenv("POSITION_MODE", "one-way").lower()

# 로그
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "live_trades_stable.json")
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"

# 데이터 캐시
DATA_CACHE = {}
CACHE_DURATION = 60


# ===== 데이터 클래스 =====
@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    direction: str
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    leverage: int
    margin: float
    liquidation_price: float
    unrealized_pnl: float = 0.0
    position_value: float = 0.0

    def get_pnl(self, current_price: float) -> float:
        if self.direction == "Long":
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size

    def get_roe(self, current_price: float) -> float:
        if self.direction == "Long":
            price_change_pct = (current_price / self.entry_price - 1) * 100
        else:
            price_change_pct = (1 - current_price / self.entry_price) * 100
        return price_change_pct * self.leverage

    def should_close(self, current_price: float, current_time: datetime) -> tuple:
        """청산 여부 판단 - TP/SL만 체크 (시간 조건 제거)"""
        # 청산가
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

        # ✅ 시간 조건 완전 제거 - TP/SL만 사용
        return False, ""


@dataclass
class Trade:
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
    roe: float
    exit_reason: str


# ===== PatchTST 모델 =====
class FlexiblePatchTST(nn.Module):
    def __init__(self, n_features, seq_len, patch_size, d_model, n_heads, n_layers, dropout=0.2):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.patch_embed = nn.Linear(patch_size * n_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Identity()

    def forward(self, x):
        B, seq_len, n_features = x.shape
        x = x.reshape(B, self.n_patches, self.patch_size * n_features)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def analyze_classifier_structure(state_dict):
    classifier_keys = [k for k in state_dict.keys() if 'classifier' in k or 'head' in k]
    layers = {}
    for key in classifier_keys:
        parts = key.split('.')
        if len(parts) >= 3 and parts[1].isdigit():
            layer_idx = int(parts[1])
            param_type = parts[2]
            if layer_idx not in layers:
                layers[layer_idx] = {}
            layers[layer_idx][param_type] = state_dict[key].shape
    return layers


def build_classifier_from_structure(layers_info):
    modules = nn.ModuleDict()
    for idx in sorted(layers_info.keys()):
        params = layers_info[idx]
        if 'weight' in params and 'bias' in params:
            weight_shape = params['weight']
            if len(weight_shape) == 1:
                modules[str(idx)] = nn.LayerNorm(weight_shape[0])
            elif len(weight_shape) == 2:
                in_features = weight_shape[1]
                out_features = weight_shape[0]
                modules[str(idx)] = nn.Linear(in_features, out_features)

    class IndexedSequential(nn.Module):
        def __init__(self, module_dict):
            super().__init__()
            self.module_dict = module_dict
            self.keys = sorted([int(k) for k in module_dict.keys()])

        def forward(self, x):
            for idx in self.keys:
                x = self.module_dict[str(idx)](x)
            return x

    return IndexedSequential(modules)


# ===== 모델 로딩 =====
print("\n🤖 PatchTST 모델 로딩 중...")
MODELS = {}
MODEL_CONFIGS = {}

for symbol in SYMBOLS:
    symbol = symbol.strip()
    if symbol not in MODEL_PATHS:
        continue

    model_path = MODEL_PATHS[symbol]
    if not os.path.exists(model_path):
        print(f"❌ {symbol}: 모델 파일 없음 - {model_path}")
        continue

    try:
        ckpt = torch.load(model_path, map_location='cpu')
        meta = ckpt['meta']
        layers_info = analyze_classifier_structure(ckpt['model'])

        model = FlexiblePatchTST(
            n_features=len(ckpt['feat_cols']),
            seq_len=meta['seq_len'],
            patch_size=meta.get('patch_size', 12),
            d_model=meta['d_model'],
            n_heads=meta['n_heads'],
            n_layers=meta['n_layers']
        )

        model.classifier = build_classifier_from_structure(layers_info)

        new_state_dict = {}
        for key, value in ckpt['model'].items():
            if key.startswith('head.'):
                new_key = key.replace('head.', 'classifier.module_dict.')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        model.load_state_dict(new_state_dict)
        model.eval()

        MODELS[symbol] = model
        MODEL_CONFIGS[symbol] = {
            'seq_len': meta['seq_len'],
            'feat_cols': ckpt['feat_cols'],
            'scaler_mu': np.array(ckpt['scaler_mu']),
            'scaler_sd': np.array(ckpt['scaler_sd'])
        }

        print(f"✅ {symbol}: 로드 완료")
    except Exception as e:
        print(f"❌ {symbol}: 로딩 실패 - {e}")

if not MODELS:
    print("\n❌ 로드된 모델이 없습니다!")
    exit(1)

print(f"✅ 총 {len(MODELS)}개 모델 로드 완료\n")


# ===== Bybit API =====
class BybitAPI:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._instrument_cache = {}
        self.base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self.recv_window = "5000"

    def _generate_signature(self, timestamp: str, params_str: str) -> str:
        sign_str = timestamp + self.api_key + self.recv_window + params_str
        return hmac.new(
            bytes(self.api_secret, 'utf-8'),
            bytes(sign_str, 'utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _request(self, method: str, endpoint: str, params: dict = None, sign: bool = False) -> dict:
        if params is None:
            params = {}

        headers = {"Content-Type": "application/json"}

        if sign:
            timestamp = str(int(time.time() * 1000))
            if method == "GET":
                params_str = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())]) if params else ""
            else:
                params_str = json.dumps(params) if params else ""

            signature = self._generate_signature(timestamp, params_str)
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
                raise ValueError(f"Unsupported method: {method}")
            return response.json()
        except Exception as e:
            return {"retCode": -1, "retMsg": str(e)}

    def get_ticker(self, symbol: str) -> dict:
        return self._request("GET", "/v5/market/tickers", {"category": "linear", "symbol": symbol})

    def get_klines(self, symbol: str, interval: str = "5", limit: int = 200) -> pd.DataFrame:
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
        result = self._request("GET", "/v5/account/wallet-balance", {
            "accountType": "UNIFIED"
        }, sign=True)

        if result.get("retCode") != 0:
            return 0.0

        try:
            coins = result["result"]["list"][0]["coin"]
            usdt = next((c for c in coins if c["coin"] == "USDT"), None)
            return float(usdt["walletBalance"]) if usdt else 0.0
        except:
            return 0.0

    def get_positions(self) -> Dict[str, Position]:
        """포지션을 딕셔너리로 반환 (TP/SL API 값 사용, 시간 무관)"""
        result = self._request("GET", "/v5/position/list", {
            "category": "linear",
            "settleCoin": "USDT"
        }, sign=True)

        if result.get("retCode") != 0:
            return {}

        positions = {}
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

                position_value = size * entry_price
                margin = position_value / leverage

                # ✅ API에서 실제 TP/SL 가져오기 (재계산하지 않음!)
                stop_loss_str = item.get("stopLoss", "")
                take_profit_str = item.get("takeProfit", "")

                if stop_loss_str and stop_loss_str != "":
                    stop_loss = float(stop_loss_str)
                else:
                    # API 값이 없을 때만 계산
                    if direction == "Long":
                        stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                    else:
                        stop_loss = entry_price * (1 + STOP_LOSS_PCT)

                if take_profit_str and take_profit_str != "":
                    take_profit = float(take_profit_str)
                else:
                    # API 값이 없을 때만 계산
                    if direction == "Long":
                        take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
                    else:
                        take_profit = entry_price * (1 - TAKE_PROFIT_PCT)

                liq_price_str = item.get("liqPrice", "0")
                if liq_price_str and liq_price_str != "":
                    liquidation_price = float(liq_price_str)
                else:
                    if direction == "Long":
                        liquidation_price = entry_price * (1 - (1 / leverage) * LIQUIDATION_BUFFER)
                    else:
                        liquidation_price = entry_price * (1 + (1 / leverage) * LIQUIDATION_BUFFER)

                # ✅ 진입 시간은 현재 시간으로 (어차피 안 씀)
                entry_time = datetime.now()

                positions[symbol] = Position(
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
                )

        except Exception as e:
            print(f"❌ 포지션 파싱 오류: {e}")

        return positions

    def get_instrument_info(self, symbol: str) -> dict:
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        result = self._request("GET", "/v5/market/instruments-info", {
            "category": "linear",
            "symbol": symbol
        })

        if result.get("retCode") != 0 or not result.get("result", {}).get("list"):
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

        self._instrument_cache[symbol] = instrument_info
        return instrument_info

    def set_leverage(self, symbol: str, leverage: int) -> tuple:
        result = self._request("POST", "/v5/position/set-leverage", {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        }, sign=True)

        success = result.get("retCode") == 0
        msg = result.get("retMsg", "")
        return success, msg

    def place_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> Optional[str]:
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "GTC",
            "positionIdx": 0
        }

        if reduce_only:
            params["reduceOnly"] = True

        result = self._request("POST", "/v5/order/create", params, sign=True)

        if result.get("retCode") == 0:
            return result["result"]["orderId"]
        else:
            print(f"❌ 주문 실패: {result.get('retMsg')}")
            return None

    def set_trading_stop(self, symbol: str, side: str, stop_loss: float = None, take_profit: float = None):
        params = {
            "category": "linear",
            "symbol": symbol,
            "positionIdx": 0
        }

        if stop_loss:
            params["stopLoss"] = str(stop_loss)
        if take_profit:
            params["takeProfit"] = str(take_profit)

        if not stop_loss and not take_profit:
            return False

        result = self._request("POST", "/v5/position/trading-stop", params, sign=True)
        return result.get("retCode") == 0


API = BybitAPI(API_KEY, API_SECRET, USE_TESTNET)


# ===== 트레이딩 매니저 =====
class TradingManager:
    def __init__(self):
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0

    def can_open_position(self, symbol: str, positions: Dict[str, Position]) -> bool:
        """포지션 진입 가능 여부 (페이퍼와 동일)"""
        if symbol in positions:
            return False
        if len(positions) >= MAX_POSITIONS:
            return False
        if abs(self.daily_pnl) >= MAX_DAILY_LOSS:
            return False
        return True

    def open_position(self, symbol: str, direction: str, price: float) -> bool:
        """포지션 진입"""
        try:
            if price <= 0 or not np.isfinite(price):
                return False

            # 레버리지 설정
            success, msg = API.set_leverage(symbol, LEVERAGE)
            if not success and "not modified" not in msg.lower():
                print(f"⚠️  레버리지 설정 실패: {msg}")

            # 심볼 정보
            instrument_info = API.get_instrument_info(symbol)
            min_qty = instrument_info["minOrderQty"]
            qty_step = instrument_info["qtyStep"]
            min_notional = instrument_info["minNotionalValue"]

            # 수량 계산
            position_value = MARGIN_PER_POSITION * LEVERAGE
            qty = position_value / price

            if qty * price < min_notional:
                print(f"❌ {symbol}: 주문 금액 부족")
                return False

            # 수량 올림 처리
            import math
            steps = math.ceil(qty / qty_step)
            qty = steps * qty_step

            # 소수점 처리
            step_str = f"{qty_step:.10f}".rstrip('0')
            decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
            qty = round(qty, decimals)

            if qty < min_qty:
                print(f"❌ {symbol}: 최소 수량 미달")
                return False

            # TP/SL 계산
            if direction == "Long":
                stop_loss = price * (1 - STOP_LOSS_PCT)
                take_profit = price * (1 + TAKE_PROFIT_PCT)
                side = "Buy"
            else:
                stop_loss = price * (1 + STOP_LOSS_PCT)
                take_profit = price * (1 - TAKE_PROFIT_PCT)
                side = "Sell"

            # 주문 생성
            order_id = API.place_order(symbol, side, qty)

            if order_id:
                time.sleep(0.5)
                # TP/SL 설정
                API.set_trading_stop(symbol, side, stop_loss, take_profit)

                print(f"\n{'=' * 90}")
                print(f"🔔 포지션 진입 성공: {symbol} | {direction}")
                print(f"   가격: ${price:,.4f} | 수량: {qty} | 레버리지: {LEVERAGE}x")
                print(f"   SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}")
                print(f"{'=' * 90}")
                return True

            return False

        except Exception as e:
            print(f"❌ 진입 실패: {e}")
            return False

    def close_position(self, position: Position, reason: str):
        """포지션 청산"""
        try:
            side = "Sell" if position.direction == "Long" else "Buy"
            order_id = API.place_order(position.symbol, side, position.size, reduce_only=True)

            if order_id:
                ticker = API.get_ticker(position.symbol)
                if ticker.get("retCode") == 0 and ticker["result"]["list"]:
                    exit_price = float(ticker["result"]["list"][0]["lastPrice"])
                else:
                    exit_price = position.entry_price

                pnl = position.get_pnl(exit_price)
                roe = position.get_roe(exit_price)
                self.daily_pnl += pnl

                trade = Trade(
                    symbol=position.symbol,
                    direction=position.direction,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    size=position.size,
                    leverage=position.leverage,
                    margin=position.margin,
                    entry_time=position.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                    exit_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    pnl=pnl,
                    roe=roe,
                    exit_reason=reason
                )

                self.trades.append(trade)

                emoji = "✅" if pnl > 0 else "❌"
                print(f"\n{'=' * 90}")
                print(f"{emoji} 포지션 청산: {position.symbol} | {reason}")
                print(f"   진입: ${position.entry_price:,.4f} → 청산: ${exit_price:,.4f}")
                print(f"   손익: ${pnl:+,.2f} (ROE: {roe:+.1f}%)")
                print(f"{'=' * 90}")

        except Exception as e:
            print(f"❌ 청산 실패: {e}")

    def save_trades(self):
        if self.trades:
            with open(TRADE_LOG_FILE, 'w') as f:
                json.dump([asdict(t) for t in self.trades], f, indent=2)
            print(f"\n💾 거래 내역 저장: {TRADE_LOG_FILE}")

    def get_stats(self) -> dict:
        if not self.trades:
            return {"total_trades": 0, "win_rate": 0, "avg_roe": 0}
        wins = sum(1 for t in self.trades if t.pnl > 0)
        win_rate = wins / len(self.trades) * 100
        avg_roe = np.mean([t.roe for t in self.trades])
        return {
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "avg_roe": avg_roe
        }


# ===== Feature 생성 =====
def generate_features(df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    g = df.copy()
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g["logc"].diff().fillna(0.0)

    for w in [8, 20, 40]:
        g[f"rv{w}"] = g["ret1"].rolling(w, min_periods=max(2, w // 3)).std().fillna(0.0)

    for w in [8, 20, 40]:
        ema = g["close"].ewm(span=w, adjust=False).mean()
        g[f"mom{w}"] = (g["close"] / ema - 1.0).fillna(0.0)

    for w in [20, 40]:
        mu = g["volume"].rolling(w, min_periods=max(2, w // 3)).mean()
        sd = g["volume"].rolling(w, min_periods=max(2, w // 3)).std().replace(0, 1.0)
        g[f"vz{w}"] = ((g["volume"] - mu) / sd).fillna(0.0)

    g["vol_spike"] = (g["volume"] / g["volume"].shift(1) - 1.0).fillna(0.0)

    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14, min_periods=5).mean().fillna(0.0)

    g["hl_spread"] = ((g["high"] - g["low"]) / g["close"]).fillna(0.0)
    g["close_pos"] = ((g["close"] - g["low"]) / (g["high"] - g["low"] + 1e-10)).fillna(0.5)
    g["body_size"] = ((g["close"] - g["open"]).abs() / g["open"]).fillna(0.0)

    for w in [2, 4, 8, 12]:
        g[f"ret{w}"] = g["logc"].diff(w).fillna(0.0)

    delta = g["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    g["rsi14"] = (100 - (100 / (1 + rs)) - 50) / 50
    g["rsi14"] = g["rsi14"].fillna(0.0)

    ema12 = g["close"].ewm(span=12, adjust=False).mean()
    ema26 = g["close"].ewm(span=26, adjust=False).mean()
    g["macd"] = ((ema12 - ema26) / g["close"]).fillna(0.0)

    hod = pd.to_datetime(g["timestamp"]).dt.hour
    g["hod_sin"] = np.sin(2 * np.pi * hod / 24.0)
    g["hod_cos"] = np.cos(2 * np.pi * hod / 24.0)
    dow = pd.to_datetime(g["timestamp"]).dt.dayofweek
    g["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    g["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    for feat in feat_cols:
        if feat in g.columns and feat not in ['hod_sin', 'hod_cos', 'dow_sin', 'dow_cos']:
            q01, q99 = g[feat].quantile([0.01, 0.99])
            g[feat] = g[feat].clip(q01, q99)

    return g[feat_cols]


def get_cached_data(symbol: str, interval: str, required_length: int) -> pd.DataFrame:
    cache_key = f"{symbol}_{interval}"
    current_time = time.time()

    if cache_key in DATA_CACHE:
        cached_df, cached_time = DATA_CACHE[cache_key]
        if current_time - cached_time < CACHE_DURATION and len(cached_df) >= required_length:
            return cached_df

    limit = min(2000, max(required_length + 50, 300))
    df = API.get_klines(symbol, interval=interval, limit=limit)

    if not df.empty:
        DATA_CACHE[cache_key] = (df, current_time)

    return df


def predict(symbol: str) -> dict:
    if symbol not in MODELS:
        return {"error": f"No model for {symbol}"}

    model = MODELS[symbol]
    config = MODEL_CONFIGS[symbol]
    seq_len = config['seq_len']

    df = get_cached_data(symbol, interval="5", required_length=seq_len + 50)

    if df.empty or len(df) < seq_len + 20:
        return {"error": "Insufficient data"}

    try:
        df_feat = generate_features(df, config['feat_cols'])
        df_feat = df_feat.dropna()

        if len(df_feat) < seq_len:
            return {"error": "Insufficient features"}

        X = df_feat.iloc[-seq_len:].values
        safe_scaler_sd = np.maximum(config['scaler_sd'], 0.01)
        X = (X - config['scaler_mu']) / (safe_scaler_sd + 1e-8)
        X = np.clip(X, -10, 10)
        X_tensor = torch.FloatTensor(X).unsqueeze(0)

        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()

        pred_class = int(probs.argmax())
        direction = "Short" if pred_class == 0 else "Long"
        confidence = float(probs.max())

        ticker = API.get_ticker(symbol)
        if ticker.get("retCode") == 0 and ticker["result"]["list"]:
            current_price = float(ticker["result"]["list"][0]["lastPrice"])
        else:
            current_price = float(df['close'].iloc[-1])

        return {
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "current_price": current_price
        }

    except Exception as e:
        return {"error": str(e)}


def print_dashboard(balance: float, positions: Dict[str, Position], prices: Dict[str, float]):
    os.system('clear' if os.name == 'posix' else 'cls')

    print("\n" + "=" * 110)
    print(f"{'🎯 PatchTST 라이브 트레이딩 (안정화) - 레버리지 ' + str(LEVERAGE) + 'x':^110}")
    print("=" * 110)

    total_margin = sum(p.margin for p in positions.values())
    unrealized_pnl = sum(p.unrealized_pnl for p in positions.values())
    total_value = balance + unrealized_pnl

    print(f"\n💰 계좌 현황")
    print(f"   현금 잔고:     ${balance:>12,.2f}")
    if positions:
        print(f"   포지션 증거금: ${total_margin:>12,.2f}")
        pnl_color = "🟢" if unrealized_pnl >= 0 else "🔴"
        print(f"   평가 손익:     {pnl_color} ${unrealized_pnl:>+12,.2f}")
    print(f"   " + "-" * 40)
    print(f"   총 자산:       ${total_value:>12,.2f}")

    if positions:
        print(f"\n📍 보유 포지션 ({len(positions)}/{MAX_POSITIONS})")
        print(f"{'심볼':^12} | {'방향':^8} | {'진입가':^12} | {'현재가':^12} | {'손익(ROE)':^22}")
        print("-" * 80)

        for symbol, pos in positions.items():
            current_price = prices.get(symbol, pos.entry_price)
            pnl = pos.get_pnl(current_price)
            roe = pos.get_roe(current_price)
            emoji = "📈" if pos.direction == "Long" else "📉"
            pnl_emoji = "🟢" if pnl > 0 else "🔴"
            print(f"{symbol:^12} | {emoji} {pos.direction:^6} | ${pos.entry_price:>10,.4f} | "
                  f"${current_price:>10,.4f} | {pnl_emoji} ${pnl:>+8,.2f} ({roe:>+6.1f}%)")

    print("\n" + "=" * 110)


# ===== 메인 함수 =====
def main():
    print(f"\n{'=' * 110}")
    print(f"{'🎯 PatchTST 라이브 트레이딩 (시간 조건 제거 버전)':^110}")
    print(f"{'🔒 청산: TP/SL만 사용 (시간 조건 없음)':^110}")
    if USE_TESTNET:
        print(f"{'⚠️  TESTNET 모드':^110}")
    else:
        print(f"{'🔴 MAINNET 모드':^110}")
    print(f"{'=' * 110}")

    manager = TradingManager()

    try:
        loop_count = 0
        while True:
            loop_count += 1
            current_time = datetime.now()

            # 현재 상태 조회
            balance = API.get_balance()
            positions = API.get_positions()  # Dict[str, Position]

            # 가격 조회
            prices = {}
            for symbol in MODELS.keys():
                ticker = API.get_ticker(symbol)
                if ticker.get("retCode") == 0 and ticker["result"]["list"]:
                    prices[symbol] = float(ticker["result"]["list"][0]["lastPrice"])

            # 🔒 포지션 관리: TP/SL/시간만 체크 (페이퍼와 동일)
            for symbol in list(positions.keys()):
                position = positions[symbol]
                current_price = prices.get(symbol, position.entry_price)
                should_close, reason = position.should_close(current_price, current_time)

                if should_close:
                    manager.close_position(position, reason)
                    time.sleep(1)
                    positions = API.get_positions()  # 재조회

            # 대시보드
            print_dashboard(balance, positions, prices)

            # 신호 스캔
            print(f"\n🔍 신호 스캔 ({len(MODELS)}개 심볼)")
            print(f"{'심볼':^12} | {'가격':^12} | {'방향':^10} | {'신뢰도':^8} | {'신호':^20}")
            print("-" * 80)

            for symbol in MODELS.keys():
                result = predict(symbol)

                if "error" in result:
                    print(f"{symbol:^12} | {'N/A':^12} | {'오류':^10} | {'N/A':^8} | ❌ {result['error']}")
                    continue

                direction = result["direction"]
                confidence = result["confidence"]
                price = result["current_price"]

                if price <= 0:
                    continue

                # 신호 표시
                dir_icon = "📈" if direction == "Long" else "📉"
                if confidence < CONF_THRESHOLD:
                    signal = f"⚠️  신호 약함"
                elif direction == "Long":
                    signal = f"🟢 매수 신호"
                else:
                    signal = f"🔴 매도 신호"

                print(f"{symbol:^12} | ${price:>10,.4f} | {dir_icon} {direction:^8} | "
                      f"{confidence:>6.1%} | {signal}")

                # 🔒 진입: 포지션 없을 때만
                if manager.can_open_position(symbol, positions) and confidence >= CONF_THRESHOLD:
                    if manager.open_position(symbol, direction, price):
                        time.sleep(1)
                        positions = API.get_positions()

            print(f"\n[스캔 #{loop_count}] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"다음 스캔까지 {INTERVAL_SEC}초... (Ctrl+C로 종료)")

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n프로그램 종료 중...")

        balance = API.get_balance()
        positions = API.get_positions()
        prices = {}
        for symbol in MODELS.keys():
            ticker = API.get_ticker(symbol)
            if ticker.get("retCode") == 0 and ticker["result"]["list"]:
                prices[symbol] = float(ticker["result"]["list"][0]["lastPrice"])

        print_dashboard(balance, positions, prices)
        manager.save_trades()

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
            print("=" * 110)

        if positions:
            print("\n⚠️  주의: 아직 포지션이 남아있습니다!")
            for symbol, pos in positions.items():
                pnl = pos.get_pnl(prices.get(symbol, pos.entry_price))
                print(f"   - {symbol}: {pos.direction} | ${pnl:+,.2f}")

        print("\n✅ 프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()