# paper_trading_patchtst_flexible.py
# -*- coding: utf-8 -*-
"""
PatchTST 페이퍼 트레이딩 (유연한 로더)
- 체크포인트 구조를 분석하여 자동으로 모델 생성
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
SYMBOLS = os.getenv("SYMBOLS", "HUSDT").split(",")
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "10"))

MODEL_PATHS = {
    "MELANIAUSDT": "./models_patchtst_improved/patchtst_best.ckpt",
}

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.55"))
USE_TESTNET = os.getenv("USE_TESTNET", "0") == "1"

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000"))
POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", "0.5"))
LEVERAGE = int(os.getenv("LEVERAGE", "25"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.03"))
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "30"))
LIQUIDATION_BUFFER = float(os.getenv("LIQUIDATION_BUFFER", "0.8"))
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "trades.json")

MAX_PRICE_CHANGE_PCT = 50.0
MAX_ROE_LIMIT = 100.0
MIN_PRICE_RATIO = 0.5
MAX_PRICE_RATIO = 2.0

DATA_CACHE = {}
CACHE_DURATION = 60


def validate_price(current_price: float, entry_price: float, symbol: str) -> tuple:
    if current_price <= 0:
        return False, f"Invalid price: {current_price}"
    price_ratio = current_price / entry_price
    if price_ratio < MIN_PRICE_RATIO or price_ratio > MAX_PRICE_RATIO:
        return False, f"Abnormal price change: {price_ratio:.2%}"
    change_pct = abs(price_ratio - 1) * 100
    if change_pct > MAX_PRICE_CHANGE_PCT:
        return False, f"Price change too large: {change_pct:.1f}%"
    return True, ""


@dataclass
class Position:
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
        if self.direction == "Long":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def get_pnl_pct(self, current_price: float) -> float:
        pnl = self.get_pnl(current_price)
        return (pnl / self.margin) * 100

    def get_roe(self, current_price: float) -> float:
        is_valid, error_msg = validate_price(current_price, self.entry_price, self.symbol)
        if not is_valid:
            return 0.0
        if self.direction == "Long":
            price_change_pct = (current_price / self.entry_price - 1) * 100
        else:
            price_change_pct = (1 - current_price / self.entry_price) * 100
        roe = price_change_pct * self.leverage
        return np.clip(roe, -MAX_ROE_LIMIT, MAX_ROE_LIMIT)

    def should_close(self, current_price: float, current_time: datetime) -> tuple:
        is_valid, error_msg = validate_price(current_price, self.entry_price, self.symbol)
        if not is_valid:
            return False, ""
        if self.direction == "Long" and current_price <= self.liquidation_price:
            return True, "Liquidation"
        if self.direction == "Short" and current_price >= self.liquidation_price:
            return True, "Liquidation"
        if self.direction == "Long" and current_price <= self.stop_loss:
            return True, "Stop Loss"
        if self.direction == "Short" and current_price >= self.stop_loss:
            return True, "Stop Loss"
        if self.direction == "Long" and current_price >= self.take_profit:
            return True, "Take Profit"
        if self.direction == "Short" and current_price <= self.take_profit:
            return True, "Take Profit"
       # hold_minutes = (current_time - self.entry_time).total_seconds() / 60
       # if hold_minutes >= MAX_HOLD_MINUTES:
       #     return True, "Time Limit"
        return False, ""


@dataclass
class Trade:
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
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.total_pnl = 0.0

    def get_available_balance(self) -> float:
        used_margin = sum(p.margin for p in self.positions.values())
        return self.balance - used_margin

    def get_total_value(self, prices: Dict[str, float]) -> float:
        unrealized_pnl = sum(
            p.get_pnl(prices.get(p.symbol, p.entry_price))
            for p in self.positions.values()
        )
        return self.balance + unrealized_pnl

    def can_open_position(self, symbol: str) -> bool:
        if symbol in self.positions:
            return False
        if len(self.positions) >= MAX_POSITIONS:
            return False
        margin_needed = self.get_available_balance() * POSITION_SIZE_PCT
        if self.get_available_balance() < margin_needed:
            return False
        return True

    def open_position(self, symbol: str, direction: str, price: float):
        if price <= 0:
            print(f"⚠️  잘못된 진입가 ({symbol}): ${price:.4f}")
            return

        margin = self.get_available_balance() * POSITION_SIZE_PCT
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
            symbol=symbol, direction=direction, entry_price=price, quantity=quantity,
            entry_time=datetime.now(), stop_loss=stop_loss, take_profit=take_profit,
            leverage=LEVERAGE, margin=margin, liquidation_price=liquidation_price
        )

        # 잔고에서 증거금 차감
        self.balance -= margin

        self.positions[symbol] = position
        print(f"\n{'=' * 90}")
        print(f"🔔 포지션 진입: {symbol} | {direction} | ${price:,.4f}")
        print(f"   레버리지: {LEVERAGE}x | 수량: {quantity:.4f} | 증거금: ${margin:,.2f}")
        print(f"   현금 잔고: ${self.balance:,.2f}")
        print(f"{'=' * 90}")

    def close_position(self, symbol: str, price: float, reason: str):
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        pnl = position.get_pnl(price)
        pnl_pct = position.get_pnl_pct(price)
        roe = position.get_roe(price)

        # 증거금 반환 + 손익
        self.balance += position.margin + pnl
        self.total_pnl += pnl

        trade = Trade(
            symbol=symbol, direction=position.direction, entry_price=position.entry_price,
            exit_price=price, quantity=position.quantity, leverage=position.leverage,
            margin=position.margin, entry_time=position.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            exit_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pnl=pnl, pnl_pct=pnl_pct, roe=roe, exit_reason=reason
        )

        self.trades.append(trade)
        del self.positions[symbol]

        emoji = "✅" if pnl > 0 else "❌"
        print(f"\n{'=' * 90}")
        print(f"{emoji} 포지션 청산: {symbol} | {reason}")
        print(f"   진입: ${position.entry_price:,.4f} → 청산: ${price:,.4f}")
        print(f"   손익: ${pnl:+,.2f} (ROE: {roe:+.1f}%) | 잔고: ${self.balance:,.2f}")
        print(f"{'=' * 90}")

    def get_stats(self) -> dict:
        if not self.trades:
            return {"total_trades": 0, "win_rate": 0, "avg_roe": 0}
        wins = sum(1 for t in self.trades if t.pnl > 0)
        win_rate = wins / len(self.trades) * 100
        avg_roe = np.mean([t.roe for t in self.trades])
        return {"total_trades": len(self.trades), "win_rate": win_rate, "avg_roe": avg_roe}


class BybitAPI:
    def __init__(self, testnet=False):
        self.base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self.session = requests.Session()

    def _request(self, endpoint, params=None):
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params, timeout=10)
            return response.json()
        except Exception as e:
            return {"retCode": -1, "retMsg": str(e)}

    def get_ticker(self, symbol):
        return self._request("/v5/market/tickers", {"category": "linear", "symbol": symbol})

    def get_klines(self, symbol, interval="5", limit=200):
        resp = self._request(
            "/v5/market/kline",
            {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
        )
        if resp.get("retCode") != 0:
            return pd.DataFrame()
        data = resp.get("result", {}).get("list", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit='ms')
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df


API = BybitAPI(testnet=USE_TESTNET)
MODELS = {}
MODEL_CONFIGS = {}


# ===== 유연한 PatchTST 모델 =====
class FlexiblePatchTST(nn.Module):
    """체크포인트 구조를 자동으로 학습하는 모델"""

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

        # classifier는 load_state_dict 후에 자동으로 생성됨
        self.classifier = nn.Identity()  # placeholder

    def forward(self, x):
        B, seq_len, n_features = x.shape
        x = x.reshape(B, self.n_patches, self.patch_size * n_features)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def analyze_classifier_structure(state_dict):
    """state_dict에서 classifier/head 구조 분석"""
    # classifier 또는 head로 시작하는 키 찾기
    classifier_keys = [k for k in state_dict.keys() if 'classifier' in k or 'head' in k]

    # 레이어별로 그룹화
    layers = {}
    for key in classifier_keys:
        parts = key.split('.')
        # classifier.0.weight 또는 head.0.weight 형태
        if len(parts) >= 3 and parts[1].isdigit():
            layer_idx = int(parts[1])
            param_type = parts[2]

            if layer_idx not in layers:
                layers[layer_idx] = {}
            layers[layer_idx][param_type] = state_dict[key].shape

    return layers


def build_classifier_from_structure(layers_info):
    """분석된 구조로 classifier 생성 (원본 인덱스 유지)"""
    # 최대 레이어 인덱스 찾기
    max_idx = max(layers_info.keys())

    # nn.Sequential 대신 nn.ModuleDict 사용하여 인덱스 유지
    modules = nn.ModuleDict()

    for idx in sorted(layers_info.keys()):
        params = layers_info[idx]

        if 'weight' in params and 'bias' in params:
            weight_shape = params['weight']

            if len(weight_shape) == 1:
                # LayerNorm
                modules[str(idx)] = nn.LayerNorm(weight_shape[0])
            elif len(weight_shape) == 2:
                # Linear
                in_features = weight_shape[1]
                out_features = weight_shape[0]
                modules[str(idx)] = nn.Linear(in_features, out_features)

    # ModuleDict를 Sequential로 변환 (순서대로 실행)
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


def load_models():
    """유연한 모델 로딩"""
    print("\n[LOAD] 모델 로딩 중 (유연한 로더)...")

    for symbol, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"⚠️  {symbol}: 모델 파일 없음 ({path})")
            continue

        try:
            ckpt = torch.load(path, map_location='cpu')
            meta = ckpt['meta']

            # 1. Classifier 구조 분석
            layers_info = analyze_classifier_structure(ckpt['model'])
            print(f"\n📊 {symbol} Classifier 구조:")
            for idx in sorted(layers_info.keys()):
                print(f"  Layer {idx}: {layers_info[idx]}")

            # 2. 기본 모델 생성 (classifier 제외)
            model = FlexiblePatchTST(
                n_features=len(ckpt['feat_cols']),
                seq_len=meta['seq_len'],
                patch_size=meta.get('patch_size', 12),
                d_model=meta['d_model'],
                n_heads=meta['n_heads'],
                n_layers=meta['n_layers']
            )

            # 3. Classifier 동적 생성
            model.classifier = build_classifier_from_structure(layers_info)

            # 4. State dict 로드 (키 변환)
            # head.x.weight -> classifier.module_dict.x.weight 변환
            new_state_dict = {}
            for key, value in ckpt['model'].items():
                if key.startswith('head.'):
                    # head.0.weight -> classifier.module_dict.0.weight
                    new_key = key.replace('head.', 'classifier.module_dict.')
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            model.load_state_dict(new_state_dict)
            model.eval()

            MODELS[symbol] = model
            MODEL_CONFIGS[symbol] = {
                'seq_len': meta['seq_len'],
                'patch_size': meta.get('patch_size', 12),
                'feat_cols': ckpt['feat_cols'],
                'scaler_mu': np.array(ckpt['scaler_mu']),
                'scaler_sd': np.array(ckpt['scaler_sd'])
            }

            print(f"✅ {symbol}: seq_len={meta['seq_len']}, features={len(ckpt['feat_cols'])}")
            print(f"   Classifier: {len(layers_info)} layers")

        except Exception as e:
            print(f"❌ {symbol}: 로딩 실패 - {e}")
            import traceback
            traceback.print_exc()

    if not MODELS:
        print("\n❌ 로딩된 모델이 없습니다!")
        exit(1)

    print(f"\n✅ 총 {len(MODELS)}개 모델 로딩 완료\n")


def generate_features(df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """Feature 생성"""
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


def predict(symbol: str, debug: bool = False) -> dict:
    symbol = symbol.strip()

    if symbol not in MODELS:
        return {"error": f"No model for {symbol}", "symbol": symbol}

    model = MODELS[symbol]
    config = MODEL_CONFIGS[symbol]
    seq_len = config['seq_len']

    df = get_cached_data(symbol, interval="5", required_length=seq_len + 50)

    if df.empty:
        return {"error": "Data fetch failed", "symbol": symbol}

    if len(df) < seq_len + 20:
        return {"error": f"Insufficient data (got {len(df)}, need {seq_len + 20})", "symbol": symbol}

    try:
        df_feat = generate_features(df, config['feat_cols'])
        df_feat = df_feat.dropna()

        if len(df_feat) < seq_len:
            return {"error": f"Insufficient features (got {len(df_feat)}, need {seq_len})", "symbol": symbol}

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
        if ticker.get("retCode") == 0 and ticker.get("result", {}).get("list"):
            current_price = float(ticker["result"]["list"][0]["lastPrice"])
        else:
            current_price = float(df['close'].iloc[-1])

        if current_price <= 0:
            return {"error": f"Invalid price: {current_price}", "symbol": symbol}

        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "current_price": current_price, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_length": len(df)
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}", "symbol": symbol}


def print_dashboard(account: Account, prices: Dict[str, float]):
    os.system('clear' if os.name == 'posix' else 'cls')

    print("\n" + "=" * 110)
    print(f"{'🎯 PatchTST 페이퍼 트레이딩 (유연한 로더) - 레버리지 ' + str(LEVERAGE) + 'x':^110}")
    print("=" * 110)

    # 계좌 정보 계산
    cash_balance = account.balance  # 현금 잔고
    total_margin = sum(p.margin for p in account.positions.values())  # 증거금
    unrealized_pnl = sum(p.get_pnl(prices.get(p.symbol, p.entry_price)) for p in account.positions.values())  # 평가 손익
    position_value = total_margin + unrealized_pnl  # 포지션 평가액
    total_value = cash_balance + position_value  # 총 자산
    total_return = (total_value / account.initial_capital - 1) * 100

    print(f"\n💰 계좌 현황")
    print(f"   현금 잔고:     ${cash_balance:>12,.2f}")
    if account.positions:
        print(f"   포지션 증거금: ${total_margin:>12,.2f}")
        pnl_color = "🟢" if unrealized_pnl >= 0 else "🔴"
        print(f"   평가 손익:     {pnl_color} ${unrealized_pnl:>+12,.2f}")
        print(f"   포지션 평가액: ${position_value:>12,.2f}")
    print(f"   " + "-" * 40)
    print(f"   총 자산:       ${total_value:>12,.2f}  ({total_return:>+6.2f}%)")

    if account.positions:
        print(f"\n📍 보유 포지션 ({len(account.positions)}/{MAX_POSITIONS})")
        print(f"{'심볼':^12} | {'방향':^8} | {'진입가':^12} | {'현재가':^12} | {'손익(ROE)':^22}")
        print("-" * 80)

        for symbol, pos in account.positions.items():
            current_price = prices.get(symbol, pos.entry_price)
            pnl = pos.get_pnl(current_price)
            roe = pos.get_roe(current_price)
            emoji = "📈" if pos.direction == "Long" else "📉"
            pnl_emoji = "🟢" if pnl > 0 else "🔴"
            print(f"{symbol:^12} | {emoji} {pos.direction:^6} | ${pos.entry_price:>10,.4f} | "
                  f"${current_price:>10,.4f} | {pnl_emoji} ${pnl:>+8,.2f} ({roe:>+6.1f}%)")

    stats = account.get_stats()
    if stats["total_trades"] > 0:
        print(f"\n📊 거래 통계")
        print(f"   총 거래: {stats['total_trades']:>3}회 | 승률: {stats['win_rate']:>6.1f}%")
        print(f"   평균 ROE: {stats['avg_roe']:>+6.1f}%")

    print("\n" + "=" * 110)


def save_trades(account: Account):
    if not account.trades:
        return
    data = [asdict(t) for t in account.trades]
    with open(TRADE_LOG_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n💾 거래 내역 저장: {TRADE_LOG_FILE}")


def main():
    print(f"\n{'=' * 110}")
    print(f"{'🎯 PatchTST 페이퍼 트레이딩 시작 (유연한 로더)':^110}")
    print(f"{'레버리지: ' + str(LEVERAGE) + 'x | 신뢰도: ' + f'{CONF_THRESHOLD:.0%}':^110}")
    print(f"{'=' * 110}")

    load_models()
    account = Account(INITIAL_CAPITAL)

    try:
        loop_count = 0
        while True:
            loop_count += 1
            current_time = datetime.now()

            prices = {}
            for symbol in MODELS.keys():
                ticker = API.get_ticker(symbol)
                if ticker.get("retCode") == 0 and ticker.get("result", {}).get("list"):
                    prices[symbol] = float(ticker["result"]["list"][0]["lastPrice"])

            for symbol in list(account.positions.keys()):
                position = account.positions[symbol]
                current_price = prices.get(symbol, position.entry_price)
                should_close, reason = position.should_close(current_price, current_time)
                if should_close:
                    account.close_position(symbol, current_price, reason)
                else:
                    result = predict(symbol)
                    if "error" not in result and result.get("confidence", 0) >= CONF_THRESHOLD:
                        signal_dir = result["direction"]
                        if (position.direction == "Long" and signal_dir == "Short") or \
                                (position.direction == "Short" and signal_dir == "Long"):
                            account.close_position(symbol, current_price, "Reverse Signal")

            print_dashboard(account, prices)

            print(f"\n🔍 신호 스캔 ({len(MODELS)}개 심볼)")
            print(f"{'심볼':^12} | {'가격':^12} | {'방향':^10} | {'신뢰도':^8} | {'데이터':^10} | {'신호':^20}")
            print("-" * 95)

            for symbol in MODELS.keys():
                result = predict(symbol)

                if "error" in result:
                    error_msg = result.get('error', 'Unknown')
                    print(f"{symbol:^12} | {'N/A':^12} | {'오류':^10} | {'N/A':^8} | {'N/A':^10} | ❌ {error_msg}")
                    continue

                direction = result["direction"]
                confidence = result["confidence"]
                price = result["current_price"]
                data_len = result.get("data_length", 0)
                dir_icon = "📈" if direction == "Long" else "📉"

                if confidence < CONF_THRESHOLD:
                    signal = f"⚠️  신호 약함"
                elif direction == "Long":
                    signal = f"🟢 매수 신호"
                else:
                    signal = f"🔴 매도 신호"

                print(f"{symbol:^12} | ${price:>10,.4f} | {dir_icon} {direction:^8} | "
                      f"{confidence:>6.1%} | {data_len:^10} | {signal}")

                if account.can_open_position(symbol) and confidence >= CONF_THRESHOLD:
                    account.open_position(symbol, direction, price)

            print(f"\n[스캔 #{loop_count}] {current_time.strftime('%H:%M:%S')} | 다음 스캔까지 {INTERVAL_SEC}초...")
            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n프로그램 종료 중...")
        print_dashboard(account, prices)
        save_trades(account)

        stats = account.get_stats()
        if stats["total_trades"] > 0:
            print(f"\n{'=' * 110}")
            print(f"{'📊 최종 결과':^110}")
            print(f"{'=' * 110}")
            final_return = (account.balance / account.initial_capital - 1) * 100
            print(f"   최종 잔고: ${account.balance:,.2f} ({final_return:+.2f}%)")
            print(f"   총 거래: {stats['total_trades']}회 | 승률: {stats['win_rate']:.1f}%")
            print(f"{'=' * 110}")

        print("\n✅ 프로그램 종료")


if __name__ == "__main__":
    main()