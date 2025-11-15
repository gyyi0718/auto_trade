# paper_trading_final_clear.py - 명확한 신호 표시
import os, time, json, warnings
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import requests, certifi

warnings.filterwarnings("ignore")
os.environ["SSL_CERT_FILE"] = certifi.where()

SYMBOLS = os.getenv("SYMBOLS", "LSKUSDT").split(",")
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "3"))
MODEL_PATHS = {
    "LSKUSDT": "D:/ygy_work/coin/multimodel/models_regression/lsk/best_model.pth",
    "XNOUSDT": "D:/ygy_work/coin/multimodel/models_regression/xno/best_model.pth",
    "ESPORTSUSDT": "D:/ygy_work/coin/multimodel/models_regression/esports/best_model.pth",
    "PARTIUSDT": "D:/ygy_work/coin/multimodel/models_regression/parti/best_model.pth",
    "BTCUSDT": "D:/ygy_work/coin/multimodel/models_regression/btc/best_model.pth",
    "ETHUSDT": "D:/ygy_work/coin/multimodel/models_regression/eth/best_model.pth",
    "SOLUSDT": "D:/ygy_work/coin/multimodel/models_regression/sol/best_model.pth",
    "EVAAUSDT": "D:/ygy_work/coin/multimodel/models_regression/evaa/best_model.pth",
    "RESOLVUSDT": "D:/ygy_work/coin/multimodel/models_regression/evaa/best_model.pth",

}
LONG_THRESHOLD = float(os.getenv("LONG_THRESHOLD", "5.0"))
SHORT_THRESHOLD = float(os.getenv("SHORT_THRESHOLD", "5.0"))
STRONG_THRESHOLD = float(os.getenv("STRONG_THRESHOLD", "15.0"))
INITIAL_CAPITAL = 1000.0
BASE_POSITION_SIZE_PCT = 0.2
LEVERAGE = 25
MAX_POSITIONS = 5
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.03
MAX_HOLD_MINUTES = 30
LIQUIDATION_BUFFER = 0.8


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


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
    predicted_bps: float
    confidence: str

    def get_pnl(self, p):
        return (p - self.entry_price) * self.quantity if self.direction == "LONG" else (
                                                                                                   self.entry_price - p) * self.quantity

    def get_roe(self, p):
        return (self.get_pnl(p) / self.margin) * 100


@dataclass
class Trade:
    symbol: str
    direction: str
    pnl: float
    roe: float
    exit_reason: str


class Portfolio:
    def __init__(self, bal):
        self.balance = bal
        self.initial = bal
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

    def get_available(self):
        return self.balance - sum(p.margin for p in self.positions.values())

    def can_open(self, m):
        return len(self.positions) < MAX_POSITIONS and self.get_available() >= m

    def open_position(self, symbol, direction, price, pred_bps, conf, size_pct):
        margin = self.balance * size_pct
        qty = (margin * LEVERAGE) / price

        if direction == "LONG":
            sl = price * (1 - STOP_LOSS_PCT)
            tp = price * (1 + TAKE_PROFIT_PCT)
            liq = price * (1 - (1 / LEVERAGE) * LIQUIDATION_BUFFER)
        else:  # SHORT
            sl = price * (1 + STOP_LOSS_PCT)
            tp = price * (1 - TAKE_PROFIT_PCT)
            liq = price * (1 + (1 / LEVERAGE) * LIQUIDATION_BUFFER)

        self.positions[symbol] = Position(
            symbol, direction, price, qty, datetime.now(),
            sl, tp, LEVERAGE, margin, liq, pred_bps, conf
        )

    def close_position(self, symbol, price, reason):
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        pnl = pos.get_pnl(price)
        self.balance += pnl
        self.trades.append(Trade(symbol, pos.direction, pnl, pos.get_roe(price), reason))
        del self.positions[symbol]

    def get_stats(self):
        if not self.trades:
            return {"total": 0, "wins": 0, "win_rate": 0, "avg_roe": 0, "liquidations": 0}
        wins = [t for t in self.trades if t.pnl > 0]
        liq = sum(1 for t in self.trades if t.exit_reason == "Liquidation")
        return {
            "total": len(self.trades),
            "wins": len(wins),
            "win_rate": len(wins) / len(self.trades) * 100,
            "avg_roe": sum(t.roe for t in self.trades) / len(self.trades),
            "liquidations": liq
        }


# 모델 정의
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = nn.functional.pad(x, (self.padding, 0))
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = nn.functional.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class AdvancedTCN_Regression(nn.Module):
    def __init__(self, in_f, hidden=64, levels=4, kernel_size=3, dropout=0.3, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(in_f, hidden)
        layers = []
        for i in range(levels):
            dilation = 2 ** i
            layers.append(TemporalBlock(hidden, hidden, kernel_size, stride=1, dilation=dilation, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.attention = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x_tcn = x.transpose(1, 2)
        x_tcn = self.tcn(x_tcn)
        x_tcn = x_tcn.transpose(1, 2)
        attn_out, _ = self.attention(x_tcn, x_tcn, x_tcn)
        x_combined = self.norm1(x_tcn + attn_out)
        x_pooled = x_combined.mean(dim=1)
        return self.fc(x_pooled).squeeze(-1)


# 모델 로드
print("\n" + "=" * 100)
print(f"{'🤖 모델 로딩':^100}")
print("=" * 100)

MODELS, MODEL_CONFIGS = {}, {}
for symbol in SYMBOLS:
    symbol = symbol.strip()
    if symbol not in MODEL_PATHS or not os.path.exists(MODEL_PATHS[symbol]):
        continue
    try:
        model_dir = os.path.dirname(MODEL_PATHS[symbol])
        meta_path = os.path.join(model_dir, "meta.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(model_dir, "meta.json")

        with open(meta_path) as f:
            meta = json.load(f)

        model = AdvancedTCN_Regression(len(meta['feat_cols']), 64, 4, 3, 0.3, 4)
        model.load_state_dict(torch.load(MODEL_PATHS[symbol], map_location="cpu")['model_state'])
        model.eval()

        MODELS[symbol] = model
        MODEL_CONFIGS[symbol] = {
            'feat_cols': meta['feat_cols'],
            'seq_len': meta['seq_len'],
            'scaler_mu': np.array(meta['scaler_mu'], dtype=np.float32),
            'scaler_sd': np.array(meta['scaler_sd'], dtype=np.float32)
        }
        print(f"  ✅ {symbol} 로드 완료 (features: {len(meta['feat_cols'])})")
    except Exception as e:
        print(f"  ❌ {symbol}: {e}")

print("=" * 100)
print(f"  총 {len(MODELS)}개 모델 로드 완료")
print("=" * 100 + "\n")

if not MODELS:
    print("❌ 로드된 모델 없음")
    exit(1)


class BybitAPI:
    def __init__(self):
        self.base_url = "https://api.bybit.com"

    def get_ticker(self, s):
        try:
            return requests.get(f"{self.base_url}/v5/market/tickers",
                                params={"category": "linear", "symbol": s}, timeout=10).json()
        except:
            return {}

    def get_klines(self, s, i="5", l=200):
        try:
            r = requests.get(f"{self.base_url}/v5/market/kline",
                             params={"category": "linear", "symbol": s, "interval": i, "limit": l},
                             timeout=10).json()
            if r.get("retCode") == 0:
                df = pd.DataFrame(r["result"]["list"],
                                  columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
                df = df.astype({"open": float, "high": float, "low": float,
                                "close": float, "volume": float, "turnover": float})
                df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit='ms')
                return df.sort_values("timestamp").reset_index(drop=True)
        except:
            return pd.DataFrame()


API = BybitAPI()


def generate_features(df, feat_cols):
    g = df.copy()
    if "turnover" in feat_cols and "turnover" not in g.columns:
        g["turnover"] = g["close"] * g["volume"]
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g["logc"].diff().fillna(0.0)
    for w in (8, 20, 40, 120):
        g[f"rv{w}"] = g["ret1"].rolling(w, min_periods=2).std().fillna(0.0)
        ema = g["close"].ewm(span=w, adjust=False).mean()
        g[f"mom{w}"] = (g["close"] / ema - 1.0).fillna(0.0)
    for w in (20, 40, 120):
        mu = g["volume"].rolling(w, min_periods=2).mean()
        sd = g["volume"].rolling(w, min_periods=2).std().replace(0, 1.0)
        g[f"vz{w}"] = ((g["volume"] - mu) / sd).fillna(0.0)
        rh, rl = g["high"].rolling(w).max(), g["low"].rolling(w).min()
        g[f"range{w}"] = ((g["high"] - g["low"]) / (rh - rl + 1e-9)).fillna(0.0)
    g["vol_spike"] = (g["volume"] / g["volume"].shift(1) - 1.0).clip(-5.0, 5.0).fillna(0.0)
    for w in (8, 14, 20):
        delta = g["close"].diff()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        rs = gain / (loss + 1e-9)
        g[f"rsi{w}"] = (100 - 100 / (1 + rs)).fillna(50.0) / 100.0
    body = (g["close"] - g["open"]).abs()
    rng = g["high"] - g["low"] + 1e-9
    g["body_ratio"] = (body / rng).fillna(0.0)
    g["upper_shadow"] = ((g["high"] - g[["open", "close"]].max(axis=1)) / rng).fillna(0.0)
    g["lower_shadow"] = ((g[["open", "close"]].min(axis=1) - g["low"]) / rng).fillna(0.0)
    g["hour"] = pd.to_datetime(g["timestamp"]).dt.hour / 24.0
    g["minute"] = pd.to_datetime(g["timestamp"]).dt.minute / 60.0
    g["day_of_week"] = pd.to_datetime(g["timestamp"]).dt.dayofweek / 7.0
    g = g.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    for col in feat_cols:
        if col not in g.columns:
            g[col] = 0.0
    return g[feat_cols].astype(np.float32)


def predict(symbol):
    if symbol not in MODELS:
        return {"error": "No model"}

    model = MODELS[symbol]
    config = MODEL_CONFIGS[symbol]

    df = API.get_klines(symbol)
    if df.empty or len(df) < config['seq_len'] + 20:
        return {"error": "No data"}

    df_feat = generate_features(df, config['feat_cols']).dropna()
    if len(df_feat) < config['seq_len']:
        return {"error": "Insufficient"}

    X = df_feat.iloc[-config['seq_len']:].values.astype(np.float32)
    X = np.clip((X - config['scaler_mu']) / config['scaler_sd'], -10.0, 10.0)

    with torch.no_grad():
        pred_bps = model(torch.from_numpy(X).unsqueeze(0)).item()

    # 신호 결정
    if pred_bps > STRONG_THRESHOLD:
        signal, confidence = "LONG", "strong"
    elif pred_bps > LONG_THRESHOLD:
        signal, confidence = "LONG", "weak"
    elif pred_bps < -STRONG_THRESHOLD:
        signal, confidence = "SHORT", "strong"
    elif pred_bps < -SHORT_THRESHOLD:
        signal, confidence = "SHORT", "weak"
    else:
        signal, confidence = "HOLD", "neutral"

    return {
        "symbol": symbol,
        "signal": signal,
        "predicted_bps": pred_bps,
        "confidence": confidence,
        "error": None
    }


def main():
    portfolio = Portfolio(INITIAL_CAPITAL)
    iteration = 0

    print(f"임계값 설정:")
    print(f"  LONG 진입: > {LONG_THRESHOLD} bps")
    print(f"  SHORT 진입: < -{SHORT_THRESHOLD} bps")
    print(f"  강한 신호: ±{STRONG_THRESHOLD} bps")
    print(f"\n시작합니다...\n")
    time.sleep(2)

    try:
        while True:
            iteration += 1
            clear_screen()

            print(f"\n{'=' * 100}")
            print(f"{'🤖 실시간 페이퍼 트레이딩':^100}")
            print(f"{'=' * 100}")
            print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 반복: {iteration}")
            print(f"{'=' * 100}\n")

            # 포트폴리오 현황
            stats = portfolio.get_stats()
            pnl = portfolio.balance - portfolio.initial
            pnl_pct = (portfolio.balance / portfolio.initial - 1) * 100

            print(f"💰 잔고: ${portfolio.balance:,.2f} | PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
            print(f"📊 포지션: {len(portfolio.positions)}/{MAX_POSITIONS} | 가용: ${portfolio.get_available():,.2f}")

            if stats['total'] > 0:
                print(
                    f"📈 거래: {stats['total']} | 승률: {stats['win_rate']:.1f}% | 평균ROE: {stats['avg_roe']:+.2f}% | 청산: {stats['liquidations']}")

            print(f"\n{'─' * 100}\n")

            # 포지션 관리
            ct = datetime.now()
            for symbol in list(portfolio.positions.keys()):
                pos = portfolio.positions[symbol]
                ticker = API.get_ticker(symbol)
                if not ticker.get("result", {}).get("list"):
                    continue

                price = float(ticker["result"]["list"][0]["lastPrice"])
                hold_min = (ct - pos.entry_time).total_seconds() / 60
                pnl = pos.get_pnl(price)
                roe = pos.get_roe(price)

                # 방향에 따른 이모지
                if pos.direction == "LONG":
                    dir_emoji = "🟢 LONG"
                else:
                    dir_emoji = "🔴 SHORT"

                pnl_emoji = "📈" if pnl > 0 else "📉"

                print(f"{dir_emoji} {symbol} | ${pos.entry_price:,.4f} → ${price:,.4f}")
                print(f"   {pnl_emoji} 미실현: ${pnl:+,.2f} ({roe:+.2f}% ROE) | ⏱️  {hold_min:.1f}분")
                print(f"   예측: {pos.predicted_bps:+.1f} bps ({pos.confidence})\n")

                # 청산 체크
                if pos.direction == "LONG":
                    if price <= pos.liquidation_price:
                        portfolio.close_position(symbol, price, "Liquidation")
                        print(f"   💀 청산됨!\n")
                    elif price >= pos.take_profit:
                        portfolio.close_position(symbol, price, "TP")
                        print(f"   ✅ 익절!\n")
                    elif price <= pos.stop_loss:
                        portfolio.close_position(symbol, price, "SL")
                        print(f"   🛑 손절!\n")
                    elif hold_min >= MAX_HOLD_MINUTES:
                        portfolio.close_position(symbol, price, "Time")
                        print(f"   ⏰ 시간 종료!\n")
                else:  # SHORT
                    if price >= pos.liquidation_price:
                        portfolio.close_position(symbol, price, "Liquidation")
                        print(f"   💀 청산됨!\n")
                    elif price <= pos.take_profit:
                        portfolio.close_position(symbol, price, "TP")
                        print(f"   ✅ 익절!\n")
                    elif price >= pos.stop_loss:
                        portfolio.close_position(symbol, price, "SL")
                        print(f"   🛑 손절!\n")
                    elif hold_min >= MAX_HOLD_MINUTES:
                        portfolio.close_position(symbol, price, "Time")
                        print(f"   ⏰ 시간 종료!\n")

            # 신규 진입 검토
            print("🔍 신호 모니터링:")
            print(f"{'─' * 100}")

            for symbol in SYMBOLS:
                symbol = symbol.strip()

                if symbol in portfolio.positions:
                    print(f"  ⏸️  {symbol}: 포지션 보유중")
                    continue

                pred = predict(symbol)

                if pred.get("error"):
                    print(f"  ⚠️  {symbol}: {pred['error']}")
                    continue

                sig = pred["signal"]
                bps = pred["predicted_bps"]
                conf = pred["confidence"]

                # 명확한 신호 표시
                if sig == "LONG":
                    if conf == "strong":
                        emoji = "🟢🔥 LONG (강함)"
                    else:
                        emoji = "🟢   LONG (약함)"
                elif sig == "SHORT":
                    if conf == "strong":
                        emoji = "🔴🔥 SHORT (강함)"
                    else:
                        emoji = "🔴   SHORT (약함)"
                else:
                    emoji = "⚪   HOLD (관망)"

                print(f"  {emoji} | {symbol} | 예측: {bps:+.2f} bps")

                # 진입 시도
                if sig != "HOLD":
                    ps = BASE_POSITION_SIZE_PCT * (1.5 if conf == "strong" else 1.0)
                    ps = min(ps, 0.3)
                    required_margin = portfolio.balance * ps

                    if portfolio.can_open(required_margin):
                        ticker = API.get_ticker(symbol)
                        if ticker.get("result", {}).get("list"):
                            price = float(ticker["result"]["list"][0]["lastPrice"])
                            portfolio.open_position(symbol, sig, price, bps, conf, ps)
                            print(f"     ✅ {sig} 포지션 진입! 가격: ${price:,.4f}")
                    else:
                        print(f"     ⛔ 진입 불가 (자금/포지션 한도)")

            print(f"{'─' * 100}")
            print(f"\n{'=' * 100}")
            print(f"다음 업데이트: {INTERVAL_SEC}초 후...")

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        clear_screen()
        stats = portfolio.get_stats()

        print(f"\n{'=' * 100}")
        print(f"{'📊 최종 통계':^100}")
        print(f"{'=' * 100}")
        print(f"총 거래: {stats['total']} | 승: {stats['wins']} | 승률: {stats['win_rate']:.1f}%")
        print(f"최종 잔고: ${portfolio.balance:,.2f} | 수익: ${portfolio.balance - portfolio.initial:+,.2f}")
        print(f"수익률: {(portfolio.balance / portfolio.initial - 1) * 100:+.2f}% | 평균 ROE: {stats['avg_roe']:+.2f}%")
        print(f"청산: {stats['liquidations']}회")
        print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()