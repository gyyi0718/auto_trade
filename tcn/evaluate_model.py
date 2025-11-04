# evaluate_model.py
# -*- coding: utf-8 -*-
"""
학습된 코인 옵션 모델 평가 스크립트
- 성능 메트릭 계산 (MAE, RMSE, 방향 정확도)
- 실전 수익성 시뮬레이션
- 예측 vs 실제 비교
"""
import os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Tuple
import matplotlib.pyplot as plt
from pandas.api.types import is_datetime64_any_dtype


# ---------- TCN 모델 정의 (학습 코드와 동일) ----------
class Chomp1d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):
        return x[:, :, :-self.c].contiguous() if self.c > 0 else x


def wconv(i, o, k, d):
    import torch.nn.utils as U
    pad = (k - 1) * d
    return U.weight_norm(nn.Conv1d(i, o, k, padding=pad, dilation=d))


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        self.conv1 = wconv(in_ch, out_ch, kernel, dilation)
        self.chomp1 = Chomp1d((kernel - 1) * dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = wconv(out_ch, out_ch, kernel, dilation)
        self.chomp2 = Chomp1d((kernel - 1) * dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.dropout1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class CryptoOptionPriceTCN(nn.Module):
    def __init__(self, in_features, hidden=128, levels=6, kernel=3, dropout=0.3):
        super().__init__()

        layers = []
        ch = in_features
        for i in range(levels):
            layers.append(TCNBlock(ch, hidden, kernel, 2 ** i, dropout))
            ch = hidden
        self.tcn = nn.Sequential(*layers)

        self.head_return = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        self.head_volatility = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.tcn(x)[:, :, -1]
        return_pct = self.head_return(h).squeeze(-1)
        volatility = self.head_volatility(h).squeeze(-1)
        return return_pct, volatility


# ---------- 피처 생성 (학습과 동일하게) ----------
def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """피처 생성"""
    g = df.copy()
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g.groupby("symbol")["logc"].diff().fillna(0.0)

    def roll_std(s, w):
        return s.rolling(w, min_periods=max(2, w // 3)).std() * np.sqrt(365 * 24)

    for w in (24, 24 * 3, 24 * 7, 24 * 14):
        g[f"rv{w}h"] = g.groupby("symbol")["ret1"].transform(lambda s: roll_std(s, w))

    g["vol_of_vol"] = g.groupby("symbol")["rv24h"].transform(
        lambda s: s.rolling(24 * 7, min_periods=24).std()
    )

    for w in (12, 24, 24 * 3, 24 * 7):
        g[f"mom{w}h"] = g.groupby("symbol")["close"].pct_change(w)

    if "funding_rate" in df.columns:
        for w in (8, 24, 24 * 3):
            g[f"funding_ma{w}"] = g.groupby("symbol")["funding_rate"].transform(
                lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean()
            )

    for w in (24, 24 * 3, 24 * 7):
        mu = g.groupby("symbol")["volume"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean()
        )
        sd = g.groupby("symbol")["volume"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).std()
        )
        g[f"vol_z{w}h"] = (g["volume"] - mu) / sd.replace(0, np.nan)

    prev_close = g.groupby("symbol")["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    g["atr24"] = tr.groupby(g["symbol"]).transform(
        lambda s: s.rolling(24, min_periods=8).mean()
    )

    g["hour"] = g["date"].dt.hour
    g["hour_sin"] = np.sin(2 * np.pi * g["hour"] / 24)
    g["hour_cos"] = np.cos(2 * np.pi * g["hour"] / 24)
    g["parkinson_vol"] = np.sqrt(1 / (4 * np.log(2)) * np.log(g["high"] / g["low"]) ** 2) * np.sqrt(365 * 24)
    g["price_norm"] = g["close"] / g.groupby("symbol")["close"].transform(
        lambda s: s.rolling(24 * 30, min_periods=24).mean()
    )

    base_feats = ["ret1", "rv24h", "rv72h", "rv168h", "rv336h", "vol_of_vol",
                  "mom12h", "mom24h", "mom72h", "mom168h",
                  "vol_z24h", "vol_z72h", "vol_z168h", "atr24",
                  "hour_sin", "hour_cos", "parkinson_vol", "price_norm"]

    if "funding_rate" in df.columns:
        base_feats += ["funding_ma8", "funding_ma24", "funding_ma72"]

    return g, base_feats


# ---------- 평가 함수 ----------
def evaluate_model(model, df_test, feat_cols, seq_len, horizon, mu, sd, device):
    """모델 평가"""
    model.eval()

    # 피처 정규화
    X = df_test[feat_cols].to_numpy(np.float32)
    X_norm = (X - mu) / sd

    # 실제 값
    current_prices = df_test["close"].to_numpy(float)

    # 미래 수익률 계산
    future_returns = []
    for sym, g in df_test.groupby("symbol", sort=False):
        close = g["close"].to_numpy(float)
        n = len(close)
        returns = np.zeros(n, dtype=np.float32)

        for i in range(n - horizon):
            returns[i] = (close[i + horizon] / close[i] - 1.0) * 100

        future_returns.extend(returns)

    future_returns = np.array(future_returns, dtype=np.float32)

    # 예측
    predictions_return = []
    predictions_vol = []
    valid_indices = []

    sym_idx = df_test.groupby("symbol").cumcount().to_numpy()

    with torch.no_grad():
        for i in range(len(df_test)):
            if sym_idx[i] < seq_len - 1:
                continue
            if np.isnan(future_returns[i]):
                continue

            s = i - seq_len + 1
            x = X_norm[s:i + 1]
            x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)

            pred_return, pred_vol = model(x_tensor)

            predictions_return.append(pred_return.cpu().item())
            predictions_vol.append(pred_vol.cpu().item())
            valid_indices.append(i)

    predictions_return = np.array(predictions_return)
    predictions_vol = np.array(predictions_vol)
    actual_returns = future_returns[valid_indices]
    actual_prices = current_prices[valid_indices]

    # 메트릭 계산
    mae = np.mean(np.abs(predictions_return - actual_returns))
    rmse = np.sqrt(np.mean((predictions_return - actual_returns) ** 2))

    # 방향 정확도
    pred_direction = (predictions_return > 0).astype(int)
    actual_direction = (actual_returns > 0).astype(int)
    direction_acc = (pred_direction == actual_direction).mean() * 100

    # 수익성 시뮬레이션 (간단 버전)
    correct_trades = (pred_direction == actual_direction).sum()
    wrong_trades = len(pred_direction) - correct_trades

    # 옵션 수익 가정: 맞으면 +50%, 틀리면 -30% (프리미엄 손실)
    expected_return = (correct_trades * 0.5 - wrong_trades * 0.3) / len(pred_direction) * 100

    return {
        'mae': mae,
        'rmse': rmse,
        'direction_accuracy': direction_acc,
        'predictions': predictions_return,
        'actuals': actual_returns,
        'prices': actual_prices,
        'dates': df_test.iloc[valid_indices]['date'].values,
        'symbols': df_test.iloc[valid_indices]['symbol'].values,
        'expected_return': expected_return,
        'total_trades': len(predictions_return),
        'correct_trades': correct_trades,
        'wrong_trades': wrong_trades
    }


# ---------- 결과 출력 ----------
def print_evaluation_report(results):
    """평가 결과 출력"""
    print("\n" + "=" * 60)
    print("📊 모델 평가 결과")
    print("=" * 60)

    print(f"\n🎯 예측 정확도:")
    print(f"  • MAE (평균 절대 오차):     {results['mae']:.3f}%")
    print(f"  • RMSE (평균 제곱근 오차):  {results['rmse']:.3f}%")
    print(f"  • 방향 정확도:              {results['direction_accuracy']:.2f}%")

    print(f"\n💰 수익성 시뮬레이션:")
    print(f"  • 총 거래 수:               {results['total_trades']}")
    print(
        f"  • 성공 거래:                {results['correct_trades']} ({results['correct_trades'] / results['total_trades'] * 100:.1f}%)")
    print(
        f"  • 실패 거래:                {results['wrong_trades']} ({results['wrong_trades'] / results['total_trades'] * 100:.1f}%)")
    print(f"  • 기대 수익률:              {results['expected_return']:.2f}%")

    print(f"\n📈 실전 판단:")
    if results['mae'] < 2.0:
        print("  ✅ 우수 - 실전 사용 강력 추천")
    elif results['mae'] < 3.0:
        print("  🟢 좋음 - 실전 사용 가능")
    elif results['mae'] < 4.0:
        print("  🟡 보통 - 신중한 사용 필요")
    else:
        print("  🔴 위험 - 추가 학습 필요")

    if results['direction_accuracy'] > 60:
        print("  ✅ 방향 예측 우수")
    elif results['direction_accuracy'] > 55:
        print("  🟢 방향 예측 양호")
    else:
        print("  🔴 방향 예측 부족")

    print("\n" + "=" * 60)


# ---------- 시각화 ----------
def plot_predictions(results, save_path=None):
    """예측 vs 실제 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Scatter plot: 예측 vs 실제
    ax1 = axes[0, 0]
    ax1.scatter(results['actuals'], results['predictions'], alpha=0.5, s=10)
    ax1.plot([-20, 20], [-20, 20], 'r--', label='Perfect Prediction')
    ax1.set_xlabel('Actual Return (%)')
    ax1.set_ylabel('Predicted Return (%)')
    ax1.set_title(f'Prediction vs Actual (MAE={results["mae"]:.2f}%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 오차 분포
    ax2 = axes[0, 1]
    errors = results['predictions'] - results['actuals']
    ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='r', linestyle='--', label='Zero Error')
    ax2.set_xlabel('Prediction Error (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Error Distribution (RMSE={results["rmse"]:.2f}%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 시계열 예측 (최근 100개)
    ax3 = axes[1, 0]
    n_show = min(100, len(results['predictions']))
    x_range = range(n_show)
    ax3.plot(x_range, results['actuals'][-n_show:], label='Actual', linewidth=2)
    ax3.plot(x_range, results['predictions'][-n_show:], label='Predicted', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Return (%)')
    ax3.set_title('Recent Predictions (Last 100)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 방향 정확도
    ax4 = axes[1, 1]
    categories = ['Correct', 'Wrong']
    values = [results['correct_trades'], results['wrong_trades']]
    colors = ['green', 'red']
    ax4.bar(categories, values, color=colors, alpha=0.7)
    ax4.set_ylabel('Number of Trades')
    ax4.set_title(f'Direction Accuracy: {results["direction_accuracy"]:.1f}%')
    for i, v in enumerate(values):
        ax4.text(i, v + 5, str(v), ha='center', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 그래프 저장: {save_path}")

    plt.show()


# ---------- 메인 ----------
def main(args):
    # 모델 로드
    ckpt_path = os.path.join(args.model_dir, "daily_simple_best.ckpt")
    meta_path = os.path.join(args.model_dir, "daily_simple_meta.json")

    print(f"🔍 모델 로딩: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    feat_cols = meta['feat_cols']
    seq_len = meta['seq_len']
    horizon = meta['horizon']
    mu = np.array(meta['scaler_mu'], dtype=np.float32)
    sd = np.array(meta['scaler_sd'], dtype=np.float32)

    # 모델 초기화
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CryptoOptionPriceTCN(in_features=len(feat_cols))
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()

    print(f"✅ 모델 로드 완료")
    print(f"   - Sequence Length: {seq_len}")
    print(f"   - Prediction Horizon: {horizon}시간")
    print(f"   - Features: {len(feat_cols)}")

    # 테스트 데이터 로드
    if args.data.lower().endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    if "date" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})

    if not is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    print(f"\n📂 테스트 데이터: {len(df)} rows")

    # 피처 생성
    df_feat, _ = make_features(df)
    df_feat = df_feat.dropna(subset=feat_cols)

    print(f"   - 피처 생성 후: {len(df_feat)} rows")

    # 평가
    print(f"\n🚀 평가 시작...")
    results = evaluate_model(model, df_feat, feat_cols, seq_len, horizon, mu, sd, device)

    # 결과 출력
    print_evaluation_report(results)

    # 시각화
    if args.plot:
        save_path = os.path.join(args.model_dir, "evaluation_results.png")
        plot_predictions(results, save_path)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="모델 디렉토리 경로")
    ap.add_argument("--data", required=True, help="테스트 데이터 (CSV/Parquet)")
    ap.add_argument("--plot", action="store_true", help="시각화 생성")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)