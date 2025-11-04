# generate_signals.py
# -*- coding: utf-8 -*-
"""
실시간 트레이딩 신호 생성 + 알림
"""
import os, json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# ========= TCN 모델 구조 =========
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
    def __init__(self, in_f, hidden=128, levels=6, k=3, drop=0.2):
        super().__init__()
        L = []
        ch = in_f
        for i in range(levels):
            L.append(Block(ch, hidden, k, 2 ** i, drop))
            ch = hidden
        self.tcn = nn.Sequential(*L)
        self.head_side = nn.Linear(hidden, 2)

    def forward(self, X):
        X = X.transpose(1, 2)
        H = self.tcn(X)[:, :, -1]
        return self.head_side(H)


# ========= 피처 생성 (train_tcn_daily.py와 동일) =========
def make_features(df):
    """train_tcn_daily.py의 make_features와 동일"""
    g = df.copy()
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g.groupby("symbol")["logc"].diff().fillna(0.0)

    def roll_std(s, w):
        return s.rolling(w, min_periods=max(2, w // 3)).std()

    for w in (5, 10, 20, 60):
        g[f"rv{w}"] = g.groupby("symbol")["ret1"].apply(lambda s: roll_std(s, w)).reset_index(level=0, drop=True)

    def mom(gp, w):
        ema = gp["close"].ewm(span=w, adjust=False).mean()
        return gp["close"] / ema - 1.0

    for w in (5, 10, 20, 60):
        g[f"mom{w}"] = g.groupby("symbol", group_keys=False).apply(lambda s: mom(s, w))

    for w in (10, 20, 60):
        mu = g.groupby("symbol")["volume"].apply(lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean()).reset_index(
            level=0, drop=True)
        sd = g.groupby("symbol")["volume"].apply(lambda s: s.rolling(w, min_periods=max(2, w // 3)).std()).reset_index(
            level=0, drop=True)
        sd = sd.replace(0, np.nan)
        g[f"vz{w}"] = (g["volume"] - mu) / sd.replace({0: np.nan}).fillna(1.0)

    prev_close = g.groupby("symbol")["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()], axis=1).max(axis=1)
    g["atr14"] = tr.groupby(g["symbol"]).transform(lambda s: s.rolling(14, min_periods=5).mean())

    feats = ["ret1", "rv5", "rv10", "rv20", "rv60", "mom5", "mom10", "mom20", "mom60", "vz10", "vz20", "vz60", "atr14"]
    g = g.dropna(subset=feats).reset_index(drop=True)
    return g, feats


# ========= 모델 로드 =========
def load_model(ckpt_path):
    """모델 로드"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feat_cols = ckpt["feat_cols"]
    mu = np.asarray(ckpt["scaler_mu"], dtype=np.float32)
    sd = np.asarray(ckpt["scaler_sd"], dtype=np.float32)
    sd[sd == 0] = 1.0

    model = TCN_Simple(in_f=len(feat_cols))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    return model, feat_cols, mu, sd


# ========= 신호 생성 =========
def generate_signals(data_path, ckpt_path, seq_len=60, min_confidence=0.05):
    """최신 데이터로 신호 생성"""

    # 데이터 로드
    print(f"📂 데이터 로드: {data_path}")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # 날짜 처리
    date_col = "date" if "date" in df.columns else "timestamp"
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
    df = df.dropna(subset=[date_col])

    print(f"   → {len(df):,}행, {df[date_col].min().date()} ~ {df[date_col].max().date()}")
    print(f"   → 심볼: {df['symbol'].nunique()}개")

    # 모델 로드
    print(f"\n🤖 모델 로드: {ckpt_path}")
    model, feat_cols, mu, sd = load_model(ckpt_path)
    print(f"   → 피처: {len(feat_cols)}개")

    # 피처 생성
    print(f"\n⚙️  피처 생성 중...")
    df_feat, _ = make_features(df)
    print(f"   → {len(df_feat):,}행 (피처 생성 후)")

    signals = []
    processed = 0

    print(f"\n🔍 신호 생성 중...")
    for symbol, g in df_feat.groupby("symbol"):
        processed += 1
        if processed % 50 == 0:
            print(f"   진행: {processed}/{df_feat['symbol'].nunique()}")

        if len(g) < seq_len:
            continue

        # 최근 60일 데이터
        recent = g.tail(seq_len)
        X = (recent[feat_cols].to_numpy(np.float32) - mu) / sd
        X_tensor = torch.from_numpy(X).unsqueeze(0)

        # 예측
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            p_short, p_long = float(probs[0]), float(probs[1])

        # 신호 생성
        side = "LONG" if p_long > p_short else "SHORT"
        confidence = abs(p_long - p_short)

        # 낮은 신뢰도 필터링
        if confidence < min_confidence:
            continue

        # 현재가 정보
        latest_date = recent.iloc[-1][date_col]
        latest_close = float(recent.iloc[-1]["close"])

        # TP/SL 계산
        if side == "LONG":
            entry_estimate = latest_close * 1.001  # 다음 open 추정
            tp_price = entry_estimate * 1.035  # +3.5%
            sl_price = entry_estimate * 0.9825  # -1.75%
        else:
            entry_estimate = latest_close * 0.999
            tp_price = entry_estimate * 0.965  # -3.5%
            sl_price = entry_estimate * 1.0175  # +1.75%

        signals.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "side": side,
            "confidence": round(confidence, 3),
            "latest_close": round(latest_close, 2),
            "entry_estimate": round(entry_estimate, 2),
            "tp_price": round(tp_price, 2),
            "sl_price": round(sl_price, 2),
            "data_date": str(latest_date.date())
        })

    # 신뢰도 순 정렬
    signals = sorted(signals, key=lambda x: -x["confidence"])

    print(f"   → 완료: {len(signals)}개 신호 생성\n")

    return signals


# ========= 출력 및 알림 =========
def print_signals(signals, top_n=20):
    """신호 출력"""
    print("\n" + "=" * 80)
    print(f"🎯 트레이딩 신호 ({len(signals)}개 심볼)")
    print(f"⏰ 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    # 사이드별 개수
    long_count = sum(1 for s in signals if s["side"] == "LONG")
    short_count = sum(1 for s in signals if s["side"] == "SHORT")
    print(f"📊 LONG: {long_count}개 | SHORT: {short_count}개\n")

    # TOP 신호
    print(f"🔥 상위 {min(top_n, len(signals))}개 신호 (신뢰도 순):")
    print("-" * 80)

    for i, sig in enumerate(signals[:top_n], 1):
        emoji = "🟢" if sig["side"] == "LONG" else "🔴"
        print(f"{i:2d}. {emoji} {sig['symbol']:10s} | {sig['side']:5s} | "
              f"신뢰도: {sig['confidence']:.3f}")
        print(f"    현재가: ${sig['latest_close']:,.2f} → "
              f"진입: ${sig['entry_estimate']:,.2f}")
        print(f"    TP: ${sig['tp_price']:,.2f} | SL: ${sig['sl_price']:,.2f}")
        print()


def save_signals(signals, output_path="trading_signals.csv"):
    """CSV 저장"""
    df = pd.DataFrame(signals)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"💾 신호 저장: {output_path}")
    print(f"   → {len(signals)}행 저장됨\n")


def create_checklist(signal):
    """거래 전 체크리스트"""
    print(f"\n📋 거래 체크리스트 - {signal['symbol']}")
    print("=" * 60)
    print(f"✅ 1. 신호 확인: {signal['side']} (신뢰도 {signal['confidence']:.1%})")
    print(f"✅ 2. 진입가 확인: ~${signal['entry_estimate']:,.2f}")
    print(f"✅ 3. TP 설정: ${signal['tp_price']:,.2f} (+3.5%)")
    print(f"✅ 4. SL 설정: ${signal['sl_price']:,.2f} (-1.75%)")
    print(f"✅ 5. 레버리지: 10배 (자본 5% 사용)")
    print(f"✅ 6. 포지션 수: 최대 5개 유지")
    print("=" * 60)
    print("\n⚠️  실행 전 확인사항:")
    print("□ 최근 뉴스/이벤트 체크")
    print("□ 현재 시장 변동성 확인")
    print("□ 다른 포지션과 상관관계 체크")
    print("□ 자금 여유 확인")
    print("\n▶️  문제없으면 마켓 주문으로 진입\n")


# ========= 메인 =========
def main():
    import argparse

    parser = argparse.ArgumentParser(description="트레이딩 신호 생성기")
    parser.add_argument("--data", required=True, help="데이터 CSV/Parquet 파일")
    parser.add_argument("--ckpt", default="./models_daily/daily_simple_ep045.ckpt", help="모델 체크포인트")
    parser.add_argument("--top", type=int, default=20, help="상위 N개 신호 표시")
    parser.add_argument("--save", action="store_true", help="CSV로 저장")
    parser.add_argument("--checklist", type=int, help="특정 신호의 체크리스트 (순위)")
    parser.add_argument("--min-conf", type=float, default=0.05, help="최소 신뢰도")
    args = parser.parse_args()

    # 파일 존재 확인
    if not os.path.exists(args.data):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {args.data}")
        print(f"\n💡 사용 가능한 데이터 파일:")
        for f in os.listdir('.'):
            if f.endswith(('.csv', '.parquet')):
                print(f"   - {f}")
        return

    if not os.path.exists(args.ckpt):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {args.ckpt}")
        return

    try:
        # 신호 생성
        signals = generate_signals(args.data, args.ckpt, min_confidence=args.min_conf)

        if not signals:
            print("⚠️  생성된 신호가 없습니다.")
            print("   → --min-conf 값을 낮춰보세요 (현재: {:.2f})".format(args.min_conf))
            return

        # 출력
        print_signals(signals, top_n=args.top)

        # 저장
        if args.save:
            save_signals(signals)

        # 체크리스트
        if args.checklist and 1 <= args.checklist <= len(signals):
            create_checklist(signals[args.checklist - 1])

        print("\n✅ 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()