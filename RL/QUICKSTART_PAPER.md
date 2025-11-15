# ⚡ Paper Trading 빠른 시작

## 🎯 1분 안에 시작하기

### Step 1: 파일 준비

필요한 파일들:
```
paper_trading.py             ← Paper trading 스크립트
rl_trading_env_final.py      ← 업로드한 환경 파일
rl_models_standalone/        ← 학습된 모델 폴더
```

### Step 2: 즉시 실행

```bash
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_final.zip
```

---

## 📊 예상 결과

```
📊 백테스트 (최근 데이터)
================================================================================

✅ 2000개 캔들 로드
   기간: 2024-01-01 ~ 2024-01-08

🚀 백테스트 실행 중...

성과:
   초기 자산: $10,000.00
   최종 자산: $10,523.45
   총 수익률: +5.23%
   총 손익: $523.45

거래:
   총 거래: 42회
   승률: 61.9%
   최대 낙폭: 8.34%

✅ 백테스트 완료!
```

---

## 🚀 추가 테스트

### 다른 코인

```bash
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_final.zip --symbol ETHUSDT
```

### 더 많은 자본

```bash
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_final.zip --balance 50000
```

---

## 💡 핵심

- **백테스트**: 즉시 결과 확인 (추천)
- **Paper Trading**: 실시간 시뮬레이션 (시간 필요)

---

**지금 바로 실행하세요! 🎉**

```bash
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_final.zip
```
