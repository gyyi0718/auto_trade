# 📊 Paper Trading 가이드

## 🎯 두 가지 모드

### 1. **백테스트 모드** (추천 - 빠름)
최근 2000개 캔들로 즉시 테스트

```bash
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_final.zip
```

**예상 결과:**
```
📊 백테스트 (최근 데이터)
================================================================================

🔄 모델 로드: rl_models_standalone/BTCUSDT_5min_final.zip
📥 BTCUSDT 5분봉 최신 데이터 다운로드 중...
✅ 2000개 캔들 로드
   기간: 2024-01-01 00:00:00 ~ 2024-01-08 12:00:00

🚀 백테스트 실행 중...

================================================================================
                            📊 백테스트 결과                            
================================================================================

성과:
   초기 자산: $10,000.00
   최종 자산: $10,523.45
   총 수익률: +5.23%
   총 손익: $523.45

거래:
   총 거래: 42회
   승률: 61.9%
   최대 낙폭: 8.34%
   샤프 비율: 1.23

행동 분포:
   LONG:  750 (42.3%)
   SHORT: 920 (51.9%)
   CLOSE: 103 (5.8%)

최근 거래 (최근 10개):
   #33: LONG  | Entry: 50123.45 | Exit: 51234.56 | PNL:   +87.30 ( +0.87%) | 보유:  15스텝 | Take Profit
   #34: SHORT | Entry: 51234.56 | Exit: 50987.23 | PNL:   +45.20 ( +0.45%) | 보유:   8스텝 | Manual
   ...
```

### 2. **실시간 Paper Trading** (시간 필요)
실제 시간 흐름대로 거래 시뮬레이션

```bash
# 1시간 동안 5분마다 업데이트
python paper_trading.py --mode paper --model rl_models_standalone/BTCUSDT_5min_final.zip --duration 60 --update 5
```

**예상 출력:**
```
📊 Paper Trading 시작
================================================================================

설정:
   모델: rl_models_standalone/BTCUSDT_5min_final.zip
   심볼: BTCUSDT
   간격: 5분
   초기 자본: $10,000
   실행 시간: 60분
   업데이트: 매 5분

🔄 모델 로드 중...
✅ 모델 로드 완료

📥 초기 데이터 다운로드 중...
✅ 500개 캔들 로드

================================================================================
                              🚀 거래 시작!                              
================================================================================

🎲 초기 포지션: SHORT @ 50123.45

[14:25:30] #1
  액션: SHORT
  포지션: short
  자산: $10,045.23
  총 거래: 1회
  승률: 100.0%
  손익: $45.23 (+0.45%)

[14:30:35] #2
  액션: LONG
  포지션: long
  자산: $10,087.56
  총 거래: 2회
  승률: 100.0%
  손익: $87.56 (+0.88%)

...

(Ctrl+C로 중단 가능)

================================================================================
                              📊 최종 결과                              
================================================================================

실행 시간: 60.3분
총 반복: 12회

성과:
   초기 자산: $10,000.00
   최종 자산: $10,234.56
   총 수익률: +2.35%
   총 손익: $234.56

거래:
   총 거래: 8회
   승률: 62.5%
```

---

## 🚀 빠른 시작

### Step 1: 백테스트 (즉시)

```bash
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_final.zip
```

### Step 2: 다른 코인 테스트

```bash
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_final.zip --symbol ETHUSDT
```

### Step 3: 실시간 시뮬레이션 (선택)

```bash
# 30분 동안 테스트
python paper_trading.py --mode paper --model rl_models_standalone/BTCUSDT_5min_final.zip --duration 30
```

---

## ⚙️ 옵션 설명

### 기본 옵션

```bash
--mode backtest           # backtest 또는 paper
--model [경로]            # 모델 파일 경로 (필수!)
--symbol BTCUSDT         # 거래 심볼
--interval 5             # 시간 간격 (분)
--balance 10000          # 초기 자본
```

### Paper Trading 옵션

```bash
--duration 60            # 실행 시간 (분)
--update 5               # 업데이트 간격 (분)
```

---

## 📊 예시 명령어

### 1. 기본 백테스트

```bash
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_final.zip
```

### 2. ETHUSDT 백테스트

```bash
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_final.zip --symbol ETHUSDT
```

### 3. 초기 자본 변경

```bash
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_final.zip --balance 50000
```

### 4. 실시간 Paper Trading (2시간)

```bash
python paper_trading.py --mode paper --model rl_models_standalone/BTCUSDT_5min_final.zip --duration 120 --update 5
```

### 5. 빠른 Paper Trading (10분마다 업데이트)

```bash
python paper_trading.py --mode paper --model rl_models_standalone/BTCUSDT_5min_final.zip --duration 60 --update 10
```

---

## ⚠️ 주의사항

### 백테스트 vs Paper Trading

| 항목 | 백테스트 | Paper Trading |
|------|---------|--------------|
| 속도 | 즉시 | 실시간 대기 |
| 용도 | 빠른 평가 | 실시간 시뮬레이션 |
| 추천 | ✅ 먼저 실행 | 옵션 |

### 모델 파일 경로

**학습 후 생성되는 모델:**
```
rl_models_standalone/BTCUSDT_5min_final.zip          ← 최종 모델
rl_models_standalone/BTCUSDT_5min_best/best_model   ← 최고 성능 모델
```

**사용 예시:**
```bash
# 최종 모델
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_final.zip

# 최고 성능 모델 (확장자 없이)
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_best/best_model
```

---

## 🎯 추천 워크플로우

### 1단계: 백테스트로 빠른 확인

```bash
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_final.zip
```

**확인할 것:**
- 총 거래 횟수 (10회 이상?)
- 승률 (50% 이상?)
- 수익률 (양수?)

### 2단계: 다른 코인으로 테스트

```bash
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_final.zip --symbol ETHUSDT
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_final.zip --symbol SOLUSDT
```

### 3단계: 실시간 시뮬레이션 (선택)

```bash
# 30분~1시간 정도 실행해보기
python paper_trading.py --mode paper --model rl_models_standalone/BTCUSDT_5min_final.zip --duration 30
```

---

## 💡 Tips

### 1. Ctrl+C로 언제든 중단 가능

Paper Trading 중 `Ctrl+C`를 누르면 안전하게 중단하고 결과를 표시합니다.

### 2. 백테스트가 더 실용적

실시간 Paper Trading은 시간이 오래 걸리므로, 대부분의 경우 백테스트로 충분합니다.

### 3. 여러 모델 비교

```bash
# 최종 모델
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_final.zip

# Best 모델
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_best/best_model
```

---

## 🐛 문제 해결

### Q: "No module named 'rl_trading_env_final'"

**A:** paper_trading.py와 rl_trading_env_final.py를 같은 폴더에 두세요.

```bash
# 확인
ls rl_trading_env_final.py
ls paper_trading.py
```

### Q: API 오류

**A:** 인터넷 연결 확인 또는 잠시 후 재시도

### Q: 거래가 없어요

**A:** 정상입니다. 최근 데이터가 학습 데이터와 다를 수 있습니다.

---

## 📊 결과 해석

### 좋은 결과 ✅
- 총 거래: 10~50회
- 승률: 55% 이상
- 수익률: +3% 이상
- 최대 낙폭: 20% 이하

### 주의 필요 ⚠️
- 총 거래: 5회 미만 또는 100회 이상
- 승률: 45% 이하
- 최대 낙폭: 50% 이상

---

**지금 바로 테스트해보세요! 🚀**

```bash
python paper_trading.py --mode backtest --model rl_models_standalone/BTCUSDT_5min_final.zip
```
