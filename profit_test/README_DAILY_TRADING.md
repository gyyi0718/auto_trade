# TCN Daily 모델 페이퍼 트레이딩 가이드

## 📋 개요

train_tcn_daily.py로 학습한 **TCN_Simple** 모델을 사용하여 페이퍼 트레이딩을 수행하는 시스템입니다.

## 🔧 주요 변경사항

### 1. 모델 호환성
- **기존**: TCN_MT (3-class: Long/Flat/Short + Time prediction)
- **변경**: TCN_Simple (2-class: Long/Short)

### 2. 데이터 타임프레임
- **기존**: 10분봉 데이터
- **변경**: 일봉(Daily) 데이터

### 3. 특징
- train_tcn_daily.py와 동일한 피처 생성 로직
- 일봉 데이터 기반 예측 (장기 트레이딩)
- 레버리지 거래 지원

---

## 🚀 사용 방법

### 1. 모델 학습

먼저 train_tcn_daily.py로 모델을 학습시킵니다:

```bash
python train_tcn_daily.py \
    --data your_daily_data.csv \
    --epochs 50 \
    --seq_len 60 \
    --horizon 1 \
    --batch 256 \
    --lr 1e-3 \
    --out_dir ./models_daily_v2
```

학습 완료 후 `./models_daily_v2/daily_simple_best.ckpt` 파일이 생성됩니다.

### 2. 페이퍼 트레이딩 실행

#### 기본 실행
```bash
python paper_trading_daily.py
```

#### 환경 변수로 설정 변경
```bash
# 심볼 설정
export SYMBOLS="BTCUSDT,ETHUSDT,SOLUSDT"

# 모델 경로
export TCN_CKPT="./models_daily_v2/daily_simple_best.ckpt"

# 초기 자본
export INITIAL_CAPITAL="10000"

# 레버리지
export LEVERAGE="10"

# 신뢰도 임계값
export CONF_THRESHOLD="0.55"

# 실행
python paper_trading_daily.py
```

---

## ⚙️ 설정 파라미터

### 모델 관련
```python
TCN_CKPT = "./models_daily_v2/daily_simple_best.ckpt"  # 모델 파일 경로
CONF_THRESHOLD = 0.55  # 신뢰도 임계값 (0.5~0.7 권장)
```

### 거래 관련
```python
INITIAL_CAPITAL = 10000       # 초기 자본 (USDT)
POSITION_SIZE_PCT = 0.1       # 포지션 크기 (10%)
LEVERAGE = 10                 # 레버리지 배율
MAX_POSITIONS = 3             # 최대 동시 포지션
STOP_LOSS_PCT = 0.02          # 손절 (2%)
TAKE_PROFIT_PCT = 0.03        # 익절 (3%)
MAX_HOLD_HOURS = 24           # 최대 보유 시간 (24시간)
LIQUIDATION_BUFFER = 0.8      # 청산 버퍼 (80%)
```

### API 관련
```python
USE_TESTNET = False           # False = 메인넷, True = 테스트넷
INTERVAL_SEC = 60             # 신호 스캔 간격 (60초)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]  # 거래 심볼
```

---

## 📊 출력 화면 예시

```
==================================================================================================================
                            🎯 페이퍼 트레이딩 시스템 (Daily TCN_Simple, 레버리지 10x)                             
==================================================================================================================

💰 계좌 현황
   초기 자본:     $   10,000.00
   현재 잔고:     $   10,450.23
   사용 가능:     $    9,450.23
   평가 손익:     $     +120.45
   총 자산:       $   10,570.68  ( +5.71%)
   실현 손익:     $     +450.23  ( +4.50%)

📍 보유 포지션 (1/3)
    심볼     |  방향    |    진입가      |    현재가      |      손익(ROE)         |    청산가      |   보유    
--------------------------------------------------------------------------------------------------------------
  BTCUSDT    | 📈 Long  | $  43,250.00   | $  43,520.00   | 🟢 $ +120.45 (+27.8%) | $  39,105.00   |    12.3h

📊 거래 통계
   총 거래:         8회
   승률:          62.5% (5승 3패)
   평균 손익:     $ +56.28
   평균 ROE:      +12.5%
   최대 수익:     $  185.30  (ROE: +41.2%)
   최대 손실:     $  -78.45  (ROE: -17.4%)
   Risk/Reward:    2.36

==================================================================================================================

🔍 신호 스캔
    심볼     |    가격      |   방향     |  신뢰도   |        확률          |       신호
----------------------------------------------------------------------------------------------------
  BTCUSDT    | $ 43,520.00  | 📈 Long    |   58.3%   |  S:0.42/L:0.58       | 🟢 매수 신호
  ETHUSDT    | $  2,245.30  | 📉 Short   |   52.1%   |  S:0.52/L:0.48       | ⚪ 신호 약함
  SOLUSDT    | $    98.45   | 📈 Long    |   61.2%   |  S:0.39/L:0.61       | 🟢 매수 신호

[스캔 #42] 2025-10-22 14:35:22
다음 스캔까지 60초... (Ctrl+C로 종료)
```

---

## 🎯 모델 차이점 정리

| 항목 | TCN_MT (기존) | TCN_Simple (신규) |
|------|---------------|-------------------|
| 출력 클래스 | 3개 (Long/Flat/Short) | 2개 (Long/Short) |
| 추가 출력 | Time to Target | 없음 |
| 데이터 | 분봉 (1분, 10분 등) | 일봉 (Daily) |
| 보유 기간 | 짧음 (분~시간) | 길음 (시간~일) |
| 거래 빈도 | 높음 | 낮음 |

---

## ⚠️ 주의사항

### 1. 일봉 데이터의 특성
- 예측이 일 단위로 업데이트됨
- 단기 변동성에 덜 민감
- 장기적인 추세 포착에 유리

### 2. 신뢰도 임계값 조정
일봉 데이터의 경우 더 높은 신뢰도가 필요할 수 있습니다:
- 보수적: 0.60 이상
- 중립적: 0.55 (기본값)
- 공격적: 0.50

### 3. 포지션 보유 시간
일봉 모델이므로 `MAX_HOLD_HOURS`를 충분히 길게 설정하세요:
- 권장: 24시간 이상
- 너무 짧으면 모델의 예측을 제대로 활용하지 못함

### 4. 레버리지 주의
일봉 데이터는 변동폭이 클 수 있으므로:
- 레버리지를 낮게 설정 (5x~10x 권장)
- 손절/익절 범위를 넓게 설정

---

## 📝 거래 기록

프로그램 종료 시 거래 내역이 `trades_daily.json`에 저장됩니다:

```json
[
  {
    "symbol": "BTCUSDT",
    "direction": "Long",
    "entry_price": 43250.0,
    "exit_price": 43520.0,
    "quantity": 2.31,
    "leverage": 10,
    "margin": 1000.0,
    "entry_time": "2025-10-22 10:15:00",
    "exit_time": "2025-10-22 22:30:00",
    "pnl": 120.45,
    "pnl_pct": 12.05,
    "roe": 27.8,
    "exit_reason": "Take Profit"
  }
]
```

---

## 🔍 트러블슈팅

### 1. "모델 로드 실패" 오류
```bash
# 모델 파일 경로 확인
ls -la ./models_daily_v2/daily_simple_best.ckpt

# 경로 수정
export TCN_CKPT="정확한/경로/daily_simple_best.ckpt"
```

### 2. "데이터 부족" 오류
- SEQ_LEN보다 충분한 과거 데이터가 필요합니다
- 기본 SEQ_LEN=60이므로 최소 60일 이상의 데이터 필요

### 3. 신호가 너무 약함
- CONF_THRESHOLD를 낮추기 (예: 0.50)
- 또는 모델 재학습

### 4. API 오류
```bash
# SSL 인증서 설치
pip install certifi requests

# 네트워크 연결 확인
curl https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT
```

---

## 📌 권장 워크플로우

1. **백테스트 먼저**: 과거 데이터로 모델 성능 검증
2. **소액으로 시작**: 초기 자본을 작게 설정
3. **파라미터 튜닝**: 손절/익절 비율 최적화
4. **로그 분석**: trades_daily.json 분석으로 전략 개선
5. **점진적 확대**: 성공적이면 자본 증액

---

## 📚 추가 정보

### 피처 정보
모델이 사용하는 피처:
- `ret1`: 일일 수익률
- `rv5, rv10, rv20, rv60`: 변동성 (5~60일)
- `mom5, mom10, mom20, mom60`: 모멘텀 (5~60일)
- `vz10, vz20, vz60`: 거래량 Z-score
- `atr14`: Average True Range (14일)

### 성능 지표
- **ROE (Return on Equity)**: 레버리지를 고려한 수익률
- **승률**: 수익 거래 / 전체 거래
- **Risk/Reward**: 평균 수익 / 평균 손실

---

## 🆘 문제 발생 시

1. 로그 파일 확인
2. 모델 파일 무결성 확인
3. API 연결 상태 확인
4. 거래 심볼이 실제로 존재하는지 확인 (Bybit Linear 선물)

---

**Happy Trading! 📈**
