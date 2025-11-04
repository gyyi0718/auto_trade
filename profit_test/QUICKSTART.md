# 🚀 빠른 시작 가이드

## 1단계: 모델 학습

```bash
# 일일 데이터로 모델 학습
python train_tcn_daily.py \
    --data daily_data.csv \
    --epochs 50 \
    --seq_len 60 \
    --horizon 1 \
    --batch 256 \
    --lr 1e-3 \
    --out_dir ./models_daily_v2
```

**데이터 포맷 (daily_data.csv)**:
```
date,symbol,open,high,low,close,volume
2024-01-01,BTCUSDT,42000,43000,41500,42500,1000000
2024-01-02,BTCUSDT,42500,43500,42000,43000,1100000
...
```

## 2단계: 페이퍼 트레이딩 실행

### 방법 1: 직접 실행
```bash
python paper_trading_daily.py
```

### 방법 2: 스크립트 사용
```bash
chmod +x run_daily_trading.sh
./run_daily_trading.sh
```

### 방법 3: 커스텀 설정
```bash
# 심볼과 초기 자본 변경
export SYMBOLS="BTCUSDT,ETHUSDT"
export INITIAL_CAPITAL="5000"
export LEVERAGE="5"

python paper_trading_daily.py
```

## 3단계: 결과 확인

프로그램 실행 중에는 실시간 대시보드가 표시됩니다.
종료 시 (Ctrl+C) `trades_daily.json` 파일에 거래 내역이 저장됩니다.

```bash
# 거래 내역 확인
cat trades_daily.json | python -m json.tool
```

## 주요 설정 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| SYMBOLS | BTCUSDT,ETHUSDT,... | 거래할 심볼 (쉼표로 구분) |
| TCN_CKPT | ./models_daily_v2/daily_simple_best.ckpt | 모델 파일 경로 |
| INITIAL_CAPITAL | 10000 | 초기 자본 (USDT) |
| LEVERAGE | 10 | 레버리지 배율 |
| CONF_THRESHOLD | 0.55 | 신호 신뢰도 임계값 |
| POSITION_SIZE_PCT | 0.1 | 포지션 크기 (10%) |
| MAX_POSITIONS | 3 | 최대 동시 포지션 |
| STOP_LOSS_PCT | 0.02 | 손절 비율 (2%) |
| TAKE_PROFIT_PCT | 0.03 | 익절 비율 (3%) |
| MAX_HOLD_HOURS | 24 | 최대 보유 시간 |
| INTERVAL_SEC | 60 | 스캔 간격 (초) |
| USE_TESTNET | 0 | 테스트넷 사용 (0=메인넷, 1=테스트넷) |

## 예제 시나리오

### 시나리오 1: 보수적 전략
```bash
export LEVERAGE="5"
export CONF_THRESHOLD="0.60"
export POSITION_SIZE_PCT="0.05"
export STOP_LOSS_PCT="0.015"
export TAKE_PROFIT_PCT="0.025"

python paper_trading_daily.py
```

### 시나리오 2: 공격적 전략
```bash
export LEVERAGE="15"
export CONF_THRESHOLD="0.50"
export POSITION_SIZE_PCT="0.15"
export STOP_LOSS_PCT="0.03"
export TAKE_PROFIT_PCT="0.05"

python paper_trading_daily.py
```

### 시나리오 3: 장기 보유
```bash
export MAX_HOLD_HOURS="72"
export STOP_LOSS_PCT="0.05"
export TAKE_PROFIT_PCT="0.10"

python paper_trading_daily.py
```

## 필수 의존성 설치

```bash
pip install torch numpy pandas requests certifi tqdm
```

## 문제 해결

### 모델 파일이 없는 경우
```bash
# 모델 파일 위치 확인
ls -la ./models_daily_v2/

# 없다면 먼저 학습 실행
python train_tcn_daily.py --data your_data.csv
```

### API 연결 오류
```bash
# 인증서 재설치
pip install --upgrade certifi requests

# 연결 테스트
curl https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT
```

### 메모리 부족
- SYMBOLS 개수를 줄이세요 (3~5개 권장)
- MAX_POSITIONS를 낮추세요

## 팁

1. **백테스트 먼저**: 먼저 과거 데이터로 모델 성능을 확인하세요
2. **소액 시작**: 처음에는 INITIAL_CAPITAL을 작게 설정
3. **로그 분석**: trades_daily.json을 주기적으로 분석
4. **파라미터 최적화**: 손절/익절 비율을 점진적으로 조정
5. **모니터링**: 실행 중 대시보드를 주의깊게 관찰

## 다음 단계

1. ✅ 모델 학습 완료
2. ✅ 페이퍼 트레이딩 시작
3. ⏳ 거래 내역 분석
4. ⏳ 파라미터 최적화
5. ⏳ 실전 투자 검토

---

**주의**: 페이퍼 트레이딩은 실제 자금이 아닌 가상 자금으로 시뮬레이션입니다. 
실전 투자 전에 충분한 테스트와 검증이 필요합니다.
