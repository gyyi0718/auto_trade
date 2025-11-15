# PatchTST 라이브 트레이딩 시스템 사용 가이드

## 📋 개요

이 시스템은 PatchTST 모델을 사용하여 Bybit 거래소에서 실제로 선물 거래를 수행하는 자동화 트레이딩 봇입니다.

**⚠️ 중요 경고: 실제 자금을 사용하므로 매우 신중하게 사용해야 합니다!**

## 🔑 주요 변경 사항

### Paper Trading → Live Trading 전환

1. **실제 API 호출**
   - 페이퍼 트레이딩의 가상 계좌 시스템 제거
   - Bybit API를 통한 실제 주문 실행

2. **인증 추가**
   - API Key와 API Secret 필요
   - HMAC SHA256 서명 방식 사용

3. **실제 포지션 관리**
   - Bybit에서 실제 포지션 조회
   - TP/SL 자동 설정

## 🚀 사용 방법

### 1. 환경 변수 설정

#### Windows PowerShell:
```powershell
# Testnet (테스트용 - 권장)
$env:BYBIT_API_KEY='your_testnet_api_key'
$env:BYBIT_API_SECRET='your_testnet_api_secret'
$env:USE_TESTNET='1'

# Mainnet (실전 - 주의!)
$env:BYBIT_API_KEY='your_mainnet_api_key'
$env:BYBIT_API_SECRET='your_mainnet_api_secret'
$env:USE_TESTNET='0'

# 거래 설정
$env:SYMBOLS='HUSDT'  # 거래할 심볼 (쉼표로 구분)
$env:LEVERAGE='20'  # 레버리지 (낮을수록 안전)
$env:MARGIN_PER_POSITION='10'  # 포지션당 증거금 (USDT)
$env:CONF_THRESHOLD='0.6'  # 신호 신뢰도 임계값
```

#### Linux/Mac:
```bash
# Testnet
export BYBIT_API_KEY='your_testnet_api_key'
export BYBIT_API_SECRET='your_testnet_api_secret'
export USE_TESTNET=1

# 거래 설정
export LEVERAGE=20
export MARGIN_PER_POSITION=10
```

### 2. API Key 발급

#### Testnet (테스트용 - 강력 권장):
1. https://testnet.bybit.com 접속
2. 회원가입/로그인
3. 우측 상단 프로필 → API Management
4. API Key 생성
   - 필수 권한: Contract Trading, Position
5. 테스트 자금 받기: Assets → Deposit

#### Mainnet (실전):
1. https://www.bybit.com 접속
2. 우측 상단 프로필 → API Management
3. API Key 생성
   - 필수 권한: Contract Trading, Position
   - **IP 화이트리스트 설정 권장**

**⚠️ API Key는 절대 공유하지 마세요!**

### 3. 모델 경로 설정

`live_trading_patchtst.py` 파일의 `MODEL_PATHS` 수정:

```python
MODEL_PATHS = {
    "HUSDT": "./models_patchtst_improved/patchtst_best.ckpt",
    # 필요한 심볼 추가
}
```

### 4. 실행

```bash
python live_trading_patchtst.py
```

## ⚙️ 주요 설정

### 리스크 관리

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `LEVERAGE` | 20 | 레버리지 배수 (낮을수록 안전) |
| `MARGIN_PER_POSITION` | 10 | 포지션당 증거금 (USDT) |
| `MAX_POSITIONS` | 5 | 최대 동시 포지션 수 |
| `STOP_LOSS_PCT` | 0.02 | 손절 비율 (2%) |
| `TAKE_PROFIT_PCT` | 0.03 | 익절 비율 (3%) |
| `MAX_DAILY_LOSS` | 100 | 일일 최대 손실 (USDT) |
| `MAX_HOLD_MINUTES` | 30 | 최대 보유 시간 (분) |

### 신호 설정

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `CONF_THRESHOLD` | 0.6 | 신호 신뢰도 임계값 |
| `INTERVAL_SEC` | 10 | 스캔 주기 (초) |

## 📊 시스템 동작

### 1. 초기화
- PatchTST 모델 로딩
- Bybit API 연결
- 레버리지 설정

### 2. 신호 스캔 (10초마다)
- 각 심볼별 예측 수행
- 신뢰도가 임계값 이상인 신호만 거래

### 3. 포지션 관리
- **진입**: 신호가 강하고 포지션이 없을 때
- **청산**:
  - TP/SL 도달
  - 반대 신호 발생
  - 최대 보유 시간 초과
  - 청산가 근접

### 4. 로그 기록
- `live_trades_patchtst.json`: 거래 내역
- `orders_patchtst.json`: 주문 내역

## 🔒 안전 장치

1. **일일 손실 제한**
   - 설정한 금액 이상 손실 시 거래 중단

2. **포지션 수 제한**
   - 동시에 최대 5개까지만 포지션 보유

3. **자동 TP/SL**
   - 모든 포지션에 자동으로 손절/익절가 설정

4. **청산가 모니터링**
   - 청산 위험 시 자동 청산

## 📈 대시보드 예시

```
==================================================
       🎯 PatchTST 라이브 트레이딩 - 레버리지 20x
==================================================

💰 계좌 현황
   현금 잔고:           $1,000.00
   포지션 증거금:         $200.00
   평가 손익:     🟢      $+50.00
   -----------------------------------------
   총 자산:             $1,250.00

📍 보유 포지션 (2/5)
   심볼    |  방향  |   진입가    |   현재가    | 손익(ROE)
------------------------------------------------------------------------
   HUSDT   | 📈 Long | $   0.2500 | $   0.2600 | 🟢 $+20.00 (+20.0%)
   BTCUSDT | 📉 Short| $70,000.00 | $69,500.00 | 🟢 $+30.00 (+30.0%)

🔍 신호 스캔 (2개 심볼)
   심볼    |    가격    |   방향   | 신뢰도 |       신호
------------------------------------------------------------------------------
   HUSDT   | $  0.2600  | 📈 Long  |  75.5% | 🟢 매수 신호
   BTCUSDT | $69,500.00 | 📉 Short |  80.2% | 🔴 매도 신호
```

## ⚠️ 주의사항

### 🔴 필수 확인 사항

1. **반드시 Testnet에서 먼저 테스트**
   - 실제 자금 사용 전 충분히 테스트

2. **API Key 보안**
   - API Key는 환경변수로만 관리
   - 절대 코드에 하드코딩 금지
   - Git에 업로드 금지

3. **레버리지 주의**
   - 높은 레버리지는 큰 손실 위험
   - 초보자는 10배 이하 권장

4. **시장 변동성**
   - 급격한 시장 변동 시 큰 손실 가능
   - 항상 모니터링 필요

5. **포지션 모드 확인**
   - Bybit 웹사이트 설정과 일치해야 함
   - 기본값: One-Way Mode (단방향)

### 🟡 권장 사항

1. **소액으로 시작**
   - 처음에는 최소 금액으로 테스트

2. **점진적 증가**
   - 성과가 검증된 후 서서히 증액

3. **정기적 모니터링**
   - 하루 2-3회 이상 확인

4. **손실 관리**
   - 일일/주간 손실 한도 설정
   - 연속 손실 시 거래 중단

## 🐛 문제 해결

### API 인증 실패
```
ERROR: API 인증 실패
```
→ API Key와 Secret 확인, Testnet/Mainnet 구분 확인

### 주문 실패
```
주문 실패: Insufficient balance
```
→ 계좌 잔고 확인

### 모델 로딩 실패
```
모델 파일 없음
```
→ MODEL_PATHS 경로 확인

### 포지션이 자동으로 청산됨
```
포지션이 사라졌습니다!
```
→ Bybit가 TP/SL을 자동 실행했을 가능성. 웹사이트에서 확인

## 📞 지원

- Bybit 고객센터: https://www.bybit.com/help-center
- Bybit API 문서: https://bybit-exchange.github.io/docs/

## 📄 라이선스

이 코드는 교육 및 연구 목적으로 제공됩니다.
실제 거래로 인한 손실에 대해 개발자는 책임지지 않습니다.

**투자는 본인의 책임입니다. 신중하게 결정하세요!**
