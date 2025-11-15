# 🚀 PatchTST 라이브 트레이딩 빠른 시작

## ⚡ 5분 안에 시작하기

### Step 1: Testnet API Key 발급 (2분)

1. **https://testnet.bybit.com** 접속 및 가입
2. 우측 상단 **프로필 아이콘** → **API Management**
3. **Create New Key** 클릭
   - Name: `MyTradingBot`
   - Permissions: **Contract Trading** 체크
   - Submit 클릭
4. **API Key**와 **API Secret** 복사 (안전하게 보관!)
5. **Assets** → **Deposit** → 테스트 자금 받기 (무료)

### Step 2: 환경 변수 설정 (1분)

**Windows PowerShell:**
```powershell
$env:BYBIT_API_KEY='여기에_복사한_API_KEY_입력'
$env:BYBIT_API_SECRET='여기에_복사한_API_SECRET_입력'
$env:USE_TESTNET='1'
```

**Linux/Mac Terminal:**
```bash
export BYBIT_API_KEY='여기에_복사한_API_KEY_입력'
export BYBIT_API_SECRET='여기에_복사한_API_SECRET_입력'
export USE_TESTNET=1
```

### Step 3: 실행 (1분)

```bash
python live_trading_patchtst.py
```

**완료!** 🎉 시스템이 자동으로 거래를 시작합니다.

---

## 🎮 기본 사용법

### 프로그램 종료
- **Ctrl + C** 누르기

### 실시간 모니터링
- 터미널 화면에서 실시간 포지션 및 손익 확인
- 10초마다 자동으로 업데이트

### 거래 내역 확인
- `live_trades_patchtst.json`: 완료된 거래 내역
- `orders_patchtst.json`: 모든 주문 내역

---

## ⚙️ 주요 설정 변경

### 레버리지 변경 (기본: 20배)
```powershell
$env:LEVERAGE='10'  # 10배로 낮춤 (더 안전)
```

### 포지션당 증거금 변경 (기본: $10)
```powershell
$env:MARGIN_PER_POSITION='5'  # $5로 줄임
```

### 신호 신뢰도 변경 (기본: 60%)
```powershell
$env:CONF_THRESHOLD='0.7'  # 70%로 높임 (더 신중)
```

### 거래 심볼 추가
```powershell
$env:SYMBOLS='HUSDT,BTCUSDT,ETHUSDT'  # 여러 심볼 거래
```

---

## 📊 대시보드 읽는 법

```
💰 계좌 현황
   현금 잔고:           $1,000.00  ← 사용 가능한 현금
   포지션 증거금:         $200.00  ← 포지션에 묶인 증거금
   평가 손익:     🟢      $+50.00  ← 현재 미실현 손익
   총 자산:             $1,250.00  ← 현금 + 포지션 평가액

📍 보유 포지션
   HUSDT   | 📈 Long  | 진입: $0.25 | 현재: $0.26 | 🟢 $+20 (+20%)
            ↑           ↑               ↑            ↑        ↑
         심볼      매수/매도        진입가격      현재가격    손익  ROE
```

**ROE (Return on Equity)**: 증거금 대비 수익률
- 레버리지 20배 → 가격 1% 상승 시 ROE 20%

---

## ⚠️ 초보자 주의사항

### ✅ 해야 할 것

1. **항상 Testnet에서 먼저 테스트**
   - 실제 돈 쓰기 전에 충분히 연습

2. **낮은 레버리지로 시작**
   - 5배~10배 권장 (기본 20배는 위험)

3. **소액으로 시작**
   - 포지션당 $5~$10로 시작

4. **정기적으로 확인**
   - 하루 2~3회 확인 필수

5. **손실 한도 설정**
   ```powershell
   $env:MAX_DAILY_LOSS='50'  # 하루 최대 $50 손실
   ```

### ❌ 하지 말아야 할 것

1. **높은 레버리지 사용** (50배 이상)
   - 순식간에 청산될 수 있음

2. **전 재산 투자**
   - 잃어도 괜찮은 금액만

3. **방치**
   - 시장 급변 시 큰 손실 가능

4. **API Key 공유**
   - 절대 다른 사람에게 알려주면 안됨

5. **감정적 거래**
   - 연속 손실 시 설정 변경 자제

---

## 🆘 자주 묻는 질문

### Q: Testnet과 Mainnet의 차이는?
**A:** 
- **Testnet**: 가짜 돈으로 연습 (무료, 안전)
- **Mainnet**: 실제 돈으로 거래 (위험!)

### Q: 얼마나 벌 수 있나요?
**A:** 수익을 보장할 수 없습니다. 손실도 발생할 수 있으니 주의하세요.

### Q: 24시간 실행해야 하나요?
**A:** 원하는 시간만 실행하면 됩니다. 종료 시 Ctrl+C로 안전하게 종료하세요.

### Q: 여러 컴퓨터에서 동시 실행 가능?
**A:** 가능하지만 같은 심볼은 중복되지 않게 설정하세요.

### Q: 인터넷이 끊기면?
**A:** 프로그램이 멈춥니다. 포지션은 Bybit 웹사이트에서 수동으로 관리해야 합니다.

---

## 📱 Bybit 모바일 앱

- **iOS**: App Store에서 "Bybit" 검색
- **Android**: Play Store에서 "Bybit" 검색

앱으로 실시간 포지션 확인 및 수동 청산 가능!

---

## 🎓 추가 학습 자료

1. **레버리지 이해하기**
   - 낮을수록 안전하지만 수익도 적음
   - 높을수록 위험하지만 수익도 큼

2. **손절/익절의 중요성**
   - 항상 자동으로 설정됨
   - 큰 손실 방지

3. **포지션 크기 관리**
   - 한 번에 전체 자금의 10% 이하 권장

---

## 💡 성공 팁

1. **작게 시작하기**
   - 처음 1주일은 최소 금액으로

2. **기록하기**
   - 어떤 설정에서 잘 됐는지 메모

3. **감정 배제**
   - 손실 후 바로 설정 바꾸지 말기

4. **지속적 학습**
   - 시장 상황 공부

5. **백업 계획**
   - 시스템 오류 대비 수동 청산 방법 숙지

---

## 🔄 Mainnet 전환 (신중히!)

테스트에서 충분히 만족스러운 결과가 나왔다면:

```powershell
# 1. Mainnet API Key 새로 발급
# 2. 환경변수 변경
$env:BYBIT_API_KEY='mainnet_api_key'
$env:BYBIT_API_SECRET='mainnet_api_secret'
$env:USE_TESTNET='0'  # Mainnet으로 변경

# 3. 낮은 레버리지로 시작
$env:LEVERAGE='5'
$env:MARGIN_PER_POSITION='5'
```

**⚠️ 주의**: 실제 돈을 잃을 수 있습니다!

---

## 📞 도움이 필요하면

1. **README_LIVE_TRADING.md** 참고
2. Bybit 고객센터 문의
3. 커뮤니티 포럼 검색

---

**행운을 빕니다! 🍀**

**면책조항**: 이 시스템은 교육 목적으로 제공되며, 수익을 보장하지 않습니다.
투자 손실에 대한 책임은 전적으로 사용자에게 있습니다.
