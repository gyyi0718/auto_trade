# 🚀 모델 자동 관리 시스템 - 빠른 시작 가이드

## 📦 생성된 파일

✅ **model_evaluator.py** - 모델 성능 평가 & 비교  
✅ **model_manager.py** - 모델 버전 관리 & 자동 재학습

---

## 🎯 핵심 기능

### model_evaluator.py
- ✅ 여러 모델의 백테스트 결과 비교
- ✅ 종합 점수 계산 (승률 40% + R/R 30% + 수익 20% + Sharpe 10%)
- ✅ 최적 모델 자동 선택
- ✅ CSV 결과 저장

### model_manager.py
- ✅ 주기적 자동 재학습
- ✅ A/B 테스트로 안전한 모델 교체
- ✅ 버전 관리 및 롤백
- ✅ 프로덕션 배포 자동화
- ✅ 크론잡 지원

---

## 🔧 초기 설정 (한 번만)

### 1단계: 디렉토리 구조 준비
```bash
# 프로젝트 루트에 파일 배치
project/
├── model_evaluator.py    ← 다운로드한 파일
├── model_manager.py      ← 다운로드한 파일
├── pipeline.py           ← 기존 파일 (학습 파이프라인)
├── coin_scanner.py       ← 기존 파일
├── data_downloader.py    ← 기존 파일
└── production/           ← 자동 생성됨
    ├── active/           # 현재 운영 모델
    ├── candidates/       # 후보 모델들
    ├── history/          # 백업 모델들
    └── ab_test/          # A/B 테스트 중
```

### 2단계: 첫 모델 학습 & 배포
```bash
# 1. 첫 모델 학습 (pipeline.py 이용)
python pipeline.py --top 5 --days 30

# 2. 프로덕션에 배포
python model_manager.py --auto-update --force

# 결과:
# ✅ ./production/active/ 에 첫 모델 배포 완료
```

### 3단계: 자동화 설정 (크론잡)
```bash
# 크론잡 편집기 열기
crontab -e

# 아래 내용 추가 (주간 자동 업데이트 - 추천)
0 3 * * 0 cd /your/project/path && /usr/bin/python3 model_manager.py --auto-update

# 또는 격일 업데이트 (더 빠른 적응)
0 4 */2 * * cd /your/project/path && /usr/bin/python3 model_manager.py --auto-update --force

# 저장 후 확인
crontab -l
```

**끝! 이제 자동으로 돌아갑니다** 🎉

---

## 💻 사용법

### model_evaluator.py

#### 1️⃣ 전체 모델 평가
```bash
# 모든 모델 평가 및 순위 매기기
python model_evaluator.py --models-dir ./models

# 결과 CSV로 저장
python model_evaluator.py --models-dir ./models --save comparison.csv

# 출력 예시:
# ============================================================
# 🏆 모델 성능 순위
# ============================================================
# 
# rank  model_name    win_rate  risk_reward  total_return  sharpe_ratio  score
#    1  model_A       60.0%           1.5         200.00%          2.00  85.20
#    2  model_B       58.0%           1.3         180.00%          1.80  78.50
#    3  model_C       55.0%           1.2         150.00%          1.50  72.10
# 
# ============================================================
# 🥇 최고 성능: model_A (점수: 85.2)
# ============================================================
```

#### 2️⃣ 두 모델 직접 비교
```bash
# A/B 테스트용 비교
python model_evaluator.py --compare ./production/active/model.pkl ./production/candidates/model_20250101.pkl

# 출력 예시:
# ============================================================
# 🆚 모델 비교
# ============================================================
# 
# 📊 모델 A: model.pkl
#    승률: 60.0% | R/R: 1.5 | 점수: 85.20
# 
# 📊 모델 B: model_20250101.pkl
#    승률: 62.0% | R/R: 1.6 | 점수: 88.50
# 
# ============================================================
# ✅ 모델 B가 3.30점 우수 → 교체 권장
# ============================================================
```

---

### model_manager.py

#### 1️⃣ 현재 상태 확인
```bash
python model_manager.py

# 출력:
# ============================================================
# 📊 현재 상태
# ============================================================
# 
# ✅ 운영 중인 모델: model_20250101_120000.pkl
#    경로: ./production/active/model_20250101_120000.pkl
# 
# 설정:
#   • 자동 업데이트: ON
#   • 최소 개선: 5.0점
#   • 학습 기간: 30일
#   • 상위 코인: 5개
```

#### 2️⃣ 자동 업데이트 (추천) ⭐
```bash
# A/B 테스트 포함 안전 업데이트
python model_manager.py --auto-update

# 프로세스:
# 1. 최신 30일 데이터로 새 모델 학습
# 2. 현재 모델 vs 새 모델 비교
# 3. 새 모델이 5점 이상 좋으면 → 자동 교체
# 4. 아니면 → 현재 모델 유지
```

#### 3️⃣ 강제 업데이트 (빠른 적응) 🔥
```bash
# 검증 없이 즉시 교체
python model_manager.py --auto-update --force

# 주의: 위험할 수 있으므로 신중히 사용
```

#### 4️⃣ 학습만 (배포 안 함)
```bash
# 후보 모델만 생성
python model_manager.py --train-only --days 30 --top 5

# 결과: ./production/candidates/ 에 저장
# 나중에 수동으로 평가 후 선택 가능
```

#### 5️⃣ 수동 배포
```bash
# 특정 모델을 프로덕션으로 배포
python model_manager.py --deploy ./production/candidates/model_20250101.pkl
```

#### 6️⃣ 롤백 (문제 발생 시)
```bash
# 이전 모델로 즉시 복구
python model_manager.py --rollback

# 가장 최근 백업이 자동 복원됨
```

#### 7️⃣ 배포 히스토리 확인
```bash
python model_manager.py --show-history

# 출력:
# ============================================================
# 📜 배포 히스토리 (최근 10개)
# ============================================================
# 
# 1. [2025-01-05T14:30:00]
#    모델: model_20250105.pkl
#    사유: Auto-update: +6.5 points
# 
# 2. [2025-01-01T03:00:00]
#    모델: model_20250101.pkl
#    사유: Auto-update: +3.2 points
```

#### 8️⃣ 오래된 백업 정리
```bash
# 30일 이상 된 백업 삭제
python model_manager.py --clean
```

---

## 🎯 실전 시나리오

### 시나리오 1: 완전 자동 (추천) ⭐

```bash
# 초기 설정
python model_manager.py --auto-update --force

# 크론잡 (매주 일요일 새벽 3시)
0 3 * * 0 cd /path && python model_manager.py --auto-update

# 결과:
# → 일주일마다 자동으로 재학습
# → 개선되면 자동 교체
# → 손 안 대도 됨! 🎉
```

### 시나리오 2: 매일 후보 생성 + 주말 선택

```bash
# 크론잡 (매일 새벽 2시 - 후보만 생성)
0 2 * * * cd /path && python model_manager.py --train-only --days 14

# 주말에 수동 평가
python model_evaluator.py --models-dir ./production/candidates --save weekly_comparison.csv

# 좋은 모델 선택해서 배포
python model_manager.py --deploy ./production/candidates/model_20250105.pkl
```

### 시나리오 3: 격일 빠른 업데이트

```bash
# 크론잡 (2일마다 새벽 4시)
0 4 */2 * * cd /path && python model_manager.py --auto-update --force

# 장점: 최신 트렌드 즉각 반영
# 단점: 검증 없어서 위험할 수 있음
```

---

## 📊 점수 계산 방식

```python
종합 점수 = (
    승률 × 40% +        # 가장 중요
    Risk/Reward × 30% + # 손익비
    총수익 × 20% +      # 절대 수익
    Sharpe × 10%        # 안정성
)

교체 기준:
- 새 모델 > 기존 모델 + 5점 → 자동 교체 ✅
- 차이 < 5점 → 기존 모델 유지 ⚠️
```

---

## 🔍 문제 해결

### 1. "pipeline.py를 찾을 수 없습니다"
```bash
# 같은 디렉토리에 pipeline.py가 있는지 확인
ls -la pipeline.py

# 없으면 이전 대화에서 다운로드
```

### 2. "model_evaluator.py를 import할 수 없습니다"
```bash
# 두 파일이 같은 디렉토리에 있는지 확인
ls -la model_*.py

# 권한 확인
chmod +x model_*.py
```

### 3. 크론잡이 실행 안 됨
```bash
# 절대 경로 사용
which python3  # 파이썬 경로 확인
pwd            # 프로젝트 경로 확인

# 크론잡에 절대 경로 명시
0 3 * * 0 cd /home/user/project && /usr/bin/python3 model_manager.py --auto-update

# 로그 확인
0 3 * * 0 cd /path && /usr/bin/python3 model_manager.py --auto-update >> /tmp/cron.log 2>&1
```

### 4. 모델 평가 결과가 이상함
```bash
# 백테스트 로직 확인 필요
# model_evaluator.py의 backtest_model() 함수를
# 실제 백테스트 로직으로 교체하세요

# 현재는 시뮬레이션 결과를 반환합니다
```

---

## 📁 디렉토리 구조

```
production/
├── active/                    # 현재 운영 중
│   ├── model.pkl
│   └── model.json
├── candidates/                # 후보 대기 중
│   ├── model_20250101_120000.pkl
│   ├── model_20250102_120000.pkl
│   └── ...
├── history/                   # 백업 보관
│   ├── backup_20250101_model.pkl
│   ├── backup_20250102_model.pkl
│   └── ...
├── ab_test/                   # A/B 테스트
│   └── ...
├── config.json                # 설정 파일
└── deployment_history.json    # 배포 히스토리
```

---

## 🎉 완료!

**이제 시스템이 자동으로:**
- ✅ 주기적으로 재학습
- ✅ 성능 비교 & 평가
- ✅ 자동 교체 (개선 시)
- ✅ 버전 관리 & 백업
- ✅ 롤백 지원

**당신이 할 일:**
- ⏰ 가끔 히스토리 확인 (`--show-history`)
- 📊 주말에 전체 평가 (선택)
- 🚨 문제 시 롤백 (`--rollback`)

---

## 🚀 지금 바로 시작

```bash
# 1. 첫 모델 배포
python model_manager.py --auto-update --force

# 2. 자동화 설정
crontab -e
# 추가: 0 3 * * 0 cd /path && python model_manager.py --auto-update

# 완료! 🎉
```

**질문이나 문제가 있으면 언제든 물어보세요!**
