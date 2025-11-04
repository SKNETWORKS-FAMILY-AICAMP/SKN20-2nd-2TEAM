# 🏋️ Gym Churn Prediction - 헬스장 회원 이탈 예측 프로젝트

## 1. 팀 소개 🧑‍🤝‍🧑

   - 팀명 : 2팀
   - 팀원

| | | | | |
|---|---|---|---|---|
| <img src="project\src\images\뚱이.jpg" width="120" height="120" style="object-fit: contain; background-color: black;"> <br> [박찬](https://github.com/djdjdjdfh1) | <img src="project\src\images\다람이.jpg" width="120" height="120" style="object-fit: contain; background-color: black;"> <br> [최소영](https://github.com/sy-choi25) | <img src="project\src\images\래리.jpg" width="120" height="120" style="object-fit: contain; background-color: black;"> <br> [나호성](https://github.com/BBuSang) | <img src="project\src\images\스폰지밥.jpg" width="120" height="120" style="object-fit: contain; background-color: black;"> <br> [권규리](https://github.com/gyur1eek) | <img src="project\src\images\징징이2.jpg" width="120" height="120" style="object-fit: contain; background-color: black;"> <br> [박준석](https://github.com/Ipodizar) |
---

## 📋 프로젝트 개요

헬스장 회원의 이탈(Churn)을 예측하는 머신러닝/딥러닝 프로젝트입니다. 다양한 회원 정보를 바탕으로 이탈 가능성을 사전에 예측하여, 효과적인 리텐션(Retention) 전략을 수립할 수 있도록 지원합니다.

### 🎯 프로젝트 목표
- 이탈 위험 고객 조기 식별 시스템 개발
- 데이터 기반 비즈니스 인사이트 도출

### 📊 데이터셋
- **파일명**: gym_churn_us.csv
- **샘플 수**: 4,002개
- **특성 수**: 13개 (원본) + 11개 (파생) = 24개
- **타겟**: Churn (0: 유지, 1: 이탈)
- **이탈률**: 약 30%

---

## 📚 기술 스택

### 데이터 분석
- **Pandas**: 데이터 처리 및 전처리
- **NumPy**: 수치 연산
- **Matplotlib/Seaborn**: 데이터 시각화

### 머신러닝
- **scikit-learn**: 전통적 ML 알고리즘, 전처리, 평가
- **XGBoost**: 그래디언트 부스팅
- **LightGBM**: 경량화 부스팅
- **imbalanced-learn**: SMOTE (클래스 불균형 해결)

### 딥러닝
- **TensorFlow/Keras**: 신경망 구축 및 학습

### 최적화
- **RandomizedSearchCV**: 하이퍼파라미터 자동 탐색
- **StackingClassifier**: 앙상블 학습

## 🗂️ 프로젝트 구조

```
SKN20-2ed/
├── project/
│   ├── config/                          # 설정 파일
│   ├── data/
│   │   ├── raw/                         # 원본 데이터
│   │   │   └── gym_churn_us.csv
│   │   └── processed/                   # 전처리된 데이터
│   ├── models/
│   │   └── 2024_churn_model/           # 학습된 모델 저장
│   │       ├── stacking_ultimate.pkl   # 최종 앙상블 모델
│   │       ├── scaler_enh.pkl          # 스케일러
│   │       ├── nn_model.h5             # 딥러닝 모델
│   │       └── best_threshold.txt      # 최적 임계값
│   ├── notebooks/
│   │   ├── EDA.ipynb                   # 📊 탐색적 데이터 분석
│   │   ├── Model_Training.ipynb        # 🤖 모델 학습 및 튜닝
│   │   ├── Model_Evaluation.ipynb      # 📈 모델 평가 및 분석
│   │   └── index2.ipynb                # 통합 작업 노트북
│   ├── output/
│   │   ├── predictions/                # 예측 결과
│   │   ├── reports/                    # 분석 보고서
│   │   │   └── final_evaluation_report.txt
│   │   └── visualizations/             # 시각화 결과
│   │       ├── confusion_matrices.png
│   │       ├── roc_pr_curves.png
│   │       ├── improvement_progress.png
│   │       └── feature_importance.png
│   └── src/                            # 소스 코드
└── README.md
```

---

## 🚀 주요 기능

### 1️⃣ 탐색적 데이터 분석 (EDA)
- 데이터 기본 정보 및 통계 분석
- 타겟 변수(Churn) 분포 분석
- 범주형/수치형 변수 상관관계 분석
- 주요 특성 심층 분석 (Lifetime, Contract_period 등)
- 다변량 분석 및 시각화

### 2️⃣ 모델 학습 및 최적화
- **SMOTE**를 통한 클래스 불균형 해결
- **특성 엔지니어링**: 11개의 파생 특성 생성
  - Lifetime_per_Month, Is_New_Member, Is_Long_Member
  - Class_Engagement, Recent_Activity
  - Contract_Completion, Cost_per_Visit 등
- **머신러닝 모델**: 
  - Logistic Regression, Decision Tree
  - Random Forest, Gradient Boosting
  - XGBoost, LightGBM
- **딥러닝 모델**:
  - Advanced Neural Network (BatchNormalization, Dropout)
- **하이퍼파라미터 튜닝**:
  - RandomizedSearchCV (50 iterations, 5-fold CV)
  - XGBoost & LightGBM 최적화
- **앙상블**: Ultimate Stacking (10-fold CV)

### 3️⃣ 모델 평가 및 분석
- 다양한 평가 메트릭 (Accuracy, Precision, Recall, F1, AUC-ROC)
- Confusion Matrix 분석
- ROC Curve & Precision-Recall Curve
- 성능 개선 진행 과정 시각화
- 특성 중요도 분석
- 오분류 사례 분석
- 비즈니스 인사이트 도출

---

## 📈 성능 결과

### 🏆 최종 모델 성능

| 모델 | F1 Score | AUC-ROC |
|------|----------|---------|
| Stacking Ensemble | **0.9657** | **0.9712** |

### 📊 모델 개선 과정

| 단계 | F1 Score | AUC-ROC | 설명 |
|------|----------|---------|------|
| 1. Baseline (Random Forest) | 0.7373 | 0.9635 | 기본 랜덤 포레스트 모델 |
| 2. Basic Stacking | 0.7591 | 0.9675 | 다중 모델 앙상블 |
| 3. Feature Engineering | 0.7591 | 0.9675 | 11개 파생 특성 추가 |
| 4. Hyperparameter Tuning | **0.9634** (CV) | - | XGBoost/LightGBM 최적화 |
| 5. **Ultimate Stacking** | **0.9657** | **0.9712** | 최종 최적화 모델 |

### 🔑 Top 5 중요 특성
1. **Month_to_end_contract** - 계약 만료까지 남은 기간
2. **Lifetime** - 회원 가입 기간
3. **Contract_period** - 계약 기간
4. **Avg_class_frequency_current_month** - 최근 수업 참여 빈도
5. **Class_Engagement** - 전체 수업 참여도

---

## 💻 설치 및 실행

### 필수 요구사항
```bash
Python 3.11+
pandas >= 1.5.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
xgboost >= 2.0.0
lightgbm >= 4.0.0
tensorflow >= 2.13.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
imbalanced-learn >= 0.11.0
```



---

## 📊 주요 시각화 결과

### 1. 성능 개선 진행 과정
- F1 Score와 AUC-ROC의 단계별 개선 추이
- 목표 달성 여부 시각화

### 2. Confusion Matrix
- Stacking Ensemble vs Neural Network 비교
- False Positive/False Negative 분석

### 3. ROC & Precision-Recall Curve
- 모델별 성능 비교
- Average Precision Score

### 4. 특성 중요도
- Top 15 특성 시각화
- 비즈니스 인사이트 도출

---

## 💼 비즈니스 인사이트 및 권장사항

### 🎯 핵심 발견사항
1. **신규 회원 (Lifetime ≤ 3개월)**의 이탈률이 가장 높음
2. **계약 만료 임박 회원**의 이탈 위험 증가
3. **최근 수업 참여율 저조** 시 이탈 가능성 급증
4. **장기 계약 회원**의 이탈률이 현저히 낮음
5. **그룹 활동 참여**가 높을수록 유지율 증가

### 📋 권장 액션 플랜
1. **신규 회원 온보딩 강화**
   - 가입 후 3개월간 집중 관리 프로그램
   - 1:1 PT 세션 제공

2. **리텐션 캠페인**
   - 계약 만료 2개월 전 자동 알림
   - 갱신 인센티브 제공

3. **참여율 모니터링**
   - 주간 수업 참여율 체크
   - 저조 회원 대상 맞춤형 프로그램 제안

4. **장기 계약 유도**
   - 12개월 계약 시 할인 혜택
   - 중도 해지 패널티 조정

5. **커뮤니티 활성화**
   - 그룹 수업 확대
   - 회원 간 네트워킹 이벤트

---

## 🔧 모델 활용 방안

### 1. 이탈 위험 고객 자동 식별
```python
# 모델 로드
import pickle
with open('project/models/2024_churn_model/stacking_ultimate.pkl', 'rb') as f:
    model = pickle.load(f)

# 예측
churn_probability = model.predict_proba(new_customer_data)[:, 1]
high_risk_customers = churn_probability >= 0.50  # 최적 임계값
```

### 2. 월간 리스크 리포트 자동화
- 예측 확률 0.50 이상 회원 리스트 생성
- 특성별 리스크 요인 분석
- 담당자에게 자동 전송

### 3. 맞춤형 리텐션 프로그램
- 리스크 수준별 차별화된 접근
- 개인화된 혜택 제공
- ROI 측정 및 최적화

---

### 설치 방법
```bash
# 저장소 클론
git clone https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN20-2nd-2TEAM.git
cd SKN20-2ed

# 필요한 패키지 설치
pip install pandas numpy scikit-learn xgboost lightgbm tensorflow matplotlib seaborn imbalanced-learn
```

### 실행 방법

#### 1. 전체 분석 파이프라인 실행
```bash
# 1단계: 탐색적 데이터 분석
jupyter notebook project/notebooks/EDA.ipynb

# 2단계: 모델 학습 및 튜닝
jupyter notebook project/notebooks/Model_Training.ipynb

# 3단계: 모델 평가 및 분석
jupyter notebook project/notebooks/Model_Evaluation.ipynb
```

#### 2. 통합 노트북 실행
```bash
jupyter notebook project/notebooks/index2.ipynb
```