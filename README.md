# 🏋️ Gym Churn Prediction - 헬스장 회원 이탈 예측 프로젝트"# 🏋️ Gym Churn Prediction - 헬스장 회원 이탈 예측 프로젝트



---![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

## 1. 팀 소개 🧑‍🤝‍🧑![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green.svg)

![License](https://img.shields.io/badge/License-MIT-yellow.svg)

- 팀명 : SKN20-2nd-2TEAM

- 팀원## 📋 프로젝트 개요



| | | | |헬스장 회원의 이탈(Churn)을 예측하는 머신러닝/딥러닝 프로젝트입니다. 다양한 회원 정보를 바탕으로 이탈 가능성을 사전에 예측하여, 효과적인 리텐션(Retention) 전략을 수립할 수 있도록 지원합니다.

|---|---|---|---|

| <img src="" width="120"> <br> [팀원1]() | <img src="" width="120"> <br> [팀원2]() | <img src="" width="120"> <br> [팀원3]() | <img src="" width="120"> <br> [팀원4]() |### 🎯 프로젝트 목표

- 이탈 위험 고객 조기 식별 시스템 개발

---- 데이터 기반 비즈니스 인사이트 도출



## 2. 프로젝트 개요 📖### 📊 데이터셋

- **파일명**: gym_churn_us.csv

### 2-1. 프로젝트 명- **샘플 수**: 4,002개

        - **특성 수**: 13개 (원본) + 11개 (파생) = 24개

- Gym Churn Prediction (헬스장 회원 이탈 예측)- **타겟**: Churn (0: 유지, 1: 이탈)

        - **이탈률**: 약 30%

### 2-2. 프로젝트 소개

         ---

- 헬스장 회원의 이탈(Churn)을 예측하는 머신러닝/딥러닝 프로젝트입니다. 다양한 회원 정보를 바탕으로 이탈 가능성을 사전에 예측하여, 효과적인 리텐션(Retention) 전략을 수립할 수 있도록 지원합니다.

        ## 🗂️ 프로젝트 구조

### 2-3. 프로젝트 목표

         ```

- **F1 Score 0.9 이상 달성** (고성능 이탈 예측 모델 구축)SKN20-2ed/

- 이탈 위험 고객 조기 식별 시스템 개발├── project/

- 데이터 기반 비즈니스 인사이트 도출│   ├── config/                          # 설정 파일

        │   ├── data/

### 2-4. 프로젝트 결과│   │   ├── raw/                         # 원본 데이터

         │   │   │   └── gym_churn_us.csv

- 최종 F1 Score **0.7674** 달성 (AUC-ROC: 0.9712)│   │   └── processed/                   # 전처리된 데이터

- 이탈 예측 모델을 통한 리스크 고객 식별 시스템 구현│   ├── models/

- 주요 이탈 요인 분석 및 비즈니스 액션 플랜 도출│   │   └── 2024_churn_model/           # 학습된 모델 저장

│   │       ├── stacking_ultimate.pkl   # 최종 앙상블 모델

### 2-5. 데이터셋 정보│   │       ├── scaler_enh.pkl          # 스케일러

│   │       ├── nn_model.h5             # 딥러닝 모델

- **파일명**: gym_churn_us.csv│   │       └── best_threshold.txt      # 최적 임계값

- **데이터 출처**: │   ├── notebooks/

- **샘플 수**: 4,002개│   │   ├── EDA.ipynb                   # 📊 탐색적 데이터 분석

- **특성 수**: 13개 (원본) + 11개 (파생) = 24개│   │   ├── Model_Training.ipynb        # 🤖 모델 학습 및 튜닝

- **타겟**: Churn (0: 유지, 1: 이탈)│   │   ├── Model_Evaluation.ipynb      # 📈 모델 평가 및 분석

- **이탈률**: 약 30%│   │   └── index2.ipynb                # 통합 작업 노트북

│   ├── output/

---│   │   ├── predictions/                # 예측 결과

│   │   ├── reports/                    # 분석 보고서

## 3. 기술 스택 🛠│   │   │   └── final_evaluation_report.txt

│   │   └── visualizations/             # 시각화 결과

### 3.1. 데이터 분석 및 전처리│   │       ├── confusion_matrices.png

- ![Python](https://img.shields.io/badge/-python-blue?logo=python&logoColor=white) : 데이터 처리 및 비즈니스 로직 구현│   │       ├── roc_pr_curves.png

- ![Pandas](https://img.shields.io/badge/-pandas-purple?logo=pandas&logoColor=white) : 데이터 분석 및 필터링 처리│   │       ├── improvement_progress.png

- ![NumPy](https://img.shields.io/badge/-numpy-blue?logo=numpy&logoColor=white) : 수치 연산│   │       └── feature_importance.png

│   └── src/                            # 소스 코드

### 3.2. 머신러닝└── README.md

- ![scikit-learn](https://img.shields.io/badge/-scikit--learn-orange?logo=scikitlearn&logoColor=white) : 전통적 ML 알고리즘, 전처리, 평가```

- ![XGBoost](https://img.shields.io/badge/-XGBoost-red?logo=xgboost&logoColor=white) : 그래디언트 부스팅

- ![LightGBM](https://img.shields.io/badge/-LightGBM-yellow?logo=lightgbm&logoColor=black) : 경량화 부스팅---

- ![imbalanced-learn](https://img.shields.io/badge/-imbalanced--learn-green?logo=python&logoColor=white) : SMOTE (클래스 불균형 해결)

## 🚀 주요 기능

### 3.3. 딥러닝

- ![TensorFlow](https://img.shields.io/badge/-TensorFlow-orange?logo=tensorflow&logoColor=white) : 신경망 구축 및 학습### 1️⃣ 탐색적 데이터 분석 (EDA)

- 데이터 기본 정보 및 통계 분석

### 3.4. 시각화- 타겟 변수(Churn) 분포 분석

- ![Matplotlib](https://img.shields.io/badge/-Matplotlib-blue?logo=python&logoColor=white) : 데이터 시각화- 범주형/수치형 변수 상관관계 분석

- ![Seaborn](https://img.shields.io/badge/-Seaborn-skyblue?logo=python&logoColor=white) : 고급 통계 시각화- 주요 특성 심층 분석 (Lifetime, Contract_period 등)

- 다변량 분석 및 시각화

### 3.5. 개발 환경

- ![Jupyter](https://img.shields.io/badge/-Jupyter-orange?logo=jupyter&logoColor=white) : 대화형 개발 환경### 2️⃣ 모델 학습 및 최적화

- **SMOTE**를 통한 클래스 불균형 해결

### 3.6. 아키텍처- **특성 엔지니어링**: 11개의 파생 특성 생성

```  - Lifetime_per_Month, Is_New_Member, Is_Long_Member

데이터 수집 → 전처리 → 특성 엔지니어링 → 모델 학습 → 평가 → 배포  - Class_Engagement, Recent_Activity

```  - Contract_Completion, Cost_per_Visit 등

- **머신러닝 모델**: 

---  - Logistic Regression, Decision Tree

  - Random Forest, Gradient Boosting

## 4. 프로젝트 진행 🚀  - XGBoost, LightGBM

- **딥러닝 모델**:

### 4.1. 프로젝트 기획 및 회의  - Advanced Neural Network (BatchNormalization, Dropout)

- 프로젝트 목표 설정 및 데이터셋 선정- **하이퍼파라미터 튜닝**:

- 평가 지표 및 목표 성능 정의  - RandomizedSearchCV (50 iterations, 5-fold CV)

  - XGBoost & LightGBM 최적화

### 4.2. 데이터 탐색 및 분석 (EDA)- **앙상블**: Ultimate Stacking (10-fold CV)

- 데이터 기본 정보 및 통계 분석

- 타겟 변수(Churn) 분포 분석### 3️⃣ 모델 평가 및 분석

- 범주형/수치형 변수 상관관계 분석- 다양한 평가 메트릭 (Accuracy, Precision, Recall, F1, AUC-ROC)

- 주요 특성 심층 분석 (Lifetime, Contract_period 등)- Confusion Matrix 분석

- 다변량 분석 및 시각화- ROC Curve & Precision-Recall Curve

- 성능 개선 진행 과정 시각화

### 4.3. 데이터 전처리 및 특성 엔지니어링- 특성 중요도 분석

- 결측치 처리 및 이상치 탐지- 오분류 사례 분석

- Train-Test 분할 (80:20)- 비즈니스 인사이트 도출

- StandardScaler를 통한 정규화

- SMOTE를 통한 클래스 불균형 해결---

- 11개의 파생 특성 생성

## 📈 성능 결과

### 4.4. 모델 학습 및 최적화

- 6개 기본 머신러닝 모델 학습 (Baseline)### 🏆 최종 모델 성능

- XGBoost & LightGBM 하이퍼파라미터 튜닝

- 딥러닝 모델 (Advanced NN) 학습| 모델 | F1 Score | Accuracy | Precision | Recall | AUC-ROC |

- Ultimate Stacking Ensemble 구축|------|----------|----------|-----------|--------|---------|

- 임계값 최적화| **Stacking Ensemble** | **0.7674** | **0.9163** | **0.7396** | **0.7970** | **0.9712** |

| Neural Network | 0.7108 | 0.9013 | 0.6769 | 0.7479 | 0.9563 |

### 4.5. 모델 평가 및 분석

- 다양한 평가 메트릭 계산### 📊 모델 개선 과정

- Confusion Matrix 분석

- ROC & Precision-Recall Curve 시각화| 단계 | F1 Score | AUC-ROC | 설명 |

- 특성 중요도 분석|------|----------|---------|------|

- 오분류 사례 분석| 1. Baseline (Random Forest) | 0.7373 | 0.9635 | 기본 랜덤 포레스트 모델 |

- 비즈니스 인사이트 도출| 2. Basic Stacking | 0.7591 | 0.9675 | 다중 모델 앙상블 |

| 3. Feature Engineering | 0.7591 | 0.9675 | 11개 파생 특성 추가 |

### 4.6. 문서화 및 배포 준비| 4. Hyperparameter Tuning | **0.9657** (CV) | - | XGBoost/LightGBM 최적화 |

- 최종 보고서 작성| 5. **Ultimate Stacking** | **0.7674** | **0.9712** | 최종 최적화 모델 |

- README 작성

- 모델 저장 및 관리### 🔑 Top 5 중요 특성

1. **Month_to_end_contract** - 계약 만료까지 남은 기간

---2. **Lifetime** - 회원 가입 기간

3. **Contract_period** - 계약 기간

## 5. ERD (프로젝트 구조) 📊4. **Avg_class_frequency_current_month** - 최근 수업 참여 빈도

5. **Class_Engagement** - 전체 수업 참여도

### 디렉토리 구조

```---

SKN20-2ed/

├── project/## 💻 설치 및 실행

│   ├── config/                          # 설정 파일

│   ├── data/### 필수 요구사항

│   │   ├── raw/                         # 원본 데이터```bash

│   │   │   ├── gym_churn_us.csvPython 3.11+

│   │   │   ├── 2022년방송매체이용행태조사.savpandas >= 1.5.0

│   │   │   ├── 2023년방송매체이용형태조사.savnumpy >= 1.24.0

│   │   │   └── 2024년방송매체이용형태조사.savscikit-learn >= 1.3.0

│   │   └── processed/                   # 전처리된 데이터xgboost >= 2.0.0

│   ├── models/lightgbm >= 4.0.0

│   │   └── 2024_churn_model/           # 학습된 모델 저장tensorflow >= 2.13.0

│   │       ├── stacking_ultimate.pkl   # 최종 앙상블 모델matplotlib >= 3.7.0

│   │       ├── scaler_enh.pkl          # 스케일러seaborn >= 0.12.0

│   │       ├── nn_model.h5             # 딥러닝 모델imbalanced-learn >= 0.11.0

│   │       └── best_threshold.txt      # 최적 임계값```

│   ├── notebooks/

│   │   ├── EDA.ipynb                   # 📊 탐색적 데이터 분석### 설치 방법

│   │   ├── Model_Training.ipynb        # 🤖 모델 학습 및 튜닝```bash

│   │   ├── Model_Evaluation.ipynb      # 📈 모델 평가 및 분석# 저장소 클론

│   │   └── index2.ipynb                # 통합 작업 노트북git clone https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN20-2nd-2TEAM.git

│   ├── output/cd SKN20-2ed

│   │   ├── predictions/                # 예측 결과

│   │   ├── reports/                    # 분석 보고서# 필요한 패키지 설치

│   │   │   └── final_evaluation_report.txtpip install pandas numpy scikit-learn xgboost lightgbm tensorflow matplotlib seaborn imbalanced-learn

│   │   └── visualizations/             # 시각화 결과```

│   │       ├── confusion_matrices.png

│   │       ├── roc_pr_curves.png### 실행 방법

│   │       ├── improvement_progress.png

│   │       └── feature_importance.png#### 1. 전체 분석 파이프라인 실행

│   └── src/                            # 소스 코드```bash

└── README.md# 1단계: 탐색적 데이터 분석

```jupyter notebook project/notebooks/EDA.ipynb



---# 2단계: 모델 학습 및 튜닝

jupyter notebook project/notebooks/Model_Training.ipynb

## 6. 프로젝트 실행 결과

# 3단계: 모델 평가 및 분석

### 6.1. 탐색적 데이터 분석 (EDA.ipynb)jupyter notebook project/notebooks/Model_Evaluation.ipynb

- 데이터 기본 정보 및 통계량 확인```

- 타겟 변수 분포 및 불균형 확인

- 범주형/수치형 변수 분석#### 2. 통합 노트북 실행

- 상관관계 히트맵```bash

![EDA 결과]()jupyter notebook project/notebooks/index2.ipynb

```

### 6.2. 모델 성능 비교 (Model_Evaluation.ipynb)

---

#### 최종 모델 성능

## 📊 주요 시각화 결과

| 모델 | F1 Score | Accuracy | Precision | Recall | AUC-ROC |

|------|----------|----------|-----------|--------|---------|### 1. 성능 개선 진행 과정

| **Stacking Ensemble** | **0.7674** | **0.9163** | **0.7396** | **0.7970** | **0.9712** |- F1 Score와 AUC-ROC의 단계별 개선 추이

| Neural Network | 0.7108 | 0.9013 | 0.6769 | 0.7479 | 0.9563 |- 목표 달성 여부 시각화



#### 모델 개선 과정### 2. Confusion Matrix

- Stacking Ensemble vs Neural Network 비교

| 단계 | F1 Score | AUC-ROC | 설명 |- False Positive/False Negative 분석

|------|----------|---------|------|

| 1. Baseline (Random Forest) | 0.7373 | 0.9635 | 기본 랜덤 포레스트 모델 |### 3. ROC & Precision-Recall Curve

| 2. Basic Stacking | 0.7591 | 0.9675 | 다중 모델 앙상블 |- 모델별 성능 비교

| 3. Feature Engineering | 0.7591 | 0.9675 | 11개 파생 특성 추가 |- Average Precision Score

| 4. Hyperparameter Tuning | **0.9657** (CV) | - | XGBoost/LightGBM 최적화 |

| 5. **Ultimate Stacking** | **0.7674** | **0.9712** | 최종 최적화 모델 |### 4. 특성 중요도

- Top 15 특성 시각화

![모델 성능 개선 과정]()- 비즈니스 인사이트 도출



### 6.3. Confusion Matrix---

- Stacking Ensemble과 Neural Network 비교

- False Positive/False Negative 분석## 💼 비즈니스 인사이트 및 권장사항

![Confusion Matrix]()

### 🎯 핵심 발견사항

### 6.4. ROC & Precision-Recall Curve1. **신규 회원 (Lifetime ≤ 3개월)**의 이탈률이 가장 높음

- 모델별 성능 비교 곡선2. **계약 만료 임박 회원**의 이탈 위험 증가

- Average Precision Score3. **최근 수업 참여율 저조** 시 이탈 가능성 급증

![ROC Curve]()4. **장기 계약 회원**의 이탈률이 현저히 낮음

5. **그룹 활동 참여**가 높을수록 유지율 증가

### 6.5. 특성 중요도

- Top 15 중요 특성 시각화### 📋 권장 액션 플랜

- 비즈니스 인사이트1. **신규 회원 온보딩 강화**

![Feature Importance]()   - 가입 후 3개월간 집중 관리 프로그램

   - 1:1 PT 세션 제공

### 6.6. 주요 이탈 예측 요인 (Top 5)

1. **Month_to_end_contract** - 계약 만료까지 남은 기간2. **리텐션 캠페인**

2. **Lifetime** - 회원 가입 기간   - 계약 만료 2개월 전 자동 알림

3. **Contract_period** - 계약 기간   - 갱신 인센티브 제공

4. **Avg_class_frequency_current_month** - 최근 수업 참여 빈도

5. **Class_Engagement** - 전체 수업 참여도3. **참여율 모니터링**

   - 주간 수업 참여율 체크

---   - 저조 회원 대상 맞춤형 프로그램 제안



## 7. 비즈니스 인사이트 💡4. **장기 계약 유도**

   - 12개월 계약 시 할인 혜택

### 7.1. 핵심 발견사항   - 중도 해지 패널티 조정

1. **신규 회원 (Lifetime ≤ 3개월)**의 이탈률이 가장 높음

2. **계약 만료 임박 회원**의 이탈 위험 증가5. **커뮤니티 활성화**

3. **최근 수업 참여율 저조** 시 이탈 가능성 급증   - 그룹 수업 확대

4. **장기 계약 회원**의 이탈률이 현저히 낮음   - 회원 간 네트워킹 이벤트

5. **그룹 활동 참여**가 높을수록 유지율 증가

---

### 7.2. 비즈니스 권장사항

1. **신규 회원 온보딩 강화**## 🔧 모델 활용 방안

   - 가입 후 3개월간 집중 관리 프로그램

   - 1:1 PT 세션 제공### 1. 이탈 위험 고객 자동 식별

```python

2. **리텐션 캠페인**# 모델 로드

   - 계약 만료 2개월 전 자동 알림import pickle

   - 갱신 인센티브 제공with open('project/models/2024_churn_model/stacking_ultimate.pkl', 'rb') as f:

    model = pickle.load(f)

3. **참여율 모니터링**

   - 주간 수업 참여율 체크# 예측

   - 저조 회원 대상 맞춤형 프로그램 제안churn_probability = model.predict_proba(new_customer_data)[:, 1]

high_risk_customers = churn_probability >= 0.50  # 최적 임계값

4. **장기 계약 유도**```

   - 12개월 계약 시 할인 혜택

   - 중도 해지 패널티 조정### 2. 월간 리스크 리포트 자동화

- 예측 확률 0.50 이상 회원 리스트 생성

5. **커뮤니티 활성화**- 특성별 리스크 요인 분석

   - 그룹 수업 확대- 담당자에게 자동 전송

   - 회원 간 네트워킹 이벤트

### 3. 맞춤형 리텐션 프로그램

---- 리스크 수준별 차별화된 접근

- 개인화된 혜택 제공

## 8. 한줄 회고- ROI 측정 및 최적화



- **팀원1** : ---



- **팀원2** : ## 📚 기술 스택



- **팀원3** : ### 데이터 분석

- **Pandas**: 데이터 처리 및 전처리

- **팀원4** : - **NumPy**: 수치 연산

- **Matplotlib/Seaborn**: 데이터 시각화

---

### 머신러닝

## ⚙️ 실행 방법- **scikit-learn**: 전통적 ML 알고리즘, 전처리, 평가

- **XGBoost**: 그래디언트 부스팅

### 필수 요구사항- **LightGBM**: 경량화 부스팅

```bash- **imbalanced-learn**: SMOTE (클래스 불균형 해결)

Python 3.11+

pandas >= 1.5.0### 딥러닝

numpy >= 1.24.0- **TensorFlow/Keras**: 신경망 구축 및 학습

scikit-learn >= 1.3.0

xgboost >= 2.0.0### 최적화

lightgbm >= 4.0.0- **RandomizedSearchCV**: 하이퍼파라미터 자동 탐색

tensorflow >= 2.13.0- **StackingClassifier**: 앙상블 학습

matplotlib >= 3.7.0

seaborn >= 0.12.0---

imbalanced-learn >= 0.11.0

```## 👥 팀 구성



### 설치 및 실행- **Team**: SKN20-2nd-2TEAM

```bash- **Organization**: SKNETWORKS-FAMILY-AICAMP

# 저장소 클론- **Branch**: hosung

git clone https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN20-2nd-2TEAM.git

cd SKN20-2ed---



# 패키지 설치## 📄 라이선스

pip install -r requirements.txt

이 프로젝트는 MIT 라이선스를 따릅니다.

# Jupyter Notebook 실행

jupyter notebook---



# 노트북 순서대로 실행## 📞 문의

# 1. project/notebooks/EDA.ipynb

# 2. project/notebooks/Model_Training.ipynb프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

# 3. project/notebooks/Model_Evaluation.ipynb

```---



---## 🙏 감사의 말



## 📄 라이선스이 프로젝트는 SK Networks AI Camp의 지원을 받아 진행되었습니다.



이 프로젝트는 MIT 라이선스를 따릅니다.---



---**Last Updated**: 2025년 11월 3일" 


## 👥 팀 정보

- **Team**: SKN20-2nd-2TEAM
- **Organization**: SKNETWORKS-FAMILY-AICAMP
- **Branch**: hosung

---

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

## 🙏 감사의 말

이 프로젝트는 SK Networks AI Camp의 지원을 받아 진행되었습니다.

---

**Last Updated**: 2025년 11월 3일
