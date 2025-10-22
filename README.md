# 시계열 예측 모델 최적화 플랫폼

## 📋 프로젝트 개요

시계열 예측 모델 최적화 플랫폼은 다양한 시계열 예측 모델을 자동으로 최적화하고 비교하는 Streamlit 기반 웹 애플리케이션입니다. LSTM, XGBoost 모델을 지원하며, Optuna를 활용한 베이즈 최적화를 통해 최적의 하이퍼파라미터를 자동으로 찾습니다.

## 🚀 주요 기능

### 1. 데이터 분석 및 시각화
- 시계열 데이터 업로드 및 전처리
- 데이터 탐색적 분석 (EDA)
- 다양한 차트와 그래프를 통한 데이터 시각화

### 2. 모델 최적화
- **LSTM**: 딥러닝 기반 시계열 예측
- **XGBoost**: 그래디언트 부스팅 기반 예측

### 3. 최적화 전략
- **빠른 전략**: 각 모델 10번씩 빠른 비교
- **균형 전략**: 모든 모델 동일하게 20번씩
- **철저 전략**: 모든 모델 충분히 50번씩
- **스마트 전략**: 단계적 최적화 (스크리닝 → 정밀 최적화)
- **커스텀 전략**: 사용자 지정 trials 수

### 4. 성능 평가
- RMSE, MAE 평가 지표
- 모델별 성능 비교 및 시각화
- 최적 모델 자동 추천

## 🛠️ 기술 스택

### Backend
- **Python 3.11.9**
- **Streamlit**: 웹 애플리케이션 프레임워크
- **Optuna**: 베이즈 최적화
- **TensorFlow/Keras**: LSTM 모델
- **XGBoost**: 그래디언트 부스팅
- **scikit-learn**: 데이터 전처리 및 평가

### Frontend
- **Streamlit**: UI 프레임워크
- **Plotly**: 인터랙티브 차트
- **Pandas**: 데이터 조작

## 📁 프로젝트 구조

```
outflow/
├── app.py                          # 메인 애플리케이션
├── app.yaml                        # 배포 설정 (gcloud 배포할 때 사용)
├── requirements.txt                 # 의존성 패키지
├── README.md                       # 프로젝트 문서
│
├── backend/                        # 백엔드 서비스
│   ├── data_service.py            # 데이터 처리 서비스
│   ├── model_service.py           # 모델 최적화 서비스
│   └── visualization_service.py   # 시각화 서비스
│
├── frontend/                       # 프론트엔드 컴포넌트
│   ├── components.py              # UI 컴포넌트
│   ├── session_state.py           # 세션 상태 관리
│   └── sidebar.py                 # 사이드바 컴포넌트
│
├── models/                         # 모델 구현
│   ├── base_model.py              # 기본 모델 클래스
│   ├── lstm_model.py              # LSTM 모델
│   ├── xgboost_model.py           # XGBoost 모델
│   └── model_factory.py           # 모델 팩토리
│
├── pages/                          # 페이지별 구현
│   ├── home.py                    # 홈페이지
│   ├── data_analysis.py           # 데이터 분석 페이지
│   └── model_training.py          # 모델 훈련 페이지
│
├── utils/                          # 유틸리티
│   ├── data_processor.py          # 데이터 전처리
│   ├── parameter_utils.py         # 파라미터 유틸리티
│   ├── singleton.py               # 싱글톤 패턴
│   └── visualizer.py              # 시각화 유틸리티
│
├── config/                         # 설정
│   ├── __init__.py
│   └── settings.py                # 애플리케이션 설정
│
├── styles/                         # 스타일 파일
│   └── NanumGothic.ttf            # 한글 폰트
│
└── templates/                      # 템플릿 파일
```

## 🚀 설치 및 실행

### 1. 저장소 클론
```bash
git clone <repository-url>
cd outflow
```

### 2. 가상환경 생성 및 활성화
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 애플리케이션 실행
```bash
streamlit run app.py
```

## 📊 사용 방법

### 1. 데이터 업로드
- 홈페이지에서 CSV 파일 업로드
- 타겟 변수 및 예측 기간 설정

### 2. 데이터 분석
- 데이터 탐색적 분석 페이지에서 데이터 확인
- 시각화를 통한 패턴 파악

### 3. 모델 훈련
- 사용할 모델 선택 (LSTM, XGBoost, CatBoost, Prophet)
- 최적화 전략 선택
- 최적화 실행

### 4. 결과 확인
- 모델별 성능 비교
- 최적 모델 자동 추천
- 예측 결과 시각화

## ⚙️ 설정

### 환경 변수
```bash
# GPU 메모리 제한 (선택사항)
CUDA_VISIBLE_DEVICES=0
```

### 모델 파라미터
각 모델의 하이퍼파라미터는 `models/` 디렉토리의 해당 파일에서 수정할 수 있습니다.

## 🔧 주요 기능 상세

### LSTM 모델
- 다층 LSTM/GRU 네트워크
- 자동 특성 선택
- 드롭아웃 및 조기 종료
- GPU 메모리 최적화

### XGBoost 모델
- 그래디언트 부스팅 최적화
- 범주형 특성 자동 처리
- 조기 종료 및 교차 검증

## 📈 성능 최적화

### GPU 사용
- TensorFlow GPU 지원
- 메모리 사용량 제한
- 배치 크기 자동 조정

### 메모리 최적화
- 스트리밍 데이터 처리
- 모델 캐싱
- 세션 상태 관리

## 🐛 문제 해결

### 일반적인 문제
1. **TensorFlow 설치 오류**: GPU 버전과 CPU 버전 확인
2. **메모리 부족**: 배치 크기 감소 또는 모델 복잡도 조정
3. **데이터 형식 오류**: CSV 파일의 인코딩 및 구분자 확인

### 로그 확인
```bash
# Streamlit 로그
streamlit run app.py --logger.level debug
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해 주세요.
