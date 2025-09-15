"""
세션 상태 관리를 위한 모듈
"""

import streamlit as st
from config.settings import app_config

def initialize_session_state():
    """
    필요한 변수들을 세션 상태에 추가합니다.
    """
    # 데이터 변수 
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None  # 원본 데이터 저장용
    if 'target' not in st.session_state:
        st.session_state.target = None

    if 'start_date' not in st.session_state:
        st.session_state.start_date = None 
    if 'end_date' not in st.session_state:
        st.session_state.end_date = None 
    if 'records_per_hour' not in st.session_state:
        st.session_state.records_per_hour = None

    if 'time_step' not in st.session_state:
        st.session_state.time_step = 168
    if 'forecast_horizon' not in st.session_state:
        st.session_state.forecast_horizon = 24

    # 시계열 데이터 변수 
    if 'series' not in st.session_state:
        st.session_state.series = None
    if 'train' not in st.session_state:
        st.session_state.train = None
    if 'test' not in st.session_state:
        st.session_state.test = None
    if 'period' not in st.session_state:
        st.session_state.period = None
    if 'decomposition' not in st.session_state:
        st.session_state.decomposition = None

    # 이상치 변수
    if 'lower_standard' not in st.session_state:
        st.session_state.lower_standard = None 
    if 'upper_standard' not in st.session_state:
        st.session_state.upper_standard = None 
    if 'lower_conservative' not in st.session_state:
        st.session_state.lower_conservative = None
    if 'upper_conservative' not in st.session_state:
        st.session_state.upper_conservative = None
    if 'outliers' not in st.session_state:
        st.session_state.outliers = {}
    if 'cleaned_series' not in st.session_state:
        st.session_state.cleaned_series = None

    # 정상성 및 ACF/PACF 관련 변수들
    if 'stationarity_result' not in st.session_state:
        st.session_state.stationarity_result = None
    if 'acf_values' not in st.session_state:
        st.session_state.acf_values = None
    if 'pacf_values' not in st.session_state:
        st.session_state.pacf_values = None

    # 주파수 분석 변수
    if 'fft_result' not in st.session_state:
        st.session_state.fft_result = {}

    # 모델 관련 변수
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None 
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = None
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = None
    if 'test_size' not in st.session_state:
        st.session_state.test_size = app_config.DEFAULT_TEST_SIZE
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'strategy' not in st.session_state:
        st.session_state.strategy = 'smart'
    if 'file_data' not in st.session_state:
        st.session_state.file_data = None
    
    # 데이터 변경 추적 변수
    if 'data_updated' not in st.session_state:
        st.session_state.data_updated = False
    if 'outliers_removed' not in st.session_state:
        st.session_state.outliers_removed = False

    # 사용자 선택 변경 추적 변수 
    if 'active_page' not in st.session_state:
        st.session_state.active_page = 'Home'
    if 'prev_target' not in st.session_state:
        st.session_state.prev_target = None

    # 하이퍼파라미터 최적화 변수 
    if 'hyperparameter_recommendations' not in st.session_state:
        st.session_state.hyperparameter_recommendations = {}
    if 'optimization_history' not in st.session_state:
        st.session_state.optimization_history = {}

    # 차분 관련 변수들
    if 'diff_order' not in st.session_state:
        st.session_state.diff_order = 0
    if 'seasonal_diff_order' not in st.session_state:
        st.session_state.seasonal_diff_order = 0
    if 'use_differencing' not in st.session_state:
        st.session_state.use_differencing = False
    if 'differenced_series' not in st.session_state:
        st.session_state.differenced_series = None
    if 'differencing_recommendation' not in st.session_state:
        st.session_state.differencing_recommendation = None
    if 'diff_train' not in st.session_state:
        st.session_state.diff_train = None
    if 'diff_test' not in st.session_state:
        st.session_state.diff_test = None

def reset_model_results():
    """
    모델 결과를 초기화합니다.
    """
    st.session_state.model_trained = False
    st.session_state.forecasts = {}
    st.session_state.metrics = {}
    st.session_state.best_model = None
    st.session_state.use_differencing = False
    st.session_state.model_results = None


def reset_data_results():
    """
    데이터 결과를 초기화합니다.
    모델 결과도 함께 초기화합니다.(reset_model_results 호출)
    """
    st.session_state.target = None  # 타겟 변수 초기화
    st.session_state.test_size = 0.2  # 테스트 사이즈 기본값으로 초기화
    st.session_state.series = None
    st.session_state.cleaned_series = None
    st.session_state.train = None
    st.session_state.test = None
    st.session_state.decomposition = None
    st.session_state.stationarity_result = None
    st.session_state.acf_values = None
    st.session_state.pacf_values = None
    st.session_state.diff_order = 0
    st.session_state.seasonal_diff_order = 0
    st.session_state.use_differencing = False
    st.session_state.differenced_series = None
    st.session_state.differencing_recommendation = None
    st.session_state.diff_train = None
    st.session_state.diff_test = None
    st.session_state.file_data = None
    reset_model_results()
