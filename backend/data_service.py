import streamlit as st
import pandas as pd

from utils.data_processor import (cached_preprocess_data,
                                 cached_analyze_outliers,
                                 cached_delete_outliers,
                                 cached_get_acf_pacf,
                                 cached_check_stationarity,
                                 cached_get_fft)

def update_series():
    """
    시계열 데이터 업데이트 함수
    """

    if st.session_state.df is not None:
        st.session_state.series = cached_preprocess_data(
            st.session_state.df,
            st.session_state.target
        )

        # 이전 결과와 현재 설정 비교 
        if st.session_state.model_trained is not None:
            if ('prev_target' in st.session_state and 
                st.session_state.prev_target != st.session_state.target):
                st.session_state.model_trained = False
                st.session_state.forecasts = {}
                st.session_state.metrics = {}

        st.session_state.prev_target = st.session_state.target

def analyze_outliers():
    if st.session_state.series is not None:
        result = cached_analyze_outliers(st.session_state.series)
        st.session_state.outliers = result
        
        return result
    return None

def delete_outliers():
    """
    이상치 제거 함수 
    """
    if st.session_state.series is not None:
        result = cached_delete_outliers(st.session_state.series)
        st.session_state.cleaned_series = result 

        return result 

    return None

def analyze_acf_pacf(nlags=40):
    """
    ACF/PACF 분석 수행
    
    Args:
        nlags: 최대 시차 (기본값: 40)
    
    Returns:
        tuple: (ACF 값, PACF 값) 튜플
    """
    if st.session_state.series is not None:
        try:
            acf_values, pacf_values = cached_get_acf_pacf(st.session_state.series, nlags)
            st.session_state.acf_values = acf_values
            st.session_state.pacf_values = pacf_values
            return acf_values, pacf_values
        
        except Exception as e:
            st.error(f"ACF/PACF 분석 중 오류 발생: {str(e)}")
            return None, None
    return None, None


def analyze_stationarity():
    """
    정상성 검정 수행
    
    Returns:
        dict: 정상성 검정 결과 딕셔너리
    """
    if st.session_state.series is not None:
        try:
            stationarity_result = cached_check_stationarity(st.session_state.series)
            st.session_state.stationarity_result = stationarity_result
            return stationarity_result
        except Exception as e:
            st.error(f"정상성 검정 중 오류 발생: {str(e)}")
            return None
    return None

def analyze_fft():
    """
    고속푸리에 변환 수행
    
    Returns:

    """
    if st.session_state.series is not None:
        try:
            # FFT 수행
            fft_result = cached_get_fft(st.session_state.series)
            st.session_state.fft_result = fft_result
            return fft_result
        except Exception as e:
            st.error(f"고속푸리에변환 중 오류 발생: {str(e)}")
            return None
        
        return None

