import streamlit as st
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from scipy.fft import fft, fftfreq

from utils.singleton import Singleton

class DataProcessor(metaclass = Singleton):
    """
    시계열 데이터 전처리 클래스 
    """

    def preprocess_data(self, df: pd.DataFrame, 
                              target_col: str) -> pd.Series:
        """
        Args:
            df (pd.DataFrame): 입력 데이터프레임
            target_col (str): 타겟 컬럼 이름

        Returns:
            전처리된 시계열 데이터 
        """

        # 시간 인덱스 설정 
        df = df.set_index('logTime')

        # 타겟 컬럼만 선택 
        series = df[target_col] if target_col in df.columns else df.iloc[:, 0]

        # 시간순으로 정렬
        series = series.sort_index()

        # temp = series.copy()

        # # 이상치를 결측치로 처리 
        # q1 = temp.quantile(0.25)
        # q3 = temp.quantile(0.75)
        # iqr = q3 - q1
        
        # # 이상치(Q1 - 1.5*IQR 미만 또는 Q3 + 1.5*IQR 초과)를 NaN으로 처리
        # series.loc[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)] = np.nan

        # 결측치 처리 (선형 보간)
        series = series.interpolate(method='time').fillna('ffill').fillna('bfill')

        return series 
    
    def analyze_outliers(self, series: pd.Series) -> dict:
        """
        Args:
            series (pd.Series): 시계열 데이터

        Returns:
            dict: 이상치 분석 결과
        """
        temp = series.copy()
        temp.dropna(inplace=True)

        # IQR 방법을 사용한 이상치 분석 로직 구현
        Q1 = temp.quantile(0.25)
        Q3 = temp.quantile(0.75)
        IQR = Q3 - Q1

        # 표준 기준 (1.5 × IQR)
        lower_standard = Q1 - 1.5 * IQR
        upper_standard = Q3 + 1.5 * IQR

        # 보수적 기준 (3.0 × IQR)
        lower_conservative = Q1 - 3.0 * IQR
        upper_conservative = Q3 + 3.0 * IQR

        total_standard = len(st.session_state.series[(st.session_state.series < lower_standard)|(st.session_state.series > upper_standard)])
        total_conservative = len(st.session_state.series[(st.session_state.series < lower_conservative)|(st.session_state.series > upper_conservative)])

        return {
            'lower_standard': lower_standard,
            'upper_standard': upper_standard,
            'lower_conservative': lower_conservative,
            'upper_conservative': upper_conservative,
            'total_standard' : total_standard,
            'total_conservative' : total_conservative
        }
    
    def delete_outliers(self, series):
        temp = series.copy()
        temp[(temp < st.session_state.outliers['lower_standard'])|(temp > st.session_state.outliers['upper_standard'])]= np.nan
        st.session_state.cleaned_series = temp.interpolate(method='time').fillna('ffill').fillna('bfill')

        return None
    
    def get_acf_pacf(self, 
                    series: pd.Series, 
                    nlags: int = 40) -> tuple[np.ndarray, np.ndarray]:
        """
        시계열 데이터의 ACF와 PACF를 계산합니다.
        
        Args:
            series: 분석할 시계열 데이터
            nlags: 계산할 최대 지연값
            
        Returns:
            (ACF, PACF) 튜플
        """
        
        # 결측치 제거
        series_clean = series.dropna()
        
        # statsmodels는 기본적으로 시계열 길이의 절반 이하의 lag만 허용
        max_lags_allowed = len(series_clean) // 2
        
        # 요청한 nlags와 허용 가능 maximum lag 중 더 작은 값 선택
        safe_nlags = min(nlags, max_lags_allowed)
        
        # ACF, PACF 계산
        acf_values = acf(series_clean, nlags=safe_nlags)
        pacf_values = pacf(series_clean, nlags=safe_nlags)
        
        return acf_values, pacf_values
    
    def check_stationarity(self, series: pd.Series) -> dict:
        """
        시계열 데이터의 정상성을 검정합니다.
        
        Args:
            series: 검정할 시계열 데이터
            
        Returns:
            검정 결과를 담은 딕셔너리
        """
        # ADF 검정 수행
        result = adfuller(series.dropna())
        
        # 결과 정리
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'num_observations': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def get_fft(self, series: pd.Series) -> dict:
        fft_values, frequencies = fft(st.session_state.series.values), fftfreq(len(st.session_state.series))
        
        # 양의 주파수만 선택
        positive_freq_idx = frequencies > 0
        freqs = frequencies[positive_freq_idx]
        power = np.abs(fft_values[positive_freq_idx])**2
        
        # 상위 주파수 성분 찾기
        top_freq_idx = np.argsort(power)[-5:]  # 상위 5개

        return {
            'idx' : positive_freq_idx,
            'freqs' : freqs,
            'power' : power,
            'top_freq_idx' : top_freq_idx
        }
    

@st.cache_data(ttl=3600)
def cached_preprocess_data(df, target_col):
    """
    시계열 그래프 캐싱
    """
    processor = DataProcessor()
    return processor.preprocess_data(df, target_col)

@st.cache_data(ttl=3600)
def cached_analyze_outliers(series):
    """
    이상치 분석 캐싱
    Args:
        series (pd.Series): 시계열 데이터

    Returns:
        pd.DataFrame: 이상치 분석 결과
    """
    processor = DataProcessor()
    return processor.analyze_outliers(series)

@st.cache_data(ttl=3600)
def cached_delete_outliers(series):
    processor = DataProcessor()
    return processor.delete_outliers(series)

@st.cache_data(ttl=3600)
def cached_get_acf_pacf(series, nlags=40):
    """ACF/PACF 결과 캐싱"""
    processor = DataProcessor()
    return processor.get_acf_pacf(series, nlags)

@st.cache_data(ttl=3600)
def cached_check_stationarity(series):
    """정상성 검정 결과 캐싱"""
    processor = DataProcessor()
    return processor.check_stationarity(series)

@st.cache_data(ttl=3600)
def cached_get_fft(series):
    """고속푸리에변환 결과 캐싱"""
    processor = DataProcessor()
    return processor.get_fft(series)