import streamlit as st
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fft import fft, fftfreq

from utils.singleton import Singleton

from config.settings import app_config

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
    
    def delete_outliers(self, series, mode):
        temp = series.copy()
        temp[(temp < st.session_state.outliers['lower_'+mode])|(temp > st.session_state.outliers['upper_'+mode])] = np.nan
        cleaned_series = temp.interpolate(method='time').fillna('ffill').fillna('bfill')

        return cleaned_series
    
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
    
    def perform_differencing(self, series: pd.Series, diff_order: int = 1, seasonal_diff_order: int = 0, seasonal_period: int = None) -> pd.Series:
        """
        시계열 데이터에 차분을 적용합니다.
        
        Args:
            series: 차분할 시계열 데이터
            diff_order: 일반 차분 차수 (기본값: 1)
            seasonal_diff_order: 계절 차분 차수 (기본값: 0)
            seasonal_period: 계절성 주기 (None인 경우 계절 차분 미적용)
            
        Returns:
            차분된 시계열 데이터
        """
        differenced_series = series.copy()
        
        # 계절 차분 적용 (seasonal_period가 있는 경우)
        if seasonal_diff_order > 0 and seasonal_period is not None:
            for _ in range(seasonal_diff_order):
                differenced_series = differenced_series.diff(seasonal_period).dropna()
        
        # 일반 차분 적용
        for _ in range(diff_order):
            differenced_series = differenced_series.diff().dropna()
        
        return differenced_series

    def recommend_differencing(self, series: pd.Series, acf_values: np.ndarray = None, pacf_values: np.ndarray = None) -> dict:
        """
        시계열 데이터의 ACF, PACF 및 정상성 검정 결과를 기반으로 차분 추천을 제공합니다.
        
        Args:
            series: 시계열 데이터
            acf_values: ACF 값 배열 (None인 경우 계산)
            pacf_values: PACF 값 배열 (None인 경우 계산)
            
        Returns:
            추천 정보를 담은 딕셔너리
        """
        # 정상성 검정
        stationarity_result = self.check_stationarity(series)
        is_stationary = stationarity_result['is_stationary']
        
        # ACF/PACF가 제공되지 않은 경우 계산
        if acf_values is None or pacf_values is None:
            acf_values, pacf_values = self.get_acf_pacf(series)
        
        # 초기 추천 사항
        recommendation = {
            'needs_differencing': not is_stationary,
            'diff_order': 0,
            'seasonal_diff_order': 0,
            'seasonal_period': None,
            'reason': []
        }
        
        # 비정상이면 차분 추천
        if not is_stationary:
            recommendation['diff_order'] = 1
            recommendation['reason'].append("시계열이 정상성을 만족하지 않습니다 (ADF 검정 p-value > 0.05).")
        
        # ACF 감소 속도가 느리면 차분 추천
        slow_decay = all(acf_values[i] > 0.5 for i in range(1, min(5, len(acf_values))))
        if slow_decay:
            recommendation['diff_order'] = max(recommendation['diff_order'], 1)
            recommendation['reason'].append("ACF가 천천히 감소하는 패턴을 보입니다 (추세 존재 가능성).")
        
        # 계절성 확인 (ACF에서 특정 lag에서 높은 값 발견)
        for period in [24, 168, 720]:  # 일별(24시간), 주별(168시간), 월별(30일) 주기
            if len(acf_values) > period and acf_values[period] > 0.3:
                recommendation['seasonal_period'] = period
                recommendation['seasonal_diff_order'] = 1
                recommendation['reason'].append(f"{period}시간 주기의 계절성이 감지되었습니다.")
                break
        
        return recommendation

    
    def decompose_timeseries(self, series: pd.Series, period):
        one_day = int(24*st.session_state.records_per_hour)

        if len(series) >= one_day: # 최소 하루치 이상의 데이터
            try:
                # 주기 자동 감지 또는 기본값 사용
                period = min(one_day, len(series)//2)
                decomposition = seasonal_decompose(series, model='additive', period=period)

                return {
                    'observed': decomposition.observed.tolist(),
                    'trend': decomposition.trend.tolist(),
                    'seasonal': decomposition.seasonal.tolist(),
                    'resid': decomposition.resid.tolist(),
                    'index': decomposition.observed.index.astype(str).tolist()  # datetime index도 string으로
                }
            
            except Exception as e:
                return None
        else:
            return None
        
    def train_test_split(self, 
                         series: pd.Series, 
                         test_size: float = app_config.DEFAULT_TEST_SIZE) -> tuple[pd.Series, pd.Series]:
        """
        시계열 데이터를 훈련 세트와 테스트 세트로 분할합니다.
        
        Args:
            series: 분할할 시계열 데이터
            test_size: 테스트 세트의 비율
            
        Returns:
            (훈련 데이터, 테스트 데이터) 튜플
        """
        # 분할 지점 계산
        split_idx = int(len(series) * (1 - test_size))
        
        # 시간 순서대로 분할
        train = series[:split_idx]
        test = series[split_idx:]
        
        return train, test
    
            
    

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
def cached_delete_outliers(series, mode):
    processor = DataProcessor()
    return processor.delete_outliers(series, mode)

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

@st.cache_data(ttl=3600)
def cached_decompose_timeseries(series, period):
    """계절성 분해 결과 캐싱"""
    processor = DataProcessor()
    return processor.decompose_timeseries(series, period)

@st.cache_data(ttl=3600)
def cached_perform_differencing(series, diff_order=1, seasonal_diff_order=0, seasonal_period=None):
    """차분 적용 결과 캐싱"""
    processor = DataProcessor()
    return processor.perform_differencing(series, diff_order, seasonal_diff_order, seasonal_period)

@st.cache_data(ttl=3600)
def cached_recommend_differencing(series, acf_values=None, pacf_values=None):
    """차분 추천 결과 캐싱"""
    processor = DataProcessor()
    return processor.recommend_differencing(series, acf_values, pacf_values)

@st.cache_data(ttl=3600)
def cached_train_test_split(series, test_size):
    """훈련/테스트 분할 캐싱"""
    processor = DataProcessor()
    return processor.train_test_split(series, test_size)
