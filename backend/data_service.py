import streamlit as st
import pandas as pd

from utils.data_processor import (cached_preprocess_data,
                                 cached_analyze_outliers,
                                 cached_delete_outliers,
                                 cached_get_acf_pacf,
                                 cached_check_stationarity,
                                 cached_get_fft,
                                 cached_decompose_timeseries)
@st.cache_data(ttl=3600)
def load_data(file_path):
    if file_path.name.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.name.endswith('.xlsx'):
        df = pd.read_excel(file_path)

    # 날짜 형식 컬럼 확인 후 에러 처리
    if 'logTime' not in df.columns:
        first_col = df.columns[0]
        df.rename(columns={first_col: 'logTime'}, inplace=True)
        
    if 'logTime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['logTime']):
        df['logTime'] = pd.to_datetime(df['logTime'])
    st.session_state.df = df

    # 새 데이터가 업로드되면 관련 session state 초기화
    st.session_state.target = None  # 타겟 변수 초기화
    st.session_state.test_size = 0.2  # 테스트 사이즈 기본값으로 초기화
    
    # # 날짜 범위 선택
            # default_end_date = pd.Timestamp(df['logTime'].max())
            # default_start_date = default_end_date - timedelta(days=30)
    
            # st.sidebar.markdown("##### 📅 분석 기간 선택", help="시계열 분석을 위한 데이터 기간을 선택하세요. (최대 30일)")
            
            # date_col1, date_col2 = st.sidebar.columns(2)
            
            # with date_col1:
            #     start_date = pd.Timestamp(st.date_input(
            #         "시작 날짜",
            #         default_start_date
            #     ))
                
            # with date_col2:
                    
            #     end_date = pd.Timestamp(st.date_input(
            #         "종료 날짜",
            #         min_value=start_date,
            #         max_value=default_end_date
            #     ))
            
            # # 선택된 날짜 범위 일수 계산
            # date_range_days = (end_date - start_date).days
            
            # # 기간 표시 정보 및 시각화
            # progress_value = min(date_range_days / 30, 1.0)
            # st.sidebar.progress(progress_value)
            # st.sidebar.text(f"선택된 기간: {date_range_days + 1}일 / 최대 30일")
            
            # if date_range_days > 25:
            #     st.sidebar.warning("데이터 양이 많을수록 분석 시간이 길어질 수 있습니다.")
    
        #     # 데이터 가져오기 버튼
        #     if st.sidebar.button("데이터 가져오기"):
        #         try:
        #             filtered_df = df.loc[(df['logTime'] >= start_date) & (df['logTime'] <= end_date)]
        #             if filtered_df is not None and not filtered_df.empty:
        #                 st.session_state.df = filtered_df
        #                 st.rerun()  # 화면 갱신
        #         except Exception as e:
        #             st.sidebar.error(f"데이터 필터링 중 오류가 발생했습니다: {str(e)}")

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

        st.session_state.start_date = st.session_state.df['logTime'].min()
        st.session_state.end_date = st.session_state.df['logTime'].max()

        # 시간당 측정 빈도
        hours_span = (st.session_state.end_date - st.session_state.start_date).total_seconds() / 3600
        st.session_state.records_per_hour = st.session_state.df.shape[0] / max(hours_span, 1) # 최소 1시간으로 나누기


def analyze_outliers():
    if st.session_state.series is not None:
        result = cached_analyze_outliers(st.session_state.series)
        st.session_state.outliers = result
        
        return result
    return None

def delete_outliers(mode):
    """
    이상치 제거 함수 
    """
    if st.session_state.series is not None:
        cleaned_series = cached_delete_outliers(st.session_state.series, mode)
        st.session_state.cleaned_series = cleaned_series 

        return cleaned_series 

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


def analyze_decomposition(period=None):
    """
    시계열 분해 분석 수행
    
    Args:
        period: 계절성 주기 (기본값: None)
    
    Returns:
        dict: 분해 결과 딕셔너리
    """

    if period is None:
        period = st.session_state.period

    if st.session_state.series is not None:       
        try:
            decomposition = cached_decompose_timeseries(st.session_state.series, period)
            st.session_state.decomposition = decomposition
            return decomposition
        
        except Exception as e:
            st.error(f"시계열 분해 중 오류 발생: {str(e)}")
            return None    
        
    return None
