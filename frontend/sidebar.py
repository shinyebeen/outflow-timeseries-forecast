import streamlit as st

import pandas as pd 
from datetime import datetime, timedelta

from backend.data_service import update_series

def initialize_sidebar():
    """
    사이드바를 초기화하고 필요한 변수들을 설정합니다.
    """
    # 데이터 로드 섹션
    render_data_load_section()
    
    # 데이터가 있을 경우 분석 옵션 섹션 표시
    if st.session_state.df is not None and not st.session_state.df.empty:
        render_analysis_options()

def render_data_load_section():
    st.sidebar.header("배수지 유출유량 예측")

    file = st.sidebar.file_uploader("upload file", 
                            type=["csv", "xlsx"],
                            help="배수지 유출유량 데이터를 포함한 CSV 또는 엑셀 파일을 업로드하세요.")
    
    if file is not None:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        
        # 날짜 형식 컬럼 확인 후 에러 처리
        try:
            if 'logTime' not in df.columns:
                first_col = df.columns[0]
                df.rename(columns={first_col: 'logTime'}, inplace=True)
                
            if 'logTime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['logTime']):
                df['logTime'] = pd.to_datetime(df['logTime'])
                    
            # 새 데이터가 업로드되면 관련 session state 초기화
            st.session_state.target = None  # 타겟 변수 초기화
            st.session_state.test_size = 0.2  # 테스트 사이즈 기본값으로 초기화
                
            # 날짜 범위 선택
            default_end_date = pd.Timestamp(df['logTime'].max())
            default_start_date = default_end_date - timedelta(days=30)
    
            st.sidebar.markdown("##### 📅 분석 기간 선택", help="시계열 분석을 위한 데이터 기간을 선택하세요. (최대 30일)")
            
            date_col1, date_col2 = st.sidebar.columns(2)
            
            with date_col1:
                start_date = pd.Timestamp(st.date_input(
                    "시작 날짜",
                    default_start_date
                ))
                
            with date_col2:
                    
                end_date = pd.Timestamp(st.date_input(
                    "종료 날짜",
                    min_value=start_date,
                    max_value=default_end_date
                ))
            
            # 선택된 날짜 범위 일수 계산
            date_range_days = (end_date - start_date).days
            
            # 기간 표시 정보 및 시각화
            progress_value = min(date_range_days / 30, 1.0)
            st.sidebar.progress(progress_value)
            st.sidebar.text(f"선택된 기간: {date_range_days + 1}일 / 최대 30일")
            
            if date_range_days > 25:
                st.sidebar.warning("데이터 양이 많을수록 분석 시간이 길어질 수 있습니다.")
    
            # 데이터 가져오기 버튼
            if st.sidebar.button("데이터 가져오기"):
                try:
                    filtered_df = df.loc[(df['logTime'] >= start_date) & (df['logTime'] <= end_date)]
                    if filtered_df is not None and not filtered_df.empty:
                        st.session_state.df = filtered_df
                        st.rerun()  # 화면 갱신
                except Exception as e:
                    st.sidebar.error(f"데이터 필터링 중 오류가 발생했습니다: {str(e)}")
        
        except ValueError as e:
            st.session_state.df = None
            st.sidebar.error("날짜 형식 열의 이름을 'logTime'으로 변경하거나, 첫 번째 순서로 오도록 수정한 후 데이터를 업로드해주세요.")
        
    else:
        st.sidebar.warning("파일을 선택해주세요.")

def render_analysis_options():
    """
    분석 옵션 설정 섹션 렌더링
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 시계열 분석 옵션", help="숫자형 변수만 선택할 수 있습니다.")

    # 타겟 변수 선택
    import numpy as np
    numeric_columns = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
    target_options = numeric_columns
    
    if target_options:
        selected_target = st.sidebar.selectbox(
            "분석할 변수 선택", 
            target_options,
            index=0 if st.session_state.target is None else target_options.index(st.session_state.target)
        )
        st.session_state.target = selected_target
    else:
        st.sidebar.error("분석 가능한 숫자형 변수가 없습니다.")
        return
    
    # 테스트 데이터 비율 설정
    test_size = st.sidebar.slider(
        "테스트 데이터 비율",
        min_value=0.1,
        max_value=0.5,
        value=st.session_state.test_size,
        step=0.05
    )
    st.session_state.test_size = test_size
    
    # 시리즈 데이터 업데이트
    update_series()