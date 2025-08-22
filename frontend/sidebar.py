import streamlit as st

import pandas as pd 
from datetime import datetime, timedelta

from backend.data_service import load_data, update_series
from frontend.components import show_memory_usage

def initialize_sidebar():
    """
    사이드바를 초기화하고 필요한 변수들을 설정합니다.
    """
    # 데이터 로드 섹션
    render_data_load_section()
    
    # 데이터가 있을 경우 분석 옵션 섹션 표시
    if st.session_state.df is not None and not st.session_state.df.empty:
        render_analysis_options()
    
    # 메모리 사용량 표시
    show_memory_usage()

def render_data_load_section():
    st.sidebar.header("시계열 예측")

    file = st.sidebar.file_uploader("upload file", 
                            type=["csv", "xlsx"],
                            help="시계열 데이터를 포함한 CSV 또는 엑셀 파일을 업로드하세요.")
    
    if file is not None:
        try:
            load_data(file)
        
        except ValueError as e:
            st.session_state.df = None
            st.sidebar.error("날짜 형식 열을 포함한 데이터를 업로드해주세요.")
        
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