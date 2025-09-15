import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from frontend.components import render_data_summary
from backend.visualization_service import visualize_timeseries

st.title("데이터 요약")

# 데이터 변경 상태 표시
if hasattr(st.session_state, 'outliers_removed') and st.session_state.outliers_removed:
    st.info("ℹ️ 이상치가 제거된 데이터를 표시하고 있습니다.")

if st.session_state.df is not None and not st.session_state.df.empty:
    # 데이터 요약 정보 표시
    render_data_summary(st.session_state.df)

    # 시계열 그래프 생성
    if hasattr(st.session_state, 'series') and st.session_state.series is not None:
        fig = visualize_timeseries(st.session_state.series, st.session_state.target)
        if fig:
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        else:
            st.error("시계열 그래프 생성 실패")
    elif hasattr(st.session_state, 'target') and st.session_state.target is not None:
        # 시계열 데이터가 없지만 target이 있으면 자동 생성
        try:
            from backend.data_service import cached_preprocess_data
            st.session_state.series = cached_preprocess_data(
                st.session_state.df,
                st.session_state.target
            )
            fig = visualize_timeseries(st.session_state.series, st.session_state.target)
            if fig:
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            else:
                st.error("시계열 그래프 생성 실패")
        except Exception as e:
            st.warning(f"시계열 데이터 생성 중 오류: {str(e)}")
    else:
        st.warning("시계열 데이터가 없습니다. 사이드바에서 분석할 변수를 선택해주세요.")

    # # 데이터 상태 정보
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     st.metric("총 데이터 수", f"{len(st.session_state.df):,}개")
    # with col2:
    #     if hasattr(st.session_state, 'outliers_removed') and st.session_state.outliers_removed:
    #         st.metric("데이터 상태", "이상치 제거됨", delta="정제됨")
    #     else:
    #         st.metric("데이터 상태", "원본 데이터")
    # with col3:
    #     if hasattr(st.session_state, 'series') and st.session_state.series is not None:
    #         st.metric("시계열 데이터 수", f"{len(st.session_state.series):,}개")

    st.markdown("**데이터 확인하기**")    
    st.dataframe(st.session_state.df, use_container_width=True, hide_index=True)

else:
    st.info("업로드된 데이터가 없습니다. 사이드바에서 CSV 파일을 업로드하세요.")