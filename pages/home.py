import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from frontend.components import render_data_summary
from backend.visualization_service import visualize_timeseries

st.title("데이터 요약")

if st.session_state.df is not None and not st.session_state.df.empty:
    # 데이터 요약 정보 표시
    render_data_summary(st.session_state.df)

    # 기본통계량
    stats_df = st.session_state.series.describe().to_frame().T
    st.dataframe(stats_df, use_container_width=True)

    fig = visualize_timeseries(st.session_state.series, st.session_state.target)
    if fig:
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    else:
        st.error("시계열 그래프 생성 실패")


else:
    st.info("업로드된 데이터가 없습니다. 사이드바에서 CSV 파일을 업로드하세요.")