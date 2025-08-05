"""
시계열 데이터 시각화 모듈
"""

import streamlit as st
from utils.visualizer import (cached_plot_timeseries, 
                              cached_boxplot, 
                              cached_plot_acf_pacf,
                              cached_plot_fft,
                              cached_plot_decomposition)

def visualize_timeseries(series, target):
    """
    시계열 데이터를 시각화합니다.
    :return: Plotly Figure 객체
    """
    if series is not None:
        fig = cached_plot_timeseries(
            data=series,
            title=f"{target} 데이터 시각화",
            xlabel="측정 시간",
            ylabel=target
        )
        return fig
    
    return None

def visualize_boxplot():
    """
    이상치를 시각화합니다.
    :return: Plotly Figure 객체
    """
    if st.session_state.series is not None:
        # series를 직접 전달하여 캐싱이 작동하도록 함
        series = st.session_state.series
        target = st.session_state.target
        
        fig = cached_boxplot(
            data=series,  # session_state 대신 직접 값을 전달
            title=f"{target} 박스 플롯",
            xlabel=target,
            ylabel=""
        )
        return fig
    return None

def visualize_acf_pacf():
    """
    ACF/PACF 시각화
    
    Returns:
        plotly.graph_objects.Figure: ACF/PACF 그래프
    """
    if st.session_state.acf_values is not None and st.session_state.pacf_values is not None:
        acf_pacf_fig = cached_plot_acf_pacf(st.session_state.acf_values, st.session_state.pacf_values)
        return acf_pacf_fig
    return None

def visualize_fft():
    if st.session_state.fft_result is not None:
        fft_fig = cached_plot_fft(st.session_state.fft_result)
        return fft_fig
    
def visualize_decomposition():
    if st.session_state.decomposition is not None:
        decomposition_fig = cached_plot_decomposition(st.session_state.decomposition)
        return decomposition_fig