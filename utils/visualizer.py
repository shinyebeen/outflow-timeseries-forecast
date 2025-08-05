import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from matplotlib import font_manager, rc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.singleton import Singleton

class TimeSeriesVisualizer(metaclass=Singleton):

    """
    시계열 데이터를 시각화하는 클래스
    """
    
    def __init__(self):
        """
        시각화 클래스 초기화 메소드
        """
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 12
        plt.rcParams['figure.figsize'] = (20, 6)

        # 한글 폰트 설정
        font_path = 'C:/outflow/styles/NanumGothic.ttf'
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False

    def plot_timeseries(self, 
                        data: pd.Series, 
                        title: str ='시계열 그래프 (Time Series Plot)', 
                        xlabel: str ='측정 시간', 
                        ylabel: str ='값', 
                        color='#1f77b4') -> go.Figure:
        """
        기본 시계열 그래프
        """
        fig = px.line(
            x = data.index,
            y = data.values,
            labels={'x': xlabel, 'y': ylabel},
            title=title
        )

        fig.update_layout(
            title=title,
            title_font_size=14,
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title=xlabel,
            yaxis_title=ylabel
        )

        fig.update_xaxes(
            tickformat='%Y-%m-%d'
        )

        return fig
    
    def plot_boxplot(self, 
                        data: pd.Series, 
                        title: str ='박스플롯 (Box Plot)', 
                        xlabel: str = st.session_state.target, 
                        ylabel: str ='값', 
                        ) -> go.Figure:
        """
        박스플롯 시각화
        """
        # DataFrame으로 변환하여 px.box 사용
        df = data.to_frame()
        
        fig = px.box(
            df,
            y=st.session_state.target,
            title=title
        )

        fig.update_layout(
            title=title,
            title_font_size=14,
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            showlegend=False
        )

        return fig
    
    def plot_acf_pacf(self,
                    acf_values: np.ndarray,
                    pacf_values: np.ndarray,
                    lags: int = 40,
                    **kwargs) -> go.Figure:
        """
        ACF 및 PACF 플롯을 생성합니다 (Plotly 버전).
        
        Args:
            acf_values: ACF 값
            pacf_values: PACF 값
            lags: 지연값 수
            
        Returns:
            Plotly Figure 객체
        """
        # 서브플롯 생성
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                '자기상관 함수 (Autocorrelation Function)',
                '부분 자기상관 함수 (Partial Autocorrelation Function)'
            )
        )
        
        # x축 값 (lags)
        x = list(range(len(acf_values)))
        
        # 신뢰 구간 계산 (95%)
        confidence = 1.96 / np.sqrt(len(acf_values))
        
        # ACF 플롯 - stem 효과 (마커와 선 조합)
        for i in range(len(acf_values)):
            fig.add_trace(
                go.Scatter(
                    x=[i, i], 
                    y=[0, acf_values[i]], 
                    mode='lines',
                    line=dict(color='blue', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # ACF 마커
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=acf_values, 
                mode='markers',
                marker=dict(color='blue', size=8),
                name='ACF'
            ),
            row=1, col=1
        )
        
        # 신뢰 구간 추가
        fig.add_trace(
            go.Scatter(
                x=[0, len(acf_values)-1],
                y=[confidence, confidence],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0, len(acf_values)-1],
                y=[-confidence, -confidence],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # PACF 플롯 - stem 효과
        for i in range(len(pacf_values)):
            fig.add_trace(
                go.Scatter(
                    x=[i, i], 
                    y=[0, pacf_values[i]], 
                    mode='lines',
                    line=dict(color='blue', width=1),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # PACF 마커
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=pacf_values, 
                mode='markers',
                marker=dict(color='blue', size=8),
                name='PACF'
            ),
            row=1, col=2
        )
        
        # PACF 신뢰 구간
        fig.add_trace(
            go.Scatter(
                x=[0, len(pacf_values)-1],
                y=[confidence, confidence],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0, len(pacf_values)-1],
                y=[-confidence, -confidence],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
        )
        
        # x축 및 y축 레이블
        fig.update_xaxes(title_text='지연 (Lag)', row=1, col=1)
        fig.update_xaxes(title_text='지연 (Lag)', row=1, col=2)
        fig.update_yaxes(title_text='상관도 (Correlation)', row=1, col=1)
        fig.update_yaxes(title_text='상관도 (Correlation)', row=1, col=2)
        
        return fig
    
    def plot_fft(self, 
                 fft_result: dict):
        """
        주파수 분석 결과를 Plotly로 시각화하는 함수
        
        Parameters:
        - freqs: 주파수 배열
        - power: 파워 스펙트럼 배열
        - top_freq_idx: 주요 주파수 인덱스 리스트
        - eng_col_name: 분석 대상 컬럼 이름 (그래프 제목용)
        
        Returns:
        - Plotly Figure 객체
        """
        quarter_len = len(fft_result['freqs']) // 4

        # subplot 생성
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            'Power Spectrum (Low Frequency)',
            'Power Spectrum (Log Scale)'
        ])

        # 선형 스케일 그래프 (왼쪽)
        fig.add_trace(
            go.Scatter(x=fft_result['freqs'][:quarter_len], y=fft_result['freqs'][:quarter_len],
                    mode='lines', name='Linear Power'),
            row=1, col=1
        )

        # 로그 스케일 그래프 (오른쪽)
        fig.add_trace(
            go.Scatter(x=fft_result['freqs'][:quarter_len], y=fft_result['power'][:quarter_len],
                    mode='lines', name='Log Power', yaxis='y2'),
            row=1, col=2
        )

        # 로그 스케일을 수동으로 설정
        fig.update_yaxes(type='log', row=1, col=2)

        # 공통 레이아웃 설정
        fig.update_layout(
            title=f'{st.session_state.target} - Frequency Analysis',
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False
        )

        fig.update_xaxes(title_text='Frequency', row=1, col=1)
        fig.update_yaxes(title_text='Power', row=1, col=1)

        fig.update_xaxes(title_text='Frequency', row=1, col=2)
        fig.update_yaxes(title_text='Power (log)', row=1, col=2)

        return fig

@st.cache_data(ttl=3600)
def cached_plot_timeseries(data, title, xlabel, ylabel, color='#1f77b4'):
    """
    시계열 그래프 캐싱
    """ 
    viz = TimeSeriesVisualizer()
    return viz.plot_timeseries(data, title=title, xlabel=xlabel, ylabel=ylabel, color=color) 

@st.cache_data(ttl=3600)
def cached_boxplot(data, title, xlabel, ylabel):
    """
    박스플롯 캐싱
    """
    viz = TimeSeriesVisualizer()
    return viz.plot_boxplot(data, title=title, xlabel=xlabel, ylabel=ylabel)

@st.cache_data(ttl=3600)
def cached_plot_acf_pacf(acf_values, pacf_values):
    """ACF/PACF 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_acf_pacf(acf_values, pacf_values)

@st.cache_data(ttl=3600)
def cached_plot_fft(fft_result):
    """TTF 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_fft(fft_result)