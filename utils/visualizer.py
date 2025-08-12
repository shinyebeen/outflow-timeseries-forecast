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
    




    def plot_decomposition(self, 
                           decomposition,
                           color: str = '#1f77b4')  -> go.Figure:
        """
        계절성 분해 결과를 Plotly로 시각화하는 함수
        
        Parameters:
        -----------
        decomposition : statsmodels.tsa.seasonal.DecomposeResult
            seasonal_decompose() 결과 객체
        target_name : str
            대상 변수명 (기본값: 'Target Variable')
        title : str
            전체 제목 (기본값: None - 자동 생성)
        height : int
            전체 그래프 높이 (기본값: 800)
        color : str
            선 색상 (기본값: '#1f77b4')
        
        Returns:
        --------
        go.Figure : Plotly Figure 객체
        """
        
        # 리스트를 다시 pandas.Series로 복원
        index = pd.to_datetime(decomposition['index'])

        observed = pd.Series(decomposition['observed'], index=index)
        trend = pd.Series(decomposition['trend'], index=index)
        seasonal = pd.Series(decomposition['seasonal'], index=index)
        resid = pd.Series(decomposition['resid'], index=index)

        # 4행 1열 subplot 생성
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])

        fig.add_trace(go.Scatter(x=index, y=observed, mode='lines', name='Observed'),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=index, y=trend, mode='lines', name='Trend'),
                    row=2, col=1)
        fig.add_trace(go.Scatter(x=index, y=seasonal, mode='lines', name='Seasonal'),
                    row=3, col=1)
        fig.add_trace(go.Scatter(x=index, y=resid, mode='lines', name='Residual'),
                    row=4, col=1)

        fig.update_layout(height=800, showlegend=False, margin=dict(t=40, b=40))
        return fig
    
    def plot_differencing_comparison(
            self, 
            original_series: pd.Series, 
            differenced_series: pd.Series,
            title: str = "차분 비교 (Differencing Comparison)",
            **kwargs
        ) -> go.Figure:
        """
        원본 시계열과 차분된 시계열 비교 시각화 (Plotly 버전).
        
        Args:
            original_series: 원본 시계열 데이터
            differenced_series: 차분된 시계열 데이터
            title: 그래프 제목
            
        Returns:
            Plotly Figure 객체
        """
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                '원본 시계열 (Original Time Series)',
                '차분된 시계열 (Differenced Time Series)'
            ),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # 원본 시계열 그래프
        fig.add_trace(
            go.Scatter(
                x=original_series.index,
                y=original_series.values,
                mode='lines',
                name='원본 데이터',
                line=dict(color='blue', width=1.5)
            ),
            row=1, col=1
        )
        
        # 차분된 시계열 그래프
        fig.add_trace(
            go.Scatter(
                x=differenced_series.index,
                y=differenced_series.values,
                mode='lines',
                name='차분된 데이터',
                line=dict(color='red', width=1.5)
            ),
            row=2, col=1
        )
        
        # 0 라인 추가 (차분 그래프에만)
        fig.add_trace(
            go.Scatter(
                x=[differenced_series.index.min(), differenced_series.index.max()],
                y=[0, 0],
                mode='lines',
                line=dict(color='black', width=1, dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 스타일 설정
        fig.update_layout(
            title=title,
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # x축 및 y축 레이블
        fig.update_xaxes(title_text='날짜 (Date)', row=2, col=1)
        fig.update_yaxes(title_text='값 (Value)', row=1, col=1)
        fig.update_yaxes(title_text='차분값 (Differenced Value)', row=2, col=1)
        
        # 날짜 형식 지정
        fig.update_xaxes(
            tickformat="%Y-%m-%d",
            row=2, col=1
        )
        
        return fig
    
    def plot_forecast_comparison(self,
                            train: pd.Series,
                            test: pd.Series,
                            forecasts: dict[str, np.ndarray],
                            **kwargs) -> go.Figure:
        """
        여러 모델의 예측 결과를 비교하여 시각화합니다 (Plotly 버전).
        
        Args:
            train: 훈련 데이터
            test: 테스트 데이터
            forecasts: 모델별 예측값 딕셔너리
            
        Returns:
            Plotly Figure 객체
        """
        # 그래프 생성
        fig = go.Figure()
        
        # 훈련 데이터
        fig.add_trace(
            go.Scatter(
                x=train.index,
                y=train.values,
                mode='lines',
                name='Training Data',
                line=dict(color='blue', width=2)
            )
        )
        
        # 테스트 데이터
        fig.add_trace(
            go.Scatter(
                x=test.index,
                y=test.values,
                mode='lines',
                name='Actual Test Data',
                line=dict(color='green', width=2)
            )
        )
        
        # 각 모델의 예측
        colors = ['red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            fig.add_trace(
                go.Scatter(
                    x=test.index,
                    y=forecast,
                    mode='lines',
                    name=f'{model_name} Forecast',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                )
            )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title='예측 비교 (Forecast Comparison)',
            xaxis_title='날짜 (Date)',
            yaxis_title='값 (Value)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=10, r=10, t=50, b=10),
            height=500,
        )
        
        # 날짜 형식 지정
        fig.update_xaxes(
            tickformat="%Y-%m-%d"
        )
        
        return fig
    
    def plot_metrics_comparison(self, metrics: dict[str, dict[str, float]]) -> go.Figure:
        """
        여러 모델의 성능 지표를 비교하여 시각화합니다 (Plotly 버전).
        
        Args:
            metrics: 모델별 성능 지표 딕셔너리
            
        Returns:
            Plotly Figure 객체
        """
        # 데이터 준비
        models = list(metrics.keys())
        metric_names = ['RMSE', 'MAE', 'R^2', 'MAPE']
        
        # 모든 모델에 있는 지표만 선택
        available_metrics = set.intersection(*[set(m.keys()) for m in metrics.values()])
        metric_names = [m for m in metric_names if m in available_metrics]
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=len(metric_names), 
            cols=1,
            subplot_titles=[f'{metric} Comparison' for metric in metric_names]
        )
        
        # 각 메트릭별 바 차트 생성
        for i, metric in enumerate(metric_names):
            values = [metrics[model][metric] for model in models]
            
            # 특별히 R^2는 높을수록 좋음, 나머지는 낮을수록 좋음
            if metric == 'R^2':
                colors = ['green' if v > 0 else 'red' for v in values]
                # 내림차순 정렬 (높을수록 좋음)
                sorted_idx = np.argsort(values)[::-1]
            else:
                # 값을 정규화하여 색상 결정 (낮을수록 좋음)
                max_val = max(values) if values else 1
                colors = ['lightcoral' if v/max_val > 0.7 else 'lightgreen' for v in values]
                # 오름차순 정렬 (낮을수록 좋음)
                sorted_idx = np.argsort(values)
            
            # 정렬된 모델 및 값
            sorted_models = [models[i] for i in sorted_idx]
            sorted_values = [values[i] for i in sorted_idx]
            sorted_colors = [colors[i] for i in sorted_idx]
            
            # 바 차트 추가
            fig.add_trace(
                go.Bar(
                    x=sorted_models,
                    y=sorted_values,
                    text=[f'{v:.4f}' for v in sorted_values],
                    textposition='outside',
                    marker_color=sorted_colors,
                    name=metric
                ),
                row=i+1, col=1
            )
        
        # 레이아웃 업데이트
        fig.update_layout(
            height=300 * len(metric_names),
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # y축 타이틀 업데이트
        for i, metric in enumerate(metric_names):
            fig.update_yaxes(title_text=metric, row=i+1, col=1)
        
        return fig

    def plot_residuals(
            self,
            actual: pd.Series,
            predicted: np.ndarray,
            title: str = "Residual Analysis",
            **kwargs
        ) -> go.Figure:
        """
        잔차 분석 플롯을 생성합니다 (Plotly 버전).
        
        Args:
            actual: 실제 값
            predicted: 예측 값
            title: 그래프 제목
            
        Returns:
            Plotly Figure 객체
        """
        # 길이 맞춤
        min_len = min(len(actual), len(predicted))
        actual_values = actual.iloc[:min_len].values
        predicted_values = predicted[:min_len]
        
        # 잔차 계산
        residuals = actual_values - predicted_values
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '잔차 시계열 (Residuals Over Time)',
                '잔차 분포 (Residual Distribution)',
                '정규 Q-Q 플롯 (Normal Q-Q Plot)',
                '잔차 vs 예측값 (Residuals vs Predicted)'
            )
        )
        
        # 1. 잔차 시계열 플롯
        fig.add_trace(
            go.Scatter(
                x=actual.index[:min_len],
                y=residuals,
                mode='lines',
                name='Residuals'
            ),
            row=1, col=1
        )
        
        # 0 라인 추가
        fig.add_trace(
            go.Scatter(
                x=[actual.index[0], actual.index[min_len-1]],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. 잔차 히스토그램
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=20,
                marker_line_color='black',
                marker_line_width=1,
                opacity=0.7,
                name='Residual Distribution'
            ),
            row=1, col=2
        )
        
        osm, osr = stats.probplot(residuals, dist="norm", fit=False)
        
        fig.add_trace(
            go.Scatter(
                x=osm,
                y=osr,
                mode='markers',
                marker=dict(color='blue', size=6),
                name='Q-Q Plot'
            ),
            row=2, col=1
        )
        
        # 이론적인 정규분포 라인
        z = np.polyfit(osm, osr, 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=osm,
                y=p(osm),
                mode='lines',
                line=dict(color='red', width=1),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. 잔차 vs 예측값
        fig.add_trace(
            go.Scatter(
                x=predicted_values,
                y=residuals,
                mode='markers',
                marker=dict(color='blue', size=6),
                name='Residuals vs Predicted'
            ),
            row=2, col=2
        )
        
        # 0 라인 추가
        fig.add_trace(
            go.Scatter(
                x=[min(predicted_values), max(predicted_values)],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # 축 레이블 업데이트
        fig.update_xaxes(title_text='날짜 (Date)', row=1, col=1)
        fig.update_xaxes(title_text='잔차 (Residual)', row=1, col=2)
        fig.update_xaxes(title_text='이론적 분위수 (Theoretical Quantiles)', row=2, col=1)
        fig.update_xaxes(title_text='예측값 (Predicted Values)', row=2, col=2)
        
        fig.update_yaxes(title_text='잔차 (Residual)', row=1, col=1)
        fig.update_yaxes(title_text='빈도 (Frequency)', row=1, col=2)
        fig.update_yaxes(title_text='정렬된 값 (Ordered Values)', row=2, col=1)
        fig.update_yaxes(title_text='잔차 (Residuals)', row=2, col=2)
        
        return fig
    
    def plot_residual_acf(
            self,
            residuals: np.ndarray,
            max_lags: int = 20,
            title: str = "잔차의 자기상관함수 (ACF)",
            **kwargs
        ) -> go.Figure:
        """
        모델 잔차의 자기상관함수(ACF)를 시각화합니다.
        
        Args:
            residuals: 모델 잔차
            max_lags: 최대 시차
            title: 그래프 제목
            
        Returns:
            plotly Figure 객체
        """
        from statsmodels.tsa.stattools import acf
        
        # ACF 값 계산
        acf_values = acf(residuals, nlags=max_lags, fft=False)
        lags = list(range(len(acf_values)))  # range 객체를 list로 변환
        
        # 신뢰 구간 계산 (95%)
        confidence = 1.96 / np.sqrt(len(residuals))
        
        # ACF 시각화
        fig = go.Figure()
        
        # ACF 막대 그래프
        fig.add_trace(go.Bar(
            x=lags,
            y=acf_values,
            name='ACF',
            marker_color='blue'
        ))
        
        # 신뢰 구간 선
        fig.add_trace(go.Scatter(
            x=[0, max(lags)],
            y=[confidence, confidence],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='95% 신뢰구간'
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, max(lags)],
            y=[-confidence, -confidence],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False
        ))
        
        # 레이아웃 설정
        fig.update_layout(
            title=title,
            xaxis_title='시차(Lag)',
            yaxis_title='자기상관계수',
            height=400
        )
        
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

@st.cache_data(ttl=3600)
def cached_plot_decomposition(decomposition):
    """계절성 분해 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_decomposition(decomposition)

@st.cache_data(ttl=3600)
def cached_plot_differencing_comparison(original_series, differenced_series, title="차분 비교 (Differencing Comparison)"):
    """차분 비교 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_differencing_comparison(original_series, differenced_series, title=title)

@st.cache_data(ttl=3600)
def cached_plot_forecast_comparison(train, test, forecasts):
    """예측 비교 그래프 캐싱"""
    try:
        # numpy 배열을 Series로 변환 (인덱스 보장)
        for model_name, forecast in forecasts.items():
            if isinstance(forecast, np.ndarray):
                # 길이 맞춤
                min_len = min(len(test), len(forecast))
                forecasts[model_name] = pd.Series(forecast[:min_len], index=test.index[:min_len])
                
        viz = TimeSeriesVisualizer()
        return viz.plot_forecast_comparison(train, test, forecasts)
    except Exception as e:
        st.error(f"예측 비교 그래프 생성 중 오류: {str(e)}")
        import traceback
        st.error(f"상세 오류: {traceback.format_exc()}")
        return None
    
@st.cache_data(ttl=3600)
def cached_plot_metrics_comparison(metrics):
    """메트릭 비교 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_metrics_comparison(metrics)

@st.cache_data(ttl=3600)
def cached_plot_residuals(actual, predicted, title="Residual Analysis"):
    """잔차 분석 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_residuals(actual, predicted, title=title)

@st.cache_data(ttl=3600)
def cached_plot_residual_acf(residuals, max_lags=20, title="잔차의 자기상관함수 (ACF)"):
    """잔차 ACF 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_residual_acf(residuals, max_lags=max_lags, title=title)
