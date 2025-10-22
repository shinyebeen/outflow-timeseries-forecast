"""
시계열 데이터 시각화 모듈
"""

import json
import streamlit as st
import numpy as np
from utils.visualizer import (cached_plot_timeseries, 
                              cached_boxplot, 
                              cached_plot_acf_pacf,
                              cached_plot_fft,
                              cached_plot_decomposition,
                              cached_plot_forecast_comparison,
                              cached_plot_metrics_comparison,
                              cached_plot_residuals,
                              cached_plot_residual_acf
                              )

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

def visualize_acf_pacf(acf_values: np.ndarray,
                       pacf_values: np.ndarray,
                       lags: int = 10):
    """
    ACF/PACF 시각화
    
    Returns:
        plotly.graph_objects.Figure: ACF/PACF 그래프
    """
    if st.session_state.acf_values is not None and st.session_state.pacf_values is not None:
        acf_pacf_fig = cached_plot_acf_pacf(st.session_state.acf_values, st.session_state.pacf_values, lags)
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

def visualize_forecast_comparison(train_data=None, test_data=None, forecasts=None):
    """
    예측 결과 비교 시각화
    
    Args:
        train_data: 훈련 데이터 (기본값: None, 세션 상태 사용)
        test_data: 테스트 데이터 (기본값: None, 세션 상태 사용)
        forecasts: 예측 결과 딕셔너리 (기본값: None, 세션 상태 사용)
    
    Returns:
        plotly.graph_objects.Figure: 예측 비교 그래프
    """
    # 매개변수가 없으면 세션 상태 사용
    if train_data is None:
        train_data = st.session_state.train
    
    if test_data is None:
        test_data = st.session_state.test
    
    forecasts = forecasts if forecasts is not None else st.session_state.model_results
    
    # 데이터 유효성 검사
    if train_data is None or test_data is None:
        st.error("시각화에 필요한 훈련/테스트 데이터가 없습니다.")
        
        # 디버깅 정보 표시
        st.write("### 세션 상태 확인:")
        st.write(f"train: {'존재함' if hasattr(st.session_state, 'train') and st.session_state.train is not None else '없음'}")
        st.write(f"test: {'존재함' if hasattr(st.session_state, 'test') and st.session_state.test is not None else '없음'}")
    
    if not forecasts:
        st.error("시각화할 예측 결과가 없습니다.")
        return None
    
    # 유효한 예측 결과만 필터링
    valid_forecasts = {}
    for model_name, forecast in list(forecasts.items()):

        if model_name == 'best_model':
            continue
        
        # 각 모델별 best_model 결과 확인
        if forecast['result']['best_model'] is not None:
            results = forecast['result']['best_model']

            if len(results['test_actual']) != len(results['test_predictions']):
                min_len = min(results['test_actual'], len(results['test_predictions']))
                if min_len > 0:
                    st.warning(f"{model_name} 모델의 예측 길이({len(results['test_predictions'])})가 테스트 데이터 길이({len(results['test_actual'])})와 다릅니다. 최소 길이({min_len})로 조정합니다.")
                    valid_forecasts[model_name] = results['test_predictions'][:min_len]
                else:
                    st.warning(f"{model_name} 모델의 예측 결과를 시각화에서 제외합니다.")
                    continue
            else:
                valid_forecasts[model_name] = results['test_predictions']
                
    if not valid_forecasts:
        st.error("유효한 예측 결과가 없어 시각화할 수 없습니다.")
        return None
    
    try:
        comparison_fig = cached_plot_forecast_comparison(
            results['test_actual'], # 어차피 forecast_horizon 길이가 같음 
            valid_forecasts
        )

        return comparison_fig
    except Exception as e:
        st.error(f"예측 비교 시각화 중 오류 발생: {str(e)}")
        return None
 
def visualize_metrics_comparison(metrics=None):
    """
    성능 메트릭 비교 시각화
    
    Args:
        metrics: 메트릭 딕셔너리 (기본값: None, 세션 상태 사용)
    
    Returns:
        plotly.graph_objects.Figure: 메트릭 비교 그래프
    """
    metrics = metrics if metrics is not None else st.session_state.model_results
    metrics_ = {}

    for model_name, results in list(metrics.items()):

        if model_name == 'best_model':
            continue
        
        if 'result' in results and 'best_model' in results['result']:
            metrics_[model_name] = {}
            metrics_[model_name]['rmse'] = results['result']['best_model']['rmse']
            metrics_[model_name]['mae'] = results['result']['best_model']['mae']
            
        else:
            st.warning(f"{model_name} 모델의 메트릭 정보가 없습니다. 시각화에서 제외합니다.")
            del metrics[model_name]
    
    metrics_json = json.dumps(metrics_, sort_keys=True)
    
    if metrics_json:
        metrics_fig = cached_plot_metrics_comparison(metrics_json)
        return metrics_fig
    return None

def visualize_residuals(model_name=None):

    """
    잔차 분석 시각화
    
    Args:
        model_name: 모델 이름 (기본값: None, 최적 모델 사용)
        
    Returns:
        plotly.graph_objects.Figure: 잔차 분석 그래프
    """
    if model_name is None and st.session_state.best_model:
        model_name = st.session_state.best_model
        
    if (st.session_state.test is not None and 
        model_name in st.session_state.forecasts):
        best_forecast = st.session_state.forecasts[model_name]
        residuals_fig = cached_plot_residuals(st.session_state.test, best_forecast)
        return residuals_fig
    
def visualize_residual_acf(residuals, max_lags=20):
    """
    모델 잔차의 자기상관함수 시각화
    
    Args:
        residuals: 모델 잔차
        max_lags: 최대 시차
        
    Returns:
        plotly.graph_objects.Figure: 잔차 ACF 그래프
    """
    fig = cached_plot_residual_acf(residuals, max_lags)
    return fig