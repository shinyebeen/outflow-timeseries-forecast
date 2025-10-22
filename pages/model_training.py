"""
모델 학습 및 예측 페이지
"""
import streamlit as st
import pandas as pd

from frontend.session_state import reset_model_results
from frontend.components import render_model_selector
from backend.model_service import get_model_factory, run_complete_optimization
from backend.data_service import (
    prepare_train_test_data
)
from backend.visualization_service import visualize_forecast_comparison, visualize_metrics_comparison

# 페이지 제목
st.header("모델 학습 및 예측")
st.markdown("시계열 데이터에 대한 다양한 예측 모델을 학습하고 성능을 비교합니다.")
st.markdown(' ')

# 데이터 및 시계열 정보 확인
if st.session_state.df is None:
    st.warning("데이터가 로드되지 않았습니다. 사이드바에서 데이터를 로드해주세요.")
    st.stop()
elif st.session_state.series is None:
    st.warning("시계열 데이터가 생성되지 않았습니다. 사이드바에서 분석 변수와 측정소를 선택해주세요.")
    st.stop()

# 모델 팩토리 가져오기
model_factory = get_model_factory()

if model_factory is None:
    st.error("모델 팩토리 로드에 실패했습니다.")
    st.stop()

# 모델 선택기 렌더링
selected_models, strategy, trial = render_model_selector(model_factory)

# 모델 학습 버튼
results = None

col1, col2 = st.columns([3, 1])
with col1:
    if st.button("모델 학습 및 예측 시작", use_container_width=True, type="primary"):
        if not selected_models:
            st.warning("최소한 하나의 모델을 선택해주세요.")
        else:
            # 훈련/테스트 데이터 준비
            if prepare_train_test_data():
                with st.spinner("모델을 학습 중입니다..."):
                    st.session_state.selected_models = selected_models
                    st.session_state.strategy = strategy

                    # 모델 학습 실행
                    final_recommendation, results = run_complete_optimization(selected_models, strategy, trial)
                    st.success("모델 학습 완료!")
            else:
                st.error("훈련/테스트 데이터 준비 중 오류가 발생했습니다.")
            
            if results is not None:
                st.session_state.file_data = results

with col2:
    if st.button("결과 초기화", use_container_width=True):
        reset_model_results()
        st.rerun()

# 모델 학습 결과 표시
if hasattr(st.session_state, 'model_results') and st.session_state.model_results is not None:

    with open("best_lstm_model.h5", "rb") as f:
        st.download_button(
            label="LSTM 모델 다운로드",
            data=f,
            file_name="best_lstm_model.h5",
            mime="application/octet-stream"
        )

    with open("best_xgb_model.pkl", "rb") as f:
        st.download_button(
            label="XGBoost 모델 다운로드",
            data=f,
            file_name="best_xgb_model.pkl",
            mime="application/octet-stream"
        )

    # scaler_X 다운로드
    with open("scaler_X.pkl", "rb") as f:
        st.download_button(
            label="Scaler_X 다운로드",
            data=f,
            file_name="scaler_X.pkl",
            mime="application/octet-stream"
        )
        
    # scaler_y 다운로드
    with open("scaler_y.pkl", "rb") as f:
        st.download_button(
            label="Scaler_y 다운로드",
            data=f,
            file_name="scaler_y.pkl",
            mime="application/octet-stream"
        )

    if st.session_state.file_data is not None:
        st.download_button(
                            label="Download JSON",
                            file_name="model_result.json",
                            mime="application/json",
                            data=st.session_state.file_data,
                            help="모델 학습 결과를 JSON 파일로 다운로드합니다.",)

    st.header("모델 예측 결과")
    
    if 'best_model' in st.session_state.model_results:
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        best_model = st.session_state.model_results['best_model']
        outflow_avg = st.session_state.df[st.session_state.target].mean()
        best_mae = st.session_state.model_results[best_model]['mae']

        metric_col1.metric(label="우수 모델", 
                        value = best_model,
                        border=True)
        
        metric_col2.metric(label="평균유량(m³/h)", 
                        value = f'{outflow_avg:.2f}',
                        help="평균 유량을 표시합니다.",
                        border=True)
        
        metric_col3.metric(label="MAE", 
                        value = f'{best_mae:.2f}',
                        help="LSTM 모델의 평균 절대 오차(MAE)를 표시합니다.",
                        border=True)
        
        metric_col4.metric(label="오차율(%)", 
                        value = f'{(best_mae / outflow_avg * 100):.2f}%',
                        help="평균 유량 대비 MAE의 비율을 백분율로 표시합니다.",
                        border=True)

    # 예측 결과 비교 시각화
    comparison_fig = visualize_forecast_comparison()
    if comparison_fig:
        st.plotly_chart(comparison_fig, use_container_width=True, theme="streamlit")
    else:
        st.error("예측 결과 시각화에 실패했습니다.")
    
    # 메트릭 비교 시각화
    st.subheader("📈 모델 성능 비교")
    metrics_fig = visualize_metrics_comparison()
    if metrics_fig:
        st.plotly_chart(metrics_fig, use_container_width=True, theme="streamlit")
    else:
        st.error("성능 메트릭 시각화에 실패했습니다.")
    
    # 메트릭 표 표시
    st.subheader("📋 모델 성능 메트릭")
    
    # 메트릭 데이터프레임 생성
    metrics_data = {}
    print(st.session_state.model_results)
    for model_name, metrics in list(st.session_state.model_results.items()):
        if model_name == 'best_model':
            continue
        metrics_data[model_name] = {}
        metrics_data[model_name]['rmse'] = metrics['result']['best_model']['rmse']
        metrics_data[model_name]['mae'] = metrics['result']['best_model']['mae'] 
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.T, use_container_width=True)  # 전치하여 모델별로 행 표시

else:
    st.info("모델 학습을 진행하여 예측 결과를 확인하세요.")
