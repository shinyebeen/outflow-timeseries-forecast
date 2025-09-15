import pandas as pd
import streamlit as st
import json
from backend.visualization_service import visualize_forecast_comparison, visualize_metrics_comparison

st.header("모델 예측 결과")
st.markdown(' ')

# 모델 학습 결과 표시
if hasattr(st.session_state, 'model_results') and st.session_state.model_results is not None:

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
    for model_name, metrics in st.session_state.model_results.items():
        metrics_data[model_name] = {}
        metrics_data[model_name]['rmse'] = metrics['result']['best_model']['rmse']
        metrics_data[model_name]['mae'] = metrics['result']['best_model']['mae'] 
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.T, use_container_width=True)  # 전치하여 모델별로 행 표시

            #     # 모델 설명
            #     st.markdown("### 모델 해석")
            #     if "LSTM" in st.session_state.best_model:
            #         st.markdown("""
            #         **LSTM(Long Short-Term Memory) 모델**은 순환 신경망(RNN)의 일종으로, 장기 의존성 문제를 해결하기 위한 특수한 구조를 가진 딥러닝 모델입니다.
            #         - 복잡한 시계열 패턴 학습 가능
            #         - 긴 시퀀스 처리에 효과적
            #         - 비선형 관계 모델링에 강점
            #         """)
            #    elif "XGBoost" in st.session_state.best_model:
            #         st.markdown("""
            #         **LSTM(Long Short-Term Memory) 모델**은 순환 신경망(RNN)의 일종으로, 장기 의존성 문제를 해결하기 위한 특수한 구조를 가진 딥러닝 모델입니다.
            #         - 복잡한 시계열 패턴 학습 가능
            #         - 긴 시퀀스 처리에 효과적
            #         - 비선형 관계 모델링에 강점
            #         """)
            #     # 모델 성능 메트릭 설명
            #     st.markdown("### 성능 지표 해석")
            #     st.markdown("""
            #     **주요 성능 지표:**
            #     - **RMSE (Root Mean Squared Error)**: 예측 오차의 제곱평균의 제곱근. 낮을수록 좋음.
            #     - **MAE (Mean Absolute Error)**: 예측 오차의 절대값 평균. 낮을수록 좋음.
            #     """)
else:
    st.info("모델 학습을 진행하여 예측 결과를 확인하세요.")