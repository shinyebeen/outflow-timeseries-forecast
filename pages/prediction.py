import pandas as pd
import streamlit as st
import json
from backend.visualization_service import visualize_forecast_comparison, visualize_metrics_comparison

# 모델 학습 결과 표시
if hasattr(st.session_state, 'model_results') and st.session_state.model_results is not None:
    st.markdown("---")
    st.subheader("📊 모델 예측 결과")
    
    # # 차분 데이터 사용 여부 표시
    # if st.session_state.use_differencing:
    #     st.success(f"ARIMA와 지수평활법 모델에만 차분 데이터(일반 차분: {st.session_state.diff_order}차, 계절 차분: {st.session_state.seasonal_diff_order}차)를 적용했습니다. LSTM과 Prophet 모델은 항상 원본 데이터를 사용합니다.")
    
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

    
    # # 최적 모델 선택
    # if st.session_state.best_model:
    #     st.success(f"### 최적 모델 (RMSE 기준): {st.session_state.best_model}")
        
    #     # 선택한 최적 모델 상세 분석
    #     if st.session_state.best_model in st.session_state.forecasts:
    #         with st.expander("최적 모델 상세 분석", expanded=True):
    #             st.subheader(f"📈 최적 모델 ({st.session_state.best_model}) 상세 분석")
                
    #             # 잔차 분석
    #             residuals_fig = visualize_residuals()
    #             if residuals_fig:
    #                 st.plotly_chart(residuals_fig, use_container_width=True, theme="streamlit")
    #             else:
    #                 st.error("잔차 분석 시각화에 실패했습니다.")
                
    #             best_forecast = st.session_state.forecasts[st.session_state.best_model]
    #             actual = st.session_state.test
                
    #             # 길이 맞춤
    #             min_len = min(len(actual), len(best_forecast))
    #             actual_values = actual.iloc[:min_len]
    #             predicted_values = best_forecast[:min_len]
                
    #             # 잔차 계산
    #             residuals = actual_values - predicted_values
                
    #             # # Ljung-Box 테스트 수행
    #             # lb_result = perform_ljung_box_test(residuals)
                
    #             # st.markdown("### 백색잡음 검정 (Ljung-Box Test)")
                
    #             # if lb_result['is_white_noise']:
    #             #     st.success(f"Ljung-Box 테스트 결과: 잔차가 백색잡음입니다 (p-값: {lb_result['p_value']:.4f})")
    #             #     st.markdown("모델이 시계열의 패턴을 적절히 포착하고 있습니다.")
    #             # else:
    #             #     st.warning(f"Ljung-Box 테스트 결과: 잔차가 백색잡음이 아닙니다 (p-값: {lb_result['p_value']:.4f})")
    #             #     st.markdown("잔차에 여전히 패턴이 남아있어 모델 개선이 필요할 수 있습니다.")
                
    #             # 잔차 자기상관 시각화
    #             acf_fig = visualize_residual_acf(residuals)
    #             if acf_fig:
    #                 st.plotly_chart(acf_fig, use_container_width=True, theme="streamlit")

            #     # 모델 설명
            #     st.markdown("### 모델 해석")
            #     if "ARIMA" in st.session_state.best_model:
            #         st.markdown("""
            #         **ARIMA 모델**은 AutoRegressive Integrated Moving Average의 약자로, 시계열 데이터의 자기회귀(AR), 차분(I), 이동평균(MA) 특성을 모델링합니다.
            #         - AR(p): 과거 p 시점의 값들이 현재 값에 영향을 미치는 정도
            #         - I(d): 정상성을 확보하기 위해 수행한 차분의 횟수
            #         - MA(q): 과거 q 시점의 오차가 현재 값에 영향을 미치는 정도
            #         """)
            #     elif "LSTM" in st.session_state.best_model:
            #         st.markdown("""
            #         **LSTM(Long Short-Term Memory) 모델**은 순환 신경망(RNN)의 일종으로, 장기 의존성 문제를 해결하기 위한 특수한 구조를 가진 딥러닝 모델입니다.
            #         - 복잡한 시계열 패턴 학습 가능
            #         - 긴 시퀀스 처리에 효과적
            #         - 비선형 관계 모델링에 강점
            #         """)
            #     elif "Prophet" in st.session_state.best_model:
            #         st.markdown("""
            #         **Prophet 모델**은 Facebook에서 개발한 시계열 예측 모델로, 다양한 계절성과 휴일 효과를 고려할 수 있습니다.
            #         - 추세, 계절성, 휴일 효과 등을 자동으로 분해
            #         - 이상값에 강건한 특성
            #         - 직관적인 파라미터 조정 가능
            #         """)
            #     elif "지수평활법" in st.session_state.best_model or "ExpSmoothing" in st.session_state.best_model:
            #         st.markdown("""
            #         **지수평활법(Exponential Smoothing) 모델**은 과거 관측치에 지수적으로 감소하는 가중치를 부여하는 예측 기법입니다.
            #         - 단순 지수평활법: 추세나 계절성이 없는 데이터에 적합
            #         - Holt 지수평활법: 추세가 있는 데이터에 적합
            #         - Holt-Winters 지수평활법: 추세와 계절성이 모두 있는 데이터에 적합
            #         """)
            #     elif "Transformer" in st.session_state.best_model:
            #         st.markdown("""
            #         **트랜스포머(Transformer) 모델**은 어텐션 메커니즘을 활용한 딥러닝 모델로, 시계열 데이터의 장거리 의존성을 효과적으로 포착합니다.
            #         - 셀프 어텐션 메커니즘으로 시퀀스 내 모든 요소 간의 관계를 동시에 고려
            #         - 병렬 처리가 가능하여 긴 시퀀스에서도 효율적인 학습
            #         - 복잡한 시간적 패턴 및 비선형 관계 학습에 강점
            #         - 다양한 시간 규모의 패턴을 동시에 학습 가능
            #         """)
                
            #     # 모델 성능 메트릭 설명
            #     st.markdown("### 성능 지표 해석")
            #     st.markdown("""
            #     **주요 성능 지표:**
            #     - **RMSE (Root Mean Squared Error)**: 예측 오차의 제곱평균의 제곱근. 낮을수록 좋음.
            #     - **MAE (Mean Absolute Error)**: 예측 오차의 절대값 평균. 낮을수록 좋음.
            #     - **MAPE (Mean Absolute Percentage Error)**: 실제값 대비 오차의 비율(%). 낮을수록 좋음.
            #     - **R² (Coefficient of Determination)**: 모델이 설명하는 분산의 비율. 1에 가까울수록 좋음.
            #     """)
                
            # # 차분 데이터 사용 시 추가 설명
            # if st.session_state.use_differencing:
            #     if "ARIMA" in st.session_state.best_model or "지수평활법" in st.session_state.best_model or "ExpSmoothing" in st.session_state.best_model:
            #         st.markdown("### 차분 데이터 사용 정보")
            #         st.markdown(f"""
            #         이 모델은 차분 데이터를 사용하여 학습되었습니다:
            #         - 일반 차분: {st.session_state.diff_order}차
            #         - 계절 차분: {st.session_state.seasonal_diff_order}차 (주기: {st.session_state.period if st.session_state.seasonal_diff_order > 0 else '없음'})
                    
            #         **차분 사용의 이점**:
            #         - 정상성 확보: 비정상 시계열을 정상화하여 모델 정확도 향상
            #         - 추세/계절성 제거: 기본 패턴을 제거하여 숨겨진 패턴 포착
            #         - 예측 안정성: 장기 예측에서 더 안정적인 결과 제공
            #         """)
            #     elif "LSTM" in st.session_state.best_model or "Prophet" in st.session_state.best_model:
            #         st.info("LSTM과 Prophet 모델은 차분 설정과 관계없이 항상 원본 데이터를 사용합니다.")
    # else:
    #     st.warning("최적 모델을 결정할 수 없습니다.")
else:
    st.info("모델 학습을 진행하여 예측 결과를 확인하세요.")