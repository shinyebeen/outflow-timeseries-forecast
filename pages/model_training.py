"""
모델 학습 및 예측 페이지
"""
import streamlit as st
import pandas as pd

from frontend.session_state import reset_model_results
from frontend.components import render_model_selector
from backend.model_service import get_model_factory, run_complete_optimization
from backend.data_service import (
    prepare_train_test_data,
    prepare_differenced_train_test_data,
    # perform_ljung_box_test
)
from backend.visualization_service import (
    visualize_forecast_comparison, 
    visualize_metrics_comparison, 
    visualize_residuals,
    visualize_residual_acf
)

# 페이지 제목
st.title("🤖 모델 학습 및 예측")
st.markdown("시계열 데이터에 대한 다양한 예측 모델을 학습하고 성능을 비교합니다.")

# 데이터 및 시계열 정보 확인
if st.session_state.df is None:
    st.warning("데이터가 로드되지 않았습니다. 사이드바에서 데이터를 로드해주세요.")
    st.stop()
elif st.session_state.series is None:
    st.warning("시계열 데이터가 생성되지 않았습니다. 사이드바에서 분석 변수와 측정소를 선택해주세요.")
    st.stop()

# 모델 학습 섹션
st.markdown("## 모델 설정 및 학습")

# # 차분 데이터 정보 표시
# if st.session_state.differenced_series is not None:
#     diff_info = f"일반 차분: {st.session_state.diff_order}차"
#     if st.session_state.seasonal_diff_order > 0:
#         diff_info += f", 계절 차분: {st.session_state.seasonal_diff_order}차 (주기: {st.session_state.period})"
    
#     diff_col1, diff_col2 = st.columns([3, 1])
    
#     with diff_col1:
#         # 차분 데이터 사용 여부
#         st.checkbox(
#             "차분 데이터 사용",
#             value=st.session_state.use_differencing,
#             key="use_differencing_model",
#             help=f"차분 데이터({diff_info})를 사용하여 모델을 학습합니다.",
#             on_change=lambda: setattr(st.session_state, 'use_differencing', st.session_state.use_differencing_model)
#         )
        
#         if st.session_state.use_differencing:
#             st.success(f"차분 데이터({diff_info})를 사용하여 모델을 학습합니다.")
            
#             # 차분 데이터를 사용하는 경우 확인
#             if st.session_state.diff_train is None or st.session_state.diff_test is None:
#                 with st.spinner("차분 데이터 준비 중..."):
#                     prepare_differenced_train_test_data()
    
#     with diff_col2:
#         # 차분 페이지로 이동 버튼
#         st.button(
#             "차분 설정 변경",
#             help="차분 분석 페이지로 이동하여 차분 설정을 변경합니다.",
#             on_click=lambda: st.rerun()
#         )
        
#         # 차분 없이 원본 데이터 사용
#         if st.session_state.use_differencing:
#             if st.button("원본 데이터 사용", help="차분하지 않은 원본 데이터를 사용합니다."):
#                 st.session_state.use_differencing = False
#                 st.rerun()
# else:
#     # 차분 데이터가 없는 경우 안내
#     st.info("차분 분석을 수행하지 않았습니다. 정상성 문제가 있다면 '차분 분석' 페이지에서 차분을 적용해보세요.")

# 모델 팩토리 가져오기
model_factory = get_model_factory()

if model_factory is None:
    st.error("모델 팩토리 로드에 실패했습니다. pmdarima 호환성 문제일 수 있습니다.")
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
            if st.session_state.use_differencing:
                if prepare_differenced_train_test_data():
                    with st.spinner("모델을 차분 데이터로 학습 중입니다..."):
                        st.session_state.selected_models = selected_models
                        st.session_state.strategy = strategy

                        # 모델 학습 실행
                        final_recommendation, results = run_complete_optimization(selected_models, strategy, trial)
                        st.success("모델 학습 완료!")
                else:
                    st.error("차분 데이터 준비 중 오류가 발생했습니다.")
            else:
                if prepare_train_test_data():
                    with st.spinner("모델을 학습 중입니다..."):
                        st.session_state.selected_models = selected_models
                        st.session_state.strategy = strategy

                        # 모델 학습 실행
                        final_recommendation, results = run_complete_optimization(selected_models, strategy, trial)
                        st.success("모델 학습 완료!")
                else:
                    st.error("훈련/테스트 데이터 준비 중 오류가 발생했습니다.")

        # # 차분 데이터 확인
        # if st.session_state.use_differencing:
        #     with st.expander("차분 데이터 확인", expanded=False):
        #         st.write("### 원본 데이터:")
        #         st.write(f"train: {type(st.session_state.train)}, 길이: {len(st.session_state.train) if st.session_state.train is not None else 'None'}")
        #         st.write(f"test: {type(st.session_state.test)}, 길이: {len(st.session_state.test) if st.session_state.test is not None else 'None'}")
        #         st.write("### 차분 데이터:")
        #         st.write(f"diff_train: {type(st.session_state.diff_train)}, 길이: {len(st.session_state.diff_train) if st.session_state.diff_train is not None else 'None'}")
        #         st.write(f"diff_test: {type(st.session_state.diff_test)}, 길이: {len(st.session_state.diff_test) if st.session_state.diff_test is not None else 'None'}")

with col2:
    if st.button("결과 초기화", use_container_width=True):
        reset_model_results()
        st.rerun()

if results is not None:
    st.download_button(
                        label="Download JSON",
                        file_name="model_result.json",
                        mime="application/json",
                        data=results,
                    )