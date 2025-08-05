import streamlit as st
import pandas as pd 
import numpy as np

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from frontend.components import render_data_outliers
from backend.data_service import (analyze_outliers, 
                                  delete_outliers, 
                                  analyze_acf_pacf, 
                                  analyze_stationarity,
                                  analyze_fft)
from backend.visualization_service import (visualize_boxplot, 
                                           visualize_acf_pacf,
                                           visualize_fft)

st.title("Data Analysis Page")

tab1, tab2, tab3, tab4 = st.tabs(['1', '2', '3', '4'])

if st.session_state.series is not None:
    # 이상치 제거 # 박스플롯
    with tab1:
        st.subheader("박스플롯")
        outlier_col1, outlier_col2 = st.columns(2)
        with outlier_col1:
            outliers_fig = visualize_boxplot()
            if outliers_fig:
                st.plotly_chart(outliers_fig, use_container_width=True, theme="streamlit")
            else:
                st.error("이상치 시각화 생성 실패")
        with outlier_col2:
            if st.session_state.series is not None:
                outliers = analyze_outliers()
                
                if outliers:
                    render_data_outliers('standard')
                    render_data_outliers('conservative')

                    too_many_outliers = st.session_state.outliers['total_standard'] > len(st.session_state.df) * 0.1

                    if too_many_outliers:  # 10% 이상이면
                        st.text(f"\n💡 추천: 표준 기준으로 {st.session_state.outliers['total_standard']}개가 너무 많습니다. 보수적 기준 사용을 권장합니다.")
                    else:
                        st.text(f"\n💡 추천: 표준 기준으로 {st.session_state.outliers['total_standard']}개 정도면 적당합니다.")

                    if st.session_state.outliers['total_standard'] > 0:
                        # 이상치 제거 기준 선택
                        
                        st.markdown("#### 이상치 제거 기준 선택")
                        options = ['표준 기준'] if not too_many_outliers else ['표준 기준', '보수적 기준']
                        selected_criterion = st.radio("제거 기준", options, horizontal=True, label_visibility='collapsed')

                        if st.button('이상치 제거'):
                            deleted = delete_outliers()

                else:
                    st.warning("이상치 분석 도중 문제가 발생했습니다.")    

    with tab2:

        st.subheader("정상성")
        
        # 정상성 검정 수행
        stationarity_result = analyze_stationarity()
                
        if not stationarity_result:
            st.error("정상성 검정 중 오류가 발생했습니다.")

        # 정상성 검정 결과 표시
        if st.session_state.stationarity_result:
            
            # 정상성 결과 컨테이너
            with st.container():
                # 정상성 여부 먼저 큰 글씨로 표시
                if st.session_state.stationarity_result['is_stationary']:
                    st.success("### ✅ 시계열 데이터가 정상성을 만족합니다")
                else:
                    st.warning("### ⚠️ 시계열 데이터가 정상성을 만족하지 않습니다")
                    
                # 설명 추가
                with st.expander("정상성 판단 기준 설명", expanded=False):
                    st.markdown("""
                    - **ADF 통계량**이 임계값보다 **작을수록** 정상성 가능성이 높습니다
                    - **p-값**이 0.05보다 **작으면** 정상성을 만족합니다
                    - ADF 통계량이 임계값보다 작을수록, 그리고 p-값이 작을수록 정상성 가능성이 높습니다
                    """)
                
                # 메트릭 표시를 위한 3개 컬럼
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                # ADF 통계량 (첫 번째 메트릭)
                test_stat = st.session_state.stationarity_result['test_statistic']
                critical_1pct = st.session_state.stationarity_result['critical_values']['1%']
                # ADF 통계량과 1% 임계값의 차이
                delta_adf = test_stat - critical_1pct
                
                # 시각화: ADF 통계량이 임계값보다 작으면 좋은 것이므로 delta_color="inverse" 사용
                metric_col1.metric(
                    label="ADF 통계량",
                    value=f"{test_stat:.4f}",
                    delta=f"{delta_adf:.4f}",
                    delta_color="inverse",
                    help="ADF 통계량이 임계값보다 작을수록 정상성 가능성이 높습니다",
                    border=True
                )
                
                # p-값 (두 번째 메트릭)
                p_value = st.session_state.stationarity_result['p_value']
                # p-값과 0.05의 차이
                delta_p = p_value - 0.05
                
                # 시각화: p-값이 작을수록 좋은 것이므로 delta_color="inverse" 사용
                metric_col2.metric(
                    label="p-값",
                    value=f"{p_value:.4f}",
                    delta=f"{delta_p:.4f}",
                    delta_color="inverse",
                    help="p-값이 0.05보다 작으면 정상성을 만족합니다",
                    border=True
                )
                
                # 관측 수 (세 번째 메트릭)
                num_obs = st.session_state.stationarity_result['num_observations']
                metric_col3.metric(
                    label="관측 데이터 수",
                    value=f"{num_obs:,}",
                    help="정상성 검정에 사용된 데이터 수",
                    border=True
                )
                
                # 임계값 카드
                st.markdown("### 📊 임계값 (Critical Values)")
                
                # 임계값 표시를 위한 3개 컬럼
                crit_col1, crit_col2, crit_col3 = st.columns(3)
                
                # 각 임계값을 메트릭으로 표시
                for i, (key, value) in enumerate(st.session_state.stationarity_result['critical_values'].items()):
                    # ADF 통계량과 임계값의 차이
                    delta_crit = test_stat - value
                    # 색상 설정: ADF 통계량이 임계값보다 작으면 좋은 것이므로 inverse 사용
                    color_setting = "inverse"
                    
                    # 각 컬럼에 임계값 메트릭 추가
                    if i == 0:  # 1% 임계값
                        crit_col1.metric(
                            label=f"임계값 ({key})",
                            value=f"{value:.4f}",
                            delta=f"{delta_crit:.4f}",
                            delta_color=color_setting,
                            help=f"ADF 통계량이 {key} 임계값보다 작으면 {key} 유의수준에서 정상성 만족",
                            border=True
                        )
                    elif i == 1:  # 5% 임계값
                        crit_col2.metric(
                            label=f"임계값 ({key})",
                            value=f"{value:.4f}",
                            delta=f"{delta_crit:.4f}",
                            delta_color=color_setting,
                            help=f"ADF 통계량이 {key} 임계값보다 작으면 {key} 유의수준에서 정상성 만족",
                            border=True
                        )
                    elif i == 2:  # 10% 임계값
                        crit_col3.metric(
                            label=f"임계값 ({key})",
                            value=f"{value:.4f}",
                            delta=f"{delta_crit:.4f}",
                            delta_color=color_setting,
                            help=f"ADF 통계량이 {key} 임계값보다 작으면 {key} 유의수준에서 정상성 만족",
                            border=True
                        )
        
        # 시각적 구분선 추가
        st.markdown("---")

        # 정상성 평가(acf, pacf)
        st.subheader("ACF/PACF Plot")
        
        # ACF/PACF 분석 버튼
        nlags = st.slider("최대 시차(lag) 수", min_value=10, max_value=100, value=40, step=5)
        
        if st.button("ACF/PACF 분석 실행", type="primary"):
            acf_value, pacf_value = analyze_acf_pacf(nlags)
        
        # ACF/PACF 분석 결과 표시
        if st.session_state.acf_values is not None and st.session_state.pacf_values is not None:
            st.markdown("### ACF/PACF 그래프")

            acf_pacf_fig = visualize_acf_pacf()
            if acf_pacf_fig:
                st.plotly_chart(acf_pacf_fig, use_container_width=True, theme="streamlit")
            else:
                st.error("ACF/PACF 그래프 생성에 실패했습니다.")

    with tab3:
        # 변화점 탐지 
        st.title('계절성분해')
        

    with tab4:     
        # 주파수 탐지
        fft_result = analyze_fft()
        if fft_result is None:
            st.error("고속푸리에변환 중 오류가 발생했습니다.")
        
        else:  
            if st.session_state.fft_result:
                fft_fig = visualize_fft()
                if fft_fig:
                    st.plotly_chart(fft_fig, use_container_width=True, theme='streamlit')
                else:
                    st.error("FFT 그래프 생성에 실패했습니다.")
            
            # 주요 주파수 성분 출력
            st.markdown("### Frequency Analysis")
            st.markdown("#### Major frequency components (Top 3)")

            # for i, idx in enumerate(st.session_state.fft_result['top_freq_idx'][-3:]):
            #     freq = st.session_state.fft_result['freqs'][idx]
            #     period = 1 / freq if freq > 0 else float('inf')
            #     st.text(f"  {i + 1}. Frequency: {freq:.6f}, Period: {period:.2f} hours")

            # 상위 3개 주파수 성분 정보 저장
            top_freq_data = []

            for i, idx in enumerate(st.session_state.fft_result['top_freq_idx'][-3:]):
                freq = st.session_state.fft_result['freqs'][idx]
                period = 1 / freq if freq > 0 else float('inf')
                top_freq_data.append({
                    "순위": i + 1,
                    "Frequency (Hz)": f"{freq:.6f}",
                    "Period (hours)": f"{period:.2f}"
                })

            # DataFrame 생성 후 표 출력
            df_top_freq = pd.DataFrame(top_freq_data)
            st.dataframe(df_top_freq, hide_index=True)
       
else:
    st.warning('사이드바에서 데이터를 업로드해주세요.')