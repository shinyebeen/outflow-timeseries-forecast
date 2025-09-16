import streamlit as st
import pandas as pd 
import numpy as np

from frontend.components import render_data_outliers
from frontend.session_state import reset_data_results, reset_model_results
from backend.data_service import (analyze_outliers, 
                                  delete_outliers, 
                                  analyze_acf_pacf, 
                                  analyze_stationarity,
                                  analyze_fft,
                                  analyze_decomposition)
from backend.visualization_service import (visualize_boxplot, 
                                           visualize_acf_pacf,
                                           visualize_fft,
                                           visualize_decomposition)

st.header("데이터 분석")
st.markdown(' ')

tab1, tab2, tab3, tab4 = st.tabs(['이상치 처리', '정상성 판단', '시계열 분해', '주파수 분석'])

if st.session_state.series is not None:
    # 이상치 제거 # 박스플롯
    with tab1:
        outlier_col1, outlier_col2 = st.columns(2)
        
        with outlier_col1:
            outliers_fig = visualize_boxplot()
            if outliers_fig:
                st.plotly_chart(outliers_fig, use_container_width=True, theme="streamlit")
            else:
                st.error("이상치 시각화 생성 실패")
        
        with outlier_col2:
            # 변수 초기화
            outliers = None
            too_many_outliers = False
            
            try:
                outliers = analyze_outliers()
            except Exception as e:
                st.warning(f"이상치 분석 도중 문제가 발생했습니다. {str(e)}")
                outliers = None

            # outliers가 성공적으로 분석되었을 때만 실행
            if st.session_state.get('outliers') is not None:
                render_data_outliers('standard')
                render_data_outliers('conservative')
                
                # 이상치 개수 확인
                total_standard = st.session_state.outliers.get('total_standard', 0)
                total_data = len(st.session_state.series)
                too_many_outliers = total_standard > total_data * 0.1
                
                if too_many_outliers:  # 10% 이상이면
                    st.text(f"\n💡 추천: 표준 기준으로 {total_standard}개가 너무 많습니다. 보수적 기준 사용을 권장합니다.")
                else:
                    st.text(f"\n💡 추천: 표준 기준으로 {total_standard}개 정도면 적당합니다.")
                
                # 이상치가 있을 때만 제거 옵션 표시
                if total_standard > 0:
                    # 이상치 제거 기준 선택
                    st.markdown("#### 이상치 제거 기준 선택")
                    options = ['standard', 'conservative'] if too_many_outliers else ['standard']
                    selected_criterion = st.radio("제거 기준", options, horizontal=True, label_visibility='collapsed')

                    if st.button('이상치 제거'):
                        try:
                            # selected_criterion을 함수에 전달
                            cleaned_series = delete_outliers(selected_criterion)
                            
                            if cleaned_series is not None and len(cleaned_series) > 0:
                                st.success(f'이상치 제거 성공!')

                                # if st.button('앞으로 분석 및 예측에 이상치 제거 데이터 사용하기'):
                                #     reset_data_results()
                                #     reset_model_results()
                                #     st.session_state.df = cleaned_df
                                #     st.rerun()
                                if st.button('앞으로 분석 및 예측에 이상치 제거 데이터 사용하기'):
                                    reset_data_results()
                                    reset_model_results()
                                    st.session_state.series = cleaned_series  # 시계열 데이터 직접 업데이트
                                    st.session_state.df[st.session_state.target] = cleaned_series  # target_column은 실제 컬럼명으로 변경 필요
                                    st.experimental_rerun()  # 페이지 강제 새로고침

                            elif len(cleaned_series) == 0:
                                st.info('제거할 이상치가 없습니다.')
                            else:
                                st.error('이상치 제거에 실패했습니다.')
                        except Exception as e:
                            st.error(f'이상치 제거 중 오류 발생: {str(e)}')
                else:
                    st.info("제거할 이상치가 없습니다.")
            else:
                st.warning("이상치 분석을 먼저 수행해주세요.")

    with tab2:
        # 정상성 평가(acf, pacf)
        st.markdown("### ADF 정상성 검정")
        
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
                    st.success("정상성 만족")
                else:
                    st.warning("시계열 데이터가 정상성을 만족하지 않습니다")
                    
                # 설명 추가
                with st.expander("정상성 판단 기준 설명", expanded=True):
                    st.markdown("""
                    - **ADF 통계량**이 임계값보다 **작을수록** 정상성 가능성이 높습니다
                    - **p-값**이 0.05보다 **작으면** 정상성을 만족합니다
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
                    # delta=f"{delta_adf:.4f}",
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
                    # delta=f"{delta_p:.4f}",
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
        st.markdown("### ACF/PACF 그래프")
        
        # ACF/PACF 분석 버튼
        nlags = st.slider("최대 시차(lag) 수", min_value=10, max_value=100, value=10, step=5)
        
        if st.button("ACF/PACF 분석 실행", type="primary"):
            acf_value, pacf_value = analyze_acf_pacf(nlags)
        
        # ACF/PACF 분석 결과 표시
        if st.session_state.acf_values is not None and st.session_state.pacf_values is not None:

            acf_pacf_fig = visualize_acf_pacf(acf_values = st.session_state.acf_values,
                                              pacf_values = st.session_state.pacf_values,
                                              lags = nlags)
            if acf_pacf_fig:
                st.plotly_chart(acf_pacf_fig, use_container_width=True, theme="streamlit")
            else:
                st.error("ACF/PACF 그래프 생성에 실패했습니다.")

    with tab3:
        # 시계열 분해
        # 주기 자동 감지 또는 기본값 사용
        period = min(24*st.session_state.records_per_hour, len(st.session_state.series)//2)  # 시간별 데이터라 가정하고 24시간 주기
        decomposition = analyze_decomposition(period)

        try:
            if decomposition is None:
                st.error("시계열 분해 도중 오류가 발생했습니다.")
            
            else:
                st.success("시계열 분해 완료")                
                
                # 설명 추가
                with st.expander("시계열 분해 그래프 해석 방법", expanded=False):
                    st.markdown("""
                    시계열 분해는 시계열 데이터를 **트렌드(Trend)**, **계절성(Seasonal)**, **잔차(Residual)** 로 분해하여 각 구성 요소를 분석하는 방법입니다.
                    - **Observed** : 원본 시계열 데이터입니다.
                    - **Trend** : 데이터의 **장기적인 추세**입니다. 일반적으로 상승 또는 하강하는 경향을 보여줍니다.
                    - **Seasonal** : 계절성은 **주기적인 패턴**을 나타내며, 예를 들어 일별, 주별, 월별 등 반복되는 경향을 보여줍니다.
                    - **Residual** : 잔차는 트렌드와 계절성을 제거한 후 남은 데이터로, **예측할 수 없는 변동성**을 나타냅니다. 
                                
                        이 값이 작을수록 모델의 예측력이 높다고 볼 수 있습니다.
                                
                    데이터가 너무 많은 경우, 그래프가 직사각형 모양 또는 밀집된 형태로 표시될 수 있습니다. 이 경우, 그래프를 확대하거나 축소하여 확인할 수 있습니다.
                    """)
                
                if st.session_state.decomposition:
                    decomposition_fig = visualize_decomposition()
                
                if decomposition_fig:
                    st.plotly_chart(decomposition_fig, use_container_width=True, theme='streamlit')
                else:
                    st.error("계절성 분해 그래프 생성에 실패했습니다.")

        except Exception as e:
            st.error(str(e))
        
    with tab4:     
        # 주파수 탐지
        ## 📊 푸리에 변환 해석 가이드
        
        with st.expander("시계열 분해 그래프 해석 방법", expanded=False):
            st.markdown("""그래프(FFT): 시간 데이터가 어떤 주기(주파수) 성분으로 이루어져 있는지 보여줍니다. 그래프의 봉우리가 클수록 해당 주파수가 데이터에 강하게 포함되어 있음을 의미합니다.

                주요 주파수 성분 표: 데이터에서 가장 뚜렷하게 나타나는 상위 3개의 주파수를 표시합니다.

                Frequency (Hz): 1초당 반복 횟수를 의미합니다.

                Period (hours): 해당 주파수가 실제 시간에서 몇 시간 주기로 반복되는지를 나타냅니다.

                👉 즉, 값이 클수록 데이터에서 반복적으로 나타나는 주요 패턴을 설명한다고 볼 수 있습니다. 
        """)

        

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