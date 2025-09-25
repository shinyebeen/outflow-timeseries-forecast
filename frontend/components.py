"""
공통 UI 컴포넌트 모듈 
"""
import streamlit as st
import os
import gc

import psutil

def clear_memory():
    """
    메모리 비우기 기능
    - 캐시 비우기
    - 가비지 컬렉션 강제 실행
    """
    # 캐시 비우기
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # 가비지 컬렉션 강제 실행
    gc.collect()
    
    # 필요한 경우 세션 상태 초기화 기능 호출
    # (주의: 사용자 데이터가 모두 삭제됨)
    # from frontend.session_state import reset_data_results
    # reset_data_results()
    
    return True


def show_memory_usage():
    """
    메모리 사용량을 사이드바에 표시하고 메모리 비우기 버튼 제공
    """
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB 단위
    
    # 사이드바 하단에 메모리 사용량 표시
    st.sidebar.markdown("---")
    st.sidebar.progress(min(memory_usage / 4000, 1.0))  # 4GB 기준
    st.sidebar.text(f"메모리 사용량: {memory_usage:.1f} MB")
    
    # 메모리 비우기 버튼 추가
    if st.sidebar.button("메모리 비우기", help="캐시를 비우고 메모리를 정리합니다"):
        with st.spinner("메모리 정리 중..."):
            success = clear_memory()
            if success:
                st.sidebar.success("메모리 정리 완료!")
                # 페이지 새로고침 (선택적)
                st.rerun()
    
    if memory_usage > 3500:  # 3.5GB 이상일 때 경고
        st.sidebar.warning("⚠️ 메모리 사용량이 높습니다. 불필요한 모델을 제거하거나 샘플 데이터를 사용하세요.")

    # 메모리 관리 옵션 펼치기
    with st.sidebar.expander("메모리 관리"):
        # 캐시만 비우기
        if st.button("캐시 비우기", help="계산 결과 캐시만 비웁니다. 데이터는 유지됩니다."):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("캐시를 비웠습니다.")
        
        # 모델 결과 초기화
        if st.button("모델 결과 초기화", help="학습된 모델과 예측 결과를 초기화합니다."):
            from frontend.session_state import reset_model_results
            reset_model_results()
            st.success("모델 결과를 초기화했습니다.")
        
        # 전체 데이터 초기화 (위험 경고)
        danger_zone = st.checkbox("⚠️ 위험 영역 표시")
        if danger_zone:
            if st.button("모든 데이터 초기화", help="모든 데이터와 분석 결과를 초기화합니다."):
                from frontend.session_state import reset_data_results
                reset_data_results()
                st.cache_data.clear()
                st.cache_resource.clear()
                gc.collect()
                st.warning("모든 데이터가 초기화되었습니다.")
                st.rerun()

def render_data_summary(df):
    """
    데이터 요약 정보를 표시합니다.
    Args:
        df: 데이터프레임
    """

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    # 데이터 행 수
    metric_col1.metric(label="데이터 행 수", 
                       value = f'{df.shape[0]:,}',
                       help="데이터프레임의 행 수를 표시합니다.",
                       border=True)
    # 시작 날짜
    metric_col2.metric(label="시작 날짜", 
                       value = st.session_state.start_date.strftime('%Y-%m-%d'),
                       help="데이터의 시작 날짜를 표시합니다.",
                       border=True)
    # 종료 날짜
    metric_col3.metric(label="종료 날짜", 
                       value = st.session_state.end_date.strftime('%Y-%m-%d'),
                       help="데이터의 종료 날짜를 표시합니다.",
                       border=True)

    metric_col4.metric(label="측정 빈도", 
                       value=f"{st.session_state.records_per_hour:.1f}회/시간",
                       help="시간당 측정 빈도",
                       border=True)
  
def render_data_outliers(mode = 'standard'):
    if mode == 'standard':
        mode_name = '표준'
    else:
        mode_name = '보수적'
    st.subheader(mode_name + " 기준 이상치")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(label = mode_name + " 기준 이상치 하한 개수", 
                value=f"{st.session_state.outliers['lower_'+mode]:,.2f}",
                border=True
                )
    with metric_col2:
        st.metric(label = mode_name+" 기준 이상치 상한 개수", 
                value=f"{st.session_state.outliers['upper_'+mode]:,.2f}",
                border=True
                )
    
    standard_ratio = st.session_state.outliers['total_'+mode] / len(st.session_state.series) * 100
    
    with metric_col3:
        st.metric(label=mode_name+" 기준 이상치 비율(%)", 
                value=f"{standard_ratio:,.2f} %",
                border=True
                )
    with st.expander(mode_name+" 기준 이상치 데이터 보기"):
        st.dataframe(st.session_state.series[(st.session_state.series < st.session_state.outliers['lower_'+mode]) | (st.session_state.series > st.session_state.outliers['upper_'+mode])])

def render_model_selector(model_factory):
    """
    모델 선택 UI 렌더링

    Args:

    
    Returns:
        선택된 모델 목록, 모델 복잡도
    """

    with st.expander("모델 선택 및 설정", not st.session_state.trained_models):
        available_models = model_factory.get_all_available_models()

        selected_models = st.multiselect(
            "사용할 모델 선택",
            available_models,
            default=available_models[:] if not st.session_state.selected_models else st.session_state.selected_models 
        )

        # 베이지안 최적화(optuna) 반복 횟수 
        strategy = st.radio(
            "베이지안 최적화 반복 횟수",
            ["**quick** : 10회 반복 (빠른 테스트용)", "**balanced** : 20회 반복 (균형 잡힌 설정)", "**thorough** : 50회 반복 (철저한 설정)", "**smart** : 단계적 최적화 (빠른 탐색 후 세밀한 조정)", '**custom** : 사용자가 직접 설정']
        )

        col1, _ = st.columns([1, 9])

        trial = None
        with col1:
            
            if strategy.startswith('**custom'):
                trial = st.number_input(
                    "최적화 반복 횟수",
                    min_value=1,
                    max_value=100,
                    value=1,
                    key="optimization_trials",
                    help="최적화 반복 횟수를 설정합니다. 이 값이 클수록 더 많은 시간이 소요됩니다."
                )

        return selected_models, strategy, trial