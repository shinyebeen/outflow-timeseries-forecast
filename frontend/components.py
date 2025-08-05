"""
공통 UI 컴포넌트 모듈 
"""
import streamlit as st

def render_data_summary(df):
    """
    데이터 요약 정보를 표시합니다.
    Args:
        df: 데이터프레임
    """
    st.subheader("데이터 요약 정보")
    start_date = df['logTime'].min()
    end_date = df['logTime'].max()

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    # 데이터 행 수
    metric_col1.metric(label="데이터 행 수", 
                       value = f'{df.shape[0]:,}',
                       help="데이터프레임의 행 수를 표시합니다.",
                       border=True)
    # 시작 날짜
    metric_col2.metric(label="시작 날짜", 
                       value = start_date.strftime('%Y-%m-%d'),
                       help="데이터의 시작 날짜를 표시합니다.",
                       border=True)
    # 종료 날짜
    metric_col3.metric(label="종료 날짜", 
                       value = end_date.strftime('%Y-%m-%d'),
                       help="데이터의 종료 날짜를 표시합니다.",
                       border=True)
    
    # 시간당 측정 빈도
    hours_span = (end_date - start_date).total_seconds() / 3600
    records_per_hour = df.shape[0] / max(hours_span, 1) # 최소 1시간으로 나누기 

    metric_col4.metric(label="측정 빈도", 
                       value=f"{records_per_hour:.1f}회/시간",
                       help="시간당 측정 빈도",
                       border=True)
    
def render_data_outliers(mode = 'standard'):

    st.subheader(mode + " 기준 이상치")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(label=mode + " 기준 이상치 하한 개수", 
                value=f"{st.session_state.outliers['lower_'+mode]:,.2f}",
                border=True
                )
    with metric_col2:
        st.metric(label=mode+" 기준 이상치 상한 개수", 
                value=f"{st.session_state.outliers['upper_'+mode]:,.2f}",
                border=True
                )
    
    standard_ratio = st.session_state.outliers['total_'+mode] / len(st.session_state.series) * 100
    
    with metric_col3:
        st.metric(label=mode+" 기준 이상치 비율(%)", 
                value=f"{standard_ratio:,.2f} %",
                border=True
                )
    with st.expander(mode+" 기준 이상치 데이터 보기"):
        st.dataframe(st.session_state.series[(st.session_state.series < st.session_state.outliers['lower_'+mode]) | (st.session_state.series > st.session_state.outliers['upper_'+mode])])