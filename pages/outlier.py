# """
# 이상치 처리 페이지
# """

# import streamlit as st 

# from backend.data_service import analyze_outliers
# from backend.visualization_service import plot_outliers

# st.title("이상치 분석")
# st.markdown("유출유량 데이터의 이상치를 분석합니다.")

# if st.session_state.df is None:
#     st.warning("데이터를 먼저 업로드해주세요.")
#     st.stop()
# elif st.session_state.series is None:
#     st.warning("시계열 데이터가 생성되지 않았습니다. 사이드바에서 분석 변수를 선택해주세요.")
#     st.stop()

# st.subheader("이상치 분석 결과")
    
# # 이상치 분석 (보수적 기준, 일반적 기준 이상치 개수, 비율 반환 -> Dict)
# outliers = analyze_outliers(st.session_state.series)

# if outliers.empty:
#     st.info("이상치가 없습니다.")
# else:
#     st.write(outliers)
    
    
#     st.success("이상치 분석이 완료되었습니다.")

# if st.session_state.outliers:
#     st.sidebar.subheader("이상치 분석 결과")
    
#     # 보수적 기준, 일반적 기준 이상치 개수, 비율 출력

    
#     # 박스플롯 그리기 
#     st.subheader("이상치 시각화")
#     plot_outliers(st.session_state.series, outliers)