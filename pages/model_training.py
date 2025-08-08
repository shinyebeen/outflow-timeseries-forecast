import streamlit as st

st.title("Model Training Page")

if st.session_state.df is None:
    st.warning("사이드바에서 데이터를 업로드해주세요.")
elif st.session_state.series is None:
    st.warning('사이드바에서 데이터 가져오기 버튼을 눌러주세요.')

selected_models, complexity = render_model_selector()