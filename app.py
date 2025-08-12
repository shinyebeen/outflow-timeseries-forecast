"""
서비스 메인 streamlit 앱
This file is the entry point for the Streamlit application.
"""

import os 
import warnings 
import streamlit as st
import tensorflow as tf 

from config import app_config
from frontend.sidebar import initialize_sidebar
from frontend.session_state import initialize_session_state

warnings.filterwarnings("ignore")

def main():
    """
    Streamlit 애플리케이션의 메인 함수입니다.
    """

    initialize_session_state()
    initialize_sidebar()

    st.set_page_config(
        page_title=app_config.APP_NAME,
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    pages = {
        'Home': [st.Page("pages/home.py", title="데이터 요약")],
        "Data Analysis": [st.Page("pages/data_analysis.py", title="Data Analysis")],
        "Differencing" : [st.Page("pages/differencing.py", title="Differencing")],
        "Model Training": [st.Page("pages/model_training.py", title="Model Training")],
        "Prediction": [st.Page("pages/prediction.py", title="Prediction")],
    }
    
    pg = st.navigation(pages, position="sidebar")
    pg.run()

if __name__ == "__main__":
    main()