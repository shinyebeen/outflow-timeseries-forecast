"""
애플리케이션 전체에 걸쳐 사용할 설정을 정의합니다.
"""

from dataclasses import dataclass

@dataclass 
class AppConfig:
    """
    애플리케이션 설정을 정의하는 데이터 클래스입니다.
    """
    APP_NAME: str = "배수지 유출유량 예측"
    VERSION: str = "1.0.0"

    # 시각화 설정
    PLOT_BACKGROUND_COLOR: str = "#ffffff"
    PLOT_GRID_COLOR: str = "#e0e0e0"
    DEFAULT_COLOR_PALETTE: str = "viridis"    

    # 모델 학습 설정
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_RANDOM_STATE: int = 42
    MAX_EPOCHS: int = 100
    BATCH_SIZE: int = 32

# 애플리케이션 설정 인스턴스 생성
app_config = AppConfig()