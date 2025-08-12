"""
모든 시계열 모델의 기본 클래스를 정의하는 모듈
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class TimeSeriesModel(ABC):
    """
    시계열 모델을 위한 추상 기본 클래스
    모든 구체적인 모델 클래스는 이 클래스를 상속해야 합니다.
    """
    
    def __init__(self, name: str):
        """
        기본 생성자
        
        Args:
            name: 모델 이름
        """
        self.name = name
        self.model = None
        self.is_fitted = False
        self.train_data = None
        self.test_data = None
        self.forecast = None
        self.forecast_horizon = None 
        self.search_space = {}

    def _create_sequences(self, data: np.ndarray, time_step: int, forecast_horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """
        시계열 데이터를 입력 시퀀스와 타겟으로 변환
        
        Args:
            data: 시계열 데이터
            time_step: 시퀀스 길이
        
        Returns:
            (입력 시퀀스, 타겟) 튜플
        """
        pass

    @abstractmethod
    def _build_model(self, trial, input_shape):
        """
        Optuna가 제안하는 모델 아키텍처 구축

        Args:


        Returns:
            구축 모델 객체
        """
        pass 

    @abstractmethod
    def _objective(self, trial):
        """
        Optuna 목적 함수
        
        Args:

        Returns:
        
        """
        pass

    @abstractmethod
    def optimize_with_optuna(self, n_trial: int, verbose_level: int):
        """
        Optuna 베이즈 최적화 실행

        Args:

        
        Returns:
            최적 모델 등등..

        """ 
        pass   
    
    @abstractmethod
    def fit(self, train_data: pd.Series, **kwargs) -> Any:
        """
        모델을 학습합니다.
        
        Args:
            train_data: 학습 데이터
            **kwargs: 추가 매개변수
            
        Returns:
            학습된 모델 객체
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int = None, test_data: Optional[pd.Series] = None) -> np.ndarray:
        """
        미래 값을 예측합니다.
        
        Args:
            steps: 예측할 기간 수
            test_data: 테스트 데이터 (있는 경우)
            
        Returns:
            예측값 배열
        """
        pass
    
    def evaluate(self, actual: pd.Series, predicted: np.ndarray) -> Dict[str, float]:
        """
        모델 성능을 평가합니다.
        
        Args:
            actual: 실제 값
            predicted: 예측 값
            
        Returns:
            성능 지표 딕셔너리
        """
        # 길이 맞춤
        min_len = min(len(actual), len(predicted))
        actual = actual.iloc[:min_len]
        predicted = predicted[:min_len]
        
        # 성능 지표 계산
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R^2': r2
        }
    
    def fit_predict_evaluate(self, 
                            train_data: pd.Series, 
                            test_data: pd.Series, 
                            **kwargs) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        모델을 학습하고, 예측한 후, 평가까지 한번에 수행합니다.
        
        Args:
            train_data: 학습 데이터
            test_data: 테스트 데이터
            **kwargs: 추가 매개변수
            
        Returns:
            (예측값, 성능 지표) 튜플
        """
        self.train_data = train_data
        self.test_data = test_data
        
        # 모델 학습
        self.fit(train_data, **kwargs)
        
        # 예측
        forecast = self.predict(len(test_data), test_data)
        self.forecast = forecast
        
        # 평가
        metrics = self.evaluate(test_data, forecast)
        
        return forecast, metrics

    def get_params(self) -> Dict[str, Any]:
        """
        모델 파라미터를 반환합니다.
        
        Returns:
            모델 파라미터 딕셔너리
        """
        # 기본적으로 빈 딕셔너리 반환
        # 각 하위 클래스에서 이 메서드를 오버라이드해야 함
        return {}
