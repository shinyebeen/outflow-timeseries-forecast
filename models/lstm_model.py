import warnings

import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models.base_model import TimeSeriesModel 

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential 
    from tensorflow.keras.layers import Dense, LSTM, Dropout 
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    # GPU 메모리 증가 방지
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            warnings.warn(f"GPU 메모리 설정 중 오류 발생: {e}")
    
    TF_AVAILABLE = True 

except ImportError:
    TF_AVAILABLE = False 
    warnings.warn("Tensorflow 라이브러리를 사용할 수 없습니다. LSTM 모델 설치가 필요합니다.")

class LSTMModel(TimeSeriesModel):
    """
    LSTM 신경망 모델 구현 클래스
    """
    def __init__(self, name: str = 'LSTM'):
        """
        LSTM 모델 생성자

        Args:
            name: 모델 이름 
        """
        super().__init__(name)
        self.model_params = {}
        self.history = None 
        self.scaler = None
        self.time_step = None
        self.forecast_horizon = None
        
        # 베이즈 최적화를 위한 탐색 공간 정의
        self.search_space = {
            # 시간 스텝 범위
            'time_step_min': 3,
            # 'time_step_max': 336,  # 2주
            'time_step_max' : 168, # 1주

            # 모델 아키텍처
            'lstm1_units_min': 32,
            'lstm1_units_max': 256,
            'lstm2_units_min': 16,
            'lstm2_units_max': 128,
            'num_layers': [1, 2, 3],  # LSTM 레이어 개수
            'cell_type': ['LSTM', 'GRU'],  # 셀 타입

            # 정규화
            'dropout_min': 0.1,
            'dropout_max': 0.5,

            # 학습 파라미터
            'learning_rate_min': 0.0001,
            'learning_rate_max': 0.01,
            'batch_size': [8, 16, 32, 64],
            'epochs_min': 20,
            'epochs_max': 100
        }

    def _create_sequences(self, data: np.ndarray, time_step: int, forecast_horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """
        시계열 데이터를 입력 시퀀스와 타겟으로 변환
        
        Args:
            data: 시계열 데이터
            time_step: 시퀀스 길이
        
        Returns:
            (입력 시퀀스, 타겟) 튜플
        """
        X_seq, y_seq = [], []
        step = forecast_horizon

        for i in range(0, len(data) - time_step - step + 1, step):
            X_seq.append(data[i : i+time_step])
            y_seq.append(data[i+time_step : i+time_step+step])

        return np.array(X_seq), np.array(y_seq)

    def _build_model(self, trial, input_shape):
        model = Sequential()

        # 셀 타입 선택
        cell_type = trial.suggest_categorical('cell_type', self.search_space['cell_type'])
        Cell = LSTM if cell_type == 'LSTM' else 'GRU'

        # 레이어 개수
        num_layers = trial.suggest_categorial('num_layers',
                                              self.search_space['num_layers'])

        # 첫 번째 레이어
        lstm1_units = trial.suggest_int('lstm1_units',
                                       self.search_space['lstm1_units_min'],
                                       self.search_space['lstm1_units_max'])
        
        model.add(Cell(lstm1_units,
                       return_sequences=(num_layers>1),
                       input_shape=input_shape))
        
        dropout1 = trial.suggest_float('dropout1',
                                       self.search_space['dropout_min'],
                                       self.search_space['dropout_max'])

        model.add(Dropout(dropout1))

        # 추가 레이어들
        if num_layers >= 2:
            lstm2_units = trial.suggest_int('lstm2_units',
                                            self.search_space['lstm2_units_min'],
                                            self.search_space['lstm2_units_max'])
            
            model.add(Cell(lstm2_units,
                      return_sequences=(num_layers>2)))
            
            dropout2 = trial.suggest_float('dropout2',
                                          self.search_space['dropout_min'],
                                          self.search_space['dropout_max'])
            model.add(Dropout(dropout2))

        if num_layers >= 3:
            lstm3_units = trial.suggest_int('lstm3_units',
                                           self.search_space['lstm2_units_min'],
                                           self.search_space['lstm2_units_max'])

            model.add(Cell(lstm3_units, return_sequences=False))

            dropout3 = trial.suggest_float('dropout3',
                                          self.search_space['dropout_min'],
                                          self.search_space['dropout_max'])
            model.add(Dropout(dropout3))

        # 출력 레이어
        model.add(Dense(self.forecast_horizon))

        # 학습률
        learning_rate = trial.suggest_float('learning_rate',
                                            self.search_space['learning_rate_min'],
                                            self.search_space['learning_rate_max'],
                                            log=True)

        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='mean_squared_error')
        
        return model 
    
    def _objective(self, 
                   train_data: pd.Series, 
                   trial):
        if not TF_AVAILABLE:
            raise ImportError("Tensorflow 라이브러리를 먼저 설치해주세요.")
        
        self.train_data = train_data
        
        try:
            # 하이퍼파라미터 제안
            time_step = trial.suggest_int('time_step',
                                          self.search_space['time_step_min'],
                                          self.search_space['time_step_max'])
            
            X_train, y_train = self._create_sequences(self.train_data)

            model = self._build_model(trial, ())

