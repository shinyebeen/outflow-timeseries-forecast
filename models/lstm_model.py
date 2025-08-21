import warnings

import streamlit as st
import pandas as pd 
import numpy as np
import optuna

from sklearn.preprocessing import MinMaxScaler
from models.base_model import TimeSeriesModel 
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error


try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential 
    from tensorflow.keras.layers import Dense, LSTM, Dropout 
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
        self.scaler_X = None
        self.scaler_y = None
        self.time_step = None
        self.forecast_horizon = None
        
        # 베이즈 최적화를 위한 탐색 공간 정의
        self.search_space = {
            # 시간 스텝 범위
            'time_step_min': 3,
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

    def _prepare_data(self, time_step):
        """데이터 준비 및 스케일링"""
        try:
            if len(st.session_state.train) < 50 or len(st.session_state.test) < 10:
                print(f"데이터가 충분하지 않습니다. train: {len(st.session_state.train)}, test: {len(st.session_state.test)}")
                return None
            
            X_train, y_train = self._create_sequences(st.session_state.train, time_step)
            X_test, y_test = self._create_sequences(st.session_state.test, time_step)

            if len(X_train) == 0 or len(X_test) == 0:
                print(f"시퀀스 생성 실패. X_train: {len(X_train)}, X_test: {len(X_test)}")
                return None

            # 이제 X_train, y_train은 이미 3D 형태여야 함
            print(f"Original shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
            
            # 3D 데이터인지 확인
            if X_train.ndim != 3 or y_train.ndim != 3:
                print(f"경고: 예상과 다른 차원입니다. X_train: {X_train.ndim}D, y_train: {y_train.ndim}D")
                return None

            # 원본 3D 형태 저장
            X_train_shape = X_train.shape
            X_test_shape = X_test.shape
            y_train_shape = y_train.shape
            y_test_shape = y_test.shape

            # 스케일링을 위해 2D로 변환 (samples, features)
            # (n_samples, timesteps, features) -> (n_samples * timesteps, features)
            X_train_2d = X_train.reshape(-1, X_train.shape[-1])
            X_test_2d = X_test.reshape(-1, X_test.shape[-1])
            y_train_2d = y_train.reshape(-1, y_train.shape[-1])
            y_test_2d = y_test.reshape(-1, y_test.shape[-1])

            # 스케일링
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_train_scaled_2d = scaler_X.fit_transform(X_train_2d)
            X_test_scaled_2d = scaler_X.transform(X_test_2d)

            y_train_scaled_2d = scaler_y.fit_transform(y_train_2d)
            y_test_scaled_2d = scaler_y.transform(y_test_2d)

            # 다시 3D로 변환 - 원본 형태 사용
            X_train_scaled = X_train_scaled_2d.reshape(X_train_shape)
            X_test_scaled = X_test_scaled_2d.reshape(X_test_shape)
            y_train_scaled = y_train_scaled_2d.reshape(y_train_shape)
            y_test_scaled = y_test_scaled_2d.reshape(y_test_shape)

            print(f"Final shapes - X_train: {X_train_scaled.shape}, y_train: {y_train_scaled.shape}")

            return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y

        except Exception as e:
            print(f"데이터 준비 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_sequences(self, data, time_step):
        """
        시계열 데이터를 입력 시퀀스와 타겟으로 변환
        
        Args:
            data: 시계열 데이터 (pandas Series, numpy array 등)
            time_step: 시퀀스 길이
        
        Returns:
            (입력 시퀀스, 타겟) 튜플 - LSTM용 3D 형태
        """
        try:
            X_seq, y_seq = [], []
                    # LSTM과 동일하게 step 설정 추가

            time_step = st.session_state.time_step
            forecast_horizon = st.session_state.forecast_horizon  # 24시간씩 점프
            
            print(f"Creating sequences with time_step: {time_step}, forecast_horizon: {forecast_horizon}")
            print(f"Data type: {type(data)}, Data shape: {data.shape}")
            
            # pandas Series인 경우 numpy array로 변환
            if hasattr(data, 'values'):
                data = data.values
                print(f"Converted pandas to numpy: {data.shape}")
            
            # numpy array가 아닌 경우 변환
            if not isinstance(data, np.ndarray):
                data = np.array(data)
                print(f"Converted to numpy array: {data.shape}")
            
            # 데이터가 1D인 경우 2D로 변환 (n_samples, 1)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
                print(f"Reshaped data to 2D: {data.shape}")
            
            # 충분한 데이터가 있는지 확인
            min_length = time_step + forecast_horizon
            if len(data) < min_length:
                print(f"데이터 길이가 충분하지 않습니다. 필요: {min_length}, 실제: {len(data)}")
                return np.array([]).reshape(0, time_step, data.shape[1]), np.array([]).reshape(0, forecast_horizon, data.shape[1])
            
            for i in range(0, data.shape[0] - time_step - forecast_horizon + 1, time_step):
                X_seq.append(data[i : i+time_step])  # (time_step, n_features)
                y_seq.append(data[i+time_step : i+time_step+forecast_horizon])  # (forecast_horizon, n_features)
            
            # numpy 배열로 변환 - 자동으로 3D가 됨
            X_array = np.array(X_seq)  # (n_sequences, time_step, n_features)
            y_array = np.array(y_seq)  # (n_sequences, forecast_horizon, n_features)
            
            print(f"Generated sequences - X: {X_array.shape}, y: {y_array.shape}")
            
            return X_array, y_array
            
        except Exception as e:
            print(f"시퀀스 생성 중 오류: {e}")
            import traceback
            traceback.print_exc()
            # 빈 3D 배열 반환 - 안전한 기본값 사용
            try:
                n_features = 1 if data.ndim == 1 else data.shape[1]
            except:
                n_features = 1
            return np.array([]).reshape(0, time_step, n_features), np.array([]).reshape(0, forecast_horizon, n_features)

    def _build_model(self, trial, input_shape):
        """모델 구축"""
        try:
            model = Sequential()
            forecast_horizon = st.session_state.forecast_horizon

            # 셀 타입 선택
            cell_type = trial.suggest_categorical('cell_type', self.search_space['cell_type'])
            Cell = LSTM if cell_type == 'LSTM' else tf.keras.layers.GRU

            # 레이어 개수
            num_layers = trial.suggest_categorical('num_layers', self.search_space['num_layers'])

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

            # 출력 레이어 - y의 형태에 맞게 조정
            # y_train shape: (samples, forecast_horizon, n_features)
            # 출력 크기: forecast_horizon * n_features
            output_size = forecast_horizon * input_shape[1]  # forecast_horizon * n_features
            model.add(Dense(output_size))
            
            # Reshape layer 추가하여 올바른 출력 형태로 변환
            model.add(tf.keras.layers.Reshape((forecast_horizon, input_shape[1])))

            # 학습률
            learning_rate = trial.suggest_float('learning_rate',
                                                self.search_space['learning_rate_min'],
                                                self.search_space['learning_rate_max'],
                                                log=True)

            model.compile(optimizer=Adam(learning_rate=learning_rate),
                          loss='mean_squared_error')
            
            return model
            
        except Exception as e:
            print(f"모델 구축 중 오류: {e}")
            raise e
    
    def _objective(self, trial):
        """Optuna 목적 함수"""
        try:
            # 하이퍼파라미터 제안
            time_step = trial.suggest_int('time_step',
                                        self.search_space['time_step_min'],
                                        self.search_space['time_step_max'])

            print(f"Trial {trial.number}: time_step={time_step}")

            # 데이터 준비
            data_result = self._prepare_data(time_step)
            
            if data_result is None:
                print(f"Trial {trial.number}: 데이터 준비 실패")
                return float('inf')  # None 대신 큰 값 반환
            
            X_train, X_test, y_train, y_test, scaler_X, scaler_y = data_result
            
            # 모델 구축 - 이제 input_shape는 (timesteps, features)
            model = self._build_model(trial, (X_train.shape[1], X_train.shape[2]))
            
            # 학습 파라미터
            batch_size = trial.suggest_categorical('batch_size', self.search_space['batch_size'])
            epochs = trial.suggest_int('epochs',
                                    self.search_space['epochs_min'],
                                    self.search_space['epochs_max'])
            
            # 콜백 설정
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
            ]
            
            # 모델 학습
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    verbose=0
                )
            
            # 예측 및 평가
            y_pred_scaled = model.predict(X_test, verbose=0)

            # 차원 맞추기
            if y_pred_scaled.ndim == 2 and y_test.ndim == 3:
                y_pred_scaled = y_pred_scaled.reshape(y_test.shape)

            # 2D로 변환하여 스케일러 적용
            y_pred_2d = y_pred_scaled.reshape(-1, y_pred_scaled.shape[-1])
            y_test_2d = y_test.reshape(-1, y_test.shape[-1])

            # 원래 스케일로 변환
            y_pred_rescaled = scaler_y.inverse_transform(y_pred_2d)
            y_test_rescaled = scaler_y.inverse_transform(y_test_2d)
            
            rmse = root_mean_squared_error(y_test_rescaled, y_pred_rescaled)
            mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
            
            print(f"Trial {trial.number}: RMSE={rmse:.4f}, MAE={mae:.4f}")
            
            # 중간 결과 저장 (최적화 과정 추적)
            trial.set_user_attr('rmse', rmse)
            trial.set_user_attr('mae', mae)
            
            return rmse
        
        except Exception as e:
            print(f"Trial {trial.number}: 에러 발생 - {str(e)}")
            import traceback
            traceback.print_exc()
            return float('inf')  # None 대신 큰 값 반환

    def optimize_with_optuna(self, n_trials=100, verbose_level=1):
        """Optuna 베이즈 최적화 실행"""
        if verbose_level > 0:
            print("=== LSTM Optuna 베이즈 최적화 시작 ===")
            print(f"시행 횟수: {n_trials}")
            print(f"탐색 공간:")
            print(f"  - Time steps: {self.search_space['time_step_min']}~{self.search_space['time_step_max']}")
            print(f"  - LSTM units: {self.search_space['lstm1_units_min']}~{self.search_space['lstm1_units_max']}")
            print(f"  - 레이어 수: {self.search_space['num_layers']}")
            print(f"  - 셀 타입: {self.search_space['cell_type']}")

        # Optuna study 생성
        study = optuna.create_study(
            direction='minimize',  # RMSE 최소화
            sampler=optuna.samplers.TPESampler(seed=42),  # 베이즈 최적화
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        # 최적화 실행
        if verbose_level > 0:
            print("베이즈 최적화 진행 중...")

        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)

        # 완료된 trial이 있는지 확인
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            print("경고: 성공한 trial이 없습니다. 모든 trial이 실패했습니다.")
            return {
                'study': study,
                'best_trial': None,
                'best_model': None,
                'optimization_history': []
            }

        # 최적 결과 추출
        best_trial = study.best_trial
        best_params = best_trial.params
        best_rmse = best_trial.value

        if verbose_level > 0:
            print(f"\n최적화 완료!")
            print(f"성공한 trials: {len(completed_trials)}/{len(study.trials)}")
            print(f"최적 RMSE: {best_rmse:.4f}")
            print(f"최적 파라미터:")
            for key, value in best_params.items():
                print(f"  {key}: {value}")

        # 최적 모델로 최종 학습
        best_model_result = self._train_best_model(best_trial)

        return {
            'study': study,
            'best_trial': best_trial,
            'best_model': best_model_result,
            'optimization_history': self._get_optimization_history(study)
        }
    
    def _train_best_model(self, best_trial):
        """최적 파라미터로 최종 모델 학습"""
        try:
            # 최적 파라미터 추출
            params = best_trial.params

            # 데이터 준비
            data_result = self._prepare_data(params['time_step'])
            if data_result is None:
                return None

            X_train, X_test, y_train, y_test, scaler_X, scaler_y = data_result

            # 최적 모델 재구축 (더 많은 epoch으로)
            model = self._build_model(best_trial, (X_train.shape[1], X_train.shape[2]))

            # 최종 학습
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=0)
            ]

            history = model.fit(
                X_train, y_train,
                epochs=min(params['epochs'] * 2, 200),  # 더 많은 epoch
                batch_size=params['batch_size'],
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )

            # 최종 평가
            y_pred_scaled = model.predict(X_test, verbose=0)
            
            # 차원 맞추기
            if y_pred_scaled.ndim == 2 and y_test.ndim == 3:
                y_pred_scaled = y_pred_scaled.reshape(y_test.shape)

            # 2D로 변환하여 스케일러 적용
            y_pred_2d = y_pred_scaled.reshape(-1, y_pred_scaled.shape[-1])
            y_test_2d = y_test.reshape(-1, y_test.shape[-1])

            y_pred_rescaled = scaler_y.inverse_transform(y_pred_2d)
            y_test_rescaled = scaler_y.inverse_transform(y_test_2d)

            rmse = root_mean_squared_error(y_test_rescaled, y_pred_rescaled)
            mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)

            return {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'best_params': params,
                'history': history.history,
                'scalers': (scaler_X, scaler_y),
                'test_predictions': y_pred_rescaled.flatten(),
                'test_actual': y_test_rescaled.flatten()
            }
            
        except Exception as e:
            print(f"최종 모델 학습 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_optimization_history(self, study):
        """최적화 과정 히스토리 반환"""
        trials_data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_data = {
                    'trial_number': trial.number,
                    'rmse': trial.value,
                    'params': trial.params
                }
                if hasattr(trial, 'user_attrs'):
                    trial_data.update(trial.user_attrs)
                trials_data.append(trial_data)

        return trials_data