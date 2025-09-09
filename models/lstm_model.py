import warnings

import streamlit as st
import pandas as pd 
import numpy as np
import optuna
import pickle

from sklearn.preprocessing import MinMaxScaler
from models.base_model import TimeSeriesModel 
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error


try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential 
    from tensorflow.keras.layers import Dense, LSTM, Dropout 
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                # 메모리 증가 방지
                tf.config.experimental.set_memory_growth(device, True)
                # 메모리 제한 설정 (1GB로 제한)
                tf.config.set_logical_device_configuration(
                    device,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
                )
        except Exception as e:
            warnings.warn(f"GPU 메모리 설정 중 오류 발생: {e}")
    
    # CPU 메모리 사용량 제한
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    
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
        self.time_step = st.session_state.time_step
        self.forecast_horizon = st.session_state.forecast_horizon

        self.df = st.session_state.df.copy()
        self.target = st.session_state.target
        
        # 베이즈 최적화를 위한 탐색 공간 정의
        self.search_space = {
            # 시간 스텝 범위
            'time_step_min': 168,
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
                # 특성 조합 (기존 방식 + 자동 선택)
        self._prepare_feature_pool()

    def _prepare_feature_pool(self):
        """특성 풀 준비 (Optuna가 자동으로 선택)"""
        all_features = list(self.df.columns)
        all_features.remove(self.target)

        # 유량 특성들
        self.flow_features = [col for col in all_features if '유량' in col]

        # 시간 특성들
        self.time_features = [col for col in all_features
                             if col in ['weekday', 'hour', 'is_holiday', 'is_weekend', 'month']]

        # 소블록 특성들
        self.block_features = [col for col in all_features
                              if '소블록' in col or '블록' in col]

        # 전체 특성 풀
        self.feature_pool = {
            'target': [self.target],
            'flow_features': self.flow_features,
            'time_features': self.time_features,
            'block_features': self.block_features
        }

    def _suggest_features(self, trial):
        """Optuna가 특성 조합을 자동으로 제안"""
        selected_features = [self.target]  # 타겟은 항상 포함

        # 유량 특성 선택
        if self.flow_features:
            use_flow = trial.suggest_categorical('use_flow_features', [True, False])
            if use_flow:
                # 유량 특성 중 일부 선택
                num_flow = trial.suggest_int('num_flow_features', 1,
                                           min(len(self.flow_features), 5))
                selected_flow = trial.suggest_categorical('selected_flow_features',
                    [tuple(self.flow_features[:i]) for i in range(1, len(self.flow_features)+1)]
                )[:num_flow]
                selected_features.extend(list(selected_flow))

        # 시간 특성 선택
        if self.time_features:
            use_time = trial.suggest_categorical('use_time_features', [True, False])
            if use_time:
                for feature in self.time_features:
                    if trial.suggest_categorical(f'use_{feature}', [True, False]):
                        selected_features.append(feature)

        # 소블록 특성 선택
        if self.block_features:
            use_blocks = trial.suggest_categorical('use_block_features', [True, False])
            if use_blocks:
                num_blocks = trial.suggest_int('num_block_features', 1,
                                             min(len(self.block_features), 3))
                selected_blocks = self.block_features[:num_blocks]
                selected_features.extend(selected_blocks)

        return list(set(selected_features))  # 중복 제거

    def _prepare_data(self, features, time_step):
        """데이터 준비 및 스케일링"""
        try:
            # 사용 가능한 특성만 선택
            available_features = [f for f in features if f in self.df.columns]
            if not available_features or self.target not in available_features:
                return None
            
            # 스케일러 초기화
            scaler_X = MinMaxScaler(feature_range=(0, 1))
            scaler_y = MinMaxScaler(feature_range=(0, 1))

            # 특성과 타겟 분리
            X_scaled = scaler_X.fit_transform(self.df[available_features])
            y_scaled = scaler_y.fit_transform(self.df[[self.target]])

            # 시계열 데이터셋 생성
            X, y = self._create_sequences(X_scaled, y_scaled, time_step)

            if len(X) < 20:  # 최소 데이터 요구사항
                return None

            # 데이터 분할 (80:20)
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            return X_train, X_test, y_train, y_test, scaler_X, scaler_y

        except Exception as e:
            print(f"데이터 준비 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_sequences(self, X, y, time_step):
        """
        시계열 데이터를 입력 시퀀스와 타겟으로 변환
        
        Args:
            data: 시계열 데이터 (pandas Series, numpy array 등)
            time_step: 시퀀스 길이
        
        Returns:
            (입력 시퀀스, 타겟) 튜플 - LSTM용 3D 형태
        """
        try:
            """시계열 데이터셋 생성"""
            X_seq, y_seq = [], []

            for i in range(0, len(X) - time_step - self.forecast_horizon + 1, self.forecast_horizon):
                X_seq.append(X[i:i + time_step, :])
                y_seq.append(y[i + time_step:i + time_step + self.forecast_horizon, 0])

            return np.array(X_seq), np.array(y_seq)
            
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

            features = self._suggest_features(trial)

            # 데이터 준비
            data_result = self._prepare_data(features, time_step)
            if data_result is None:
                raise optuna.TrialPruned()
            
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
            y_pred = model.predict(X_test, verbose=0)
            y_pred_rescaled = scaler_y.inverse_transform(y_pred)
            y_test_rescaled = scaler_y.inverse_transform(y_test)

            rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

            # 중간 결과 저장 (최적화 과정 추적)
            trial.set_user_attr('features', features)
            trial.set_user_attr('rmse', rmse)
            trial.set_user_attr('mae', mean_absolute_error(y_test_rescaled, y_pred_rescaled))

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
            print()

        # Optuna study 생성
        study = optuna.create_study(
            direction='minimize',  # RMSE 최소화
            # sampler=optuna.samplers.TPESampler(seed=42),  # 베이즈 최적화
            # pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=3),  # 시작 trial 수 감소
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)  # 더 빠른 pruning
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
            features = best_trial.user_attrs['features']

            # 데이터 준비
            data_result = self._prepare_data(features, params['time_step'])
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
            y_pred = model.predict(X_test, verbose=0)
            y_pred_rescaled = scaler_y.inverse_transform(y_pred)
            y_test_rescaled = scaler_y.inverse_transform(y_test)

            rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
            mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)

            model.save(f"best_lstm_model.h5")
            # 스케일러 저장
            with open(f"scaler_X.pkl", "wb") as f:
                pickle.dump(scaler_X, f)
            with open(f"scaler_y.pkl", "wb") as f:
                pickle.dump(scaler_y, f)

            return {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'features': features,
                'best_params': params,
                'history': history.history,
                'scalers': (scaler_X, scaler_y),
                # 이 두 줄 추가
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