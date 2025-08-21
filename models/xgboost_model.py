import warnings
import datetime
import pandas as pd 
import numpy as np
import xgboost as xgb
import optuna

from sklearn.preprocessing import MinMaxScaler
from models.base_model import TimeSeriesModel 
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

import streamlit as st

class XGBoostModel(TimeSeriesModel):
    """
    XGBoost 모델 구현 클래스
    """
    def __init__(self, name: str = "XGBoost"):
        """
        XGBoost 모델 생성자
        Args:
            name: 모델 이름
        """
        super().__init__(name)
        self.model_params = {}
        self.history= None
        self.scaler = None
        self.forecast_horizon = None 
        
        self.X_train = None
        self.X_test = None 
        self.y_train = None
        self.y_test = None
        
        # 베이즈 최적화를 위한 탐색 공간 정의
        self.search_space = {
            # XGBoost 핵심 파라미터
            'n_estimators': (50, 500),
            'max_depth': (3, 12),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'gamma': (0, 5),
            'min_child_weight': (1, 10),
            'reg_alpha': (0, 10),  # L1 정규화
            'reg_lambda': (1, 10), # L2 정규화
            
            # 특성 엔지니어링 파라미터
            'max_lag_hours': (3, 168),        # 최대 지연 시간
            'rolling_window_size': (6, 72),   # 롤링 윈도우 크기
            'use_seasonal_features': [True, False],  # 계절성 특성 사용 여부
            'use_interaction_features': [True, False], # 상호작용 특성 사용 여부
        }

    def _create_sequences(self, data):
        """
        Multi-output 시계열 시퀀스 데이터 생성 (step 추가)
        """
        data_np = data.values if hasattr(data, "values") else np.array(data)
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        X_seq = []
        y_seq = []
        
        # LSTM과 동일하게 step 설정 추가

        time_step = st.session_state.time_step
        forecast_horizon = st.session_state.forecast_horizon  # 24시간씩 점프
        
        # 현재 시점의 특성들을 입력으로 사용
        for i in range(0, data.shape[0] - time_step - forecast_horizon + 1, time_step):  # step 추가!

            X_seq.append(data.iloc[i:i+time_step])
            y_seq.append(data.iloc[i+time_step:i+time_step+forecast_horizon])

        return np.array(X_seq), np.array(y_seq)
    
    def _prepare_data(self):
        """XGBoost Multi-output용 데이터 준비"""
        try:
            if len(st.session_state.train) < 50 or len(st.session_state.test) < 10:
                return None

            # 시퀀스 데이터 생성
            X_train, y_train = self._create_sequences(st.session_state.train)
            X_test, y_test = self._create_sequences(st.session_state.test)

            if X_train.ndim == 3:
                X_train = X_train.reshape(X_train.shape[0], -1)
            if X_test.ndim == 3:
                X_test = X_test.reshape(X_test.shape[0], -1)
            if y_train.ndim == 3:
                y_train = y_train.reshape(y_train.shape[0], -1)
            if y_test.ndim == 3:
                y_test = y_test.reshape(y_test.shape[0], -1)

            if len(X_train) == 0 or len(X_test) == 0:
                return None

            # 스케일링
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)

            y_train_scaled = scaler_y.fit_transform(y_train)
            y_test_scaled = scaler_y.transform(y_test)

            return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y

        except Exception as e:
            print(f"데이터 준비 중 오류: {e}")
            return None

    def _build_model(self, trial):
        """Optuna가 제안하는 XGBoost 모델 구축"""
        xgb_params = {
            'n_estimators': trial.suggest_int('n_estimators',
                                            self.search_space['n_estimators'][0],
                                            self.search_space['n_estimators'][1]),
            'max_depth': trial.suggest_int('max_depth',
                                        self.search_space['max_depth'][0],
                                        self.search_space['max_depth'][1]),
            'learning_rate': trial.suggest_float('learning_rate',
                                                self.search_space['learning_rate'][0],
                                                self.search_space['learning_rate'][1]),
            'subsample': trial.suggest_float('subsample',
                                            self.search_space['subsample'][0],
                                            self.search_space['subsample'][1]),
            'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                self.search_space['colsample_bytree'][0],
                                                self.search_space['colsample_bytree'][1]),
            'gamma': trial.suggest_float('gamma',
                                        self.search_space['gamma'][0],
                                        self.search_space['gamma'][1]),
            'min_child_weight': trial.suggest_int('min_child_weight',
                                                self.search_space['min_child_weight'][0],
                                                self.search_space['min_child_weight'][1]),
            'reg_alpha': trial.suggest_float('reg_alpha',
                                            self.search_space['reg_alpha'][0],
                                            self.search_space['reg_alpha'][1]),
            'reg_lambda': trial.suggest_float('reg_lambda',
                                            self.search_space['reg_lambda'][0],
                                            self.search_space['reg_lambda'][1]),
            'random_state': 42,
            'verbosity': 0
        }
        
        # XGBoost 기본 모델
        base_model = xgb.XGBRegressor(**xgb_params)
        
        # MultiOutputRegressor로 감싸기
        model = MultiOutputRegressor(base_model, n_jobs=-1)
        return model

    def _objective(self, trial):
        """Optuna 목적 함수"""
        try:
            # 데이터 준비 (고급 특성 엔지니어링 포함)
            data_result = self._prepare_data()
            
            if data_result is None:
                print(f"Trial {trial.number}: 데이터 준비 실패")
                return 1000.0

            X_train, X_test, y_train, y_test, scaler_X, scaler_y = data_result

            # XGBoost 모델 구축
            model = self._build_model(trial)

            # 모델 학습
            model.fit(X_train, y_train)

            # 예측 수행
            y_pred_scaled = model.predict(X_test)

            # 스케일 되돌리기
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_test_original = scaler_y.inverse_transform(y_test)

            # 전체 평균 성능 계산
            overall_rmse = np.sqrt(mean_squared_error(y_test_original.flatten(), y_pred.flatten()))
            overall_mae = mean_absolute_error(y_test_original.flatten(), y_pred.flatten())

            # NaN 체크
            if np.isnan(overall_rmse) or np.isinf(overall_rmse):
                print(f"Trial {trial.number}: RMSE가 NaN/Inf")
                return 1000.0
            
            # 결과 저장
            trial.set_user_attr('rmse', overall_rmse)
            trial.set_user_attr('mae', overall_mae)

            print(f"Trial {trial.number}: RMSE={overall_rmse:.4f}")

            return overall_rmse
        
        except Exception as e:
            print(f"Trial {trial.number}: 에러 발생 - {str(e)}")
            import traceback
            traceback.print_exc()
            return 1000.0
        
    def optimize_with_optuna(self, n_trials=30, verbose_level=1):
        """Optuna 베이즈 최적화 실행"""

        if verbose_level > 0:
            print("=== XGBoost Optuna 베이즈 최적화 시작 ===")
            print(f"시행 횟수: {n_trials}")
            print(f"탐색 공간:")
            print(f"  - n_estimators: {self.search_space['n_estimators']}")
            print(f"  - max_depth: {self.search_space['max_depth']}")
            print(f"  - learning_rate: {self.search_space['learning_rate']}")
            print(f"  - 지연 특성: {self.search_space['max_lag_hours']}시간")
            print(f"  - 롤링 윈도우: {self.search_space['rolling_window_size']}시간")
        
        # Optuna study 생성
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )

        # 최적화 실행
        if verbose_level > 0:
            print("베이즈 최적화 진행 중...")

        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)

        # 완료된 trial 확인
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            print("⚠️ 완료된 trial이 없습니다.")
            return None
        
        # 최적 결과 추출
        best_trial = study.best_trial
        best_params = best_trial.params
        best_rmse = best_trial.value

        if verbose_level > 0:
            print(f"\n최적화 완료!")
            print(f"최적 RMSE: {best_rmse:.4f}")
            print(f"최적 파라미터 (주요):")
            important_params = ['n_estimators', 'max_depth', 'learning_rate', 'max_lag_hours']
            for key in important_params:
                if key in best_params:
                    print(f"  {key}: {best_params[key]}")

        # # 결과 저장
        # self._save_optuna_results(study)

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
            
            # 데이터 준비 (최적 파라미터로)
            data_result = self._prepare_data()
            
            if data_result is None:
                return None
            
            X_train, X_test, y_train, y_test, scaler_X, scaler_y = data_result
            
            # 최적 모델 재구축
            model = self._build_model(best_trial)

            # 최종 학습
            model.fit(X_train, y_train)

            # 최종 평가
            y_pred_scaled = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_test_original = scaler_y.inverse_transform(y_test)

            overall_rmse = np.sqrt(mean_squared_error(y_test_original.flatten(), y_pred.flatten()))
            overall_mae = mean_absolute_error(y_test_original.flatten(), y_pred.flatten())

            return {
                'model': model,
                'rmse': overall_rmse,
                'mae': overall_mae,
                'best_params': best_trial.params,
                'scalers': (scaler_X, scaler_y),
                'test_predictions': y_pred.flatten(),
                'test_actual': y_test_original.flatten()
            }
        
        except Exception as e:
            print(f"최종 모델 학습 중 오류: {str(e)}")
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

    # def _save_optuna_results(self, study):
    #     """Optuna 결과 저장"""
    #     history = self._get_optimization_history(study)
    #     results_data = {
    #         'optimization_summary': {
    #             'best_rmse': study.best_value,
    #             'best_params': study.best_params,
    #             'best_features': study.best_trial.user_attrs.get('all_features', []),
    #             'n_trials': len(study.trials),
    #             'optimization_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #         },
    #         'trials_history': history,
    #         'search_space': self.search_space
    #     }
        

    ## visualize로 독립시키기?
    # def plot_optimization_history(self, study):
    #     """최적화 과정 시각화"""
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    #     trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    #     trial_numbers = [t.number for t in trials]
    #     rmse_values = [t.value for t in trials]
    #     # 최적화 과정
    #     ax1.plot(trial_numbers, rmse_values, 'b-', alpha=0.6, marker='o', markersize=4)
    #     ax1.set_xlabel('Trial Number')
    #     ax1.set_ylabel('RMSE')
    #     ax1.set_title('XGBoost Optimization Progress')
    #     ax1.grid(True)
    #     # 최적값 누적 업데이트
    #     best_so_far = []
    #     current_best = float('inf')
    #     for rmse in rmse_values:
    #         if rmse < current_best:
    #             current_best = rmse
    #         best_so_far.append(current_best)
    #     ax2.plot(trial_numbers, best_so_far, 'r-', linewidth=2, marker='o', markersize=4)
    #     ax2.set_xlabel('Trial Number')
    #     ax2.set_ylabel('Best RMSE So Far')
    #     ax2.set_title('Best Performance Progress')
    #     ax2.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f"{self.result_dir}/xgboost_optimization_history.png", dpi=300, bbox_inches='tight')
    #     plt.show()