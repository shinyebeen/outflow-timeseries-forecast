import warnings
import datetime
import pandas as pd 
import numpy as np
import xgboost as xgb
import optuna
import pickle

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
        self.scaler_X = None
        self.scaler_y = None

        self.df = st.session_state.df.copy()
        self.target = st.session_state.target
        self.forecast_horizon = st.session_state.forecast_horizon
        self.time_step = st.session_state.time_step
        
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
        # 데이터 준비
        self._prepare_feature_pool()

    def _prepare_feature_pool(self):
        """특성 풀 준비 - 실제 데이터에 맞게"""
        all_features = list(self.df.columns)
        all_features.remove(self.target)

        # 유량 특성들 (타겟 제외)
        self.flow_features = [col for col in all_features if '유량' in col and col != self.target]

        # 시간 특성들 (실제 데이터에 있는 것들만)
        self.time_features = [col for col in all_features if col in ['weekday', 'hour', 'is_holiday', 'is_weekend', 'month']]

        # 소블록 특성들
        self.block_features = [col for col in all_features if '소블록' in col]

        # 스마트미터 특성들
        self.smart_features = [col for col in all_features
                              if '구간사용량' in col or '구간' in col]

        print(f"총 특성 개수: 유량({len(self.flow_features)}) + 시간({len(self.time_features)}) + 소블록({len(self.block_features)})")
        print()

    def _suggest_features(self, trial):
        """Optuna가 특성 조합을 자동으로 제안"""
        selected_features = []

        # 유량 특성 선택
        if self.flow_features:
            use_flow = trial.suggest_categorical('use_flow_features', [True, False])
            if use_flow:
                selected_features.extend(self.flow_features)

        # 시간 특성 선택
        if self.time_features:
            for feature in self.time_features:
                use_this = trial.suggest_categorical(f'use_{feature}', [True, False])
                if use_this:
                    selected_features.append(feature)

        # 소블록 특성 선택
        if self.block_features:
            use_blocks = trial.suggest_categorical('use_block_features', [True, False])
            if use_blocks:
                num_blocks = trial.suggest_int('num_block_features', 1,
                                             min(len(self.block_features), 3))
                selected_blocks = self.block_features[:num_blocks]
                selected_features.extend(selected_blocks)

        # 스마트미터 특성 선택
        if self.smart_features:
            use_smart = trial.suggest_categorical('use_smart_features', [True, False])
            if use_smart:
                for feature in self.smart_features:
                    if trial.suggest_categorical(f'use_{feature}', [True, False]):
                        selected_features.append(feature)

        return list(set(selected_features))  # 중복 제거

    def _create_advanced_features(self, df, base_features, trial):
        """고급 특성 엔지니어링 (Optuna가 파라미터 제안)"""
        enhanced_df = df.copy()

        # 지연 특성 생성
        max_lag = trial.suggest_int('max_lag_hours',
                                   self.search_space['max_lag_hours'][0],
                                   self.search_space['max_lag_hours'][1])

        # 타겟 변수의 지연 특성
        # for lag in [1, 6, 12, 24]:
        for lag in [10, 30, 60, 1440]:
            if lag <= max_lag:
                enhanced_df[f'target_lag_{lag}'] = enhanced_df[self.target].shift(lag)

        # 선택된 특성들의 지연 특성 (일부만)
        for feature in base_features[:2]:  # 처음 2개만
            if feature in enhanced_df.columns and '유량' in feature:
                enhanced_df[f'{feature}_lag_1'] = enhanced_df[feature].shift(1)

        # 롤링 통계 특성
        rolling_window = trial.suggest_int('rolling_window_size',
                                          self.search_space['rolling_window_size'][0],
                                          self.search_space['rolling_window_size'][1])

        enhanced_df[f'target_rolling_mean_{rolling_window}'] = enhanced_df[self.target].rolling(window=rolling_window).mean()
        enhanced_df[f'target_rolling_std_{rolling_window}'] = enhanced_df[self.target].rolling(window=rolling_window).std()

        # 계절성 특성 (옵션)
        use_seasonal = trial.suggest_categorical('use_seasonal_features',
                                                self.search_space['use_seasonal_features'])
        if use_seasonal and 'hour' in enhanced_df.columns:
            enhanced_df['hour_sin'] = np.sin(2 * np.pi * enhanced_df['hour'] / 24)
            enhanced_df['hour_cos'] = np.cos(2 * np.pi * enhanced_df['hour'] / 24)

            if 'weekday' in enhanced_df.columns:
                enhanced_df['weekday_sin'] = np.sin(2 * np.pi * enhanced_df['weekday'] / 7)
                enhanced_df['weekday_cos'] = np.cos(2 * np.pi * enhanced_df['weekday'] / 7)

        # 상호작용 특성 (옵션)
        use_interaction = trial.suggest_categorical('use_interaction_features',
                                                   self.search_space['use_interaction_features'])
        if use_interaction and len(base_features) >= 2:
            # 첫 번째와 두 번째 특성의 상호작용
            if base_features[0] in enhanced_df.columns and base_features[1] in enhanced_df.columns:
                enhanced_df['interaction_1_2'] = enhanced_df[base_features[0]] * enhanced_df[base_features[1]]

        # 결측값 제거
        enhanced_df = enhanced_df.dropna()

        # 새로 생성된 특성들만 반환
        new_features = [col for col in enhanced_df.columns if col not in df.columns]
        print(f"생성된 고급 특성: {new_features}")

        return enhanced_df, base_features + new_features

    def _create_sequences(self, df, features):
        """Multi-output 시계열 시퀀스 데이터 생성 (step 추가)"""
        X_list = []
        y_list = []

        # LSTM과 동일하게 step 설정 추가
        step = self.forecast_horizon  # 24시간씩 점프

        # 현재 시점의 특성들을 입력으로 사용
        for i in range(0, len(df) - self.forecast_horizon, step):  # step 추가!
            # X: 현재 시점의 특성들
            X_features = []
            for feature in features:
                if feature in df.columns:
                    X_features.append(df[feature].iloc[i])
                else:
                    X_features.append(0)

            X_list.append(X_features)

            # y: 1~forecast_horizon 시간 후의 모든 타겟값들
            y_sequence = []
            for h in range(1, self.forecast_horizon + 1):
                if i + h < len(df):
                    y_sequence.append(df[self.target].iloc[i + h])
                else:
                    y_sequence.append(df[self.target].iloc[-1])

            y_list.append(y_sequence)

        return np.array(X_list), np.array(y_list)
    
    def _prepare_data(self, base_features, trial):
        """XGBoost Multi-output용 데이터 준비"""
        try:
            # 고급 특성 엔지니어링
            enhanced_train_df, all_features = self._create_advanced_features(st.session_state.train, base_features, trial)
            enhanced_test_df, _ = self._create_advanced_features(st.session_state.test, base_features, trial)

            if len(enhanced_train_df) < 50 or len(enhanced_test_df) < 10:
                return None

            # 시퀀스 데이터 생성
            X_train, y_train = self._create_sequences(enhanced_train_df, all_features)
            X_test, y_test = self._create_sequences(enhanced_test_df, all_features)

            if len(X_train) == 0 or len(X_test) == 0:
                return None

            # 스케일링
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)

            y_train_scaled = scaler_y.fit_transform(y_train)
            y_test_scaled = scaler_y.transform(y_test)

            return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y, all_features

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
            # 특성 조합 제안
            base_features = self._suggest_features(trial)

            if not base_features:
                print(f"Trial {trial.number}: 선택된 특성이 없음")
                return 1000.0

            # 데이터 준비 (고급 특성 엔지니어링 포함)
            data_result = self._prepare_data(base_features, trial)
            if data_result is None:
                print(f"Trial {trial.number}: 데이터 준비 실패")
                return 1000.0

            X_train, X_test, y_train, y_test, scaler_X, scaler_y, all_features = data_result

            # XGBoost 모델 구축
            model = self._build_model(trial)

            # 모델 학습
            model.fit(X_train, y_train)

            # 예측 수행
            y_pred_scaled = model.predict(X_test)

            # 스케일 되돌리기
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_test_original = scaler_y.inverse_transform(y_test)
            y_pred = np.where(y_pred < 0, 0, y_pred)

            # 전체 평균 성능 계산
            overall_rmse = np.sqrt(mean_squared_error(y_test_original.flatten(), y_pred.flatten()))
            overall_mae = mean_absolute_error(y_test_original.flatten(), y_pred.flatten())

            # NaN 체크
            if np.isnan(overall_rmse) or np.isinf(overall_rmse):
                print(f"Trial {trial.number}: RMSE가 NaN/Inf")
                return 1000.0
            
            # 결과 저장
            trial.set_user_attr('base_features', base_features)
            trial.set_user_attr('all_features', all_features)
            trial.set_user_attr('rmse', overall_rmse)
            trial.set_user_attr('mae', overall_mae)
            trial.set_user_attr('n_features', len(all_features))

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
            # sampler=optuna.samplers.TPESampler(seed=42),
            # pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=3),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
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
            print(f"최적 특성 수: {best_trial.user_attrs.get('n_features', 0)}")
            print(f"최적 파라미터 (주요):")
            important_params = ['n_estimators', 'max_depth', 'learning_rate', 'max_lag_hours']
            for key in important_params:
                if key in best_params:
                    print(f"  {key}: {best_params[key]}")

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
            base_features = best_trial.user_attrs.get('base_features', [])            
            
            # 데이터 준비 (최적 파라미터로)
            data_result = self._prepare_data(base_features, best_trial)
            
            if data_result is None:
                return None
            
            X_train, X_test, y_train, y_test, scaler_X, scaler_y, all_features = data_result
            
            # 최적 모델 재구축
            model = self._build_model(best_trial)

            # 최종 학습
            model.fit(X_train, y_train)

            # 최종 평가
            y_pred_scaled = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_test_original = scaler_y.inverse_transform(y_test)
            
            y_pred = np.where(y_pred < 0, 0, y_pred)

            overall_rmse = np.sqrt(mean_squared_error(y_test_original.flatten(), y_pred.flatten()))
            overall_mae = mean_absolute_error(y_test_original.flatten(), y_pred.flatten())
            
            with open('best_xgb_model.pkl', 'wb') as file:
                pickle.dump(model, file)

            return {
                'model': model,
                'rmse': overall_rmse,
                'mae': overall_mae,
                'base_features': base_features,
                'all_features': all_features,
                'best_params': best_trial.params,
                'scalers': (scaler_X, scaler_y),
                #  이 두 줄 추가
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