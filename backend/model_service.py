"""
모델 학습 및 예측 관련 서비스 모듈
"""
import traceback
from typing import List, Dict, Any

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from backend.data_service import safe_len
from utils.parameter_utils import validate_model_parameters

# 모델 팩토리 동적 로드
# streamlit cache
## cache_data : 데이터에 대한 캐싱 메커니즘
## cache_resource : 머신러닝 모델이나 데이터베이스 컨넥션 등의 리소스의 효율적 사용을 도움

# 객체 생성은 한 번만, 계산 결과는 조건부로 캐싱하여 성능 최적화
@st.cache_resource
def get_model_factory():
    """
    모델 팩토리를 동적으로 로드합니다.
    필요할 때만 import하여 시작 시 pmdarima 오류를 방지합니다.
    
    Returns:
        ModelFactory: 모델 팩토리 인스턴스 또는 None
    """
    try:
        from models.model_factory import ModelFactory
        return ModelFactory()
    
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {traceback.format_exc()}")
        return None
    
def run_complete_optimization(self, selected_models, strategy='smart', trials_per_model=None, verbose_level=1):
        """전체 최적화 실행 - 다양한 전략 지원"""
        
        if strategy == 'quick':
            return self._quick_strategy(selected_models, verbose_level)
        elif strategy == 'balanced':
            return self._balanced_strategy(selected_models, verbose_level)
        elif strategy == 'thorough':
            return self._thorough_strategy(selected_models, verbose_level)
        elif strategy == 'smart':
            return self._smart_strategy(selected_models, verbose_level)
        elif strategy == 'custom':
            return self._custom_strategy(trials_per_model, selected_models, verbose_level)
        else:
            raise ValueError("strategy는 'quick', 'balanced', 'thorough', 'smart', 'custom' 중 하나여야 합니다.")

def _smart_strategy(self, selected_models, verbose_level=1):
    """
    스마트 전략: 단계적 최적화
    """

    # 1단계: 빠른 스크리닝 (각 모델 5번씩)
    screening_results = {}
    screening_trials = 5

    for model_name in selected_models:
        print(f"\n{model_name} 스크리닝 중...")
        try:
            optimizer, result = self._run_single_model(model_name, screening_trials, verbose_level)
            if result:
                screening_results[model_name] = {
                    'optimizer': optimizer,
                    'result': result,
                    'rmse': self._extract_rmse(result),
                    'status': 'success'
                }
                print(f"{model_name} 완료 - RMSE: {screening_results[model_name]['rmse']:.4f}")
            else:
                print(f"{model_name} 실패")
        except Exception as e:
            print(f"{model_name} 오류: {str(e)}")
    if not screening_results:
        print("모든 모델 스크리닝이 실패했습니다.")
        return None
    
    # 성능 순으로 정렬
    sorted_models = sorted(screening_results.items(), key=lambda x: x[1]['rmse'])
    print(f"\n스크리닝 결과 순위:")
    for i, (model_name, result) in enumerate(sorted_models):
        print(f"  {i+1}위: {model_name} (RMSE: {result['rmse']:.4f})")

    # 2단계: 상위 2개 모델 정밀 최적화 (실제로는 2개 전부)
    print(f"\n2단계: 상위 2개 모델 정밀 최적화 (30 trials)")
    print("-" * 40)
    top_models = [name for name, _ in sorted_models[:2]]
    detailed_results = {}
    for model_name in top_models:
        print(f"\n{model_name} 정밀 최적화 중...")
        try:
            optimizer, result = self._run_single_model(model_name, 30, verbose_level)
            if result:
                detailed_results[model_name] = {
                    'optimizer': optimizer,
                    'result': result,
                    'rmse': self._extract_rmse(result),
                    'status': 'success'
                }
                print(f"{model_name} 정밀 최적화 완료 - RMSE: {detailed_results[model_name]['rmse']:.4f}")
        except Exception as e:
            print(f"{model_name} 정밀 최적화 오류: {str(e)}")

    # 모든 결과 통합
    self.model_results = {**detailed_results}
    for model_name, result in screening_results.items():
        if model_name not in self.model_results:
            self.model_results[model_name] = result

    # 최종 결과 및 추천
    final_recommendation = self._generate_final_recommendation()

    # 결과 저장 및 시각화
    self._save_complete_results()
    if verbose_level > 0:
        self._create_comprehensive_report()
    return final_recommendation

def _balanced_strategy(self, selected_models, verbose_level=1):
    """균형 전략: 모든 모델 동일하게 20번씩"""
    print("균형 전략 실행: 모든 모델 동일 조건 비교")
    trials = 20
    return self._run_all_models_equally(trials, selected_models, verbose_level)

def _thorough_strategy(self, selected_models, verbose_level=1):
    """철저 전략: 모든 모델 충분히 50번씩"""
    print("철저 전략 실행: 모든 모델 충분한 탐색")
    trials = 50
    return self._run_all_models_equally(trials, selected_models, verbose_level)

def _quick_strategy(self, selected_models, verbose_level=1):
    """빠른 전략: 모든 모델 빠르게 10번씩"""
    print("빠른 전략 실행: 모든 모델 빠른 비교")
    trials = 10
    return self._run_all_models_equally(trials, selected_models, verbose_level)

def _custom_strategy(self, trials_per_model, selected_models, verbose_level=1):
    """커스텀 전략: 사용자 지정 trials"""
    if trials_per_model is None:
        trials_per_model = 30
    print(f"커스텀 전략 실행: {trials_per_model} trials per model")
    return self._run_all_models_equally(trials_per_model, selected_models, verbose_level)

def _run_all_models_equally(self, trials, selected_models, verbose_level):
    """모든 모델을 동일한 조건으로 실행"""

    for model_name in selected_models:
        print(f"\n{model_name} 최적화 중... ({trials} trials)")
        try:
            optimizer, result = self._run_single_model(model_name, trials, verbose_level)
            if result:
                self.model_results[model_name] = {
                    'optimizer': optimizer,
                    'result': result,
                    'rmse': self._extract_rmse(result),
                    'status': 'success'
                }
                print(f"{model_name} 완료 - RMSE: {self.model_results[model_name]['rmse']:.4f}")
            else:
                print(f"{model_name} 실패")
        except Exception as e:
            print(f"{model_name} 오류: {str(e)}")

    # 최종 결과
    final_recommendation = self._generate_final_recommendation()

    # 결과 저장
    self._save_complete_results()

    if verbose_level > 0:
        self._create_comprehensive_report()

    return final_recommendation


def _run_single_model(self, model_name, trials, verbose_level):
    model_factory = get_model_factory()
    """개별 모델 실행"""
    if model_name == 'XGBoost':
        # XGBoost 클래스가 이미 정의되어 있다고 가정
        # optimizer = model_factory.get_model('xgb')
        # optimizer = XGBoostOptimizer(self.df, self.target_col, self.time_col, self.forecast_horizon)
        result = optimizer.optimize_with_optuna(trials, verbose_level)
        return optimizer, result
    
    elif model_name == 'LSTM':
        # LSTM 클래스가 이미 정의되어 있다고 가정
        optimizer = LSTMOptunaOptimizer(self.df, self.target_col, self.forecast_horizon)
        result = optimizer.optimize_with_optuna(trials, verbose_level)
        return optimizer, result
    
    elif model_name == 'CatBoost':
        # CatBoost 클래스가 이미 정의되어 있다고 가정
        optimizer = CatBoostOptunaOptimizer(self.df, self.target_col, self.time_col, self.forecast_horizon)
        result = optimizer.optimize_with_optuna(trials, verbose_level)
        return optimizer, result
    
    elif model_name == 'Prophet':
        # Prophet 클래스가 이미 정의되어 있다고 가정
        optimizer = ProphetOptunaOptimizer(self.df, self.target_col, self.time_col, self.forecast_horizon)
        result = optimizer.optimize_with_optuna(trials, verbose_level)
        return optimizer, result
    
    else:
        raise ValueError(f"Unknown model: {model_name}")