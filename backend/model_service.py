"""
모델 학습 및 예측 관련 서비스 모듈
"""
import traceback
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from backend.data_service import safe_len
from utils.parameter_utils import validate_model_parameters

# 모델 팩토리 동적 로드
# streamlit cache
# cache_data : 데이터에 대한 캐싱 메커니즘
# cache_resource : 머신러닝 모델이나 데이터베이스 컨넥션 등의 리소스의 효율적 사용을 도움

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
    
def run_complete_optimization(selected_models, strategy='smart', trials_per_model=None, verbose_level=1):
        """
        전체 최적화 실행 - 다양한 전략 지원
        """
        if hasattr(st.session_state, 'series') and st.session_state.series is not None:
            if st.session_state.train is None or st.session_state.test is None:
                from backend.data_service import cached_train_test_split
                try:
                    train, test = cached_train_test_split(st.ssestion_state.series, st.session_state.test_size)
                    st.session_state.train = train
                    st.session_state.test = test
                except Exception as e:
                    st.error(f"원본 데이터 분할 중 오류: {e}")
                    

        if 'model_results' not in st.session_state:
            st.session_state.model_results = {}
        
        if strategy == 'quick':
            return _quick_strategy(selected_models, verbose_level)
        elif strategy == 'balanced':
            return _balanced_strategy(selected_models, verbose_level)
        elif strategy == 'thorough':
            return _thorough_strategy(selected_models, verbose_level)
        elif strategy == 'smart':
            return _smart_strategy(selected_models, verbose_level)
        elif strategy == 'custom':
            return _custom_strategy(trials_per_model, selected_models, verbose_level)
        else:
            raise ValueError("strategy는 'quick', 'balanced', 'thorough', 'smart', 'custom' 중 하나여야 합니다.")

def _smart_strategy(selected_models, verbose_level=1):
    """
    스마트 전략: 단계적 최적화
    """
    # 진행 상황 표시 
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 모델 개수
    total_models = len(selected_models)
    completed_models = 0

    # 1단계: 빠른 스크리닝 (각 모델 5번씩)
    screening_results = {}
    screening_trials = 5

    for model_name in selected_models:
        status_text.text(f"\n{model_name} 스크리닝 중...")
        try:
            optimizer, result = _run_single_model(model_name, screening_trials, verbose_level)
            if result:
               screening_results[model_name] = {
                    'optimizer': optimizer,
                    'result': result,
                    'rmse': _extract_rmse(result),
                    'status': 'success'
                }
            else:
                st.error(f"{model_name} 실패")

        except Exception as e:
            st.error(f"{model_name} 오류: {str(e)}")
        
        # 진행 상황 업데이트
        completed_models += 1
        progress_bar.progress(completed_models / total_models)

    if not screening_results:
        st.error("모든 모델 스크리닝이 실패했습니다.")
        return None
    
    # 성능 순으로 정렬
    sorted_models = sorted(screening_results.items(), key=lambda x: x[1]['rmse'])
    print(f"\n스크리닝 결과 순위:")
    for i, (model_name, result) in enumerate(sorted_models):
        print(f"  {i+1}위: {model_name} (RMSE: {result['rmse']:.4f})")

    # 2단계: 상위 2개 모델 정밀 최적화 (실제로는 2개 전부)
    top_models = [name for name, _ in sorted_models[:2]]
    detailed_results = {}

    completed_models = 0
    len_top_models = len(top_models)

    for model_name in top_models:
        status_text.text(f"\n{model_name} 정밀 최적화 중...")
        try:
            optimizer, result = _run_single_model(model_name, 30, verbose_level)
            if result:
                detailed_results[model_name] = {
                    'optimizer': optimizer,
                    'result': result,
                    'rmse': _extract_rmse(result),
                    'status': 'success'
                }
                st.success(f"{model_name} 정밀 최적화 완료 - RMSE: {detailed_results[model_name]['rmse']:.4f}")
        except Exception as e:
            st.error(f"{model_name} 정밀 최적화 오류: {str(e)}")
                   
        completed_models += 1
        progress_bar.progress(completed_models / len_top_models)
    

    # 모든 결과 통합
    st.session_state.model_results = {**detailed_results}
    for model_name, result in screening_results.items():
        if model_name not in st.session_state.model_results:
            st.session_state.model_results[model_name] = result

    # 최종 결과 및 추천
    final_recommendation = _generate_final_recommendation()

    # 결과 저장 및 시각화
    results = _save_complete_results()

    # if verbose_level > 0:
    #     _create_comprehensive_report()

    return final_recommendation, results

def _balanced_strategy(selected_models, verbose_level=1):
    """균형 전략: 모든 모델 동일하게 20번씩"""
    print("균형 전략 실행: 모든 모델 동일 조건 비교")
    trials = 20
    return _run_all_models_equally(trials, selected_models, verbose_level)

def _thorough_strategy(selected_models, verbose_level=1):
    """철저 전략: 모든 모델 충분히 50번씩"""
    print("철저 전략 실행: 모든 모델 충분한 탐색")
    trials = 50
    return _run_all_models_equally(trials, selected_models, verbose_level)

def _quick_strategy(selected_models, verbose_level=1):
    """빠른 전략: 모든 모델 빠르게 10번씩"""
    print("빠른 전략 실행: 모든 모델 빠른 비교")
    trials = 10
    return _run_all_models_equally(trials, selected_models, verbose_level)

def _custom_strategy(trials_per_model, selected_models, verbose_level=1):
    """커스텀 전략: 사용자 지정 trials"""
    if trials_per_model is None:
        trials_per_model = 30
    print(f"커스텀 전략 실행: {trials_per_model} trials per model")
    return _run_all_models_equally(trials_per_model, selected_models, verbose_level)

def _run_all_models_equally(trials, selected_models, verbose_level):
    """모든 모델을 동일한 조건으로 실행"""

    # 진행 상황 표시 
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 모델 개수
    total_models = len(selected_models)
    completed_models = 0

    for model_name in selected_models:
        status_text.text(f"\n{model_name} 최적화 중... ({trials} trials)")
        # try:
        #     optimizer, result = _run_single_model(model_name, trials, verbose_level)
        #     if result:
        #         st.session_state.model_results[model_name] = {
        #             'optimizer': optimizer,
        #             'result': result,
        #             'rmse': _extract_rmse(result),
        #             'status': 'success'
        #         }
        #         print(f"{model_name} 완료 - RMSE: {st.session_state.model_results[model_name]['rmse']:.4f}")
        #     else:
        #         print(f"{model_name} 실패")
        # except Exception as e:
        #     print(f"{model_name} 오류: {str(e)}")

        optimizer, result = _run_single_model(model_name, trials, verbose_level)

        if result:
            st.session_state.model_results[model_name] = {
                'optimizer': optimizer,
                'result': result,
                'rmse': _extract_rmse(result),
                'status': 'success'
            }
            print(f"{model_name} 완료 - RMSE: {st.session_state.model_results[model_name]['rmse']:.4f}")
        else:
            print(f"{model_name} 실패")
        
        completed_models += 1
        progress_bar.progress(completed_models / total_models)

    # 최종 결과
    final_recommendation = _generate_final_recommendation()

    # 결과 저장
    results = _save_complete_results()

    # if verbose_level > 0:
    #     _create_comprehensive_report()

    return final_recommendation, results


def _run_single_model(model_name, trials, verbose_level):
    model_factory = get_model_factory()
    
    """개별 모델 실행"""
    # (self.df, self.target_col, self.time_col, self.forecast_horizon)
    if model_name == 'xgboost':
        # XGBoost 클래스가 이미 정의되어 있다고 가정
        optimizer = model_factory.get_model('xgboost')
        result = optimizer.optimize_with_optuna(trials, verbose_level)
        return optimizer, result
    
    elif model_name == 'lstm':
        # LSTM 클래스가 이미 정의되어 있다고 가정
        optimizer = model_factory.get_model('lstm')
        result = optimizer.optimize_with_optuna(trials, verbose_level)
        return optimizer, result
    
    elif model_name == 'catboost':
        # CatBoost 클래스가 이미 정의되어 있다고 가정
        optimizer = model_factory.get_model('catboost')
        result = optimizer.optimize_with_optuna(trials, verbose_level)
        return optimizer, result
    
    elif model_name == 'prophet':
        # Prophet 클래스가 이미 정의되어 있다고 가정
        optimizer = model_factory.get_model('prophet')
        result = optimizer.optimize_with_optuna(trials, verbose_level)
        return optimizer, result
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def _extract_rmse(result):
    """
    개별 모델 Optimizer의 결과 딕셔너리에서 RMSE를 추출.
    `result['best_trial'].value`를 안전하게 가져옴.
    """
    if isinstance(result, dict):
        # Case 1: `optimize_with_optuna`가 직접 반환하는 딕셔너리 구조 (예: {'study': ..., 'best_trial': ...})
        if 'best_trial' in result and result['best_trial'] is not None:
            return result['best_trial'].value
        
        # Case 2: `_run_single_model`에서 반환되어 `self.model_results`에 저장된 중첩된 딕셔너리 구조(smart strategy의 경우)
        #         (예: {'optimizer': ..., 'result': {'study': ..., 'best_trial': ...}, ...})
        elif 'result' in result and \
             isinstance(result['result'], dict) and \
             result['result'].get('best_trial') is not None:
            return result['result']['best_trial'].value
        
        # Case 3: 직접 'rmse' 키가 있는 경우 (예: _train_best_model에서 반환된 'best_model' 딕셔너리)
        elif 'rmse' in result and np.isfinite(result['rmse']): # np.isfinite로 무한대/NaN도 체크
            return result['rmse']
        
    # 유효한 RMSE를 찾지 못하면 무한대 반환
    return float('inf')

def _generate_final_recommendation():
    """최종 추천 생성"""
    st.text("\n최종 추천 분석 중...")
    successful_models = {name: result for name, result in st.session_state.model_results.items()
                       if result['status'] == 'success'}
    
    if not successful_models:
        return {'status': 'failed', 'message': '성공한 모델이 없습니다.'}
    
    # 최고 성능 모델
    best_model_name = min(successful_models.keys(),
                         key=lambda x: successful_models[x]['rmse'])
    best_model_rmse = successful_models[best_model_name]['rmse']
    
    # 추천 결정
    recommendation = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_single_model': {
            'name': best_model_name,
            'rmse': best_model_rmse,
            'details': successful_models[best_model_name]
        },
        'all_models_performance': {name: result['rmse'] for name, result in successful_models.items()},
        'recommendation': {
            'type': 'single_model',
            'model': best_model_name,
            'reason': f'{best_model_name}이 가장 좋은 성능을 보임',
            'expected_rmse': best_model_rmse
        }
    }
    return recommendation

def _extract_best_params(result):
    """결과에서 최적 하이퍼파라미터 추출"""
    try:
        if 'result' in result and result['result']:
            res = result['result']

            # CatBoost/Prophet의 경우
            if 'best_trial' in res and res['best_trial']:
                return res['best_trial'].params
            
            # XGBoost/LSTM의 경우
            elif 'best_model' in res and res['best_model']:
                best_model = res['best_model']
                if 'params' in best_model:
                    return best_model['params']
                elif 'hyperparameters' in best_model:
                    return best_model['hyperparameters']
                
            # Optuna study가 있는 경우
            elif 'study' in res and res['study']:
                return res['study'].best_params
            
    except Exception as e:
        print(f"하이퍼파라미터 추출 중 오류: {e}")

    return {}

def _extract_model_config(result):
    """결과에서 모델 설정 추출"""
    try:
        if 'result' in result and result['result']:
            res = result['result']
            
            # 공통 설정 정보
            config = {
                'target': st.session_state.target,
                'forecast_horizon': st.session_state.forecast_horizon
            }
            
            # 모델별 추가 설정
            if 'best_model' in res and res['best_model']:
                best_model = res['best_model']
                if 'model_type' in best_model:
                    config['model_type'] = best_model['model_type']
                if 'features' in best_model:
                    config['features'] = best_model['features']
                if 'preprocessing' in best_model:
                    config['preprocessing'] = best_model['preprocessing']
            return config
        
    except Exception as e:
        print(f"모델 설정 추출 중 오류: {e}")
    return {'target': st.session_state.target, 'forecast_horizon': st.session_state.forecast_horizon}

def _calculate_advantage(best_rmse, all_models):
    """최고 성능 모델의 우위 계산"""
    try:
        other_rmses = [result['rmse'] for name, result in all_models.items()
                      if result['rmse'] != best_rmse]
        if not other_rmses:
            return "유일한 성공 모델"
        
        second_best = min(other_rmses)
        improvement_pct = ((second_best - best_rmse) / second_best) * 100
        return f"2위 모델 대비 {improvement_pct:.1f}% 개선"
    
    except Exception:
        return "성능 비교 불가"    
    
import io
import json 
def _save_complete_results():
    """최고 성능 모델만 저장"""
    print(f"\n결과 저장 중...")
    
    # 최고 성능 모델 찾기
    successful_models = {name: result for name, result in st.session_state.model_results.items()
                       if result['status'] == 'success'}
    if not successful_models:
        print("저장할 성공한 모델이 없습니다.")
        return
    
    best_model_name = min(successful_models.keys(),
                         key=lambda x: successful_models[x]['rmse'])
    best_result = successful_models[best_model_name] # rmse가 가장 작은 모델 {name : result} 저장
    
    print(best_result)

    # 최고 성능 모델의 설정만 저장
    best_config = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'optimization_summary': {
            'target': st.session_state.target,
            'forecast_horizon': st.session_state.forecast_horizon,
            'data_shape': list(st.session_state.df.shape),
            'total_models_tested': len(st.session_state.model_results),
            'successful_models': len(successful_models)
        },
        'all_models_performance': {name: result['rmse'] for name, result in successful_models.items()}, # name과 rmse만 저장
        'winner': {
            'model_name': best_model_name,
            'rmse': best_result['rmse'],
            'best_params': _extract_best_params(best_result),
            'model_config': _extract_model_config(best_result),
            'performance_advantage': _calculate_advantage(best_result['rmse'], successful_models)
        },
        'usage_instructions': {
            'description': f'{best_model_name} 모델이 최고 성능을 달성했습니다.',
            'how_to_reproduce': '위의 best_params를 사용하여 동일한 성능을 재현할 수 있습니다.',
            'expected_rmse': best_result['rmse']
        }
    }

    # JSON 문자열 생성
    json_str = json.dumps(best_config, indent=2, ensure_ascii=False)

    # 메모리 파일 객체 생성
    file_obj = io.BytesIO(json_str.encode('utf-8'))

    return file_obj
    