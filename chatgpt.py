def train_models(selected_models, complexity):
    """
    선택된 모델 학습 및 예측 수행
    
    Args:
        selected_models: 선택된 모델 목록
        complexity: 모델 복잡도 설정
    
    Returns:
        bool: 학습 성공 여부
    """
    # 복잡도별 파라미터 설정
    if complexity == '간단 (빠름, 저메모리)':
        arima_params = {
            'max_p': 1, 'max_q': 1, 'max_P': 0, 'max_Q': 0,
            'stepwise': True, 'n_jobs': 1
        }
        lstm_params = {
            'n_steps': min(24, safe_len(st.session_state.train, 100) // 20),
            'lstm_units': [32],
            'epochs': 30
        }
        prophet_params = {
            'daily_seasonality': False,
            'weekly_seasonality': True,
            'changepoint_prior_scale': 0.01
        }
        transformer_params = {
            'window_size': min(24, safe_len(st.session_state.train, 100) // 20),
            'embed_dim': 32,
            'num_heads': 2,
            'ff_dim': 64,
            'num_layers': 1,
            'epochs': 30
        }
    elif complexity == '중간':
        arima_params = {
            'max_p': 2, 'max_q': 2, 'max_P': 1, 'max_Q': 1,
            'stepwise': True, 'n_jobs': 1
        }
        lstm_params = {
            'n_steps': min(48, safe_len(st.session_state.train, 100) // 10),
            'lstm_units': [50],
            'epochs': 50
        }
        prophet_params = {
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'changepoint_prior_scale': 0.05
        }
        transformer_params = {
            'window_size': min(48, safe_len(st.session_state.train, 100) // 10),
            'embed_dim': 64,
            'num_heads': 4,
            'ff_dim': 128,
            'num_layers': 2,
            'epochs': 50
        }
    else:  # 복잡 (정확도 높음, 고메모리)
        arima_params = {
            'max_p': 5, 'max_q': 5, 'max_P': 2, 'max_Q': 2,
            'stepwise': True, 'n_jobs': 1
        }
        lstm_params = {
            'n_steps': min(72, safe_len(st.session_state.train, 100) // 8),
            'lstm_units': [50, 50],
            'epochs': 100
        }
        prophet_params = {
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'yearly_seasonality': True,
            'changepoint_prior_scale': 0.05
        }
        transformer_params = {
            'window_size': min(72, safe_len(st.session_state.train, 100) // 8),
            'embed_dim': 128,
            'num_heads': 8,
            'ff_dim': 256,
            'num_layers': 3,
            'epochs': 100
        }
    
    # 데이터 준비 및 키 생성 (캐싱용)
    if st.session_state.use_differencing:
        # 차분 데이터가 없으면 생성
        if st.session_state.differenced_series is None:
            from backend.data_service import perform_differencing
            perform_differencing()
            
        if st.session_state.diff_train is None or st.session_state.diff_test is None:
            from backend.data_service import prepare_differenced_train_test_data
            prepare_differenced_train_test_data()
        
        # 모든 데이터 키 생성 (차분 및 원본 모두)
        if st.session_state.diff_train is not None and st.session_state.diff_test is not None:
            train_data_key = hash(tuple(st.session_state.diff_train.values.tolist()))
            test_data_key = hash(tuple(st.session_state.diff_test.values.tolist()))
            
            # 모델마다 다른 데이터를 사용할 수 있으므로 원본 데이터 키도 생성
            if st.session_state.train is not None and st.session_state.test is not None:
                original_train_key = hash(tuple(st.session_state.train.values.tolist()))
                original_test_key = hash(tuple(st.session_state.test.values.tolist()))
            else:
                # 원본 데이터 없으면 차분 데이터 키로 대체
                original_train_key = train_data_key
                original_test_key = test_data_key
        else:
            st.error("차분 데이터를 준비할 수 없습니다. 원본 데이터를 사용합니다.")
            st.session_state.use_differencing = False
            if st.session_state.train is not None and st.session_state.test is not None:
                train_data_key = hash(tuple(st.session_state.train.values.tolist()))
                test_data_key = hash(tuple(st.session_state.test.values.tolist()))
            else:
                st.error("모델 학습에 필요한 데이터가 없습니다.")
                return False
    else:
        # 원본 데이터 학습 시
        if st.session_state.train is not None and st.session_state.test is not None:
            train_data_key = hash(tuple(st.session_state.train.values.tolist()))
            test_data_key = hash(tuple(st.session_state.test.values.tolist()))
        else:
            st.error("모델 학습에 필요한 데이터가 없습니다.")
            return False

    # 차분 데이터 사용 시 원본 데이터도 세션 상태에 유지
    if st.session_state.use_differencing:
        # 차분 데이터로 학습하더라도 시각화를 위해 원본 데이터 유지
        if hasattr(st.session_state, 'series') and st.session_state.series is not None:
            if st.session_state.train is None or st.session_state.test is None:
                # 원본 시리즈로부터 train/test 데이터 분할
                from backend.data_service import cached_train_test_split
                try:
                    train, test = cached_train_test_split(st.session_state.series, st.session_state.test_size)
                    st.session_state.train = train
                    st.session_state.test = test
                    st.info("차분 데이터 사용 시 시각화를 위해 원본 데이터도 준비했습니다.")
                except Exception as e:
                    st.error(f"원본 데이터 분할 중 오류: {e}")

    # 진행 상황 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 예측 결과 및 메트릭 저장
    forecasts = {}
    metrics = {}
    
    # 모델 개수
    total_models = len(selected_models)
    completed_models = 0

    # 모델 파라미터 저장을 위한 초기화
    if 'model_params' not in st.session_state:
        st.session_state.model_params = {}
    
    # 각 모델 학습 및 예측
    for model_type in selected_models:
        status_text.text(f"{model_type} 모델 학습 중...")
        
        try:
            # 모델별 캐싱된 학습 함수 호출
            if model_type == 'arima':
                # 파라미터 저장
                st.session_state.model_params[model_type] = {
                    'order': arima_params.get('order', (1, 1, 1)),
                    'seasonal_order': arima_params.get('seasonal_order', (1, 1, 1, st.session_state.period)),
                    'seasonal': True,
                    'm': st.session_state.period,
                    **arima_params
                }
                
                # 차분 데이터 사용 여부에 따라 다른 함수 호출
                if st.session_state.use_differencing and st.session_state.diff_train is not None and st.session_state.diff_test is not None:
                    # ARIMA 모델은 차분 기능을 내장하고 있으므로,
                    # 차분을 이미 했다면 차수를 줄일 수 있음
                    modified_arima_params = arima_params.copy()
                    if 'order' in modified_arima_params:
                        p, d, q = modified_arima_params['order']
                        # 이미 차분을 했으므로 d를 줄임
                        modified_arima_params['order'] = (p, max(0, d - st.session_state.diff_order), q)
                    
                    try:
                        forecast, model_metrics = cached_train_arima_differenced(
                            train_data_key, 
                            test_data_key,
                            seasonal=True,
                            m=st.session_state.period,
                            **modified_arima_params
                        )
                        
                        # 역변환 적용
                        if forecast is not None:
                            from backend.data_service import inverse_transform_forecast
                            try:
                                forecast = inverse_transform_forecast(forecast)
                            except Exception as e:
                                st.error(f"ARIMA 예측 결과 역변환 중 오류: {e}")
                                forecast = None
                                model_metrics = None
                    except Exception as e:
                        st.error(f"차분 데이터로 ARIMA 모델 학습 중 오류: {e}")
                        # 원본 데이터로 학습 시도
                        if st.session_state.train is not None and st.session_state.test is not None:
                            st.warning("원본 데이터로 ARIMA 모델 학습을 시도합니다.")
                            train_key = hash(tuple(st.session_state.train.values.tolist()))
                            test_key = hash(tuple(st.session_state.test.values.tolist()))
                            forecast, model_metrics = cached_train_arima(
                                train_key, 
                                test_key,
                                seasonal=True,
                                m=st.session_state.period,
                                **arima_params
                            )
                        else:
                            forecast = None
                            model_metrics = None
                else:
                    # 원본 데이터로 학습
                    if st.session_state.train is not None and st.session_state.test is not None:
                        forecast, model_metrics = cached_train_arima(
                            train_data_key, 
                            test_data_key,
                            seasonal=True,
                            m=st.session_state.period,
                            **arima_params
                        )
                    else:
                        st.error(f"ARIMA 모델 학습을 위한 데이터가 없습니다.")
                        forecast = None
                        model_metrics = None
                    
            elif model_type == 'exp_smoothing':
                # 파라미터 저장
                st.session_state.model_params[model_type] = {
                    'model_type': 'hw',
                    'trend': 'add',
                    'seasonal': 'add',
                    'seasonal_periods': st.session_state.period,
                    'damped_trend': False
                }
                
                # 차분 데이터 사용 여부에 따라 다른 함수 호출
                if st.session_state.use_differencing and st.session_state.diff_train is not None and st.session_state.diff_test is not None:
                    try:
                        forecast, model_metrics = cached_train_exp_smoothing_differenced(
                            train_data_key, 
                            test_data_key,
                            seasonal_periods=st.session_state.period
                        )
                        
                        # 역변환 적용
                        if forecast is not None:
                            from backend.data_service import inverse_transform_forecast
                            try:
                                forecast = inverse_transform_forecast(forecast)
                            except Exception as e:
                                st.error(f"지수평활법 예측 결과 역변환 중 오류: {e}")
                                forecast = None
                                model_metrics = None
                    except Exception as e:
                        st.error(f"차분 데이터로 지수평활법 모델 학습 중 오류: {e}")
                        # 원본 데이터로 학습 시도
                        if st.session_state.train is not None and st.session_state.test is not None:
                            st.warning("원본 데이터로 지수평활법 모델 학습을 시도합니다.")
                            train_key = hash(tuple(st.session_state.train.values.tolist()))
                            test_key = hash(tuple(st.session_state.test.values.tolist()))
                            forecast, model_metrics = cached_train_exp_smoothing(
                                train_key, 
                                test_key,
                                seasonal_periods=st.session_state.period
                            )
                        else:
                            forecast = None
                            model_metrics = None
                else:
                    # 원본 데이터로 학습
                    if st.session_state.train is not None and st.session_state.test is not None:
                        forecast, model_metrics = cached_train_exp_smoothing(
                            train_data_key, 
                            test_data_key,
                            seasonal_periods=st.session_state.period
                        )
                    else:
                        st.error(f"지수평활법 모델 학습을 위한 데이터가 없습니다.")
                        forecast = None
                        model_metrics = None
                    
            elif model_type == 'prophet':
                # 파라미터 저장
                st.session_state.model_params[model_type] = {
                    'daily_seasonality': prophet_params.get('daily_seasonality', False),
                    'weekly_seasonality': prophet_params.get('weekly_seasonality', True),
                    'yearly_seasonality': prophet_params.get('yearly_seasonality', False),
                    'changepoint_prior_scale': prophet_params.get('changepoint_prior_scale', 0.05)
                }
                
                # Prophet은 항상 원본 데이터 사용 (차분 데이터 무시)
                if st.session_state.train is not None and st.session_state.test is not None:
                    train_hash = hash(tuple(st.session_state.train.values.tolist()))
                    test_hash = hash(tuple(st.session_state.test.values.tolist()))
                    forecast, model_metrics = cached_train_prophet(
                        train_hash, 
                        test_hash,
                        **prophet_params
                    )
                else:
                    st.error(f"Prophet 모델 학습을 위한 데이터가 없습니다.")
                    forecast = None
                    model_metrics = None
                
            elif model_type == 'lstm':
                # 파라미터 저장
                st.session_state.model_params[model_type] = {
                    'n_steps': lstm_params.get('n_steps', 24),
                    'lstm_units': lstm_params.get('lstm_units', [50]),
                    'epochs': lstm_params.get('epochs', 50)
                }
                
                # LSTM은 항상 원본 데이터로 학습 (차분 데이터 무시)
                if st.session_state.train is not None and st.session_state.test is not None:
                    train_key = hash(tuple(st.session_state.train.values.tolist()))
                    test_key = hash(tuple(st.session_state.test.values.tolist()))
                    forecast, model_metrics = cached_train_lstm(
                        train_key, 
                        test_key,
                        **lstm_params
                    )
                else:
                    st.error(f"LSTM 모델 학습을 위한 데이터가 없습니다.")
                    forecast = None
                    model_metrics = None
                    
            elif model_type == 'transformer':
                # 파라미터 저장
                st.session_state.model_params[model_type] = {
                    'window_size': transformer_params.get('window_size', 24),
                    'embed_dim': transformer_params.get('embed_dim', 64),
                    'num_heads': transformer_params.get('num_heads', 4),
                    'ff_dim': transformer_params.get('ff_dim', 128),
                    'num_layers': transformer_params.get('num_layers', 2),
                    'epochs': transformer_params.get('epochs', 50)
                }
                
                # 트랜스포머는 항상 원본 데이터로 학습 (차분 데이터 무시)
                if st.session_state.train is not None and st.session_state.test is not None:
                    train_key = hash(tuple(st.session_state.train.values.tolist()))
                    test_key = hash(tuple(st.session_state.test.values.tolist()))
                    forecast, model_metrics = cached_train_transformer(
                        train_key, 
                        test_key,
                        **transformer_params
                    )
                else:
                    st.error(f"트랜스포머 모델 학습을 위한 데이터가 없습니다.")
                    forecast = None
                    model_metrics = None
            
            # 유효한 결과만 저장
            if forecast is not None and model_metrics is not None:
                # 차분 데이터로 학습한 경우 메트릭 재계산 (원본 스케일로)
                if st.session_state.use_differencing:
                    model_name = model_metrics.get('name', model_type)
                    
                    # 차분 역변환 후 원본 test와 비교 (test가 있는 경우만)
                    if hasattr(st.session_state, 'test') and st.session_state.test is not None:
                        try:
                            metrics_from_test = evaluate_prediction(st.session_state.test, forecast)
                            model_metrics.update(metrics_from_test)
                        except Exception as e:
                            st.error(f"원본 스케일 메트릭 계산 중 오류: {e}")
                            # 오류 발생 시 기존 메트릭 유지
                            pass
                    else:
                        # 원본 test가 없으면 NaN 값으로 대체
                        model_metrics.update({
                            'MSE': float('nan'),
                            'RMSE': float('nan'),
                            'MAE': float('nan'),
                            'R^2': float('nan'),
                            'MAPE': float('nan')
                        })
                    
                    model_metrics['name'] = model_name  # 이름 복원
                
                forecasts[model_metrics.get('name', model_type)] = forecast
                metrics[model_metrics.get('name', model_type)] = model_metrics
            
            # 진행 상황 업데이트
            completed_models += 1
            progress_bar.progress(completed_models / total_models)
            
        except Exception as e:
            st.error(f"{model_type} 모델 학습 중 오류 발생: {traceback.format_exc()}")
    
    # 모든 모델 학습 완료 후 결과 저장
    if forecasts:
        st.session_state.forecasts = forecasts
        st.session_state.metrics = metrics
        st.session_state.models_trained = True
        
        # 최적 모델 선택
        rmse_values = {model: metrics[model]['RMSE'] for model in metrics}
        st.session_state.best_model = min(rmse_values.items(), key=lambda x: x[1])[0]
        
        status_text.text("모든 모델 학습 완료!")
        return True
    else:
        st.error("모델 학습 중 오류가 발생했습니다.")
        return False