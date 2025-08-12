def validate_model_parameters(model_type: str, parameters: dict[str, any]) -> dict[str, any]:
    """
    모델 타입에 따른 파라미터 유효성 검사 및 형식 변환
    """
    if model_type == 'prophet':
        return validate_prophet_parameters(parameters)
    elif model_type == 'lstm':
        return validate_lstm_parameters(parameters)
    else:
        return parameters
    
def validate_prophet_parameters(parameters: dict[str, any]) -> dict[str, any]:
    """
    Prophet 모델 파라미터 검증
    """
    valid_params = {}
    
    # 유효한 Prophet 파라미터만 추출
    valid_keys = [
        'daily_seasonality', 'weekly_seasonality', 'yearly_seasonality',
        'seasonality_mode', 'changepoint_prior_scale', 'seasonality_prior_scale',
        'holidays_prior_scale'
    ]
    
    for key in valid_keys:
        if key in parameters:
            valid_params[key] = parameters[key]
    
    return valid_params

def validate_lstm_parameters(parameters: dict[str, any]) -> dict[str, any]:
    """
    LSTM 모델 파라미터 검증
    """
    valid_params = {}
    
    # 유효한 파라미터만 추출
    valid_keys = [
        'n_steps', 'lstm_units', 'dropout_rate', 'epochs', 
        'batch_size', 'validation_split', 'early_stopping', 'patience'
    ]
    
    for key in valid_keys:
        if key in parameters:
            valid_params[key] = parameters[key]
    
    return valid_params