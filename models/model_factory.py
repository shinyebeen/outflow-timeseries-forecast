"""
시계열 모델 생성을 위한 팩토리 패턴 구현 모듈
"""
import warnings

from models.base_model import TimeSeriesModel
from utils.singleton import Singleton

def get_prophet():
    return None 

def get_lstm():
    return None 

def get_xgb():
    return None 

def get_cat():
    return None 

class ModelFactory(metaclass = Singleton):
    """
    시계열 모델 생성을 위한 팩토리 클래스
    싱글턴 패턴을 적용하여 메모리 효율성 확보 
    """

    def __init__(self):
        """
        모델 팩토리 초기화
        """
        self.avaliable_models = {}

        prophet = get_prophet()
        if prophet:
            self.avaliable_models['prophet'] = prophet 
        
        lstm = get_lstm()
        if lstm:
            self.avaliable_models['lstm'] = lstm 
        
        xgb = get_xgb()
        if xgb:
            self.avaliable_models['xgb'] = xgb

        cat = get_cat()
        if cat:
            self.avaliable_models['cat'] = cat
    
    def get_model(self, model_type: str, **kwargs) -> TimeSeriesModel:
        """
        지정된 유형의 모델 인스턴스를 생성합니다.

        Args:
            model_type: 모델 유형
            **kwargs: 모델 생성자에 전달할 추가 인자 

        Returns:
            생성된 모델 인스턴스

        Raises:
            ValueError: 존재하지 않는 모델 유형인 경우 
        """

        model_type = model_type.lower()

        if model_type not in self.avaliable_models:
            available_types = ", ".join(self.avaliable_models.keys())
            raise ValueError(f'존재하지 않는 모델 유형입니다: {model_type}. f"사용 가능한 모델 유형: {available_types}')
        
        # 사용 가능한 모델 클래스 
        model_class = self.avaliable_models[model_type]

        # 모델 인스턴스 생성 
        model = model_class(**kwargs)

        return model
    
    def get_all_available_models(self) -> list[str]:
        """
        사용 가능한 모델 유형 반환
        
        Returns:
            사용 가능한 모델 유형 목록
        """
        return list(self.avaliable_models.keys())

    def create_all_models(self, **kwargs) -> dict[str, TimeSeriesModel]:
        """
        Args:
            **kwargs: 모델 생성자에 전달할 추가 인자
        
        Returns:
            모델 유형을 키로, 모델 인스턴스를 값으로 하는 딕셔너리
        """

        models = {}

        for model_type in self.avaliable_models:
            try:
                models[model_type] = self.get_model(model_type, **kwargs)
            except Exception as e:
                warnings.warn(f"{model_type} 모델 생성 중 오류 발생: {e}")
        
        return models 