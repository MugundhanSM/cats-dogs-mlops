from src.inference import Predictor

def test_model_load():
    predictor = Predictor()
    assert predictor.model is not None
