from src.model import LogisticRegressionModel

def test_model_creation():
    model = LogisticRegressionModel(128*128*3, 2)
    assert model is not None
