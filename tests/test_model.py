from src.model import LogisticRegressionModel


def test_model_creation():
    model = LogisticRegressionModel()
    assert model is not None
