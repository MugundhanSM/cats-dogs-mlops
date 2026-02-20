from src.model import LogisticRegressionModel

def test_model_creation():
    model = LogisticRegressionModel(input_dim=128*128*3, num_classes=2)
    assert model is not None
