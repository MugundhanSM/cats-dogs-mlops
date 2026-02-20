from src.model import LogisticRegressionModel


def test_model_forward_pass():
    import torch

    model = LogisticRegressionModel(128 * 128 * 3, 2)
    dummy_input = torch.randn(1, 3, 128, 128)

    output = model(dummy_input)

    assert output.shape == (1, 2)
