from src.config import Config

def test_image_size():
    cfg = Config()
    assert cfg.image_size == 128
