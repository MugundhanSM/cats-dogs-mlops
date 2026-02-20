from dataclasses import dataclass

@dataclass
class Config:
    image_size: int = 128
    batch_size: int = 32
    lr: float = 1e-5
    epochs: int = 15
    num_classes: int = 2

    train_dir: str = "data/processed/train"
    val_dir: str = "data/val"
    model_path: str = "models/model.pt"