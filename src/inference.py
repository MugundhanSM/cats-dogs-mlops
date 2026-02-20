import torch
import json
from PIL import Image

from src.model import LogisticRegressionModel
from src.preprocess import get_val_transforms
from src.config import Config


class Predictor:
    def __init__(self):
        self.cfg = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = self.cfg.image_size * self.cfg.image_size * 3

        self.model = LogisticRegressionModel(
            input_dim=input_dim,
            num_classes=self.cfg.num_classes
        )

        self.model.load_state_dict(
            torch.load(self.cfg.model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        self.transform = get_val_transforms(self.cfg.image_size)

        # Load class names
        with open("class_names.json", "r") as f:
            self.class_names = json.load(f)

        print("Loaded classes:", self.class_names)

    def predict(self, image_path: str):
        # Load image
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        label = self.class_names[predicted.item()]

        return {
            "label": label,
            "confidence": float(confidence.item())
        }
