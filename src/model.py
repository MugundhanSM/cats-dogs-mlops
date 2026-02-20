import torch.nn as nn


class LogisticRegressionModel(nn.Module):
    def __init__(self, image_size=128, num_classes=2):
        super(LogisticRegressionModel, self).__init__()

        input_dim = 3 * image_size * image_size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
