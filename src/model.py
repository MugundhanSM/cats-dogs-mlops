import torch
import torch.nn as nn


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim=128*128*3, num_classes=2):
        super().__init__()

        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)   
        return self.classifier(x)
