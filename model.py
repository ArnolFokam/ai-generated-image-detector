import torch
import torch.nn as nn

from transformers import PreTrainedModel
from torchvision.models import resnet18

from config import AIGeneratedImageDetectorConfig

class AIGeneratedImageDetector(PreTrainedModel):
    config_class = AIGeneratedImageDetectorConfig

    def __init__(self, config):
        super().__init__(config)
        self.backbone = resnet18()
        self.classifier = nn.Linear(self.backbone.fc.weight.shape[0], 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x).flatten()
        return torch.sigmoid(x)