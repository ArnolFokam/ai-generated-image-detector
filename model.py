import torch.nn.functional as F

from transformers import PreTrainedModel
from torchvision import models

from config import AIGeneratedImageDetectorConfig

    
class AIGeneratedImageDetectorForClassification(PreTrainedModel):
    config_class = AIGeneratedImageDetectorConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = models.__dict__[config.backbone_name](
            weights=config.weights,
            num_classes=config.num_classes
        )

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}