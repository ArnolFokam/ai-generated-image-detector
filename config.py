from typing import Optional

from transformers import PretrainedConfig
from torchvision.models._api import WeightsEnum

class AIGeneratedImageDetectorConfig(PretrainedConfig):
    model_type = "resnet"
    
    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 2,
        weights: Optional[WeightsEnum] = None,
        **kwargs):
        assert backbone_name.startswith("resnet")
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.weights = weights
        super().__init__(**kwargs)