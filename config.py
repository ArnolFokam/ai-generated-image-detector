from transformers import PretrainedConfig

class AIGeneratedImageDetectorConfig(PretrainedConfig):
    model_type = "resnet"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)