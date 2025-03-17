from transformers import ViTModel as TransformersViTModel
import torch
from .mlp import ClassifierHead
import os

cache_dir = os.environ.get("PSCRATCH", '/tmp')

class ViTModel(torch.nn.Module):
    def __init__(self, pretrained_model_name: str, num_classes: int, dropout: float = 0.1, 
                 hidden_layers: int = 0,
                 hidden_dim: int = 512,
                 batch_norm: bool = False,
                 layer_norm: bool = False,
                 hidden_activation: str = "relu",
                 output_activation = None):
        
        super().__init__()
        self.backbone = TransformersViTModel.from_pretrained(pretrained_model_name, cache_dir=cache_dir)
        self.classifier = ClassifierHead(
            input_dim=self.backbone.config.hidden_size,
            num_classes=num_classes,
            dropout=dropout,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )

    def forward(self, x):
        last_hidden_state = self.backbone(x).last_hidden_state[:, 0, :]
        logits = self.classifier(last_hidden_state)
        return logits
    

