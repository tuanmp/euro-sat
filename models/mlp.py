import torch

class ClassifierHead(torch.nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 num_classes: int, 
                 dropout: float = 0.1, 
                 hidden_layers: int = 0,
                 hidden_dim: int = 512,
                 batch_norm: bool = False,
                 layer_norm: bool = False,
                 hidden_activation: str = "relu",
                 output_activation = None):
        
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes 
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        if self.batch_norm and self.layer_norm:
            raise ValueError("Cannot use both batch normalization and layer normalization at the same time.")

        self.classifier = self.make_mlp()
        
    def make_mlp(self):
        layers = []
        in_dim = self.input_dim
        
        for i in range(self.hidden_layers):
            layers.append(torch.nn.Linear(in_dim, self.hidden_dim))
            if self.batch_norm:
                layers.append(torch.nn.BatchNorm1d(self.hidden_dim))
            if self.layer_norm:
                layers.append(torch.nn.LayerNorm(self.hidden_dim))
            if self.hidden_activation:
                layers.append(getattr(torch.nn, self.hidden_activation)())
            if self.dropout > 0:
                layers.append(torch.nn.Dropout(self.dropout))
            in_dim = self.hidden_dim
            
        layers.append(torch.nn.Linear(in_dim, self.num_classes))
        if self.output_activation:
            layers.append(getattr(torch.nn, self.output_activation)())
        
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)