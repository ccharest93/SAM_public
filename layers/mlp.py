import torch
import torch.nn as nn

from typing import Type

#MLP for inference
class mlp(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 hidden_features: int = None, 
                 out_features: int = None, 
                 act: Type[nn.Module] = nn.GELU,
                 bias = True) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.lin1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act
        self.lin2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        return x
    