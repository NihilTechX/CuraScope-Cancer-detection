import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 256, output_dim: int = 128, dropout_rate: float = 0.3):
        """
        Projects a high-dimensional image embedding (e.g., 2048-D from ResNet-50) 
        down to a compact representation for survival prediction or fusion.
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, input_dim)
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
