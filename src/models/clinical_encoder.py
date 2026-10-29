import torch
import torch.nn as nn

class ClinicalEncoder(nn.Module):
    """
    Encodes tabular clinical features into a dense embedding.
    Uses a 2-layer MLP with BatchNorm and Dropout to prevent overfitting.
    """
    def __init__(self, d_clin: int, d_emb: int = 128, dropout: float = 0.3):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(d_clin, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            
            nn.Linear(64, d_emb),
            nn.ReLU(),
            nn.BatchNorm1d(d_emb)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, d_clin]
        Returns:
            Tensor of shape [batch_size, d_emb]
        """
        return self.net(x)
