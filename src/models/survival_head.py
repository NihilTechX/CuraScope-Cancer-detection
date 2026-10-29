import torch
import torch.nn as nn

class CoxSurvivalHead(nn.Module):
    """
    Maps a hidden representation to a scalar log-hazard risk score.
    Higher score --> higher predicted risk of death.
    """
    def __init__(self, d_input: int):
        super().__init__()
        # In standard Cox PH, there is no bias term in the linear predictor
        self.risk_layer = nn.Linear(d_input, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden representation tensor of shape [batch_size, d_input]
        Returns:
            Risk scores: Tensor of shape [batch_size]
        """
        # Squeeze the last dimension to return a 1D tensor of risk scores
        return self.risk_layer(h).squeeze(-1)
