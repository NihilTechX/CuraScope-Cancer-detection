import torch
import torch.nn as nn
from src.models.clinical_encoder import ClinicalEncoder
from src.models.image_encoder import ImageEncoder
from src.models.survival_head import CoxSurvivalHead

class ConcatFusionModel(nn.Module):
    """
    Simple Multimodal Concatenation Baseline (M3).
    Encodes clinical features and image embeddings into a shared hidden dimension,
    concatenates them, and passes through a fusion MLP into the Cox survival head.
    """
    def __init__(
        self,
        clinical_input_dim: int,
        image_input_dim: int = 2048,
        hidden_dim: int = 128,
        dropout_rate: float = 0.3
    ):
        super().__init__()

        self.clinical_encoder = ClinicalEncoder(
            d_clin=clinical_input_dim,
            d_emb=hidden_dim,
            dropout=dropout_rate
        )

        self.image_encoder = ImageEncoder(
            input_dim=image_input_dim,
            hidden_dim=256,
            output_dim=hidden_dim,
            dropout_rate=dropout_rate
        )

        # Fusion MLP after naive concatenation: hidden_dim * 2 -> hidden_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.head = CoxSurvivalHead(d_input=hidden_dim)

    def forward(self, clinical_x: torch.Tensor, image_x: torch.Tensor) -> torch.Tensor:
        h_clin = self.clinical_encoder(clinical_x)
        h_img = self.image_encoder(image_x)

        fused = torch.cat([h_clin, h_img], dim=1)
        h_final = self.fusion_mlp(fused)

        risk = self.head(h_final)
        return risk
