import torch
import torch.nn as nn
from src.models.clinical_encoder import ClinicalEncoder
from src.models.image_encoder import ImageEncoder
from src.models.survival_head import CoxSurvivalHead
from src.models.concat_fusion import ConcatFusionModel
from src.models.attention_fusion import LegacyAttentionFusionModel, SpatialAttentionFusionModel

class ClinicalSurvivalModel(nn.Module):
    def __init__(self, encoder: ClinicalEncoder, head: CoxSurvivalHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

class ImageSurvivalModel(nn.Module):
    def __init__(self, encoder: ImageEncoder, head: CoxSurvivalHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

def create_model(model_name: str, **kwargs) -> nn.Module:
    """
    Unified factory to instantiate models M1, M2, M3, M4.
    """
    name = model_name.lower()

    if name in ["m1", "clinical", "clinical_only"]:
        enc = ClinicalEncoder(
            d_clin=kwargs["clinical_input_dim"],
            d_emb=kwargs.get("hidden_dim", 128),
            dropout=kwargs.get("dropout_rate", 0.3)
        )
        head = CoxSurvivalHead(d_input=kwargs.get("hidden_dim", 128))
        return ClinicalSurvivalModel(enc, head)

    elif name in ["m2", "image", "image_only"]:
        enc = ImageEncoder(
            input_dim=kwargs.get("image_input_dim", 2048),
            hidden_dim=kwargs.get("img_hidden_dim", 256),
            output_dim=kwargs.get("hidden_dim", 128),
            dropout_rate=kwargs.get("dropout_rate", 0.3)
        )
        head = CoxSurvivalHead(d_input=kwargs.get("hidden_dim", 128))
        return ImageSurvivalModel(enc, head)

    elif name in ["m3", "concat", "concat_fusion"]:
        return ConcatFusionModel(
            clinical_input_dim=kwargs["clinical_input_dim"],
            image_input_dim=kwargs.get("image_input_dim", 2048),
            hidden_dim=kwargs.get("hidden_dim", 128),
            dropout_rate=kwargs.get("dropout_rate", 0.3)
        )

    elif name in ["m4", "attention", "attention_fusion"]:
        return SpatialAttentionFusionModel(
            clinical_input_dim=kwargs["clinical_input_dim"],
            image_input_dim=kwargs.get("image_input_dim", 2048),
            hidden_dim=kwargs.get("hidden_dim", 128),
            dropout_rate=kwargs.get("dropout_rate", 0.4)
        )

    elif name in ["legacy_m4"]:
        return LegacyAttentionFusionModel(
            clinical_input_dim=kwargs["clinical_input_dim"],
            image_input_dim=kwargs.get("image_input_dim", 2048),
            hidden_dim=kwargs.get("hidden_dim", 128),
            dropout_rate=kwargs.get("dropout_rate", 0.3)
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from ['m1', 'm2', 'm3', 'm4'].")
