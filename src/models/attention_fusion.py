import torch
import torch.nn as nn
from src.models.clinical_encoder import ClinicalEncoder
from src.models.image_encoder import ImageEncoder
from src.models.survival_head import CoxSurvivalHead

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.3):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: Tensor of shape (batch, 1, d_model)
            key_value: Tensor of shape (batch, 1, d_model)
        """
        # MultiHeadAttention returns (attn_output, attn_weights)
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        
        # Add & Norm
        out = self.layer_norm(query + self.dropout(attn_output))
        return out

class LegacyAttentionFusionModel(nn.Module):
    def __init__(self, clinical_input_dim: int, image_input_dim: int = 2048, hidden_dim: int = 128, dropout_rate: float = 0.3):
        """
        Legacy Bidirectional Cross-Modal Attention Model (100-Patient Baseline).
        Operated on a single 1D mean-pooled image vector.
        """
        super().__init__()
        
        # 1. Modality-specific Encoders (both map to hidden_dim)
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
        
        # 2. Cross-Attention Blocks
        self.image_q_clinical_kv = CrossAttention(d_model=hidden_dim, nhead=4, dropout=dropout_rate)
        self.clinical_q_image_kv = CrossAttention(d_model=hidden_dim, nhead=4, dropout=dropout_rate)
        
        # 3. Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 4. Survival Head
        self.head = CoxSurvivalHead(d_input=hidden_dim)
        
    def forward(self, clinical_x: torch.Tensor, image_x: torch.Tensor) -> torch.Tensor:
        h_clin = self.clinical_encoder(clinical_x) 
        h_img = self.image_encoder(image_x)        
        
        h_clin_seq = h_clin.unsqueeze(1)
        h_img_seq = h_img.unsqueeze(1)
        
        attended_img = self.image_q_clinical_kv(query=h_img_seq, key_value=h_clin_seq).squeeze(1)
        attended_clin = self.clinical_q_image_kv(query=h_clin_seq, key_value=h_img_seq).squeeze(1)
        
        fused = torch.cat([attended_clin, attended_img], dim=1) 
        h_final = self.fusion_mlp(fused)
        return self.head(h_final)

class ABMILCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.w_img = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_clin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, img_patches: torch.Tensor, clin_context: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            img_patches: (Batch, N, hidden_dim)
            clin_context: (Batch, hidden_dim)
            mask: (Batch, N) boolean tensor, True for valid patches, False for padding
        Returns:
            attention_weights: (Batch, N)
            context_vector: (Batch, hidden_dim)
        """
        # Expand clinical context to match N patches
        # clin_context: (Batch, 1, hidden_dim)
        clin_expanded = clin_context.unsqueeze(1)
        
        # Additive attention: v^T * tanh(W_img * img + W_clin * clin)
        # projected_img: (Batch, N, hidden_dim)
        # projected_clin: (Batch, 1, hidden_dim)
        energy = self.tanh(self.w_img(img_patches) + self.w_clin(clin_expanded)) # (Batch, N, hidden_dim)
        
        # scores: (Batch, N, 1)
        scores = self.v(energy)
        
        if mask is not None:
            # Mask out invalid patches by setting their score to -inf before softmax
            scores = scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        
        # Softmax over N patches
        attention_weights = torch.softmax(scores, dim=1) # (Batch, N, 1)
        
        # Handle case where entire sequence might be masked (e.g. edge cases, though prevented by N>0)
        # Softmax of -inf is 0, but if all are -inf, softmax returns NaN.
        # We replace NaNs with 0 if that happens
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Weighted sum: sum(weights * img_patches, dim=1)
        # (Batch, N, 1) * (Batch, N, hidden_dim) -> (Batch, N, hidden_dim) -> sum -> (Batch, hidden_dim)
        context_vector = torch.sum(attention_weights * img_patches, dim=1)
        
        return attention_weights.squeeze(-1), context_vector

class SpatialAttentionFusionModel(nn.Module):
    def __init__(self, clinical_input_dim: int, image_input_dim: int = 2048, hidden_dim: int = 128, dropout_rate: float = 0.4):
        """
        Lightweight Spatial Cross-Modal Attention Model (ABMIL-inspired).
        Accepts raw patch-level sequences [Batch, N, 2048] and computes clinically-guided attention per patch.
        Designed specifically for low-event survival datasets to heavily restrict capacity and prevent memorization.
        """
        super().__init__()
        
        self.clinical_encoder = ClinicalEncoder(
            d_clin=clinical_input_dim, 
            d_emb=hidden_dim, 
            dropout=dropout_rate
        )
        
        # Lightweight patch projection: project each patch independently 
        # (Batch, N, 2048) -> (Batch, N, hidden_dim)
        self.patch_projector = nn.Sequential(
            nn.Linear(image_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Clinically-guided ABMIL attention
        self.cross_attention = ABMILCrossAttention(hidden_dim=hidden_dim)
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Survival Head
        self.head = CoxSurvivalHead(d_input=hidden_dim)
        
        # State variable for interpretability
        self._last_attention_weights = None

    def forward(self, clinical_x: torch.Tensor, image_patches: torch.Tensor, image_mask: torch.Tensor = None, return_attention: bool = False):
        """
        Args:
            clinical_x: (Batch, clinical_input_dim)
            image_patches: (Batch, N, 2048)
            image_mask: (Batch, N) boolean
            return_attention: bool, if True returns tuple of (risk_score, attention_weights)
        """
        # Ensure image_patches is 3D (Batch, N, 2048). 
        # If it happens to be 2D (e.g. N=1 during edge cases), fix it.
        if image_patches.dim() == 2:
            image_patches = image_patches.unsqueeze(1)
            
        h_clin = self.clinical_encoder(clinical_x) # (Batch, hidden_dim)
        h_patches = self.patch_projector(image_patches) # (Batch, N, hidden_dim)
        
        attn_weights, img_context = self.cross_attention(h_patches, h_clin, mask=image_mask)
        self._last_attention_weights = attn_weights
        
        # Concatenate clinically-guided image context with original clinical features
        fused = torch.cat([h_clin, img_context], dim=1) # (Batch, hidden_dim * 2)
        h_final = self.fusion_mlp(fused)
        risk = self.head(h_final)
        
        if return_attention:
            return risk, attn_weights
        return risk

    def get_attention_weights(self) -> torch.Tensor:
        """
        Retrieves the attention weights for the patches from the last forward pass.
        Useful for interpretability and spatial visualization.
        """
        return self._last_attention_weights
