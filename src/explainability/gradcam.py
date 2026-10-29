import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for convolutional backbones
    (e.g., ResNet-50 layer4) predicting Cox hazard/survival risk.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)
        
    def _save_activations(self, module, input, output):
        self.activations = output
        
    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output is a tuple; first element contains the gradient wrt output
        self.gradients = grad_output[0]
        
    def generate_cam(self, input_tensor: torch.Tensor, target_risk: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Generates a 2D Grad-CAM heatmap for the given input image tensor [B, C, H, W].
        """
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        if target_risk is None:
            # For Cox survival, higher score indicates higher mortality risk
            score = output.sum()
        else:
            score = target_risk.sum()
            
        score.backward(retain_graph=True)
        
        # [B, C, H, W]
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling of gradients to compute channel weights alpha_k
        alpha = torch.mean(gradients, dim=(2, 3), keepdim=True) # [B, C, 1, 1]
        
        # Weighted combination of feature maps
        cam = torch.sum(alpha * activations, dim=1, keepdim=True) # [B, 1, H, W]
        cam = F.relu(cam)
        
        # Upsample to input tensor resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.detach().cpu().numpy()
        
        # Normalize between 0 and 1 per sample
        b, _, h, w = cam.shape
        cams_norm = np.zeros((b, h, w), dtype=np.float32)
        for i in range(b):
            c_min, c_max = cam[i, 0].min(), cam[i, 0].max()
            if c_max > c_min:
                cams_norm[i] = (cam[i, 0] - c_min) / (c_max - c_min)
            else:
                cams_norm[i] = cam[i, 0]
                
        return cams_norm
        
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def overlay_heatmap_on_image(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays a 2D Grad-CAM heatmap onto an RGB image (values 0-255 or 0-1).
    """
    import matplotlib.cm as cm
    color_map = cm.get_cmap("jet")
    colored_cam = color_map(heatmap)[:, :, :3]  # Drop alpha channel [H, W, 3]
    
    if image_rgb.max() > 1.0:
        image_rgb = image_rgb / 255.0
        
    overlay = (1 - alpha) * image_rgb + alpha * colored_cam
    return np.clip(overlay, 0.0, 1.0)
