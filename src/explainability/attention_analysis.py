import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze_m4_attention(
    m4_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extracts and analyzes patch-level spatial cross-modal attention weights
    from M4 (SpatialAttentionFusionModel).
    """
    m4_model.eval()
    
    all_attentions = []
    all_masks = []
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in dataloader:
            clin = batch['clinical_features'].to(device)
            img = batch['image_features'].to(device)
            mask = batch['image_mask'].to(device)
            
            risks = m4_model(clin, img, image_mask=mask)
            attn = m4_model.get_attention_weights()  # [Batch, 120]
            
            all_attentions.append(attn.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
            all_risks.append(risks.cpu().numpy().flatten())
            all_times.append(batch['time'].cpu().numpy())
            all_events.append(batch['event'].cpu().numpy())
            
    attentions = np.concatenate(all_attentions, axis=0) # [N_patients, 120]
    masks = np.concatenate(all_masks, axis=0)           # [N_patients, 120]
    risks = np.concatenate(all_risks, axis=0)           # [N_patients]
    times = np.concatenate(all_times, axis=0)           # [N_patients]
    events = np.concatenate(all_events, axis=0)         # [N_patients]
    
    n_patients = len(risks)
    median_risk = np.median(risks)
    high_risk = risks > median_risk
    
    # Compute per-patient attention statistics
    entropies = []
    normalized_entropies = []
    max_weights = []
    top_patch_indices = []
    valid_patch_counts = []
    
    eps = 1e-12
    for i in range(n_patients):
        valid_n = int(masks[i].sum())
        valid_patch_counts.append(valid_n)
        
        valid_attn = attentions[i, :valid_n]
        if valid_n > 0:
            valid_attn = valid_attn / (np.sum(valid_attn) + eps)
            ent = -np.sum(valid_attn * np.log(valid_attn + eps))
            max_ent = np.log(valid_n) if valid_n > 1 else 1.0
            norm_ent = ent / max_ent if max_ent > 0 else 1.0
            max_w = float(np.max(valid_attn))
            top_idx = int(np.argmax(valid_attn))
        else:
            ent, norm_ent, max_w, top_idx = 0.0, 1.0, 0.0, 0
            
        entropies.append(float(ent))
        normalized_entropies.append(float(norm_ent))
        max_weights.append(max_w)
        top_patch_indices.append(top_idx)
        
    entropies = np.array(entropies)
    normalized_entropies = np.array(normalized_entropies)
    max_weights = np.array(max_weights)
    
    # Stratified stats
    high_ent_mean = float(np.mean(normalized_entropies[high_risk]))
    low_ent_mean = float(np.mean(normalized_entropies[~high_risk]))
    high_max_mean = float(np.mean(max_weights[high_risk]))
    low_max_mean = float(np.mean(max_weights[~high_risk]))
    
    stats_summary = {
        "n_patients": n_patients,
        "mean_valid_patches": float(np.mean(valid_patch_counts)),
        "mean_normalized_entropy": float(np.mean(normalized_entropies)),
        "high_risk_mean_normalized_entropy": high_ent_mean,
        "low_risk_mean_normalized_entropy": low_ent_mean,
        "high_risk_mean_max_attention": high_max_mean,
        "low_risk_mean_max_attention": low_max_mean,
        "overall_mean_max_attention": float(np.mean(max_weights))
    }
    
    # Plotting
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
        
        # Plot 1: Normalized Attention Entropy (Concentration vs Uniformity)
        ax1 = axes[0]
        data_to_plot = [normalized_entropies[~high_risk], normalized_entropies[high_risk]]
        bp1 = ax1.boxplot(data_to_plot, tick_labels=["Low Risk", "High Risk"], patch_artist=True)
        colors = ["#1f77b4", "#d62728"]
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax1.set_ylabel("Normalized Attention Entropy (0=Hotspot, 1=Uniform)", fontsize=10)
        ax1.set_title("Attention Entropy by Risk Group", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.3)
        
        # Plot 2: Max Patch Attention Weight (Peak Saliency)
        ax2 = axes[1]
        data_to_plot_max = [max_weights[~high_risk], max_weights[high_risk]]
        bp2 = ax2.boxplot(data_to_plot_max, tick_labels=["Low Risk", "High Risk"], patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax2.set_ylabel("Peak Patch Attention Weight", fontsize=10)
        ax2.set_title("Peak Saliency by Risk Group", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.3)
        
        # Plot 3: Attention distribution heatmap across representative patients
        ax3 = axes[2]
        # Sort patients by risk score
        sort_order = np.argsort(risks)
        sample_subset = sort_order[np.linspace(0, n_patients - 1, 30, dtype=int)]
        subset_attn = attentions[sample_subset, :60]  # first 60 patch positions
        
        im = ax3.imshow(subset_attn, aspect="auto", cmap="magma", interpolation="nearest")
        ax3.set_xlabel("Patch Position Index", fontsize=10)
        ax3.set_ylabel("Patients (Ranked Low -> High Risk)", fontsize=10)
        ax3.set_title("Patch Attention Weights Heatmap", fontsize=11, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label("Attention Weight", fontsize=9)
        
        plt.suptitle("M4 Spatial Cross-Modal Histology Patch Attention Analysis", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
    return stats_summary
