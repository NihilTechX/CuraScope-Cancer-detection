import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from src.data.clinical_preprocessing import ClinicalPreprocessor

def get_clean_feature_names(preprocessor: ClinicalPreprocessor) -> List[str]:
    """
    Extracts clean, readable clinical feature names from the fitted preprocessor.
    """
    try:
        raw_names = preprocessor.preprocessor.get_feature_names_out()
        cleaned = []
        name_map = {
            "age_at_index": "Age",
            "year_of_diagnosis": "Year of Diagnosis",
            "ajcc_pathologic_stage": "AJCC Stage",
            "ajcc_pathologic_t": "Pathologic T",
            "ajcc_pathologic_n": "Pathologic N",
            "ajcc_pathologic_m": "Pathologic M",
            "gender": "Gender",
            "race": "Race",
            "primary_diagnosis": "Primary Diagnosis"
        }
        for n in raw_names:
            # strip pipeline prefixes like 'num__', 'ord__', 'ohe__'
            parts = n.split("__")
            name = parts[-1] if len(parts) > 1 else n
            name = name.replace(".demographic", "").replace(".diagnoses", "")
            
            # Map known names
            found = False
            for k, v in name_map.items():
                if name.startswith(k):
                    remainder = name[len(k):]
                    name = v + remainder
                    found = True
                    break
            cleaned.append(name)
        return cleaned
    except Exception:
        dim = preprocessor.get_feature_dim()
        return [f"Feature_{i}" for i in range(dim)]

def explain_clinical_model(
    model: nn.Module,
    X_background: np.ndarray,
    X_eval: np.ndarray,
    preprocessor: ClinicalPreprocessor,
    device: torch.device,
    save_path: Optional[str] = None,
    max_eval_samples: int = 50,
    background_samples: int = 20
) -> Dict[str, Any]:
    """
    Computes SHAP feature importance values for the clinical survival model
    and exports summary visualization plots.
    """
    model.eval()

    def predict_risk(x_numpy: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_tensor = torch.tensor(x_numpy, dtype=torch.float32, device=device)
            if hasattr(model, 'encoder') and hasattr(model, 'head'):
                # M1 model (ClinicalSurvivalModel)
                out = model(x_tensor)
            elif hasattr(model, 'clinical_encoder'):
                # Model with clinical_encoder
                h = model.clinical_encoder(x_tensor)
                if hasattr(model, 'head'):
                    out = model.head(h)
                else:
                    out = h.mean(dim=-1, keepdim=True)
            else:
                out = model(x_tensor)
            return out.cpu().numpy().flatten()

    # Create background dataset for KernelExplainer
    if len(X_background) > background_samples:
        indices = np.random.RandomState(42).choice(len(X_background), background_samples, replace=False)
        bg = X_background[indices]
    else:
        bg = X_background

    # Select evaluation samples
    if len(X_eval) > max_eval_samples:
        eval_indices = np.random.RandomState(42).choice(len(X_eval), max_eval_samples, replace=False)
        eval_data = X_eval[eval_indices]
    else:
        eval_data = X_eval

    feature_names = get_clean_feature_names(preprocessor)

    # Initialize SHAP KernelExplainer
    explainer = shap.KernelExplainer(predict_risk, bg)
    shap_values = explainer.shap_values(eval_data, nsamples=100)

    # Calculate global feature importance (mean absolute SHAP value)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    ranking = [
        {"feature": feature_names[i], "mean_abs_shap": float(mean_abs_shap[i])}
        for i in sorted_idx
    ]

    # Generate summary plot
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(9, 6), dpi=300)
        shap.summary_plot(
            shap_values,
            eval_data,
            feature_names=feature_names,
            show=False,
            max_display=15
        )
        plt.title("Clinical Feature Attribution (SHAP) on Survival Risk", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return {
        "feature_names": feature_names,
        "feature_ranking": ranking,
        "mean_abs_shap": {feature_names[i]: float(mean_abs_shap[i]) for i in sorted_idx}
    }
