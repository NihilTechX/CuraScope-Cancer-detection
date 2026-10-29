import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path

class ClinicalDataset(Dataset):
    """
    PyTorch Dataset for clinical tabular data.
    """
    def __init__(self, clinical_features: np.ndarray, times: np.ndarray, events: np.ndarray):
        self.features = torch.tensor(clinical_features, dtype=torch.float32)
        self.times = torch.tensor(times, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'time': self.times[idx],
            'event': self.events[idx]
        }

class MultimodalDataset(Dataset):
    """
    PyTorch Dataset that loads both clinical tabular features (NumPy) 
    and pre-computed image features (PyTorch .pt files) simultaneously.
    """
    def __init__(self, clinical_features: np.ndarray, times: np.ndarray, events: np.ndarray, patient_ids: pd.Series, features_dir: str, pool_image: bool = True):
        self.clinical_features = torch.tensor(clinical_features, dtype=torch.float32)
        self.times = torch.tensor(times, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.float32)
        self.patient_ids = patient_ids.reset_index(drop=True)
        self.features_dir = Path(features_dir)
        self.pool_image = pool_image
        
    def __len__(self):
        return len(self.clinical_features)
        
    def __getitem__(self, idx):
        patient_id = self.patient_ids.iloc[idx]
        clinical_feat = self.clinical_features[idx]
        
        patch_dir = self.features_dir.parent / "patch_features"
        patch_path = patch_dir / f"{patient_id}_patches.pt"
        legacy_path = self.features_dir / f"{patient_id}_image.pt"
        
        if patch_path.exists():
            image_feat = torch.load(patch_path, weights_only=True)
            if self.pool_image and image_feat.dim() > 1:
                image_feat = image_feat.mean(dim=0)
                mask = torch.ones(1, dtype=torch.bool)
            elif not self.pool_image and image_feat.dim() > 1:
                N = image_feat.shape[0]
                mask = torch.zeros(120, dtype=torch.bool)
                if N <= 120:
                    mask[:N] = True
                    if N < 120:
                        padding = torch.zeros((120 - N, 2048), dtype=torch.float32)
                        image_feat = torch.cat([image_feat, padding], dim=0)
                else:
                    mask[:] = True
                    image_feat = image_feat[:120]
            else:
                # Fallback if somehow dim == 1 but requested sequence
                if not self.pool_image:
                    image_feat = image_feat.unsqueeze(0)
                    mask = torch.zeros(120, dtype=torch.bool)
                    mask[0] = True
                    padding = torch.zeros((119, 2048), dtype=torch.float32)
                    image_feat = torch.cat([image_feat, padding], dim=0)
                else:
                    mask = torch.ones(1, dtype=torch.bool)
        elif legacy_path.exists():
            image_feat = torch.load(legacy_path, weights_only=True)
            if image_feat.dim() > 1:
                image_feat = image_feat.squeeze()
            if not self.pool_image:
                image_feat = image_feat.unsqueeze(0)
                mask = torch.zeros(120, dtype=torch.bool)
                mask[0] = True
                padding = torch.zeros((119, 2048), dtype=torch.float32)
                image_feat = torch.cat([image_feat, padding], dim=0)
            else:
                mask = torch.ones(1, dtype=torch.bool)
        else:
            raise FileNotFoundError(f"Missing feature file for patient {patient_id}. Searched {patch_path} and {legacy_path}")
            
        return {
            'clinical_features': clinical_feat,
            'image_features': image_feat,
            'image_mask': mask,
            'time': self.times[idx],
            'event': self.events[idx]
        }
