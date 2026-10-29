import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

class WSIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features_dir: str | Path):
        """
        Args:
            df: DataFrame containing at least 'patient_id', 'OS.time', 'OS'.
            features_dir: Directory containing the .pt image feature files.
        """
        self.df = df.reset_index(drop=True)
        self.features_dir = Path(features_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['patient_id']
        
        # Determine path (try new patch_features first, fallback to legacy features)
        patch_dir = self.features_dir.parent / "patch_features"
        patch_path = patch_dir / f"{patient_id}_patches.pt"
        legacy_path = self.features_dir / f"{patient_id}_image.pt"
        
        if patch_path.exists():
            image_feat = torch.load(patch_path, weights_only=True)
            if image_feat.dim() > 1:
                image_feat = image_feat.mean(dim=0)
        elif legacy_path.exists():
            image_feat = torch.load(legacy_path, weights_only=True)
            if image_feat.dim() > 1:
                image_feat = image_feat.squeeze()
        else:
            raise FileNotFoundError(f"Missing feature file for patient {patient_id}. Searched {patch_path} and {legacy_path}")
            
        os_time = torch.tensor(row['OS.time'], dtype=torch.float32)
        os_event = torch.tensor(row['OS'], dtype=torch.float32)
        
        return {
            'features': image_feat,
            'time': os_time,
            'event': os_event
        }
