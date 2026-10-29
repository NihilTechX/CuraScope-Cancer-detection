import os
import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.wsi_dataset import WSIDataset
from src.models.image_encoder import ImageEncoder
from src.models.survival_head import CoxSurvivalHead
from src.training.cox_loss import CoxPHLoss
from src.training.trainer import SurvivalTrainer
from src.utils.logger import setup_logger

class ImageSurvivalModel(nn.Module):
    def __init__(self, encoder: ImageEncoder, head: CoxSurvivalHead):
        super().__init__()
        self.encoder = encoder
        self.head = head
        
    def forward(self, x):
        features = self.encoder(x)
        risk = self.head(features)
        return risk

def main():
    print("="*50)
    print("TRAINING IMAGE-ONLY BASELINE (M2)")
    print("="*50)
    
    # 1. Setup
    set_seed(42)
    config = load_config(project_root / "configs" / "image_only.yaml")
    logger = setup_logger("M2_Image_Only")
    
    # 2. Load Cohort and Filter
    dataset_dir = project_root / "dataset"
    cohort_df = pd.read_csv(dataset_dir / "patient_cohort.csv")
    features_dir = dataset_dir / "features"
    
    # Only keep patients that actually have an image feature file downloaded
    def has_image(patient_id):
        return (features_dir / f"{patient_id}_image.pt").exists()
        
    cohort_df['has_image'] = cohort_df['patient_id'].apply(has_image)
    valid_df = cohort_df[cohort_df['has_image']].copy()
    
    print(f"Total cohort size: {len(cohort_df)}")
    print(f"Patients with downloaded image features: {len(valid_df)}")
    
    if len(valid_df) == 0:
        print("Error: No image features found in dataset/features/")
        sys.exit(1)
        
    # Split
    train_df = valid_df[valid_df['split'] == 'train']
    val_df = valid_df[valid_df['split'] == 'val']
    test_df = valid_df[valid_df['split'] == 'test']
    
    print(f"\nSplits -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # 3. Datasets and Loaders
    train_dataset = WSIDataset(train_df, features_dir)
    val_dataset = WSIDataset(val_df, features_dir)
    test_dataset = WSIDataset(test_df, features_dir)
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Model Setup
    encoder = ImageEncoder(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    head = CoxSurvivalHead(d_input=config['model']['output_dim'])
    model = ImageSurvivalModel(encoder, head)
    
    criterion = CoxPHLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # 5. Training
    trainer = SurvivalTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        logger=logger,
        patience=config['training']['early_stopping_patience'],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    trainer.train(train_loader, val_loader, max_epochs=config['training']['epochs'])
    
    # 6. Evaluation
    print("\nEvaluating on Test Set...")
    _, test_cindex = trainer.evaluate(test_loader)
    print(f"Test C-index: {test_cindex:.4f}")
    
    # Save Checkpoint
    ckpt_dir = project_root / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "M2_best.pt")
    print(f"Model weights saved to {ckpt_dir / 'M2_best.pt'}")

    # Save Results
    results_dir = project_root / "results" / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "model": "M2_Image_Only",
        "cohort_size": len(valid_df),
        "test_c_index": test_cindex,
        "best_val_c_index": trainer.best_val_cindex,
        "best_epoch": trainer.best_epoch
    }
    
    with open(results_dir / "image_only.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Results saved to {results_dir / 'image_only.json'}")

if __name__ == "__main__":
    main()
