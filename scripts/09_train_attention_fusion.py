import os
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data.clinical_preprocessing import ClinicalPreprocessor
from src.data.multimodal_dataset import MultimodalDataset
from src.models.attention_fusion import AttentionFusionModel
from src.training.cox_loss import CoxPHLoss
from src.training.trainer import SurvivalTrainer

def build_clinical_merge(cohort_df: pd.DataFrame, clinical_raw: pd.DataFrame) -> pd.DataFrame:
    rename_map = {"sample_id.samples": "sample_id"}
    clinical_for_merge = clinical_raw.rename(columns=rename_map)
    if "patient_id" in clinical_for_merge.columns:
        clinical_for_merge = clinical_for_merge.drop(columns=["patient_id"])

    merged = pd.merge(cohort_df, clinical_for_merge, on="sample_id", how="left")
    
    unmatched = merged["age_at_index.demographic"].isna()
    if unmatched.any():
        cohort_subset = cohort_df[unmatched].copy()
        clinical_patient = (
            clinical_raw.drop_duplicates(subset=["submitter_id"])
            .rename(columns={"submitter_id": "patient_id"})
        )
        subset_merged = pd.merge(cohort_subset, clinical_patient, on="patient_id", how="left")
        merged.loc[unmatched, subset_merged.columns] = subset_merged.values
        
    return merged

def main():
    print("="*50)
    print("TRAINING MULTIMODAL ATTENTION FUSION (M4)")
    print("="*50)
    
    set_seed(42)
    config = load_config(project_root / "configs" / "multimodal_attention.yaml")
    logger = setup_logger("M4_Attention")
    
    # 1. Load Data
    dataset_dir = project_root / "dataset"
    features_dir = dataset_dir / "features"
    
    cohort_df = pd.read_csv(dataset_dir / "patient_cohort.csv")
    clinical_raw = pd.read_csv(dataset_dir / "TCGA-BRCA.clinical.tsv", sep="\t")
    
    merged_df = build_clinical_merge(cohort_df, clinical_raw)
    
    # Filter to patients with image features
    def has_image(patient_id):
        return (features_dir / f"{patient_id}_image.pt").exists()
        
    merged_df['has_image'] = merged_df['patient_id'].apply(has_image)
    valid_df = merged_df[merged_df['has_image']].copy()
    
    print(f"Total cohort: {len(cohort_df)}")
    print(f"Valid Multimodal Patients: {len(valid_df)}")
    
    if len(valid_df) == 0:
        print("Error: No valid patients with both clinical and image data!")
        sys.exit(1)
        
    # Split
    train_df = valid_df[valid_df['split'] == 'train']
    val_df = valid_df[valid_df['split'] == 'val']
    test_df = valid_df[valid_df['split'] == 'test']
    
    print(f"Splits -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # 2. Fit Clinical Preprocessor
    preprocessor = ClinicalPreprocessor()
    preprocessor.fit(train_df)
    
    X_train = preprocessor.transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)
    
    clinical_input_dim = X_train.shape[1]
    
    # 3. Datasets & Loaders
    train_dataset = MultimodalDataset(
        clinical_features=X_train, times=train_df['OS.time'].values, events=train_df['OS'].values,
        patient_ids=train_df['patient_id'], features_dir=features_dir
    )
    val_dataset = MultimodalDataset(
        clinical_features=X_val, times=val_df['OS.time'].values, events=val_df['OS'].values,
        patient_ids=val_df['patient_id'], features_dir=features_dir
    )
    test_dataset = MultimodalDataset(
        clinical_features=X_test, times=test_df['OS.time'].values, events=test_df['OS'].values,
        patient_ids=test_df['patient_id'], features_dir=features_dir
    )
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Model Setup
    model = AttentionFusionModel(
        clinical_input_dim=clinical_input_dim,
        image_input_dim=config['model']['image_input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    criterion = CoxPHLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # Custom trainer logic wrapper for Multimodal
    class MultimodalTrainer(SurvivalTrainer):
        def train_epoch(self, dataloader):
            self.model.train()
            total_loss = 0.0
            
            for batch in dataloader:
                clin = batch['clinical_features'].to(self.device)
                img = batch['image_features'].to(self.device)
                times = batch['time'].to(self.device)
                events = batch['event'].to(self.device)
                
                self.optimizer.zero_grad()
                risk_scores = self.model(clin, img)
                loss = self.criterion(risk_scores, times, events)
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(clin)
                
            return total_loss / len(dataloader.dataset)
            
        def evaluate(self, dataloader):
            from src.evaluation.metrics import compute_c_index
            import numpy as np
            self.model.eval()
            total_loss = 0.0
            all_risks, all_times, all_events = [], [], []
            
            with torch.no_grad():
                for batch in dataloader:
                    clin = batch['clinical_features'].to(self.device)
                    img = batch['image_features'].to(self.device)
                    times = batch['time'].to(self.device)
                    events = batch['event'].to(self.device)
                    
                    risk_scores = self.model(clin, img)
                    loss = self.criterion(risk_scores, times, events)
                    
                    total_loss += loss.item() * len(clin)
                    all_risks.append(risk_scores.cpu().numpy())
                    all_times.append(times.cpu().numpy())
                    all_events.append(events.cpu().numpy())
                    
            avg_loss = total_loss / len(dataloader.dataset)
            risks = np.concatenate(all_risks)
            times = np.concatenate(all_times)
            events = np.concatenate(all_events)
            
            c_index = compute_c_index(risks, times, events)
            return avg_loss, c_index
            
    trainer = MultimodalTrainer(
        model=model, criterion=criterion, optimizer=optimizer, logger=logger,
        patience=config['training']['early_stopping_patience'],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 5. Training
    trainer.train(train_loader, val_loader, max_epochs=config['training']['epochs'])
    
    # 6. Evaluation
    print("\nEvaluating on Test Set...")
    _, test_cindex = trainer.evaluate(test_loader)
    print(f"Test C-index: {test_cindex:.4f}")
    
    # Save Checkpoint
    ckpt_dir = project_root / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "M4_best.pt")
    print(f"Model weights saved to {ckpt_dir / 'M4_best.pt'}")

    # Save Results
    results_dir = project_root / "results" / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "M4_Multimodal_Attention",
        "cohort_size": len(valid_df),
        "test_c_index": test_cindex,
        "best_val_c_index": trainer.best_val_cindex,
        "best_epoch": trainer.best_epoch
    }
    
    with open(results_dir / "multimodal_attention.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Results saved to {results_dir / 'multimodal_attention.json'}")

if __name__ == "__main__":
    main()
