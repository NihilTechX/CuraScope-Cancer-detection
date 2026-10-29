import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data.clinical_preprocessing import ClinicalPreprocessor
from src.data.multimodal_dataset import MultimodalDataset
from src.data.wsi_dataset import WSIDataset
from src.models.model_factory import create_model
from src.training.cox_loss import CoxPHLoss
from src.training.trainer import SurvivalTrainer
from src.evaluation.metrics import compute_c_index

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

def count_comparable_pairs(times: np.ndarray, events: np.ndarray) -> int:
    pairs = 0
    n = len(times)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if events[i] == 1 and times[i] < times[j]:
                pairs += 1
            elif events[i] == 1 and events[j] == 1 and times[i] == times[j]:
                pairs += 0.5
    return int(pairs)

class M4Trainer(SurvivalTrainer):
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            clin = batch['clinical_features'].to(self.device)
            img = batch['image_features'].to(self.device)
            mask = batch.get('image_mask')
            if mask is not None:
                mask = mask.to(self.device)
            times = batch['time'].to(self.device)
            events = batch['event'].to(self.device)
            self.optimizer.zero_grad()
            if mask is not None:
                loss = self.criterion(self.model(clin, img, image_mask=mask), times, events)
            else:
                loss = self.criterion(self.model(clin, img), times, events)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(clin)
        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        risks, times, events = [], [], []
        with torch.no_grad():
            for b in dataloader:
                clin = b['clinical_features'].to(self.device)
                img = b['image_features'].to(self.device)
                mask = b.get('image_mask')
                if mask is not None:
                    mask = mask.to(self.device)
                    r = self.model(clin, img, image_mask=mask)
                else:
                    r = self.model(clin, img)
                risks.append(r.cpu().numpy())
                times.append(b['time'].cpu().numpy())
                events.append(b['event'].cpu().numpy())
        risks = np.concatenate(risks)
        times = np.concatenate(times)
        events = np.concatenate(events)
        try:
            c_index = compute_c_index(risks, times, events)
        except ZeroDivisionError:
            c_index = 0.5
        return total_loss, c_index

def main():
    print("=" * 80)
    print("M4 SPATIAL ATTENTION TUNING (300-PATIENT DATASET)")
    print("=" * 80)

    logger = setup_logger("M4_Tuning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_dir = project_root / "dataset"
    features_dir = dataset_dir / "features"

    cohort_df = pd.read_csv(dataset_dir / "wsi_manifest_300.csv")
    clinical_raw = pd.read_csv(dataset_dir / "TCGA-BRCA.clinical.tsv", sep="\t")

    merged_df = build_clinical_merge(cohort_df, clinical_raw)

    def has_image(patient_id):
        patch_path = features_dir.parent / "patch_features" / f"{patient_id}_patches.pt"
        legacy_path = features_dir / f"{patient_id}_image.pt"
        return patch_path.exists() or legacy_path.exists()

    merged_df['has_image'] = merged_df['patient_id'].apply(has_image)
    valid_df = merged_df[merged_df['has_image']].copy().reset_index(drop=True)

    print(f"Total Cohort (N={len(valid_df)} patients)")
    print(f"Total Deaths (OS=1): {valid_df['OS'].sum()} ({valid_df['OS'].mean()*100:.1f}%)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    configs = [
        {"name": "M4_c1_baseline", "lr": 5e-4, "dropout": 0.4, "wd": 1e-4},
        {"name": "M4_c2", "lr": 1e-3, "dropout": 0.4, "wd": 1e-4},
        {"name": "M4_c3", "lr": 1e-4, "dropout": 0.4, "wd": 1e-4},
        {"name": "M4_c4", "lr": 5e-4, "dropout": 0.3, "wd": 1e-4},
        {"name": "M4_c5", "lr": 5e-4, "dropout": 0.5, "wd": 1e-4},
        {"name": "M4_c6", "lr": 5e-4, "dropout": 0.4, "wd": 1e-3},
        {"name": "M4_c7", "lr": 1e-3, "dropout": 0.5, "wd": 1e-3},
        {"name": "M4_c8", "lr": 1e-4, "dropout": 0.5, "wd": 1e-3},
    ]

    results = {c["name"]: {"fold_cindices": [], "oof_risks": np.zeros(len(valid_df)), "config": c} for c in configs}
    fold_stats = []

    for fold, (dev_idx, test_idx) in enumerate(skf.split(valid_df, valid_df['OS'])):
        print(f"\n--- Running Fold {fold + 1} / 5 ---")

        dev_df = valid_df.iloc[dev_idx].copy()
        test_df = valid_df.iloc[test_idx].copy()

        # Reserve ~15-20% of dev_df for inner validation (early stopping)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42 + fold)
        train_sub_idx, val_sub_idx = next(sss.split(dev_df, dev_df['OS']))

        train_df = dev_df.iloc[train_sub_idx].copy()
        val_df = dev_df.iloc[val_sub_idx].copy()

        test_events = int(test_df['OS'].sum())
        test_pairs = count_comparable_pairs(test_df['OS.time'].values, test_df['OS'].values)
        
        fold_stat = {
            "fold": fold + 1,
            "train_n": len(train_df),
            "train_events": int(train_df['OS'].sum()),
            "train_censored": len(train_df) - int(train_df['OS'].sum()),
            "val_n": len(val_df),
            "val_events": int(val_df['OS'].sum()),
            "val_censored": len(val_df) - int(val_df['OS'].sum()),
            "test_n": len(test_df),
            "test_events": test_events,
            "test_censored": len(test_df) - test_events,
            "test_comparable_pairs": test_pairs
        }
        fold_stats.append(fold_stat)
        
        preprocessor = ClinicalPreprocessor()
        preprocessor.fit(train_df)
        X_train = preprocessor.transform(train_df)
        X_val = preprocessor.transform(val_df)
        X_test = preprocessor.transform(test_df)
        clin_dim = X_train.shape[1]

        # Sequence for M4
        train_ds_mm_seq = MultimodalDataset(X_train, train_df['OS.time'].values, train_df['OS'].values, train_df['patient_id'], features_dir, pool_image=False)
        val_ds_mm_seq = MultimodalDataset(X_val, val_df['OS.time'].values, val_df['OS'].values, val_df['patient_id'], features_dir, pool_image=False)
        test_ds_mm_seq = MultimodalDataset(X_test, test_df['OS.time'].values, test_df['OS'].values, test_df['patient_id'], features_dir, pool_image=False)

        batch_size = 16
        tl_mm_seq, vl_mm_seq, tel_mm_seq = DataLoader(train_ds_mm_seq, batch_size=batch_size, shuffle=True), DataLoader(val_ds_mm_seq, batch_size=batch_size, shuffle=False), DataLoader(test_ds_mm_seq, batch_size=batch_size, shuffle=False)
        max_epochs = 60

        for config in configs:
            print(f"  -> Testing {config['name']} (LR={config['lr']}, Drop={config['dropout']}, WD={config['wd']})")
            set_seed(42 + fold)
            m4 = create_model("m4", clinical_input_dim=clin_dim, image_input_dim=2048, hidden_dim=128, dropout_rate=config["dropout"]).to(device)
            opt_m4 = torch.optim.Adam(m4.parameters(), lr=config["lr"], weight_decay=config["wd"])
            tr_m4 = M4Trainer(m4, CoxPHLoss(), opt_m4, device, logger, patience=15)
            tr_m4.train(tl_mm_seq, vl_mm_seq, max_epochs=max_epochs)
            m4.eval()
            with torch.no_grad():
                m4_risks = np.concatenate([m4(b['clinical_features'].to(device), b['image_features'].to(device), image_mask=b['image_mask'].to(device)).cpu().numpy() for b in tel_mm_seq])
            c_m4 = compute_c_index(m4_risks, test_df['OS.time'].values, test_df['OS'].values) if test_pairs > 0 else np.nan
            results[config["name"]]["fold_cindices"].append(float(c_m4))
            results[config["name"]]["oof_risks"][test_idx] = m4_risks
            print(f"     C-Index: {c_m4:.4f}")

    # Final Evaluation & Aggregation
    times_all = valid_df['OS.time'].values
    events_all = valid_df['OS'].values
    total_pairs_all = count_comparable_pairs(times_all, events_all)

    final_output = {"fold_statistics": fold_stats, "configurations": {}}
    
    for config in configs:
        c_name = config["name"]
        c_list = results[c_name]["fold_cindices"]
        valid_c = [c for c in c_list if not np.isnan(c)]
        mean_c = float(np.mean(valid_c)) if valid_c else np.nan
        std_c = float(np.std(valid_c)) if valid_c else np.nan
        med_c = float(np.median(valid_c)) if valid_c else np.nan

        oof_risks = results[c_name]["oof_risks"]
        pooled_cindex = float(compute_c_index(oof_risks, times_all, events_all))

        final_output["configurations"][c_name] = {
            "hyperparameters": config,
            "per_fold_cindex": [float(c) if not np.isnan(c) else None for c in c_list],
            "mean_cindex": mean_c,
            "std_cindex": std_c,
            "median_cindex": med_c,
            "pooled_oof_cindex": pooled_cindex,
            "total_oof_pairs": total_pairs_all
        }

    out_dir = project_root / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "m4_tuning_300.json", "w") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\nSaved results to {out_dir / 'm4_tuning_300.json'}")

if __name__ == "__main__":
    main()
