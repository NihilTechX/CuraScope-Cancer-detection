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

class M3Trainer(SurvivalTrainer):
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            clin = batch['clinical_features'].to(self.device)
            img = batch['image_features'].to(self.device)
            times = batch['time'].to(self.device)
            events = batch['event'].to(self.device)
            self.optimizer.zero_grad()
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

class SafeSurvivalTrainer(SurvivalTrainer):
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        risks, times, events = [], [], []
        with torch.no_grad():
            for b in dataloader:
                r = self.model(b['features'].to(self.device))
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

def run_sanity_check(df, features_dir, clin_dim, device):
    print("\n" + "="*80)
    print("M4 SPATIAL ATTENTION SANITY CHECK")
    print("="*80)
    
    test_ds = MultimodalDataset(np.random.randn(5, clin_dim), np.ones(5), np.ones(5), df['patient_id'].head(5), features_dir, pool_image=False)
    dl = DataLoader(test_ds, batch_size=2, shuffle=False)
    batch = next(iter(dl))
    
    img = batch['image_features']
    mask = batch['image_mask']
    clin = batch['clinical_features']
    
    print(f"Batch Image Shape: {img.shape} (Expected [2, 120, 2048])")
    print(f"Batch Mask Shape: {mask.shape} (Expected [2, 120])")
    
    if img.shape[1] != 120 or mask.shape[1] != 120:
        print("FAIL: Shapes do not match expected 120 padding.")
        sys.exit(1)
        
    m4 = create_model("m4", clinical_input_dim=clin_dim, image_input_dim=2048, hidden_dim=128, dropout_rate=0.4).to(device)
    m4.train()
    
    out = m4(clin.to(device), img.to(device), image_mask=mask.to(device))
    attn = m4.get_attention_weights()
    print(f"Attention Weights Shape: {attn.shape} (Expected [2, 120])")
    
    for i in range(2):
        valid_count = mask[i].sum().item()
        attn_sum = attn[i, :valid_count].sum().item()
        padded_sum = attn[i, valid_count:].sum().item()
        print(f"Sample {i}: Valid Patches = {valid_count}")
        print(f"  Sum of attention on VALID patches: {attn_sum:.4f}")
        print(f"  Sum of attention on PADDED patches: {padded_sum:.4f}")
        
        if abs(attn_sum - 1.0) > 1e-4:
            print("FAIL: Attention weights on valid patches do not sum to 1!")
            sys.exit(1)
        if padded_sum > 1e-4:
            print("FAIL: Padded patches received attention weight!")
            sys.exit(1)
            
    loss = out.sum()
    loss.backward()
    
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in m4.parameters())
    if not has_grad:
        print("FAIL: Gradients did not propagate through M4!")
        sys.exit(1)
    
    print("SANITY CHECK PASSED. Proceeding to Cross-Validation...\n")

def main():
    print("=" * 80)
    print("5-FOLD CV ON 300-PATIENT DATASET (SCALING EXPERIMENT)")
    print("=" * 80)

    logger = setup_logger("5Fold_CV_100")
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

    model_names = ["M2_Image", "M3_Concat", "M4_Attention"]
    results = {m: {"fold_cindices": [], "oof_risks": np.zeros(len(valid_df))} for m in model_names}
    fold_stats = []

    # Run Sanity Check first
    preprocessor = ClinicalPreprocessor()
    preprocessor.fit(valid_df)
    dummy_X = preprocessor.transform(valid_df.head(5))
    clin_dim = dummy_X.shape[1]
    run_sanity_check(valid_df, features_dir, clin_dim, device)

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
        print(f"Fold {fold+1} Stats: Train={len(train_df)} (E={fold_stat['train_events']}), Val={len(val_df)} (E={fold_stat['val_events']}), Test={len(test_df)} (E={test_events}, Comparable Pairs={test_pairs})")

        # Preprocessing strictly on train_df
        preprocessor = ClinicalPreprocessor()
        preprocessor.fit(train_df)
        X_train = preprocessor.transform(train_df)
        X_val = preprocessor.transform(val_df)
        X_test = preprocessor.transform(test_df)
        clin_dim = X_train.shape[1]

        # Datasets
        train_ds_m2 = WSIDataset(train_df, features_dir)
        val_ds_m2 = WSIDataset(val_df, features_dir)
        test_ds_m2 = WSIDataset(test_df, features_dir)

        # Pooled for M3
        train_ds_mm_pooled = MultimodalDataset(X_train, train_df['OS.time'].values, train_df['OS'].values, train_df['patient_id'], features_dir, pool_image=True)
        val_ds_mm_pooled = MultimodalDataset(X_val, val_df['OS.time'].values, val_df['OS'].values, val_df['patient_id'], features_dir, pool_image=True)
        test_ds_mm_pooled = MultimodalDataset(X_test, test_df['OS.time'].values, test_df['OS'].values, test_df['patient_id'], features_dir, pool_image=True)

        # Sequence for M4
        train_ds_mm_seq = MultimodalDataset(X_train, train_df['OS.time'].values, train_df['OS'].values, train_df['patient_id'], features_dir, pool_image=False)
        val_ds_mm_seq = MultimodalDataset(X_val, val_df['OS.time'].values, val_df['OS'].values, val_df['patient_id'], features_dir, pool_image=False)
        test_ds_mm_seq = MultimodalDataset(X_test, test_df['OS.time'].values, test_df['OS'].values, test_df['patient_id'], features_dir, pool_image=False)

        batch_size = 16
        tl_m2, vl_m2, tel_m2 = DataLoader(train_ds_m2, batch_size=batch_size, shuffle=True), DataLoader(val_ds_m2, batch_size=batch_size, shuffle=False), DataLoader(test_ds_m2, batch_size=batch_size, shuffle=False)
        tl_mm_pooled, vl_mm_pooled, tel_mm_pooled = DataLoader(train_ds_mm_pooled, batch_size=batch_size, shuffle=True), DataLoader(val_ds_mm_pooled, batch_size=batch_size, shuffle=False), DataLoader(test_ds_mm_pooled, batch_size=batch_size, shuffle=False)
        tl_mm_seq, vl_mm_seq, tel_mm_seq = DataLoader(train_ds_mm_seq, batch_size=batch_size, shuffle=True), DataLoader(val_ds_mm_seq, batch_size=batch_size, shuffle=False), DataLoader(test_ds_mm_seq, batch_size=batch_size, shuffle=False)

        max_epochs = 60

        # M2
        set_seed(42 + fold)
        m2 = create_model("m2", image_input_dim=2048, hidden_dim=128, dropout_rate=0.4)
        opt_m2 = torch.optim.Adam(m2.parameters(), lr=0.001, weight_decay=1e-4)
        tr_m2 = SafeSurvivalTrainer(m2, CoxPHLoss(), opt_m2, device, logger, patience=15)
        tr_m2.train(tl_m2, vl_m2, max_epochs=max_epochs)
        m2.eval()
        with torch.no_grad():
            m2_risks = np.concatenate([m2(b['features'].to(device)).cpu().numpy() for b in tel_m2])
        c_m2 = compute_c_index(m2_risks, test_df['OS.time'].values, test_df['OS'].values) if test_pairs > 0 else np.nan
        results["M2_Image"]["fold_cindices"].append(float(c_m2))
        results["M2_Image"]["oof_risks"][test_idx] = m2_risks

        # M3
        set_seed(42 + fold)
        m3 = create_model("m3", clinical_input_dim=clin_dim, image_input_dim=2048, hidden_dim=128, dropout_rate=0.4)
        opt_m3 = torch.optim.Adam(m3.parameters(), lr=0.0005, weight_decay=1e-4)
        tr_m3 = M3Trainer(m3, CoxPHLoss(), opt_m3, device, logger, patience=15)
        tr_m3.train(tl_mm_pooled, vl_mm_pooled, max_epochs=max_epochs)
        m3.eval()
        with torch.no_grad():
            m3_risks = np.concatenate([m3(b['clinical_features'].to(device), b['image_features'].to(device)).cpu().numpy() for b in tel_mm_pooled])
        c_m3 = compute_c_index(m3_risks, test_df['OS.time'].values, test_df['OS'].values) if test_pairs > 0 else np.nan
        results["M3_Concat"]["fold_cindices"].append(float(c_m3))
        results["M3_Concat"]["oof_risks"][test_idx] = m3_risks

        # M4
        set_seed(42 + fold)
        m4 = create_model("m4", clinical_input_dim=clin_dim, image_input_dim=2048, hidden_dim=128, dropout_rate=0.4)
        opt_m4 = torch.optim.Adam(m4.parameters(), lr=0.0005, weight_decay=1e-4)
        tr_m4 = M4Trainer(m4, CoxPHLoss(), opt_m4, device, logger, patience=15)
        tr_m4.train(tl_mm_seq, vl_mm_seq, max_epochs=max_epochs)
        m4.eval()
        with torch.no_grad():
            m4_risks = np.concatenate([m4(b['clinical_features'].to(device), b['image_features'].to(device), image_mask=b['image_mask'].to(device)).cpu().numpy() for b in tel_mm_seq])
        c_m4 = compute_c_index(m4_risks, test_df['OS.time'].values, test_df['OS'].values) if test_pairs > 0 else np.nan
        results["M4_Attention"]["fold_cindices"].append(float(c_m4))
        results["M4_Attention"]["oof_risks"][test_idx] = m4_risks

        print(f"Fold {fold+1} C-Index: M2={c_m2:.4f} | M3={c_m3:.4f} | M4={c_m4:.4f}")

    # Final Evaluation & Aggregation
    times_all = valid_df['OS.time'].values
    events_all = valid_df['OS'].values
    total_pairs_all = count_comparable_pairs(times_all, events_all)

    final_output = {"fold_statistics": fold_stats, "models": {}}
    
    md_report = [
        "# 5-Fold Cross-Validation: 100-Patient Cohort",
        "**NOTE: This is a DEVELOPMENT/BASELINE experiment.** The cohort size is only 100 patients with a total of 12 OS events. As a result, out-of-fold metrics are subject to extreme variance.",
        f"\n**Total Pooled Comparable Pairs:** {total_pairs_all}\n",
        "## Per-Fold Demographics",
        "| Fold | Train (N/Events) | Val (N/Events) | Test (N/Events) | Test Pairs |",
        "|---|---|---|---|---|"
    ]
    for s in fold_stats:
        md_report.append(f"| {s['fold']} | {s['train_n']} / {s['train_events']} | {s['val_n']} / {s['val_events']} | {s['test_n']} / {s['test_events']} | {s['test_comparable_pairs']} |")

    md_report.extend([
        "\n## Model Performance Summary",
        "| Model | Mean C-Index | Std Dev | Median C-Index | Pooled OOF C-Index |",
        "|---|---|---|---|---|"
    ])

    for model_name in model_names:
        c_list = results[model_name]["fold_cindices"]
        valid_c = [c for c in c_list if not np.isnan(c)]
        mean_c = float(np.mean(valid_c)) if valid_c else np.nan
        std_c = float(np.std(valid_c)) if valid_c else np.nan
        med_c = float(np.median(valid_c)) if valid_c else np.nan

        oof_risks = results[model_name]["oof_risks"]
        pooled_cindex = float(compute_c_index(oof_risks, times_all, events_all))

        final_output["models"][model_name] = {
            "per_fold_cindex": [float(c) if not np.isnan(c) else None for c in c_list],
            "mean_cindex": mean_c,
            "std_cindex": std_c,
            "median_cindex": med_c,
            "pooled_oof_cindex": pooled_cindex,
            "total_oof_pairs": total_pairs_all
        }

        md_report.append(f"| {model_name} | {mean_c:.4f} | {std_c:.4f} | {med_c:.4f} | {pooled_cindex:.4f} |")

    # Save JSON and MD
    out_dir = project_root / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "cv_300_scaling.json", "w") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\nSaved results to {out_dir / 'cv_300_scaling.json'}")

if __name__ == "__main__":
    main()
