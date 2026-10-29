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
from src.evaluation.survival_plots import plot_kaplan_meier_curves, plot_all_models_km_comparison

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

# --- Model Trainers ---

class M1Trainer(SurvivalTrainer):
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            clin = batch['clinical_features'].to(self.device)
            times = batch['time'].to(self.device)
            events = batch['event'].to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(clin), times, events)
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
                r = self.model(clin)
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

class M2Trainer(SurvivalTrainer):
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            img = batch['features'].to(self.device)
            times = batch['time'].to(self.device)
            events = batch['event'].to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(img), times, events)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(img)
        return total_loss / len(dataloader.dataset)

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

def main():
    print("=" * 80)
    print("MODULE 5: UNIFIED 5-FOLD CV BENCHMARKING (M1, M2, M3, M4)")
    print("=" * 80)

    logger = setup_logger("Final_Evaluation_All_Models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_dir = project_root / "dataset"
    features_dir = dataset_dir / "features"
    checkpoints_dir = project_root / "results" / "checkpoints"
    figures_dir = project_root / "results" / "figures"
    metrics_dir = project_root / "results" / "metrics"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    cohort_df = pd.read_csv(dataset_dir / "wsi_manifest_300.csv")
    clinical_raw = pd.read_csv(dataset_dir / "TCGA-BRCA.clinical.tsv", sep="\t")
    merged_df = build_clinical_merge(cohort_df, clinical_raw)

    def has_image(patient_id):
        patch_path = features_dir.parent / "patch_features" / f"{patient_id}_patches.pt"
        legacy_path = features_dir / f"{patient_id}_image.pt"
        return patch_path.exists() or legacy_path.exists()

    merged_df['has_image'] = merged_df['patient_id'].apply(has_image)
    valid_df = merged_df[merged_df['has_image']].copy().reset_index(drop=True)

    n_patients = len(valid_df)
    n_events = int(valid_df['OS'].sum())
    print(f"Total Cohort: {n_patients} patients | Deaths: {n_events} ({n_events/n_patients*100:.1f}%)")

    # 2. Setup 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_keys = ["M1_Clinical", "M2_Image", "M3_Concat", "M4_Attention"]
    results = {
        m: {
            "fold_cindices": [],
            "oof_risks": np.zeros(n_patients),
            "best_model_state": None,
            "best_cindex": -1.0
        }
        for m in model_keys
    }
    fold_stats = []

    # Track best fitted preprocessor across folds
    best_preprocessor = None
    best_m4_score = -1.0

    for fold, (dev_idx, test_idx) in enumerate(skf.split(valid_df, valid_df['OS'])):
        print(f"\n" + "-" * 40)
        print(f"RUNNING FOLD {fold + 1} / 5")
        print("-" * 40)

        dev_df = valid_df.iloc[dev_idx].copy()
        test_df = valid_df.iloc[test_idx].copy()

        # Inner validation split (15%) for early stopping
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42 + fold)
        train_sub_idx, val_sub_idx = next(sss.split(dev_df, dev_df['OS']))

        train_df = dev_df.iloc[train_sub_idx].copy()
        val_df = dev_df.iloc[val_sub_idx].copy()

        test_events = int(test_df['OS'].sum())
        test_pairs = count_comparable_pairs(test_df['OS.time'].values, test_df['OS'].values)

        f_stat = {
            "fold": fold + 1,
            "train_n": len(train_df),
            "train_events": int(train_df['OS'].sum()),
            "val_n": len(val_df),
            "val_events": int(val_df['OS'].sum()),
            "test_n": len(test_df),
            "test_events": test_events,
            "test_comparable_pairs": test_pairs
        }
        fold_stats.append(f_stat)
        print(f"Fold {fold+1} Counts: Train={len(train_df)} (E={f_stat['train_events']}), Val={len(val_df)} (E={f_stat['val_events']}), Test={len(test_df)} (E={test_events}, Pairs={test_pairs})")

        # Fit preprocessor strictly on train_df
        preprocessor = ClinicalPreprocessor()
        preprocessor.fit(train_df)
        X_train = preprocessor.transform(train_df)
        X_val = preprocessor.transform(val_df)
        X_test = preprocessor.transform(test_df)
        clin_dim = X_train.shape[1]

        batch_size = 16
        max_epochs = 60
        patience = 15

        # --- Datasets ---
        # M1: Clinical Only (using MultimodalDataset for aligned interface)
        train_ds_clin = MultimodalDataset(X_train, train_df['OS.time'].values, train_df['OS'].values, train_df['patient_id'], features_dir, pool_image=True)
        val_ds_clin = MultimodalDataset(X_val, val_df['OS.time'].values, val_df['OS'].values, val_df['patient_id'], features_dir, pool_image=True)
        test_ds_clin = MultimodalDataset(X_test, test_df['OS.time'].values, test_df['OS'].values, test_df['patient_id'], features_dir, pool_image=True)

        # M2: Image Only (WSIDataset)
        train_ds_m2 = WSIDataset(train_df, features_dir)
        val_ds_m2 = WSIDataset(val_df, features_dir)
        test_ds_m2 = WSIDataset(test_df, features_dir)

        # M3: Concat Fusion (Multimodal pooled)
        train_ds_m3 = train_ds_clin
        val_ds_m3 = val_ds_clin
        test_ds_m3 = test_ds_clin

        # M4: Spatial Attention Fusion (Multimodal sequence [120, 2048])
        train_ds_m4 = MultimodalDataset(X_train, train_df['OS.time'].values, train_df['OS'].values, train_df['patient_id'], features_dir, pool_image=False)
        val_ds_m4 = MultimodalDataset(X_val, val_df['OS.time'].values, val_df['OS'].values, val_df['patient_id'], features_dir, pool_image=False)
        test_ds_m4 = MultimodalDataset(X_test, test_df['OS.time'].values, test_df['OS'].values, test_df['patient_id'], features_dir, pool_image=False)

        # Loaders
        tl_clin = DataLoader(train_ds_clin, batch_size=batch_size, shuffle=True)
        vl_clin = DataLoader(val_ds_clin, batch_size=batch_size, shuffle=False)
        tel_clin = DataLoader(test_ds_clin, batch_size=batch_size, shuffle=False)

        tl_m2 = DataLoader(train_ds_m2, batch_size=batch_size, shuffle=True)
        vl_m2 = DataLoader(val_ds_m2, batch_size=batch_size, shuffle=False)
        tel_m2 = DataLoader(test_ds_m2, batch_size=batch_size, shuffle=False)

        tl_m4 = DataLoader(train_ds_m4, batch_size=batch_size, shuffle=True)
        vl_m4 = DataLoader(val_ds_m4, batch_size=batch_size, shuffle=False)
        tel_m4 = DataLoader(test_ds_m4, batch_size=batch_size, shuffle=False)

        # -----------------
        # Model 1: M1 Clinical
        # -----------------
        set_seed(42 + fold)
        m1 = create_model("m1", clinical_input_dim=clin_dim, hidden_dim=128, dropout_rate=0.3)
        opt_m1 = torch.optim.Adam(m1.parameters(), lr=1e-4, weight_decay=1e-4)
        tr_m1 = M1Trainer(m1, CoxPHLoss(), opt_m1, device, logger, patience=patience)
        tr_m1.train(tl_clin, vl_clin, max_epochs=max_epochs)
        m1.eval()
        with torch.no_grad():
            m1_risks = np.concatenate([m1(b['clinical_features'].to(device)).cpu().numpy() for b in tel_clin]).flatten()
        c_m1 = compute_c_index(m1_risks, test_df['OS.time'].values, test_df['OS'].values)
        results["M1_Clinical"]["fold_cindices"].append(float(c_m1))
        results["M1_Clinical"]["oof_risks"][test_idx] = m1_risks
        if c_m1 > results["M1_Clinical"]["best_cindex"]:
            results["M1_Clinical"]["best_cindex"] = c_m1
            results["M1_Clinical"]["best_model_state"] = m1.state_dict()

        # -----------------
        # Model 2: M2 Image Only
        # -----------------
        set_seed(42 + fold)
        m2 = create_model("m2", image_input_dim=2048, hidden_dim=128, dropout_rate=0.4)
        opt_m2 = torch.optim.Adam(m2.parameters(), lr=1e-3, weight_decay=1e-4)
        tr_m2 = M2Trainer(m2, CoxPHLoss(), opt_m2, device, logger, patience=patience)
        tr_m2.train(tl_m2, vl_m2, max_epochs=max_epochs)
        m2.eval()
        with torch.no_grad():
            m2_risks = np.concatenate([m2(b['features'].to(device)).cpu().numpy() for b in tel_m2]).flatten()
        c_m2 = compute_c_index(m2_risks, test_df['OS.time'].values, test_df['OS'].values)
        results["M2_Image"]["fold_cindices"].append(float(c_m2))
        results["M2_Image"]["oof_risks"][test_idx] = m2_risks
        if c_m2 > results["M2_Image"]["best_cindex"]:
            results["M2_Image"]["best_cindex"] = c_m2
            results["M2_Image"]["best_model_state"] = m2.state_dict()

        # -----------------
        # Model 3: M3 Concat Fusion
        # -----------------
        set_seed(42 + fold)
        m3 = create_model("m3", clinical_input_dim=clin_dim, image_input_dim=2048, hidden_dim=128, dropout_rate=0.4)
        opt_m3 = torch.optim.Adam(m3.parameters(), lr=5e-4, weight_decay=1e-4)
        tr_m3 = M3Trainer(m3, CoxPHLoss(), opt_m3, device, logger, patience=patience)
        tr_m3.train(tl_clin, vl_clin, max_epochs=max_epochs)
        m3.eval()
        with torch.no_grad():
            m3_risks = np.concatenate([m3(b['clinical_features'].to(device), b['image_features'].to(device)).cpu().numpy() for b in tel_clin]).flatten()
        c_m3 = compute_c_index(m3_risks, test_df['OS.time'].values, test_df['OS'].values)
        results["M3_Concat"]["fold_cindices"].append(float(c_m3))
        results["M3_Concat"]["oof_risks"][test_idx] = m3_risks
        if c_m3 > results["M3_Concat"]["best_cindex"]:
            results["M3_Concat"]["best_cindex"] = c_m3
            results["M3_Concat"]["best_model_state"] = m3.state_dict()

        # -----------------
        # Model 4: M4 Spatial Attention Fusion (FROZEN CONFIG)
        # -----------------
        set_seed(42 + fold)
        m4 = create_model("m4", clinical_input_dim=clin_dim, image_input_dim=2048, hidden_dim=128, dropout_rate=0.5)
        opt_m4 = torch.optim.Adam(m4.parameters(), lr=5e-4, weight_decay=1e-4)
        tr_m4 = M4Trainer(m4, CoxPHLoss(), opt_m4, device, logger, patience=patience)
        tr_m4.train(tl_m4, vl_m4, max_epochs=max_epochs)
        m4.eval()
        with torch.no_grad():
            m4_risks = np.concatenate([m4(b['clinical_features'].to(device), b['image_features'].to(device), image_mask=b['image_mask'].to(device)).cpu().numpy() for b in tel_m4]).flatten()
        c_m4 = compute_c_index(m4_risks, test_df['OS.time'].values, test_df['OS'].values)
        results["M4_Attention"]["fold_cindices"].append(float(c_m4))
        results["M4_Attention"]["oof_risks"][test_idx] = m4_risks
        if c_m4 > results["M4_Attention"]["best_cindex"]:
            results["M4_Attention"]["best_cindex"] = c_m4
            results["M4_Attention"]["best_model_state"] = m4.state_dict()
            best_preprocessor = preprocessor

        print(f"Fold {fold+1} C-Index: M1={c_m1:.4f} | M2={c_m2:.4f} | M3={c_m3:.4f} | M4={c_m4:.4f}")

    # 3. Final Overall Metrics & Pooled OOF Evaluation
    times_all = valid_df['OS.time'].values
    events_all = valid_df['OS'].values
    total_pairs = count_comparable_pairs(times_all, events_all)

    metrics_output = {
        "cohort_size": n_patients,
        "event_count": n_events,
        "event_rate": float(n_events / n_patients),
        "total_oof_comparable_pairs": total_pairs,
        "fold_statistics": fold_stats,
        "models": {}
    }

    oof_risks_dict = {}

    print("\n" + "=" * 80)
    print("FINAL 5-FOLD BENCHMARK COMPARISON TABLE (298-PATIENT TCGA-BRCA COHORT)")
    print("=" * 80)
    header = f"{'Model':<20} | {'Mean C-Index':<14} | {'Std Dev':<10} | {'Median':<10} | {'Pooled OOF':<12} | {'Folds (1..5)'}"
    print(header)
    print("-" * 95)

    for key in model_keys:
        c_folds = results[key]["fold_cindices"]
        mean_c = float(np.mean(c_folds))
        std_c = float(np.std(c_folds))
        median_c = float(np.median(c_folds))
        oof_r = results[key]["oof_risks"]
        pooled_c = float(compute_c_index(oof_r, times_all, events_all))

        oof_risks_dict[key] = oof_r

        metrics_output["models"][key] = {
            "per_fold_cindex": c_folds,
            "mean_cindex": mean_c,
            "std_cindex": std_c,
            "median_cindex": median_c,
            "pooled_oof_cindex": pooled_c
        }

        folds_str = ", ".join([f"{f:.4f}" for f in c_folds])
        row = f"{key:<20} | {mean_c:.4f}{'':<8} | ±{std_c:.4f}{'':<4} | {median_c:.4f}{'':<4} | {pooled_c:.4f}{'':<6} | [{folds_str}]"
        print(row)

    print("=" * 80)

    # 4. Save Best Model Checkpoints
    for key in model_keys:
        ckpt_path = checkpoints_dir / f"{key.split('_')[0]}_best.pt"
        torch.save(results[key]["best_model_state"], ckpt_path)
    if best_preprocessor is not None:
        best_preprocessor.save(str(checkpoints_dir / "clinical_preprocessor.joblib"))
    print(f"Model checkpoints saved to {checkpoints_dir}")

    # 5. Kaplan-Meier Survival Analysis & Log-Rank Statistical Tests
    print("\nGenerating Kaplan-Meier Survival Curves and Log-Rank Tests...")
    km_comparison_path = figures_dir / "km_all_models_comparison.png"
    km_stats = plot_all_models_km_comparison(
        models_oof_dict=oof_risks_dict,
        times=times_all,
        events=events_all,
        save_path=str(km_comparison_path)
    )

    # Generate individual KM plots for each model
    for key in model_keys:
        single_km_path = figures_dir / f"km_{key.lower()}.png"
        plot_kaplan_meier_curves(
            risk_scores=oof_risks_dict[key],
            times=times_all,
            events=events_all,
            model_name=key,
            save_path=str(single_km_path)
        )

    metrics_output["log_rank_tests"] = km_stats

    # 6. Save JSON Metrics
    final_metrics_path = metrics_dir / "final_model_evaluation.json"
    with open(final_metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=4)
    print(f"Metrics saved to {final_metrics_path}")
    print(f"Figures saved to {figures_dir}")

    print("\nLog-Rank Statistical Significance:")
    for key, stat in km_stats.items():
        p_str = "< 0.0001" if stat['p_value'] < 0.0001 else f"= {stat['p_value']:.4f}"
        print(f"  {key:<18}: Log-rank Stat = {stat['test_statistic']:6.2f} | p-value {p_str} | High/Low N = {stat['high_risk_n']}/{stat['low_risk_n']}")

if __name__ == "__main__":
    main()
