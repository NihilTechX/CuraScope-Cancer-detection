import os
import sys
from pathlib import Path
import pandas as pd
import glob

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.clinical_preprocessing import ClinicalPreprocessor

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
    print("=" * 80)
    print("FULL TCGA-BRCA COHORT VERIFICATION")
    print("=" * 80)
    
    dataset_dir = project_root / "dataset"
    patch_features_dir = dataset_dir / "patch_features"
    legacy_features_dir = dataset_dir / "features"
    
    # 1. Check full clinical data
    clinical_raw = pd.read_csv(dataset_dir / "TCGA-BRCA.clinical.tsv", sep="\t")
    cohort_df = pd.read_csv(dataset_dir / "patient_cohort.csv")
    
    print(f"Total entries in patient_cohort.csv: {len(cohort_df)}")
    print(f"Total distinct patients in patient_cohort.csv: {cohort_df['patient_id'].nunique()}")
    
    merged_df = build_clinical_merge(cohort_df, clinical_raw)
    
    # Check survival completeness
    has_survival = merged_df[['OS', 'OS.time']].notna().all(axis=1)
    print(f"Patients with valid Survival Data: {has_survival.sum()}")
    
    # Check clinical completeness (via preprocessor)
    preprocessor = ClinicalPreprocessor()
    df_valid_surv = merged_df[has_survival].copy().reset_index(drop=True)
    
    try:
        preprocessor.fit(df_valid_surv)
        has_clinical = True
        print(f"Patients with valid Clinical Data (imputable/processable): {len(df_valid_surv)}")
    except Exception as e:
        print(f"Error processing clinical data: {e}")
        has_clinical = False
    
    # 2. Check usable patch features
    patch_files = glob.glob(str(patch_features_dir / "*_patches.pt"))
    legacy_files = glob.glob(str(legacy_features_dir / "*_image.pt"))
    
    # We want to know how many of the VALID clinical/survival patients have features
    def has_image(patient_id):
        patch_path = patch_features_dir / f"{patient_id}_patches.pt"
        legacy_path = legacy_features_dir / f"{patient_id}_image.pt"
        return patch_path.exists() or legacy_path.exists()
        
    df_valid_surv['has_image'] = df_valid_surv['patient_id'].apply(has_image)
    df_final_eligible = df_valid_surv[df_valid_surv['has_image']].copy().reset_index(drop=True)
    
    print(f"Patients with matching Patch/Image Features: {df_valid_surv['has_image'].sum()}")
    
    print("-" * 80)
    print(f"FINAL ELIGIBLE COHORT SIZE: {len(df_final_eligible)} patients")
    print(f"Total Events (OS=1): {df_final_eligible['OS'].sum()} ({df_final_eligible['OS'].mean()*100:.1f}%)")
    
    print("\nPROPOSED SPLIT STRATEGY (5-Fold CV):")
    print("- Method: StratifiedKFold (n_splits=5, shuffle=True) on 'OS'")
    print("- Stratification: Ensures event proportion is preserved across folds.")
    print("- Splitting Level: PATIENT-LEVEL (no patient leakage).")
    print("- Validation: 15-20% of training fold reserved via StratifiedShuffleSplit for early stopping.")
    print("- Test Fold: Strictly held out. Used only for final C-Index.")
    print("- Alignment: Clinical, survival, and image data are strictly index-aligned in MultimodalDataset.")
    print("- M4 Parameters: Frozen to LR=5e-4, Dropout=0.5, WD=1e-4, max_epochs=60, patience=15.")
    print("- M4 Masking: pool_image=False generates [120, 2048] patched sequences with exact boolean padding masks.")
    print("- Risk direction check: compute_c_index is used with properly structured event ordering.")

if __name__ == "__main__":
    main()
