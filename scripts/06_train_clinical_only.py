import sys
from pathlib import Path
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.utils.logger import setup_logger, log_metrics
from src.utils.config import load_config
from src.data.clinical_preprocessing import ClinicalPreprocessor
from src.data.multimodal_dataset import ClinicalDataset
from src.models.clinical_encoder import ClinicalEncoder
from src.models.survival_head import CoxSurvivalHead
from src.training.cox_loss import CoxPHLoss
from src.training.trainer import SurvivalTrainer
from src.evaluation.metrics import compute_c_index


def build_clinical_merge(cohort_df: pd.DataFrame, clinical_raw: pd.DataFrame) -> pd.DataFrame:
    """Attach clinical features to the cohort using the PRIMARY TUMOR sample.

    Joins via the sample_id stored in patient_cohort.csv (the primary tumor
    sample selected by cohort_builder) rather than the naive patient-level
    'keep first row' dedup, which can pick normal-tissue rows.
    """
    rename_map = {"sample_id.samples": "sample_id"}
    # Drop columns that would collide (e.g., an existing sample_id) before rename
    clinical_for_merge = clinical_raw.rename(columns=rename_map)

    # Some data frames may have both sample_id.samples and other sample cols;
    # the merge must be unambiguous on 'sample_id'.
    if "patient_id" in clinical_for_merge.columns:
        clinical_for_merge = clinical_for_merge.drop(columns=["patient_id"])

    merged = pd.merge(
        cohort_df, clinical_for_merge, on="sample_id", how="left"
    )

    # Fallback: any patient that had no primary sample match still needs data.
    # Left on patient_id guarantees a row exists (all cohort IDs come from clinical).
    unmatched = merged["age_at_index.demographic"].isna()
    if unmatched.any():
        cohort_subset = cohort_df[unmatched].copy()
        clinical_patient = (
            clinical_raw.drop_duplicates(subset=["submitter_id"])
            .rename(columns={"submitter_id": "patient_id"})
        )
        subset_merged = pd.merge(
            cohort_subset, clinical_patient, on="patient_id", how="left"
        )
        merged.loc[unmatched, subset_merged.columns] = subset_merged.values

    return merged


def main():
    # Setup
    config = load_config(PROJECT_ROOT / "configs" / "clinical_only.yaml")
    set_seed(config.get("seed", 42))
    logger = setup_logger("train_m1_clinical", log_dir=PROJECT_ROOT / "results" / "logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Data
    logger.info("Loading patient cohort and raw data...")
    dataset_dir = PROJECT_ROOT / "dataset"
    cohort_df = pd.read_csv(dataset_dir / "patient_cohort.csv")
    clinical_raw = pd.read_csv(dataset_dir / "TCGA-BRCA.clinical.tsv", sep="\t")

    # Merge clinical features onto the cohort (primary-tumor sample join)
    merged_df = build_clinical_merge(cohort_df, clinical_raw)

    train_df = merged_df[merged_df["split"] == "train"].copy()
    val_df = merged_df[merged_df["split"] == "val"].copy()
    test_df = merged_df[merged_df["split"] == "test"].copy()

    # 2. Preprocess Clinical Features (Leakage-Proof)
    logger.info("Preprocessing clinical features...")
    preprocessor = ClinicalPreprocessor()

    # Fit only on train
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)

    d_clin = preprocessor.get_feature_dim()
    logger.info(f"Extracted {d_clin} clinical features per patient.")

    # Save preprocessor
    preprocessor.save(PROJECT_ROOT / "results" / "checkpoints" / "clinical_preprocessor.joblib")

    # 3. Create Datasets and DataLoaders
    train_dataset = ClinicalDataset(X_train, train_df["OS.time"].values, train_df["OS"].values)
    val_dataset = ClinicalDataset(X_val, val_df["OS.time"].values, val_df["OS"].values)
    test_dataset = ClinicalDataset(X_test, test_df["OS.time"].values, test_df["OS"].values)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # 4. Build Model M1
    logger.info("Building Model M1 (Clinical-Only)...")

    class ModelM1(torch.nn.Module):
        def __init__(self, d_clin, d_emb, dropout):
            super().__init__()
            self.encoder = ClinicalEncoder(d_clin, d_emb, dropout)
            self.head = CoxSurvivalHead(d_emb)

        def forward(self, x):
            emb = self.encoder(x)
            return self.head(emb)

    model = ModelM1(d_clin, config["d_emb"], config["dropout"])
    criterion = CoxPHLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )

    # 5. Train Model
    trainer = SurvivalTrainer(model, criterion, optimizer, device, logger, config["patience"])
    train_results = trainer.train(train_loader, val_loader, config["max_epochs"])

    # 6. Final Evaluation on Test Set
    logger.info("Evaluating on Test Set...")
    # The best model was automatically loaded by the trainer
    test_loss, test_cindex = trainer.evaluate(test_loader)
    logger.info(f"Test C-index: {test_cindex:.4f}")

    # 7. Log Results
    final_results = {
        "model": "M1_Clinical_Only",
        "config": config,
        "best_val_cindex": train_results["best_val_cindex"],
        "test_cindex": test_cindex,
    }
    log_metrics(final_results, PROJECT_ROOT / "results" / "metrics" / "clinical_only.json")
    torch.save(model.state_dict(), PROJECT_ROOT / "results" / "checkpoints" / "M1_best.pt")
    logger.info("Experiment M1 completed successfully.")


if __name__ == "__main__":
    main()