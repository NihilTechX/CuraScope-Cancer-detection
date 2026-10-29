import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.cohort_builder import build_patient_cohort


def main():
    dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
    survival_file = dataset_dir / "TCGA-BRCA.survival.tsv"
    clinical_file = dataset_dir / "TCGA-BRCA.clinical.tsv"
    output_file = dataset_dir / "patient_cohort.csv"

    surv_df = pd.read_csv(survival_file, sep="\t")
    clin_df = pd.read_csv(clinical_file, sep="\t")

    print("Building patient cohort...")
    cohort = build_patient_cohort(surv_df, clin_df, output_file)

    print(f"\nCohort built successfully! Total patients: {len(cohort)}")
    print(f"Saved to: {output_file}")

    print("\nSplit Distribution:")
    split_counts = cohort["split"].value_counts()
    for split, count in split_counts.items():
        event_rate = cohort[cohort["split"] == split]["OS"].mean()
        print(f"  {split.capitalize()}: {count} patients ({event_rate:.2%} event rate)")


if __name__ == "__main__":
    main()
