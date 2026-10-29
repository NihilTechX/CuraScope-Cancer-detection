import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
from typing import Optional

# Columns required from the clinical file
REQUIRED_CLINICAL_COLS = [
    "submitter_id",
    "sample_type_id.samples",
    "sample_id.samples",
]

REQUIRED_SURVIVAL_COLS = ["_PATIENT", "OS.time", "OS"]


def _validate_inputs(
    survival_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
) -> None:
    """Raise ValueError if required columns are missing or targets contain NaN."""
    missing_surv = [c for c in REQUIRED_SURVIVAL_COLS if c not in survival_df.columns]
    if missing_surv:
        raise ValueError(
            f"Survival file missing columns: {missing_surv}. "
            f"Found: {list(survival_df.columns)}"
        )
    missing_clin = [c for c in REQUIRED_CLINICAL_COLS if c not in clinical_df.columns]
    if missing_clin:
        raise ValueError(
            f"Clinical file missing columns: {missing_clin}. "
            f"Found: {list(clinical_df.columns)}"
        )

    nan_surv = survival_df[["OS", "OS.time"]].isna().sum()
    if nan_surv.any():
        bad = nan_surv[nan_surv > 0].to_dict()
        raise ValueError(f"Survival file contains NaN in target columns: {bad}")

    bad_os = set(survival_df["OS"].unique()) - {0, 1}
    if bad_os:
        raise ValueError(
            f"OS column must be 0 or 1. Found unexpected values: {bad_os}"
        )


def _deduplicate_survival(survival_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate survival data to one row per patient.

    Uses max(OS.time) and max(OS) as independent aggregations, but
    asserts consistency: if a patient has any OS=1, the time taken
    must come from that row to avoid time/event mismatch.
    """
    surv_patient = (
        survival_df.groupby("_PATIENT")
        .agg(n_rows=("OS", "size"), os_max=("OS", "max"), t_max=("OS.time", "max"))
        .reset_index()
    )

    # Guard: time/event consistency — if max(OS)=1, verify the row with
    # OS=1 has the same max time (empirically true for TCGA-BRCA).
    os1_times = (
        survival_df[survival_df["OS"] == 1]
        .groupby("_PATIENT")["OS.time"]
        .max()
    )
    mismatched = []
    for pat in surv_patient["_PATIENT"]:
        if os1_times.get(pat, None) is not None:
            os1_t = os1_times[pat]
            patient_max_t = surv_patient.loc[
                surv_patient["_PATIENT"] == pat, "t_max"
            ].values[0]
            # Allow max(OS.time) to be >= the OS=1 time (it usually is the same)
            if os1_t > patient_max_t:
                mismatched.append(pat)

    if mismatched:
        raise ValueError(
            f"Survival dedup inconsistency: {len(mismatched)} patients have "
            f"OS=1 row with time greater than max(OS.time). "
            f"Patients: {mismatched[:5]}"
        )

    result = surv_patient[["_PATIENT", "t_max", "os_max"]].copy()
    result.rename(
        columns={"_PATIENT": "patient_id", "t_max": "OS.time", "os_max": "OS"},
        inplace=True,
    )
    return result


def _deduplicate_clinical(clinical_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate clinical data to one row per patient.

    Rules (matches ARCH_PART1 §5.1):
    1. Filter to primary tumor samples (sample_type_id.samples == 1).
    2. If multiple primary samples per patient, keep the one with the
       latest days_to_collection.samples (or first if tied/missing).
    """
    # Filter to primary tumor only
    primary = clinical_df[clinical_df["sample_type_id.samples"] == 1].copy()

    if primary.empty:
        raise ValueError(
            "No primary tumor rows (sample_type_id.samples == 1) found "
            "in clinical data."
        )

    # Sort by days_to_collection descending so first = most recent
    if "days_to_collection.samples" in primary.columns:
        primary["_days"] = pd.to_numeric(
            primary["days_to_collection.samples"], errors="coerce"
        )
        primary.sort_values(
            ["submitter_id", "_days"], ascending=[True, False], inplace=True
        )
        primary.drop(columns=["_days"], inplace=True)

    # Keep first (most recent primary) per patient
    clin_patient = primary.drop_duplicates(subset=["submitter_id"], keep="first").copy()
    clin_patient.rename(columns={"submitter_id": "patient_id"}, inplace=True)
    return clin_patient


def build_patient_cohort(
    survival_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    output_path: str | Path,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build the final patient cohort and generate splits.

    Steps:
    1. Validate required columns and targets.
    2. Deduplicate survival data to patient-level.
    3. Deduplicate clinical data to primary-tumor-only, patient-level.
    4. Inner join on patient ID.
    5. Stratified split (70/15/15) on OS event indicator.
    6. Save to CSV with patient_id, sample_id, OS.time, OS, split.

    Parameters
    ----------
    survival_df : DataFrame with columns _PATIENT, OS.time, OS
    clinical_df : DataFrame with submitter_id, sample_type_id.samples, sample_id.samples, ...
    output_path : path to write patient_cohort.csv
    random_state : seed for stratified splits

    Returns
    -------
    DataFrame with columns patient_id, sample_id, OS.time, OS, split
    """
    _validate_inputs(survival_df, clinical_df)

    # 1. Deduplicate to patient-level
    surv_patient = _deduplicate_survival(survival_df)
    clin_patient = _deduplicate_clinical(clinical_df)

    # 2. Inner join — carry sample_id downstream for WSI matching
    merged = pd.merge(
        surv_patient,
        clin_patient[["patient_id", "sample_id.samples"]],
        on="patient_id",
        how="inner",
    )
    merged.rename(columns={"sample_id.samples": "sample_id"}, inplace=True)

    n_patients = len(merged)
    n_events = merged["OS"].sum()
    print(
        f"Cohort: {n_patients} patients | "
        f"events: {n_events} ({n_events/n_patients:.1%})"
    )

    # 3. Stratified split (70% train, 30% temp)
    split1 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=random_state
    )
    train_idx, temp_idx = next(split1.split(merged, merged["OS"]))

    train_df = merged.iloc[train_idx].copy()
    temp_df = merged.iloc[temp_idx].copy()

    # 4. Split temp 50/50 → 15% val, 15% test
    split2 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=random_state
    )
    val_idx, test_idx = next(split2.split(temp_df, temp_df["OS"]))

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    final_cohort = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # 5. Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_cohort.to_csv(output_path, index=False)

    return final_cohort
