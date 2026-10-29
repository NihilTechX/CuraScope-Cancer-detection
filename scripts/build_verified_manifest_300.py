import json
import sys
from pathlib import Path
import pandas as pd
import requests

def main():
    print("=" * 70)
    print("BUILDING VERIFIED 300-PATIENT MANIFEST (HTTP 200 OK)")
    print("=" * 70)

    dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
    cohort_path = dataset_dir / "patient_cohort.csv"
    old_manifest_path = dataset_dir / "wsi_manifest_100.csv"
    out_manifest_path = dataset_dir / "wsi_manifest_300.csv"

    cohort_df = pd.read_csv(cohort_path)
    old_manifest_df = pd.read_csv(old_manifest_path)

    existing_pids = set(old_manifest_df['patient_id'])
    print(f"Existing 100 baseline patients preserved: {len(existing_pids)}")

    # Remaining eligible candidates from cohort
    remaining_cohort = cohort_df[~cohort_df['patient_id'].isin(existing_pids)].copy()
    print(f"Remaining eligible cohort pool: {len(remaining_cohort)} patients")

    # Sample candidates (we need 200, so we sample 400 to account for broken GDC links)
    candidate_sample = remaining_cohort.sample(n=400, random_state=42)
    candidate_pids = candidate_sample['patient_id'].tolist()

    print("Querying GDC for released open-access SVS files by patient ID...")
    endpoint = "https://api.gdc.cancer.gov/files"
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.submitter_id",
                    "value": candidate_pids
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "data_format",
                    "value": "SVS"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "access",
                    "value": "open"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "state",
                    "value": "released"
                }
            }
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.samples.sample_id,cases.submitter_id,file_size",
        "format": "JSON",
        "size": "2000"
    }

    resp = requests.post(endpoint, json={"filters": filters, "fields": params["fields"], "size": params["size"]}, timeout=30)
    if resp.status_code != 200:
        print(f"GDC API Error: {resp.status_code}")
        sys.exit(1)

    hits = resp.json()['data']['hits']
    print(f"Retrieved {len(hits)} released open-access SVS files from GDC.")

    patient_slides = {}
    for hit in hits:
        pid = hit['cases'][0]['submitter_id']
        sid = hit['cases'][0]['samples'][0]['sample_id']
        fid = hit['file_id']
        fname = hit['file_name']
        fsize = hit['file_size']

        if pid not in patient_slides:
            patient_slides[pid] = []
            
        patient_slides[pid].append({
            'patient_id': pid,
            'sample_id': sid,
            'file_id': fid,
            'file_name': fname,
            'file_size_bytes': fsize
        })

    print(f"Patients with released SVS files: {len(patient_slides)}")

    verified_200 = []
    print("\nVerifying live download URLs (HTTP 200)... this may take a few minutes...")

    for pid, slides in patient_slides.items():
        if pid in existing_pids:
            continue
        if len(verified_200) >= 200:
            break

        # Try slides for this patient until one gives HTTP 200
        for slide in slides:
            url = f"https://api.gdc.cancer.gov/data/{slide['file_id']}"
            try:
                # We just ask for the headers to ensure the file exists without downloading
                with requests.get(url, stream=True, timeout=8) as r:
                    if r.status_code == 200:
                        verified_200.append(slide)
                        print(f"  [{len(verified_200):3d}/200] VERIFIED 200 OK: {pid}")
                        break
            except Exception as err:
                pass

    print(f"\nSuccessfully verified {len(verified_200)} / 200 new patients with live HTTP 200 OK!")
    
    if len(verified_200) < 200:
        print(f"WARNING: Only found {len(verified_200)} valid patients. Expected 200.")

    new_200_df = pd.DataFrame(verified_200)
    new_200_merged = pd.merge(
        new_200_df,
        remaining_cohort[['patient_id', 'split', 'OS.time', 'OS']],
        on='patient_id',
        how='inner'
    )

    cols = ['patient_id', 'sample_id', 'file_id', 'file_name', 'file_size_bytes', 'split', 'OS.time', 'OS']
    new_200_final = new_200_merged[cols].copy()
    old_100_final = old_manifest_df[cols].copy()

    manifest_300 = pd.concat([old_100_final, new_200_final], ignore_index=True)

    assert len(manifest_300) == (100 + len(verified_200))
    assert manifest_300['patient_id'].nunique() == len(manifest_300)

    manifest_300.to_csv(out_manifest_path, index=False)
    print(f"\nSUCCESS: Saved {len(manifest_300)}-patient verified manifest to {out_manifest_path}")

if __name__ == "__main__":
    main()
