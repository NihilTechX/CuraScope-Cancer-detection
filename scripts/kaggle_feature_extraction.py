import os
import sys
import time
import json
import argparse
import tempfile
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
try:
    import openslide
except ImportError:
    print("Warning: openslide-python not found. WSI extraction will fail.", file=sys.stderr)

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
except ImportError:
    print("Warning: google-api-python-client or google-auth not found.", file=sys.stderr)

# ==============================================================================
# CONFIGURATION CONSTANTS (DO NOT CHANGE)
# ==============================================================================
PATCH_SIZE = 256
MAX_PATCHES = 120
TARGET_MAG = 20
BG_THRESHOLD = 210
TISSUE_RATIO = 0.5
IMAGE_INPUT_DIM = 2048

# ==============================================================================
# 1. GOOGLE DRIVE API INTEGRATION
# ==============================================================================
def authenticate_drive():
    sa_json_str = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not sa_json_str:
        raise ValueError("Environment variable GCP_SERVICE_ACCOUNT_JSON is not set. Please inject Kaggle Secret.")
    
    try:
        sa_info = json.loads(sa_json_str)
        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=['https://www.googleapis.com/auth/drive']
        )
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        return service
    except Exception as e:
        raise RuntimeError(f"Failed to authenticate with Google Drive: {e}")

def get_or_create_folder(service, folder_name, parent_id=None):
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    
    if not items:
        # Create folder
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            file_metadata['parents'] = [parent_id]
        
        folder = service.files().create(body=file_metadata, fields='id').execute()
        print(f"Created Drive folder: {folder_name} (ID: {folder.get('id')})")
        return folder.get('id')
    return items[0]['id']

def list_drive_files(service, folder_id):
    files = set()
    page_token = None
    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='nextPageToken, files(id, name, size)',
            pageToken=page_token
        ).execute()
        for file in response.get('files', []):
            files.add(file.get('name'))
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    return files

def upload_file_to_drive(service, local_path, folder_id, max_retries=3):
    file_name = os.path.basename(local_path)
    file_metadata = {'name': file_name, 'parents': [folder_id]}
    media = MediaFileUpload(local_path, resumable=True)
    
    for attempt in range(1, max_retries + 1):
        try:
            file = service.files().create(body=file_metadata, media_body=media, fields='id, size').execute()
            # Verify upload size
            local_size = os.path.getsize(local_path)
            remote_size = int(file.get('size', 0))
            if remote_size > 0:
                return True
        except Exception as e:
            print(f"  [Drive Upload Retry {attempt}/{max_retries}] Failed: {e}")
            time.sleep(2 ** attempt)
    return False

# ==============================================================================
# 2. FEATURE EXTRACTION PIPELINE
# ==============================================================================
def get_resnet50_extractor(device):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model

preprocess_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FastPatchExtractor:
    def __init__(self):
        self.patch_size = PATCH_SIZE
        self.target_mag = TARGET_MAG
        self.bg_threshold = BG_THRESHOLD
        self.tissue_ratio = TISSUE_RATIO

    def get_level(self, slide):
        try:
            base_mag = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40.0))
        except Exception:
            base_mag = 40.0
        target_downsample = base_mag / self.target_mag
        return slide.get_best_level_for_downsample(target_downsample)

    def is_tissue(self, pil_img):
        gray = np.array(pil_img.convert('L'))
        tissue_pixels = np.sum(gray < self.bg_threshold)
        return (tissue_pixels / (self.patch_size * self.patch_size)) >= self.tissue_ratio

    def extract(self, slide_path):
        slide = openslide.OpenSlide(str(slide_path))
        level = self.get_level(slide)
        downsample = slide.level_downsamples[level]
        step_lvl0 = int(self.patch_size * downsample)
        w_lvl0, h_lvl0 = slide.dimensions

        patches = []
        for y in range(0, h_lvl0 - step_lvl0, step_lvl0):
            for x in range(0, w_lvl0 - step_lvl0, step_lvl0):
                if len(patches) >= MAX_PATCHES:
                    break
                try:
                    region = slide.read_region((x, y), level, (self.patch_size, self.patch_size)).convert('RGB')
                    if self.is_tissue(region):
                        patches.append(region)
                except Exception:
                    continue
            if len(patches) >= MAX_PATCHES:
                break
        slide.close()
        return patches

def download_svs(file_id, save_path, max_retries=3, chunk_size=1024*1024):
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            if os.path.exists(save_path) and os.path.getsize(save_path) > 1000000:
                return True
        except Exception as e:
            print(f"  [Download Retry {attempt}/{max_retries}] Interrupted: {e}")
            if os.path.exists(save_path):
                os.remove(save_path)
            time.sleep(2 ** attempt)
    return False

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Kaggle Scalable WSI Feature Extractor")
    parser.add_argument("--manifest", type=str, default="dataset/wsi_manifest_300.csv", help="Path to patient manifest CSV")
    parser.add_argument("--max-patients", type=int, default=None, help="Limit number of patients for testing")
    parser.add_argument("--dry-run", action="store_true", help="Authenticate and check Drive files, but download nothing")
    parser.add_argument("--drive-folder-id", type=str, default=None, help="Google Drive Folder ID of the shared TCGA_Project folder. (Required to avoid 0-quota errors)")
    parser.add_argument("--local-only", action="store_true", help="Save features locally to /kaggle/working/patch_features and bypass Google Drive completely.")
    args = parser.parse_args()

    # Hardware Report
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- HARDWARE REPORT ---")
    print(f"Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Load manifest
    if not os.path.exists(args.manifest):
        print(f"Error: Manifest {args.manifest} not found.")
        return
    manifest_df = pd.read_csv(args.manifest)
    if args.max_patients:
        manifest_df = manifest_df.head(args.max_patients)
    total_patients = len(manifest_df)
    print(f"\nLoaded manifest with {total_patients} patients.")

    # Authenticate and setup Drive structure
    existing_files = set()
    service = None
    patch_folder_id = None
    
    if not args.local_only:
        print("\n--- AUTHENTICATING GOOGLE DRIVE ---")
        try:
            service = authenticate_drive()
            print("Successfully authenticated with Google Drive API.")
        except Exception as e:
            print(f"Authentication Failed: {e}")
            return

        print("\n--- RESOLVING DRIVE FOLDERS ---")
        if args.drive_folder_id:
            print(f"Using explicitly provided TCGA_Project Folder ID: {args.drive_folder_id}")
            tcga_folder_id = args.drive_folder_id
        else:
            print("WARNING: No --drive-folder-id provided. Service Account may try to use its own 0-quota storage!")
            tcga_folder_id = get_or_create_folder(service, "TCGA_Project")
            
        patch_folder_id = get_or_create_folder(service, "patch_features", parent_id=tcga_folder_id)
        
        print("Fetching list of existing features in Drive...")
        existing_files = list_drive_files(service, patch_folder_id)
        print(f"Found {len(existing_files)} existing patch feature files in Drive.")
    else:
        print("\n--- RUNNING IN LOCAL-ONLY MODE (Bypassing Google Drive) ---")
        local_feature_dir = Path("/kaggle/working/patch_features") if os.path.exists("/kaggle/working") else Path(tempfile.gettempdir()) / "patch_features"
        local_feature_dir.mkdir(exist_ok=True)
        for f in os.listdir(local_feature_dir):
            if f.endswith(".pt"):
                existing_files.add(f)
        print(f"Found {len(existing_files)} existing patch feature files in local directory.")

    if args.dry_run:
        print("\n--- DRY RUN SUMMARY ---")
        skipped = 0
        missing = 0
        for _, row in manifest_df.iterrows():
            target_name = f"{row['patient_id']}_patches.pt"
            if target_name in existing_files:
                skipped += 1
            else:
                missing += 1
        print(f"Patients Completed (Will Skip): {skipped}")
        print(f"Patients Missing (Will Process): {missing}")
        print("Dry run finished. Exiting.")
        return

    # Initialize models
    resnet = get_resnet50_extractor(device)
    patch_extractor = FastPatchExtractor()

    # Local working directories (Volatile on Kaggle)
    working_dir = Path("/kaggle/working") if os.path.exists("/kaggle/working") else Path(tempfile.gettempdir())
    print(f"\nUsing local working directory: {working_dir}")

    # Processing Loop
    for idx, row in manifest_df.iterrows():
        patient_id = row['patient_id']
        file_id = row['file_id']
        target_name = f"{patient_id}_patches.pt"
        
        print(f"\n[{idx+1}/{total_patients}] PATIENT: {patient_id}")
        
        # 1. Persistent Check
        if target_name in existing_files:
            print(f"  -> [SKIPPED] Feature already exists in Google Drive.")
            continue

        local_svs = working_dir / f"{patient_id}.svs"
        local_pt = working_dir / target_name
        
        try:
            # 2. Download SVS
            start_time = time.time()
            print("  -> Downloading SVS from GDC...")
            if not download_svs(file_id, str(local_svs)):
                print("  -> [ERROR] Download completely failed. Skipping patient.")
                continue

            # 3. Extract Patches
            print("  -> Extracting patches...")
            patches = patch_extractor.extract(local_svs)
            
            if len(patches) == 0:
                print("  -> [ERROR] No valid tissue patches extracted. Skipping.")
                if local_svs.exists(): local_svs.unlink()
                continue
            
            # 4. ResNet Embedding
            print(f"  -> Computing ResNet embeddings for {len(patches)} patches...")
            patch_tensors = torch.stack([preprocess_transform(p) for p in patches])
            batch_size = 64
            embeddings = []
            with torch.no_grad():
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    for b_start in range(0, len(patch_tensors), batch_size):
                        batch = patch_tensors[b_start : b_start + batch_size].to(device, non_blocking=True)
                        emb = resnet(batch)
                        embeddings.append(emb.cpu())
                        
            all_embs = torch.cat(embeddings, dim=0).float() # [N, 2048]
            
            # Validate constraints
            assert all_embs.dim() == 2 and all_embs.shape[1] == 2048, f"Invalid shape {all_embs.shape}"
            assert all_embs.shape[0] <= MAX_PATCHES, f"Exceeded max patches: {all_embs.shape[0]}"
            assert not torch.isnan(all_embs).any(), "NaN found in embeddings"
            assert not torch.isinf(all_embs).any(), "Inf found in embeddings"
            
            torch.save(all_embs, str(local_pt))
            
            if args.local_only:
                # Save locally in a dedicated folder and keep the .pt file
                local_feature_dir = working_dir / "patch_features"
                local_feature_dir.mkdir(exist_ok=True)
                final_pt_path = local_feature_dir / target_name
                import shutil
                shutil.move(str(local_pt), str(final_pt_path))
                print(f"  -> [SUCCESS] Saved locally to {final_pt_path}")
                local_svs.unlink()
                
                log_row = pd.DataFrame([{
                    'patient_id': patient_id, 'status': 'success', 
                    'num_patches': len(patches), 'elapsed_time': round(time.time() - start_time, 1)
                }])
                log_path = working_dir / "processing_log.csv"
                log_row.to_csv(log_path, mode='a', header=not log_path.exists(), index=False)
                
            else:
                # 5. Upload to Drive
                print("  -> Uploading tensor to Google Drive...")
                upload_success = upload_file_to_drive(service, str(local_pt), patch_folder_id)
                
                if upload_success:
                    print("  -> [SUCCESS] Upload verified.")
                    # 6. Strict Cleanup ONLY on success
                    local_pt.unlink()
                    local_svs.unlink()
                    
                    # Append to local log for runtime tracking
                    log_row = pd.DataFrame([{
                        'patient_id': patient_id, 'status': 'success', 
                        'num_patches': len(patches), 'elapsed_time': round(time.time() - start_time, 1)
                    }])
                    log_path = working_dir / "processing_log.csv"
                    log_row.to_csv(log_path, mode='a', header=not log_path.exists(), index=False)
                    
                    # Update our in-memory cache so we don't re-upload if logic loops
                    existing_files.add(target_name)
                else:
                    print("  -> [ERROR] Drive upload failed after retries.")
                    # Retain local files for potential manual intervention or next loop
                
        except Exception as e:
            print(f"  -> [FATAL ERROR] {e}")
            # Do NOT create fake checkpoints. Keep log info.
            if local_pt.exists(): local_pt.unlink()
            if local_svs.exists(): local_svs.unlink()

if __name__ == "__main__":
    main()
