# Project Progress Tracker

This document tracks the implementation progress of the "Attention-Based Multimodal Deep Learning for Breast Cancer Detection and Survival Prognosis" project.

## Implementation Phases

### Module 0: Project Scaffolding
- [x] Create directory structure (`src/`, `scripts/`, `configs/`, `notebooks/`, `results/`, `dataset/`)
- [x] `src/utils/config.py`: YAML configuration loader
- [x] `src/utils/logger.py`: Logging utilities for metrics and experiments
- [x] `src/utils/seed.py`: Random seed setting for reproducibility
- [x] `requirements.txt`: Define project dependencies

### Module 1: Data Cohort Pipeline
- [x] `scripts/01_inspect_datasets.py`: Audit script to verify column names and event rates
- [x] `src/data/cohort_builder.py`: Logic to join clinical/survival data, filter to primary tumors, and split into train/val/test
- [x] `scripts/02_build_cohort.py`: Generates the `patient_cohort.csv` tracking split assignments
- [x] Generate `patient_cohort.csv` (1,076 patients)

### Module 2: Clinical Baseline (M1)
- [x] `src/data/clinical_preprocessing.py`: Scikit-learn imputation and encoding pipeline (leakage-proof)
- [x] `src/models/clinical_encoder.py`: 2-layer MLP for clinical features
- [x] `src/models/survival_head.py`: Linear Cox predictor
- [x] `src/training/cox_loss.py`: Negative partial log-likelihood loss with Breslow tie handling
- [x] `src/training/trainer.py`: PyTorch training loop with early stopping
- [x] `src/evaluation/metrics.py`: Concordance index (C-index) calculation
- [x] `configs/clinical_only.yaml`: Configuration for clinical baseline
- [x] `scripts/06_train_clinical_only.py`: Execution script for Model M1
- [x] Evaluate M1 and log results to `results/metrics/clinical_only.json`

### Module 3: WSI Processing & Image Baseline (M2)
- [x] `scripts/03_gdc_query.py`: Fetch slide manifest for the cohort
- [x] Select prototype patients and download WSI SVS files (Completed via Google Colab)
- [x] `src/data/patch_extractor.py`: OpenSlide tissue segmentation and patch extraction (Completed via Google Colab)
- [x] `scripts/05_extract_patches.py`: Execute patch extraction (Completed via Google Colab)
- [x] `scripts/06_extract_features.py`: Offline pipeline to save `.pt` patch embeddings using pretrained ResNet-50 (Completed via Google Colab)
- [x] `src/data/wsi_dataset.py`: PyTorch dataset for loading `.pt` patch embeddings
- [x] `src/models/image_encoder.py`: Linear layer mapping 2048-D to 128-D
- [x] `configs/image_only.yaml`: Configuration for image baseline
- [x] `scripts/07_train_image_only.py`: Execution script for Model M2
- [x] Evaluate M2 and log results

### Module 4: Multimodal Fusion Models (M3 & M4)
- [x] `src/data/multimodal_dataset.py`: PyTorch Dataset delivering both clinical vectors and patch embeddings
- [x] `src/models/model_factory.py`: Unified factory to instantiate M1/M2/M3/M4
- [x] `configs/concat_fusion.yaml` & `configs/multimodal_attention.yaml`: Configurations for both fusion architectures
- [x] **Concat Fusion (M3):**
  - [x] `src/models/concat_fusion.py`: Implement simple concatenation baseline
  - [x] `scripts/08_train_concat_fusion.py`: Train and log M3
- [x] **Attention Fusion (M4):**
  - [x] `src/models/attention_fusion.py`: Bidirectional Q/K/V cross-modal attention module
  - [x] `scripts/09_train_attention_fusion.py`: Train and log M4
  - [x] Train M4 Model and evaluate
- [x] Persist model checkpoints (`M2_best.pt`, `M3_best.pt`, `M4_best.pt`) in `results/checkpoints/`
- [x] 5-Fold patient-level cross-validation benchmark on 60 prototype patients (`scripts/10_cross_validate_prototype.py`)

### Module 5: Evaluation & Explainability
- [x] `src/evaluation/survival_plots.py`: Kaplan-Meier curve generation and log-rank tests
- [x] `scripts/14_evaluate_all_models.py`: Unified 5-fold CV runner across all models (M1, M2, M3, M4) on the 298-patient cohort
- [x] `src/explainability/gradcam.py`: Interpretability hooks and overlay generator for convolutional feature maps
- [x] `src/explainability/shap_clinical.py`: Clinical feature attribution using KernelExplainer and summary plots
- [x] `src/explainability/attention_analysis.py`: Spatial patch attention entropy and saliency analysis for M4
- [x] `scripts/15_run_explainability.py`: Execute SHAP, Grad-CAM, and patch attention weight explainability suite
- [x] Full performance metrics persisted to `results/metrics/final_model_evaluation.json` and `results/metrics/explainability_summary.json`
- [x] High-resolution figures generated in `results/figures/` (Kaplan-Meier comparison, SHAP summary, Patch attention distribution, Grad-CAM demo)

### Module 6: Clinical Prognosis Web Application & Dashboard
- [x] `app/schemas.py`: Pydantic input/output schemas for patient clinical features, survival predictions, and metrics
- [x] `app/api.py`: FastAPI server serving endpoints `/api/health`, `/api/metrics`, `/api/samples`, `/api/explainability`, `/api/predict`, and static `/figures`
- [x] `frontend/`: Vite 5 + React 18 + Tailwind CSS 3.4 single-page application adhering to Stitch clinical theme tokens
- [x] `frontend/src/views/DashboardView.jsx`: Cohort KPIs, model benchmark table, and interactive Kaplan-Meier curves
- [x] `frontend/src/views/PatientPredictionView.jsx`: Clinical parameter form, preloaded sample presets, and WSI sequence selector
- [x] `frontend/src/views/PredictionResultView.jsx`: Calculated hazard score, risk classification badge, and 10-year dynamic survival curve projection
- [x] `frontend/src/views/ExplainabilityView.jsx`: Interactive 120-patch WSI spatial attention grid with hover tooltips, top saliency patches, and global SHAP attributions
- [x] `frontend/src/views/ArchitectureView.jsx`: 4-stage multimodal pipeline architecture and frozen hyperparameter specifications
- [x] End-to-end automated browser verification across all 5 views with video recording and UI screenshots captured

---
*Tracker updated upon full completion of Module 5 & Module 6.*
