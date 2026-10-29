# Attention-Based Multimodal Deep Learning for Breast Cancer Survival Prognosis Using Histopathology Images and Clinical Data
## Complete System Architecture and Implementation Plan

**Project Title:** Attention-Based Multimodal Deep Learning for Breast Cancer Survival Prognosis Using Histopathology Images and Clinical Data  
**Dataset:** TCGA-BRCA (The Cancer Genome Atlas - Breast Invasive Carcinoma)  
**Primary Modalities:** H&E Whole-Slide Histopathology Images + Structured Clinical Data  
**Primary Task:** Overall Survival Prediction (Cox-based risk scoring)  
**Evaluation Metric:** Concordance Index (C-index), Kaplan-Meier stratification  
**Status:** Implementation Plan — No results fabricated. All metrics derived from experiments.

---

## Table of Contents

1. [Dataset Analysis (Real Data Audit)](#1-dataset-analysis-real-data-audit)
2. [Research Questions](#2-research-questions)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Project Directory Structure](#4-project-directory-structure)
5. [Data Pipeline & Cohort Selection](#5-data-pipeline--cohort-selection)
6. [Clinical Feature Engineering](#6-clinical-feature-engineering)
7. [Image Processing Pipeline (WSI)](#7-image-processing-pipeline-wsi)
8. [Model Architectures](#8-model-architectures)
9. [Training Strategy](#9-training-strategy)
10. [Evaluation Strategy](#10-evaluation-strategy)
11. [Explainability Design](#11-explainability-design)
12. [Baseline and Ablation Experiments](#12-baseline-and-ablation-experiments)
13. [Technical Risk Analysis](#13-technical-risk-analysis)
14. [Computational Requirements](#14-computational-requirements)
15. [Implementation Roadmap](#15-implementation-roadmap)
16. [Literature Grounding](#16-literature-grounding)
17. [Changelog](#17-changelog)

---

## 1. Dataset Analysis (Real Data Audit)

> All figures were computed directly from the actual downloaded files. No values are fabricated.

### 1.1 Survival File: TCGA-BRCA.survival.tsv

| Property | Value |
|---|---|
| Total sample rows | 1,232 |
| Unique patients (_PATIENT column) | **1,076** |
| OS.time range | 1.0 to 8,605.0 days |
| Median OS.time | 922.0 days |
| OS events (OS=1, deceased) | 201 of 1,232 = **16.3%** |
| OS censored (OS=0, alive at last follow-up) | 1,031 of 1,232 = 83.7% |

**Columns confirmed present:** sample, OS.time, OS, _PATIENT

**Critical observations:**
- Multiple sample IDs map to the same _PATIENT. Patient-level deduplication is mandatory before any train/val/test split.
- Low event rate (16.3%) means the dataset is heavily right-censored — Cox-based survival modeling is the correct approach.

### 1.2 Clinical File: TCGA-BRCA.clinical.tsv

| Property | Value |
|---|---|
| Total sample rows | 1,255 |
| Unique patients (submitter_id column) | **1,098** |
| Total columns | 85 |
| Confirmed patient overlap with survival file | **1,076 patients** |

> [!CAUTION]
> **Target variables — MUST NOT be used as model input features under any circumstances:**
> - `OS.time` — survival/follow-up time (used only inside the Cox loss function as a label)
> - `OS` — event indicator (used only inside the Cox loss function as a label)

### 1.3 Confirmed Usable Cohort

- **1,076 patients** have both clinical information AND survival data.
- Among these 1,076 patients, a subset will have matched H&E whole-slide images available on GDC.
- Only primary tumor slides (sample type -01) should be used for imaging.

---

## 2. Research Questions

| ID | Research Question |
|---|---|
| **RQ1** | How well does structured clinical information alone predict overall survival in TCGA-BRCA patients? |
| **RQ2** | How well does H&E histopathology image information alone predict OS? |
| **RQ3** | Does combining imaging and clinical modalities outperform either unimodal approach? |
| **RQ4** | Does attention-based multimodal fusion outperform simple feature concatenation? |
| **RQ5** | Which clinical features and image regions contribute most to survival predictions? |
| **RQ6** | How stable and consistent are model results under patient-level cross-validation? |

---

## 3. System Architecture Overview

### 3.1 Four Models for Systematic Comparison

| Model | Architecture Chain |
|---|---|
| **M1: Clinical-Only** | Clinical features &rarr; MLP &rarr; Cox survival head &rarr; risk score |
| **M2: Image-Only** | H&E patches &rarr; ResNet-50 (offline) &rarr; patch embeddings &rarr; mean pooling aggregation &rarr; Cox survival head &rarr; risk score |
| **M3: Concat-Fusion** | Clinical embedding + Image embedding &rarr; concatenate &rarr; prediction layer &rarr; Cox survival head &rarr; risk score |
| **M4: Attention-Fusion (Proposed)** | Clinical embedding + Image embedding &rarr; cross-modal attention &rarr; fused representation &rarr; prediction layer &rarr; Cox survival head &rarr; risk score |

### 3.2 Cross-Modal Attention Fusion (Core Proposed Contribution)

The cross-modal attention module implements **bidirectional attention** between modalities using query/key/value projections. 

> [!NOTE]
> **Design Clarification:** This is a lightweight cross-modal interaction mechanism operating on meaningful modality representations suitable for this dataset size. It is **NOT** a highly expressive sequence-level Transformer over a long token sequence. 

**Conceptual Structure:**
- **Clinical-to-image attention:** Clinical representation &rarr; query projection. Image representation(s) &rarr; key/value projections.
- **Image-to-clinical attention:** Image representation &rarr; query projection. Clinical representation &rarr; key/value projections.
- **Fusion:** Both attention contexts are concatenated &rarr; normalization/projection layer &rarr; Cox survival head.

This distinct Q/K/V structure generates explicit attention weights, allowing us to inspect learned cross-modal interactions (e.g., how much the clinical embedding relies on the image embedding).

---

## 4. Project Directory Structure

*(Structure remains preserved: `dataset/`, `slides/`, `patches/`, `features/`, `src/`, `scripts/`, `notebooks/`, `configs/`, `results/`)*

---

## 5. Data Pipeline & Cohort Selection

### 5.1 Development vs Final Research Experiments

This project explicitly separates prototype development from the final research evaluation to manage computational expense without compromising experimental integrity.

**A. Development / Prototype Stage:**
- **Cohort:** A small prototype cohort (approx 50–100 patients).
- **Purpose:** Strictly used to verify the engineering pipeline (downloading WSI, extracting patches, extracting offline ResNet-50 features, implementing models, and debugging).
- **Selection:** Representative patient-level selection. *Do not artificially enrich this cohort by oversampling deceased (OS=1) patients.* If the low event rate causes instability, document it as a known artifact of the development cohort.

**B. Final Research Experiment:**
- **Cohort:** The largest practical matched cohort available from the 1,076 clinical+survival patients that have corresponding H&E slides.
- **Purpose:** Used for all final reported C-index metrics, p-values, and statistical evaluations.
- **Selection:** Reproducible, strictly patient-level selection. Report the actual event and censoring distribution. *Never upsample or oversample OS=1 patients for the final evaluation.*

### 5.2 Patient-Level Train/Val/Test Split

Data splitting applies to the defined cohort (prototype or final) with strict leakage protection:
- **Patient-Level:** All splits occur strictly at the patient ID level. No patient may appear in more than one partition.
- **No WSI Leakage:** No patch from a test patient is ever visible during training.
- **No Clinical Leakage:** Clinical preprocessing (e.g., imputation, scaling) must be fitted on training data only, and applied to validation/test sets via the fitted pipeline.

---

## 6. Clinical Feature Engineering

Based on actual TCGA-BRCA data:
- **Features Used:** Age, Gender, Race, AJCC Stage, T Stage, N Stage, M Stage, Histotype, Year of Diagnosis.
- **Encoding:** Scikit-learn pipeline (imputation, ordinal/one-hot encoding, standard scaling).
- **Target Variables (OS.time, OS):** Never used as model input.

---

## 7. Image Processing Pipeline (WSI)

### 7.1 Acquisition & Patch Extraction
- SVS files downloaded via GDC client.
- Tissue segmentation via HSV Otsu thresholding on a downsampled thumbnail.
- Extract valid 256x256 tissue patches at 20x magnification.

### 7.2 Offline Feature Extraction (ResNet-50)
- **Model:** Pretrained ResNet-50 (ImageNet weights), fully-connected layer removed.
- **Process:** Generate 2048-dimensional features per patch.
- **Important:** Feature extraction is offline. ResNet-50 is NOT repeatedly run during survival-model training.

### 7.3 Slide-Level Aggregation
- **Primary Baseline (M2):** Mean pooling across all patch features to create the slide representation.
- **Ablation Experiment:** Attention-Based Multiple Instance Learning (ABMIL) will be tested as an optional ablation against mean pooling.

---

## 8. Model Architectures

**(Definitions align with Section 3.1)**
- **M1:** 2-layer MLP for clinical features.
- **M2:** MeanPool aggregator + projection layer.
- **M3:** Concatenation + linear layer.
- **M4:** Bidirectional Q/K/V cross-modal attention module.
- **Cox Survival Head:** A simple linear risk head over the learned representation mapping to a scalar risk score.

---

## 9. Training Strategy

### 9.1 Cox Proportional Hazards Loss
```
L_cox = -(1 / N_events) * SUM_i [ delta_i * (eta_i - log(SUM_j: t_j >= t_i [exp(eta_j)])) ]
```
- Targets `OS.time` and `OS event indicator` are used strictly as survival labels.
- Implements negative partial log-likelihood with Breslow approximation for ties.

### 9.2 Validation
- **Patient-Level 5-Fold Cross Validation:** Prevents leakage and provides stable C-index estimates across folds. 

---

## 10. Evaluation Strategy

- **Primary Metric:** Concordance Index (C-index). Evaluates ranking of predicted risk scores against actual survival times.
- **Secondary Metrics:** Kaplan-Meier survival curves, Log-rank test p-values.
- **Disclaimer:** All performance values remain [TBD] until experiments are completed. No metrics will be fabricated.

---

## 11. Explainability Design

These methods are used to inspect model behavior and learned interactions. **They do not constitute or prove clinical validity.**

- **Grad-CAM:** Applied to the final convolutional layer of ResNet-50 to visualize image regions associated with the model's risk predictions.
- **SHAP:** Used on the clinical MLP (M1) to analyze the contribution of individual clinical features to the risk score.
- **Attention Weights:** Used in M4 to inspect learned cross-modal interactions between clinical and imaging data.

---

## 12. Baseline and Ablation Experiments

- **Core Experiments:** E1(M1), E2(M2), E3(M3), E4(M4).
- **Ablations:** ABMIL vs Mean pooling, Embedding dimensions, Feature subsets.

---

## 13. Technical Risk Analysis

- Low event rate (16.3%) handled by appropriate Cox modeling and 5-fold CV.
- Computational bottlenecks handled by separating the small development cohort from the final scale-up experiment, and running ResNet-50 extraction strictly offline.

---

## 14. Computational Requirements

- Core stack remains unchanged: Python, PyTorch, Torchvision, scikit-learn, OpenSlide, Pillow, NumPy, Pandas, SciPy, scikit-image, lifelines, scikit-survival, SHAP, Matplotlib.

---

## 15. Implementation Roadmap

The execution priority specifically starts with clinical data to validate the survival modeling pipeline before attempting expensive WSI processing.

1. Validate clinical + survival data
2. Build patient-level cohort
3. Train clinical-only Cox baseline (M1 on full clinical cohort)
4. Query and obtain matched H&E WSI (for prototype)
5. Extract patches
6. Extract ResNet-50 features
7. Train image-only baseline (M2)
8. Train concatenation fusion (M3)
9. Train proposed attention fusion (M4)
10. Compare models
11. Explainability
12. Optional ablations
13. Optional scale-up (Final Research Experiment)

---

## 16. Literature Grounding

*(Existing references to Sucre et al. 2025 and Nishizawa et al. 2025 maintained)*

---

## 17. Changelog

**Changes applied based on user review:**
1. **Title Change:** Updated title to clearly state "Survival Prognosis Using Histopathology Images and Clinical Data" instead of generic "Detection", as the implemented task is strictly survival-risk prediction.
2. **Prototype Selection:** Removed the instruction to artificially oversample/enrich OS=1 patients. Prototype must use representative selection. 
3. **Cohort Distinction:** Added Section 5.1 to explicitly separate the small Development/Prototype cohort from the Final Research Experiment cohort.
4. **Clinical Baseline Scope:** Clarified that the M1 Clinical-Only baseline must utilize the full available clinical cohort (1,076 patients), not just the restricted WSI prototype overlap.
5. **Model Definitions:** Explicitly detailed the architectures for M1, M2, M3, and M4 in Section 3.1.
6. **Attention Module Clarification:** Added explicit documentation (Section 3.2) stating that the cross-modal attention is a lightweight Q/K/V interaction mechanism on modality representations, NOT a sequence-level Transformer.
7. **ResNet-50:** Clarified that feature extraction is strictly offline.
8. **Aggregation:** Confirmed Mean Pooling as the primary baseline, keeping ABMIL as an ablation experiment.
9. **Explainability Context:** Added disclaimers that Grad-CAM, SHAP, and Attention weights visualize model behavior, but do not prove clinical validity.
10. **Implementation Order:** Streamlined roadmap to explicitly show clinical baseline validation occurs before WSI downloads.

---
*Version 1.1 — September 2026*  
*No patient records, model weights, or performance metrics have been fabricated.*
