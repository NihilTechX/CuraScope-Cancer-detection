import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Play,
  RotateCcw,
  Microscope,
  FileText,
  Activity,
  ArrowRight,
  UploadCloud,
  CheckCircle2,
  Info
} from 'lucide-react';
import Input from '../components/Input';
import Select from '../components/Select';
import Button from '../components/Button';
import ImageUpload from '../components/ImageUpload';
import RiskAssessment from '../components/RiskAssessment';
import SurvivalCurve from '../components/SurvivalCurve';

// Sample case metadata
const SAMPLE_METADATA = {
  'TCGA-A2-A0T2': {
    label: 'Early-stage case',
    tag: 'Favorable · Stage I',
    image: '/samples/sample_he_stage1.jpg',
    description: '48yo Female · Stage I · Grade 1 · T1c N0 M0 · ER+ PR+ HER2-',
  },
  'TCGA-A2-A0CM': {
    label: 'Intermediate case',
    tag: 'Moderate · Stage IIA',
    image: '/samples/sample_he_stage2.jpg',
    description: '59yo Female · Stage IIA · Grade 2 · T2 N0 M0 · ER+ PR- HER2-',
  },
  'TCGA-A2-A0T0': {
    label: 'Advanced case',
    tag: 'Elevated · Stage IV',
    image: '/samples/sample_he_stage4.jpg',
    description: '74yo Female · Stage IV · Grade 3 · T4b N3a M1 · ER- PR- HER2+',
  },
};

// Clinical Staging Options
const STAGE_OPTIONS = [
  'Stage I',
  'Stage IIA',
  'Stage IIB',
  'Stage IIIA',
  'Stage IIIB',
  'Stage IIIC',
  'Stage IV',
];

const T_OPTIONS = [
  { value: 'T1', label: 'T1 — Small tumor (≤ 20mm)' },
  { value: 'T1c', label: 'T1c — Contained (10-20mm)' },
  { value: 'T2', label: 'T2 — Medium tumor (20-50mm)' },
  { value: 'T3', label: 'T3 — Large tumor (> 50mm)' },
  { value: 'T4', label: 'T4 — Chest wall / skin extension' },
  { value: 'T4b', label: 'T4b — Extensive ulceration / edema' },
];

const N_OPTIONS = [
  { value: 'N0', label: 'N0 — No regional lymph node metastasis' },
  { value: 'N1', label: 'N1 — 1 to 3 axillary lymph nodes' },
  { value: 'N2', label: 'N2 — 4 to 9 axillary lymph nodes' },
  { value: 'N3', label: 'N3 — 10+ axillary or infraclavicular nodes' },
  { value: 'N3a', label: 'N3a — 10+ axillary nodes (> 2mm)' },
];

const M_OPTIONS = [
  { value: 'M0', label: 'M0 — No clinical or radiographic metastasis' },
  { value: 'M1', label: 'M1 — Distant organ metastasis present' },
];

const GRADE_OPTIONS = [
  { value: 'Grade 1', label: 'Grade 1 — Well differentiated (Low)' },
  { value: 'Grade 2', label: 'Grade 2 — Moderately differentiated' },
  { value: 'Grade 3', label: 'Grade 3 — Poorly differentiated (High)' },
];

const STATUS_OPTIONS = ['Positive', 'Negative', 'Unknown'];

export default function PrognosisView({
  imageMode = 'sample',
  setImageMode,
  uploadedImageBase64,
  setUploadedImageBase64,
  uploadedFileName,
  setUploadedFileName,
  selectedSampleId,
  setSelectedSampleId,
  samples = [],
  formData,
  setFormData,
  onPredict,
  predictionResult,
  isLoading = false,
  onReset,
  onNewAssessment,
  onNavigateToInsights,
}) {
  const handleInputChange = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleImageUpload = (base64, fileName) => {
    setUploadedImageBase64(base64);
    setUploadedFileName(fileName);
    setImageMode('upload');
  };

  const handleImageRemove = () => {
    setUploadedImageBase64(null);
    setUploadedFileName(null);
  };

  const handleSampleSelect = (patientId) => {
    setSelectedSampleId(patientId);
    setImageMode('sample');

    const sample = samples.find((s) => s.patient_id === patientId);
    if (sample?.clinical) {
      setFormData((prev) => ({
        ...prev,
        age: sample.clinical.age ?? prev.age,
        ajcc_stage: sample.clinical.ajcc_stage ?? prev.ajcc_stage,
        pathologic_t: sample.clinical.pathologic_t ?? prev.pathologic_t,
        pathologic_n: sample.clinical.pathologic_n ?? prev.pathologic_n,
        pathologic_m: sample.clinical.pathologic_m ?? prev.pathologic_m,
        er_status: sample.clinical.er_status ?? prev.er_status,
        pr_status: sample.clinical.pr_status ?? prev.pr_status,
        her2_status: sample.clinical.her2_status ?? prev.her2_status,
        tumor_grade: sample.clinical.tumor_grade ?? prev.tumor_grade,
      }));
    }

    if (onReset) onReset();
  };

  const currentSampleMeta = SAMPLE_METADATA[selectedSampleId] || {
    label: selectedSampleId || 'Sample Patient',
    tag: 'Research Case',
    image: '/samples/sample_he_stage1.jpg',
    description: 'TCGA-BRCA multimodal cohort specimen',
  };

  // Active specimen thumbnail to display in the result section
  const activeThumbnail =
    imageMode === 'upload' && uploadedImageBase64
      ? uploadedImageBase64
      : currentSampleMeta.image;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* ========================================================
            LEFT COLUMN: PATIENT ASSESSMENT INPUT PANEL
            ======================================================== */}
        <div
          className="lg:col-span-6 rounded-2xl overflow-hidden transition-shadow"
          style={{
            backgroundColor: 'var(--surface-elevated)',
            border: '1px solid var(--slate-200)',
            boxShadow: 'var(--shadow-sm)',
          }}
        >
          {/* Card Header */}
          <div
            className="px-6 py-4 flex items-center justify-between"
            style={{
              borderBottom: '1px solid var(--slate-100)',
              backgroundColor: 'var(--slate-50)',
            }}
          >
            <div className="flex items-center gap-3">
              <div
                className="w-9 h-9 rounded-lg flex items-center justify-center text-white"
                style={{
                  background: 'linear-gradient(135deg, var(--clinical-blue-700) 0%, var(--clinical-blue-900) 100%)',
                  boxShadow: 'var(--shadow-sm)',
                }}
              >
                <Microscope className="w-5 h-5" />
              </div>
              <div>
                <h2
                  className="font-bold tracking-tight"
                  style={{
                    fontSize: 'var(--text-base)',
                    color: 'var(--clinical-blue-900)',
                  }}
                >
                  Patient Assessment
                </h2>
                <p
                  style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--slate-600)',
                  }}
                >
                  Biopsy histopathology imaging & Nottingham/AJCC clinical staging
                </p>
              </div>
            </div>

            <span
              className="px-2.5 py-1 rounded-md text-xs font-medium"
              style={{
                backgroundColor: 'var(--clinical-blue-100)',
                color: 'var(--clinical-blue-700)',
              }}
            >
              Model M4 Staging
            </span>
          </div>

          <div className="p-6 space-y-6">

            {/* --- Histopathology Input Section --- */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <label
                  className="flex items-center gap-2 font-semibold"
                  style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--slate-900)',
                  }}
                >
                  <Activity
                    className="w-4 h-4"
                    style={{ color: 'var(--clinical-blue-700)' }}
                  />
                  Histopathology Specimen
                </label>

                {/* Mode Segmented Toggle */}
                <div
                  className="inline-flex p-1 rounded-lg"
                  style={{
                    backgroundColor: 'var(--slate-100)',
                    border: '1px solid var(--slate-200)',
                  }}
                >
                  <button
                    type="button"
                    onClick={() => setImageMode('sample')}
                    className="px-2.5 py-1 text-xs font-medium rounded-md transition-all"
                    style={{
                      backgroundColor:
                        imageMode === 'sample' ? 'white' : 'transparent',
                      color:
                        imageMode === 'sample'
                          ? 'var(--clinical-blue-900)'
                          : 'var(--slate-600)',
                      boxShadow:
                        imageMode === 'sample' ? 'var(--shadow-sm)' : 'none',
                    }}
                  >
                    Research Case
                  </button>
                  <button
                    type="button"
                    onClick={() => setImageMode('upload')}
                    className="px-2.5 py-1 text-xs font-medium rounded-md transition-all"
                    style={{
                      backgroundColor:
                        imageMode === 'upload' ? 'white' : 'transparent',
                      color:
                        imageMode === 'upload'
                          ? 'var(--clinical-blue-900)'
                          : 'var(--slate-600)',
                      boxShadow:
                        imageMode === 'upload' ? 'var(--shadow-sm)' : 'none',
                    }}
                  >
                    Upload Custom
                  </button>
                </div>
              </div>

              {/* Research Case Mode */}
              {imageMode === 'sample' ? (
                <div className="space-y-3">
                  <div
                    className="p-3.5 rounded-xl transition-all"
                    style={{
                      backgroundColor: 'var(--surface-base)',
                      border: '1px solid var(--slate-200)',
                    }}
                  >
                    <Select
                      label="Select Curated TCGA-BRCA Patient"
                      value={selectedSampleId || ''}
                      onChange={(e) => handleSampleSelect(e.target.value)}
                      options={samples.map((s) => ({
                        value: s.patient_id,
                        label: `${s.patient_id} — ${SAMPLE_METADATA[s.patient_id]?.label || 'Cohort Case'} (${s.clinical?.ajcc_stage || 'Staged'})`,
                      }))}
                      fullWidth
                    />

                    {/* Selected Patient Details Card */}
                    <div
                      className="mt-3 p-3 rounded-lg flex items-center gap-3.5"
                      style={{
                        backgroundColor: 'white',
                        border: '1px solid var(--clinical-blue-100)',
                      }}
                    >
                      <img
                        src={currentSampleMeta.image}
                        alt="Biopsy Tissue"
                        className="w-14 h-14 rounded-lg object-cover border border-slate-200 shrink-0"
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2 mb-0.5">
                          <span
                            className="font-bold tracking-tight text-xs"
                            style={{ color: 'var(--clinical-blue-900)' }}
                          >
                            {selectedSampleId}
                          </span>
                          <span
                            className="px-2 py-0.5 rounded text-[11px] font-semibold"
                            style={{
                              backgroundColor: 'var(--diagnostic-teal-100)',
                              color: 'var(--diagnostic-teal-700)',
                            }}
                          >
                            {currentSampleMeta.tag}
                          </span>
                        </div>
                        <p
                          className="text-xs text-slate-600 truncate"
                          style={{ fontSize: 'var(--text-xs)' }}
                        >
                          {currentSampleMeta.description}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                /* Custom Upload Mode */
                <ImageUpload
                  onImageUpload={handleImageUpload}
                  onImageRemove={handleImageRemove}
                  uploadedImage={uploadedImageBase64}
                  uploadedFileName={uploadedFileName}
                />
              )}
            </div>

            {/* --- Clinical Parameters Section --- */}
            <div
              className="pt-4 space-y-4"
              style={{ borderTop: '1px solid var(--slate-100)' }}
            >
              <div className="flex items-center justify-between">
                <label
                  className="flex items-center gap-2 font-semibold"
                  style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--slate-900)',
                  }}
                >
                  <FileText
                    className="w-4 h-4"
                    style={{ color: 'var(--clinical-blue-700)' }}
                  />
                  Clinical Staging & Biomarkers
                </label>
                <span
                  style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--slate-500)',
                  }}
                >
                  AJCC 8th Edition
                </span>
              </div>

              {/* Form Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                <Input
                  label="Age at Diagnosis"
                  type="number"
                  value={formData.age}
                  onChange={(e) =>
                    handleInputChange(
                      'age',
                      e.target.value === '' ? '' : Number(e.target.value)
                    )
                  }
                  min="20"
                  max="100"
                  fullWidth
                />

                <Select
                  label="AJCC Stage"
                  value={formData.ajcc_stage}
                  onChange={(e) => handleInputChange('ajcc_stage', e.target.value)}
                  options={STAGE_OPTIONS}
                  fullWidth
                />

                <Select
                  label="Tumor (T)"
                  value={formData.pathologic_t}
                  onChange={(e) => handleInputChange('pathologic_t', e.target.value)}
                  options={T_OPTIONS}
                  fullWidth
                />

                <Select
                  label="Nodes (N)"
                  value={formData.pathologic_n}
                  onChange={(e) => handleInputChange('pathologic_n', e.target.value)}
                  options={N_OPTIONS}
                  fullWidth
                />

                <Select
                  label="Metastasis (M)"
                  value={formData.pathologic_m}
                  onChange={(e) => handleInputChange('pathologic_m', e.target.value)}
                  options={M_OPTIONS}
                  fullWidth
                />

                <Select
                  label="Tumor Grade"
                  value={formData.tumor_grade}
                  onChange={(e) => handleInputChange('tumor_grade', e.target.value)}
                  options={GRADE_OPTIONS}
                  fullWidth
                />

                <Select
                  label="ER Status"
                  value={formData.er_status}
                  onChange={(e) => handleInputChange('er_status', e.target.value)}
                  options={STATUS_OPTIONS}
                  fullWidth
                />

                <Select
                  label="PR Status"
                  value={formData.pr_status}
                  onChange={(e) => handleInputChange('pr_status', e.target.value)}
                  options={STATUS_OPTIONS}
                  fullWidth
                />

                <Select
                  label="HER2 Status"
                  value={formData.her2_status}
                  onChange={(e) => handleInputChange('her2_status', e.target.value)}
                  options={STATUS_OPTIONS}
                  fullWidth
                />
              </div>

              <p
                className="leading-relaxed"
                style={{
                  fontSize: 'var(--text-xs)',
                  color: 'var(--slate-500)',
                }}
              >
                Receptors and grading calibrated using standard Nottingham Histologic Scoring.
              </p>
            </div>

            {/* --- Primary Action Buttons --- */}
            <div className="flex flex-col sm:flex-row gap-3 pt-2">
              <Button
                variant="primary"
                size="large"
                fullWidth
                onClick={onPredict}
                disabled={isLoading}
                loading={isLoading}
                icon={!isLoading && <Play className="w-4 h-4 fill-current" />}
              >
                {isLoading ? 'Computing Multimodal Staging...' : 'Generate Multimodal Prognosis'}
              </Button>

              {predictionResult && (
                <Button
                  variant="secondary"
                  size="large"
                  onClick={onReset}
                  icon={<RotateCcw className="w-4 h-4" />}
                >
                  Clear
                </Button>
              )}
            </div>

          </div>
        </div>

        {/* ========================================================
            RIGHT COLUMN: SURVIVAL ANALYSIS RESULTS PANEL
            ======================================================== */}
        <div
          className="lg:col-span-6 rounded-2xl overflow-hidden flex flex-col min-h-[620px] transition-shadow"
          style={{
            backgroundColor: 'var(--surface-elevated)',
            border: '1px solid var(--slate-200)',
            boxShadow: 'var(--shadow-sm)',
          }}
        >
          {/* Panel Header */}
          <div
            className="px-6 py-4 flex items-center justify-between"
            style={{
              borderBottom: '1px solid var(--slate-100)',
              backgroundColor: 'var(--slate-50)',
            }}
          >
            <div className="flex items-center gap-3">
              <div
                className="w-9 h-9 rounded-lg flex items-center justify-center text-white"
                style={{
                  background: 'linear-gradient(135deg, var(--diagnostic-teal-700) 0%, var(--clinical-blue-700) 100%)',
                  boxShadow: 'var(--shadow-sm)',
                }}
              >
                <Activity className="w-5 h-5" />
              </div>
              <div>
                <h2
                  className="font-bold tracking-tight"
                  style={{
                    fontSize: 'var(--text-base)',
                    color: 'var(--clinical-blue-900)',
                  }}
                >
                  Survival Analysis
                </h2>
                <p
                  style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--slate-600)',
                  }}
                >
                  Multimodal Deep Learning Staging (Model M4)
                </p>
              </div>
            </div>

            {predictionResult && (
              <span
                className="px-2.5 py-1 rounded-md text-xs font-semibold"
                style={{
                  backgroundColor:
                    predictionResult.risk_tier === 'LOW'
                      ? 'rgba(46, 125, 50, 0.1)'
                      : predictionResult.risk_tier === 'HIGH'
                      ? 'rgba(198, 40, 40, 0.1)'
                      : 'rgba(245, 124, 0, 0.1)',
                  color:
                    predictionResult.risk_tier === 'LOW'
                      ? 'var(--risk-low)'
                      : predictionResult.risk_tier === 'HIGH'
                      ? 'var(--risk-high)'
                      : 'var(--risk-medium)',
                }}
              >
                {predictionResult.risk_tier} RISK
              </span>
            )}
          </div>

          {/* Panel Content */}
          <div className="p-6 flex-1 flex flex-col">
            <AnimatePresence mode="wait">
              {!predictionResult ? (
                /* Empty / Awaiting Prediction State */
                <motion.div
                  key="empty-state"
                  initial={{ opacity: 0, scale: 0.98 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.98 }}
                  className="flex-1 flex flex-col items-center justify-center text-center p-8 border-2 border-dashed rounded-xl"
                  style={{
                    borderColor: 'var(--slate-200)',
                    backgroundColor: 'var(--surface-base)',
                  }}
                >
                  <div
                    className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
                    style={{
                      backgroundColor: 'var(--clinical-blue-100)',
                      color: 'var(--clinical-blue-700)',
                    }}
                  >
                    <Microscope className="w-8 h-8" />
                  </div>
                  <h3
                    className="font-bold mb-1"
                    style={{
                      fontSize: 'var(--text-base)',
                      color: 'var(--slate-900)',
                    }}
                  >
                    Ready for Prognosis Assessment
                  </h3>
                  <p
                    className="max-w-md mt-1 leading-relaxed"
                    style={{
                      fontSize: 'var(--text-xs)',
                      color: 'var(--slate-600)',
                    }}
                  >
                    Select an example patient or upload a biopsy image and clinical staging parameters, then press <strong>Generate Multimodal Prognosis</strong> to view 5-year survival probability and risk curve.
                  </p>

                  <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
                    <Button
                      variant="outline"
                      size="small"
                      onClick={onPredict}
                      icon={<Play className="w-3.5 h-3.5 fill-current" />}
                    >
                      Quick Run with Example Case
                    </Button>
                  </div>
                </motion.div>
              ) : (
                /* Full Prediction Result Display */
                <motion.div
                  key="results-state"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -12 }}
                  className="space-y-6"
                >
                  {/* Evaluated Specimen & Multimodal Summary */}
                  <div
                    className="p-3.5 rounded-xl flex items-center justify-between gap-4"
                    style={{
                      backgroundColor: 'var(--slate-50)',
                      border: '1px solid var(--slate-200)',
                    }}
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <img
                        src={activeThumbnail}
                        alt="Evaluated Specimen"
                        className="w-12 h-12 rounded-lg object-cover border border-slate-200 shrink-0"
                      />
                      <div className="min-w-0">
                        <span
                          className="font-bold truncate block"
                          style={{
                            fontSize: 'var(--text-xs)',
                            color: 'var(--slate-900)',
                          }}
                        >
                          {imageMode === 'upload' && uploadedFileName
                            ? uploadedFileName
                            : `Specimen ${selectedSampleId || 'Sample'}`}
                        </span>
                        <p
                          className="truncate"
                          style={{
                            fontSize: 'var(--text-xs)',
                            color: 'var(--slate-500)',
                          }}
                        >
                          {predictionResult.modality_balance
                            ? `${predictionResult.modality_balance.clinical}% Clinical · ${predictionResult.modality_balance.histology}% Histology`
                            : 'Multimodal Attention Ensembled'}
                        </p>
                      </div>
                    </div>

                    <div className="text-right shrink-0">
                      <span
                        className="block font-black tabular-nums"
                        style={{
                          fontSize: 'var(--text-sm)',
                          color: 'var(--clinical-blue-700)',
                        }}
                      >
                        {predictionResult.percentile}th %ile
                      </span>
                      <span
                        className="text-[10px] uppercase font-semibold text-slate-500"
                      >
                        Cohort Hazard
                      </span>
                    </div>
                  </div>

                  {/* Core Risk Assessment Component */}
                  <RiskAssessment
                    riskTier={predictionResult.risk_tier}
                    survival5yr={predictionResult.survival_5yr}
                    survival1yr={predictionResult.survival_1yr}
                    survival3yr={predictionResult.survival_3yr}
                    survival10yr={predictionResult.survival_10yr}
                    summary={predictionResult.survival_summary}
                  />

                  {/* Kaplan-Meier Survival Curve Component */}
                  <SurvivalCurve
                    survivalData={predictionResult.survival_curve}
                    patientSurvival={{
                      time: 5,
                      survival: predictionResult.survival_5yr,
                    }}
                  />

                  {/* CTA to Explainability View */}
                  {onNavigateToInsights && (
                    <div
                      className="p-4 rounded-xl flex items-center justify-between gap-4"
                      style={{
                        backgroundColor: 'var(--clinical-blue-100)',
                        border: '1px solid var(--clinical-blue-500)',
                        borderOpacity: 0.2,
                      }}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
                          style={{
                            backgroundColor: 'white',
                            color: 'var(--clinical-blue-700)',
                          }}
                        >
                          <Microscope className="w-4 h-4" />
                        </div>
                        <div>
                          <h4
                            className="font-semibold text-xs"
                            style={{ color: 'var(--clinical-blue-900)' }}
                          >
                            Explore Attention Heatmaps
                          </h4>
                          <p
                            className="text-[11px]"
                            style={{ color: 'var(--clinical-blue-700)' }}
                          >
                            Inspect which tissue patches the deep learning model focused on.
                          </p>
                        </div>
                      </div>

                      <Button
                        variant="primary"
                        size="small"
                        onClick={onNavigateToInsights}
                        icon={<ArrowRight className="w-3.5 h-3.5" />}
                        iconPosition="right"
                      >
                        View Insights
                      </Button>
                    </div>
                  )}

                </motion.div>
              )}
            </AnimatePresence>

            {/* Clinical Disclaimer */}
            <div className="pt-4 mt-auto border-t border-slate-100 text-center">
              <p className="text-[11px] text-slate-400 italic">
                {predictionResult?.disclaimer ||
                  'Research prototype — not a clinical diagnosis. Always correlate with certified histopathologic review.'}
              </p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
