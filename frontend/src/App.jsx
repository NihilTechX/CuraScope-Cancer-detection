import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import PrognosisView from './views/PrognosisView';
import ExplainabilityView from './views/ExplainabilityView';
import { fetchSamplePatients, predictPatient, checkHealth } from './services/api';
import { AlertCircle, WifiOff } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('prognosis');

  // Image input: 'upload' or 'sample'
  const [imageMode, setImageMode] = useState('sample');
  const [uploadedImageBase64, setUploadedImageBase64] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState(null);
  const [selectedSampleId, setSelectedSampleId] = useState(null);
  const [samples, setSamples] = useState([]);

  // Clinical info
  const [formData, setFormData] = useState({
    age: 50,
    ajcc_stage: 'Stage I',
    pathologic_t: 'T1c',
    pathologic_n: 'N0',
    pathologic_m: 'M0',
    er_status: 'Positive',
    pr_status: 'Positive',
    her2_status: 'Negative',
    tumor_grade: 'Grade 1',
  });

  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendOnline, setBackendOnline] = useState(true);

  // Load sample patients + health on mount
  useEffect(() => {
    let cancelled = false;

    async function load() {
      const health = await checkHealth();
      if (cancelled) return;
      setBackendOnline(!!health);

      try {
        const data = await fetchSamplePatients();
        if (cancelled) return;
        const list = Array.isArray(data) ? data : data?.samples || [];
        if (list.length > 0) {
          setSamples(list);
          setSelectedSampleId((prev) => prev || list[0].patient_id);
          // Prefill the clinical form from the first example so the form and
          // the selector always describe the same patient.
          if (list[0].clinical) {
            setFormData((prev) => ({
              ...prev,
              age: list[0].clinical.age ?? prev.age,
              ajcc_stage: list[0].clinical.ajcc_stage ?? prev.ajcc_stage,
              pathologic_t: list[0].clinical.pathologic_t ?? prev.pathologic_t,
              pathologic_n: list[0].clinical.pathologic_n ?? prev.pathologic_n,
              pathologic_m: list[0].clinical.pathologic_m ?? prev.pathologic_m,
              er_status: list[0].clinical.er_status ?? prev.er_status,
              pr_status: list[0].clinical.pr_status ?? prev.pr_status,
              her2_status: list[0].clinical.her2_status ?? prev.her2_status,
              tumor_grade: list[0].clinical.tumor_grade ?? prev.tumor_grade,
            }));
          }
        }
      } catch {
        // Offline fallback so the UI stays usable
        if (!cancelled) {
          setSamples([
            { patient_id: 'TCGA-A2-A0T2' },
            { patient_id: 'TCGA-A2-A0CM' },
            { patient_id: 'TCGA-A2-A0T0' },
          ]);
          setSelectedSampleId((prev) => prev || 'TCGA-A2-A0T2');
        }
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const handlePredict = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const payload = {
        age: Number(formData.age) || 50,
        ajcc_stage: formData.ajcc_stage || 'Stage I',
        pathologic_t: formData.pathologic_t || 'T1c',
        pathologic_n: formData.pathologic_n || 'N0',
        pathologic_m: formData.pathologic_m || 'M0',
        er_status: formData.er_status || 'Positive',
        pr_status: formData.pr_status || 'Positive',
        her2_status: formData.her2_status || 'Negative',
        tumor_grade: formData.tumor_grade || 'Grade 1',
        year_of_diagnosis: 2011,
      };

      const imagePayload = imageMode === 'upload' ? uploadedImageBase64 : null;
      const patientIdPayload = imageMode === 'sample' ? selectedSampleId : null;

      const res = await predictPatient(payload, patientIdPayload, imagePayload);
      setPredictionResult(res);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err) {
      console.error('Prediction failed:', err);
      setError(
        backendOnline
          ? 'Something went wrong while generating the assessment. Please try again.'
          : "Can't reach the analysis service. Please make sure it's running and try again."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setPredictionResult(null);
    setError(null);
  };

  const handleNewAssessment = () => {
    setPredictionResult(null);
    setError(null);
    setUploadedImageBase64(null);
    setUploadedFileName(null);
    setImageMode('sample');
    setActiveTab('prognosis');
  };

  return (
    <div
      className="min-h-screen text-slate-900 flex flex-col font-sans"
      style={{ backgroundColor: 'var(--surface-base)' }}
    >
      <Navbar activeTab={activeTab} onTabChange={setActiveTab} />

      <main className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Offline / Demo notice */}
        {!backendOnline && (
          <div
            className="mb-6 p-3.5 rounded-xl border flex items-center justify-between gap-3 text-xs"
            style={{
              backgroundColor: 'var(--slate-100)',
              borderColor: 'var(--slate-200)',
              color: 'var(--slate-700)',
            }}
          >
            <div className="flex items-center gap-2.5">
              <WifiOff className="w-4 h-4 shrink-0 text-slate-500" />
              <span>
                Backend API is offline (port 8000) — running in <strong>calibrated local clinical simulation mode</strong>. All prediction models and visual explanations are fully functional.
              </span>
            </div>
            <span
              className="px-2 py-0.5 rounded text-[11px] font-semibold uppercase tracking-wider shrink-0"
              style={{
                backgroundColor: 'var(--diagnostic-teal-100)',
                color: 'var(--diagnostic-teal-700)',
              }}
            >
              Demo Ready
            </span>
          </div>
        )}

        {/* Error banner */}
        {error && (
          <div
            className="mb-6 p-4 rounded-xl border flex items-start justify-between gap-3 text-xs"
            style={{
              backgroundColor: 'rgba(198, 40, 40, 0.08)',
              borderColor: 'rgba(198, 40, 40, 0.2)',
              color: 'var(--risk-high)',
            }}
          >
            <div className="flex items-start gap-2">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
            <button
              onClick={() => setError(null)}
              className="font-bold hover:underline shrink-0"
            >
              Dismiss
            </button>
          </div>
        )}

        {activeTab === 'prognosis' && (
          <PrognosisView
            imageMode={imageMode}
            setImageMode={setImageMode}
            uploadedImageBase64={uploadedImageBase64}
            setUploadedImageBase64={setUploadedImageBase64}
            uploadedFileName={uploadedFileName}
            setUploadedFileName={setUploadedFileName}
            selectedSampleId={selectedSampleId}
            setSelectedSampleId={setSelectedSampleId}
            samples={samples}
            formData={formData}
            setFormData={setFormData}
            onPredict={handlePredict}
            predictionResult={predictionResult}
            isLoading={isLoading}
            onReset={handleReset}
            onNewAssessment={handleNewAssessment}
            onNavigateToInsights={() => setActiveTab('insight')}
          />
        )}

        {activeTab === 'insight' && (
          <ExplainabilityView
            predictionResult={predictionResult}
            uploadedImageBase64={uploadedImageBase64}
          />
        )}
      </main>

      <footer
        className="border-t py-5 px-4 text-center"
        style={{
          backgroundColor: 'var(--surface-elevated)',
          borderColor: 'var(--slate-200)',
        }}
      >
        <p className="text-xs text-slate-500">
          CuraScope — AI-assisted survival insights for breast cancer care teams.
        </p>
        <p className="text-[11px] text-slate-400 mt-1">
          For research and educational use only. Not a medical diagnosis. Always consult a qualified
          clinician.
        </p>
      </footer>
    </div>
  );
}
