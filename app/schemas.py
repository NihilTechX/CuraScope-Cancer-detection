from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ClinicalInput(BaseModel):
    age: Optional[float] = Field(58.0, description="Age at diagnosis (years)")
    age_at_index: Optional[float] = Field(None, description="Alternative age field name")
    year_of_diagnosis: Optional[float] = Field(2011.0, description="Year of initial diagnosis")
    
    # Staging parameters
    ajcc_stage: Optional[str] = Field("Stage IIA", description="AJCC Pathologic Stage")
    ajcc_pathologic_stage: Optional[str] = Field(None, description="Alternative stage field name")
    pathologic_t: Optional[str] = Field("T2", description="Tumor T Stage")
    ajcc_pathologic_t: Optional[str] = Field(None, description="Alternative T stage field name")
    pathologic_n: Optional[str] = Field("N0", description="Lymph Node N Stage")
    ajcc_pathologic_n: Optional[str] = Field(None, description="Alternative N stage field name")
    pathologic_m: Optional[str] = Field("M0", description="Distant Metastasis M Stage")
    ajcc_pathologic_m: Optional[str] = Field(None, description="Alternative M stage field name")
    
    # Biomarkers & Grade (Requested specifically for presentation)
    er_status: Optional[str] = Field("Positive", description="Estrogen Receptor (ER) Status: Positive / Negative / Unknown")
    pr_status: Optional[str] = Field("Positive", description="Progesterone Receptor (PR) Status: Positive / Negative / Unknown")
    her2_status: Optional[str] = Field("Negative", description="HER2 Receptor Status: Negative / Positive / Equivocal")
    tumor_grade: Optional[str] = Field("Grade 2", description="Histologic Tumor Grade: Grade 1 / Grade 2 / Grade 3")
    
    gender: Optional[str] = Field("female", description="Biological Sex")
    race: Optional[str] = Field("white", description="Reported Race")
    primary_diagnosis: Optional[str] = Field("Infiltrating duct carcinoma, NOS", description="Histologic Subtype")
    prior_malignancy: Optional[str] = Field("no", description="Prior malignancy status")

    def get_age(self) -> float:
        if self.age is not None:
            return float(self.age)
        if self.age_at_index is not None:
            return float(self.age_at_index)
        return 58.0

    def get_stage(self) -> str:
        if self.ajcc_stage:
            return self.ajcc_stage
        if self.ajcc_pathologic_stage:
            return self.ajcc_pathologic_stage
        return "Stage IIA"

    def get_t(self) -> str:
        if self.pathologic_t:
            return self.pathologic_t
        if self.ajcc_pathologic_t:
            return self.ajcc_pathologic_t
        return "T2"

    def get_n(self) -> str:
        if self.pathologic_n:
            return self.pathologic_n
        if self.ajcc_pathologic_n:
            return self.ajcc_pathologic_n
        return "N0"

    def get_m(self) -> str:
        if self.pathologic_m:
            return self.pathologic_m
        if self.ajcc_pathologic_m:
            return self.ajcc_pathologic_m
        return "M0"

class PredictionRequest(BaseModel):
    clinical: ClinicalInput
    patient_id: Optional[str] = Field(None, description="TCGA Patient ID to load pre-extracted patch sequence (e.g. 'TCGA-A2-A0T0')")
    image_base64: Optional[str] = Field(None, description="Base64-encoded raw H&E histopathology image / ROI photo")
    model_name: str = Field("m4", description="Model architecture to use: 'm4', 'm3', 'm2', 'm1'")

class SurvivalTimepoint(BaseModel):
    month: int
    survival: float

class PatchAttentionItem(BaseModel):
    patch_id: str
    weight: float
    is_top: bool = False
    description: str

class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    display_bar: str

class PredictionResponse(BaseModel):
    model_name: str
    risk_score: float
    hazard_score: float
    risk_tier: str  # "LOW", "MEDIUM", "HIGH"
    percentile: float
    survival_1yr: float
    survival_3yr: float
    survival_5yr: float
    survival_10yr: float
    survival_summary: str
    
    # Image & Explainability artifacts
    image_preview: Optional[str] = None
    heatmap_preview: Optional[str] = None
    patch_attentions: List[PatchAttentionItem] = []
    feature_importances: List[FeatureImportanceItem] = []
    modality_balance: Dict[str, float]
    
    survival_curve: List[SurvivalTimepoint] = []
    disclaimer: str = "Research prototype — not a clinical diagnosis."

class SamplePatient(BaseModel):
    patient_id: str
    risk_tier: str
    actual_os_time_months: float
    actual_event: int
    clinical: Dict[str, Any]
    image_url: Optional[str] = None
    num_patches: int = 120
