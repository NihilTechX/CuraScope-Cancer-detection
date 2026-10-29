import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.api import load_artifacts, predict_survival
from app.schemas import PredictionRequest, ClinicalInput
import base64

with open('dataset/sample_he_images/sample_he_stage1.jpg', 'rb') as f:
    b64 = 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode('utf-8')

req = PredictionRequest(
    clinical=ClinicalInput(age=48, ajcc_stage='Stage I', pathologic_t='T1c', pathologic_n='N0', pathologic_m='M0'),
    image_base64=b64,
    model_name='m4'
)
res = predict_survival(req)
print('TEST PREDICT ON UPLOADED H&E IMAGE:')
print('  risk_tier:', res.risk_tier)
print('  hazard_score:', res.hazard_score)
print('  survival_5yr:', res.survival_5yr)
print('  has image_preview:', bool(res.image_preview))
print('  has heatmap_preview:', bool(res.heatmap_preview))
print('  patch_attentions count:', len(res.patch_attentions))
print('  feature_importances count:', len(res.feature_importances))
