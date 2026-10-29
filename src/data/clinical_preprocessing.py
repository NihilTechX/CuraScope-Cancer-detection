import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import joblib

class ClinicalPreprocessor:
    def __init__(self):
        # Define feature groups
        self.num_features = ['age_at_index.demographic', 'year_of_diagnosis.diagnoses']
        
        self.cat_ordinal_features = [
            'ajcc_pathologic_stage.diagnoses',
            'ajcc_pathologic_t.diagnoses',
            'ajcc_pathologic_n.diagnoses',
            'ajcc_pathologic_m.diagnoses',
            'gender.demographic'
        ]
        
        self.cat_onehot_features = [
            'race.demographic',
            'primary_diagnosis.diagnoses'
        ]
        
        self.all_features = self.num_features + self.cat_ordinal_features + self.cat_onehot_features
        
        # Numeric pipeline: Impute median -> Scale
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Ordinal pipeline: Impute most frequent -> Ordinal Encode (handle unknown)
        # Using handle_unknown='use_encoded_value' is robust for unseen categories in test
        ord_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        # OneHot pipeline: Impute constant 'unknown' -> OneHot Encode
        ohe_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine into ColumnTransformer
        self.preprocessor = ColumnTransformer([
            ('num', num_pipeline, self.num_features),
            ('ord', ord_pipeline, self.cat_ordinal_features),
            ('ohe', ohe_pipeline, self.cat_onehot_features)
        ], remainder='drop')
        
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """Fits the preprocessing pipeline on the training dataframe."""
        self.preprocessor.fit(df[self.all_features])
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms a dataframe using the fitted pipeline."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")
        
        # Ensure missing columns are filled with NaN if they don't exist.
        # Work on a copy so the caller's DataFrame is not mutated.
        missing = [col for col in self.all_features if col not in df.columns]
        if missing:
            df = df.copy()
            for col in missing:
                df[col] = np.nan
                
        return self.preprocessor.transform(df[self.all_features])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def get_feature_dim(self):
        """Returns the output dimension of the transformed features."""
        if not self.is_fitted:
            raise RuntimeError("Cannot get feature dim before fitting.")
        # Transform a dummy row to get the output shape
        dummy = pd.DataFrame({col: [np.nan] for col in self.all_features})
        return self.preprocessor.transform(dummy).shape[1]

    def save(self, filepath: str):
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: str):
        return joblib.load(filepath)
