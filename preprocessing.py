"""
Data Preprocessing Pipeline for Banking Credit Default Risk & Cross-Sell Engine
Student ID: IBMQ2DST1296
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN

class CreditDataPreprocessor:
    """Central Data Preprocessing Pipeline preserving data integrity and avoiding data leakage."""
    def __init__(self):
        self.num_cols = NUMERICAL_FEATURES
        self.cat_cols = CATEGORICAL_FEATURES
        self.target_col = TARGET_COLUMN
        
        # Pipelines for Numerical and Categorical processing
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', num_pipeline, self.num_cols),
            ('cat', cat_pipeline, self.cat_cols)
        ])
        
        self.is_fitted = False
        self.feature_names_ = []

    def fit(self, df: pd.DataFrame):
        """Fits the preprocessing pipeline on training dataset."""
        X = df.drop(columns=[self.target_col], errors='ignore')
        self.preprocessor.fit(X)
        self.is_fitted = True
        
        # Extract feature names post encoding
        num_names = self.num_cols
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_names = cat_encoder.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names_ = num_names + cat_names
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms features using the fitted pipeline."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transformation.")
        X = df.drop(columns=[self.target_col], errors='ignore')
        return self.preprocessor.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fits and transforms dataset in one step."""
        self.fit(df)
        return self.transform(df)

    def prepare_train_test_split(self, df: pd.DataFrame, test_size=0.2, random_state=42):
        """Executes data leakage-free train/test split."""
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train_processed = self.fit_transform(X_train)
        X_test_processed = self.transform(X_test)
        
        return X_train, X_test, X_train_processed, X_test_processed, y_train, y_test

    def transform_single_customer(self, customer_dict: dict) -> np.ndarray:
        """Converts a single customer dictionary safely into processed numpy array."""
        if customer_dict is None:
            customer_dict = {}
            
        defaults = {
            'person_age': 30,
            'person_income': 50000,
            'person_home_ownership': 'RENT',
            'person_emp_length': 3.0,
            'loan_intent': 'PERSONAL',
            'loan_grade': 'A',
            'loan_amnt': 10000,
            'loan_int_rate': 10.0,
            'loan_percent_income': 0.2,
            'cb_person_default_on_file': 'N',
            'cb_person_cred_hist_length': 5.0
        }
        
        # Merge with defaults to prevent Missing Key exceptions
        full_dict = defaults.copy()
        full_dict.update(customer_dict)
        
        df_single = pd.DataFrame([full_dict])
        
        # Compute loan_percent_income if needed
        income = df_single['person_income'].iloc[0]
        loan_amnt = df_single['loan_amnt'].iloc[0]
        if pd.isna(income) or income <= 0:
            income = 50000
        if pd.isna(loan_amnt):
            loan_amnt = 10000
            
        df_single['loan_percent_income'] = round(loan_amnt / max(income, 1), 3)
            
        return self.transform(df_single)
