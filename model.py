"""
Machine Learning Engine for Banking Credit Default Risk & Cross-Sell Engine
Student ID: IBMQ2DST1296
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from config import RISK_THRESHOLDS
from preprocessing import CreditDataPreprocessor

class CreditRiskModelEngine:
    """Central ML Module for model training, comparative evaluation, and risk classification."""
    def __init__(self):
        self.preprocessor = CreditDataPreprocessor()
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        }
        self.evaluations = {}
        self.best_model_name = None
        self.best_model = None
        self.is_trained = False

    def train_and_evaluate_all(self, df: pd.DataFrame):
        """Trains and evaluates all candidate models on actual dataset."""
        X_train, X_test, X_train_proc, X_test_proc, y_train, y_test = self.preprocessor.prepare_train_test_split(df)
        
        best_auc = -1.0
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train_proc, y_train)
            
            # Predict labels & probabilities
            y_pred = model.predict(X_test_proc)
            y_proba = model.predict_proba(X_test_proc)[:, 1]
            
            # Calculate actual evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba)
            cm = confusion_matrix(y_test, y_pred).tolist() # [[TN, FP], [FN, TP]]
            
            self.evaluations[name] = {
                'Accuracy': float(acc),
                'Precision': float(prec),
                'Recall': float(rec),
                'F1 Score': float(f1),
                'ROC-AUC': float(auc),
                'Confusion Matrix': cm,
                'model_obj': model,
                'y_test': y_test,
                'y_proba': y_proba
            }
            
            if auc > best_auc:
                best_auc = auc
                self.best_model_name = name
                self.best_model = model
                
        self.is_trained = True
        return self.evaluations

    def classify_risk_category(self, proba: float) -> tuple[str, str, str]:
        """Classifies default probability into LOW RISK, MEDIUM RISK, or HIGH RISK.
        Returns (risk_category, color_hex, description)
        """
        if proba <= RISK_THRESHOLDS["LOW_RISK_MAX"]:
            return "LOW RISK", "#059669", "Customer displays high creditworthiness and low risk of default."
        elif proba <= RISK_THRESHOLDS["MEDIUM_RISK_MAX"]:
            return "MEDIUM RISK", "#D97706", "Customer displays moderate credit risk requiring structured terms."
        else:
            return "HIGH RISK", "#DC2626", "Customer displays elevated default risk. Restrict unsecured credit products."

    def predict_credit_risk(self, customer_dict: dict) -> dict:
        """Predicts loan default risk probability and category for a single customer."""
        if not self.is_trained:
            raise RuntimeError("ML engine must be trained before predicting risk.")
            
        x_single = self.preprocessor.transform_single_customer(customer_dict)
        proba = float(self.best_model.predict_proba(x_single)[0, 1])
        risk_cat, color, desc = self.classify_risk_category(proba)
        
        # Risk driver identification (Model Influence)
        risk_factors = []
        income = customer_dict.get('person_income', 0)
        loan_amnt = customer_dict.get('loan_amnt', 0)
        pct_inc = loan_amnt / max(income, 1)
        int_rate = customer_dict.get('loan_int_rate', 0)
        prev_default = customer_dict.get('cb_person_default_on_file', 'N')
        grade = customer_dict.get('loan_grade', 'A')
        
        if pct_inc > 0.30:
            risk_factors.append(f"High loan-to-income ratio ({pct_inc*100:.1f}%) [High Influence]")
        if int_rate > 14.0:
            risk_factors.append(f"Elevated loan interest rate ({int_rate:.1f}%) [High Influence]")
        if prev_default == 'Y':
            risk_factors.append("Prior credit default record on file [Strong Influence]")
        if grade in ['D', 'E', 'F', 'G']:
            risk_factors.append(f"Subprime loan risk grade ({grade}) [Moderate Influence]")
        if income < 35000:
            risk_factors.append(f"Lower annual income threshold (${income:,}) [Moderate Influence]")
            
        if not risk_factors:
            risk_factors.append("Stable income and strong credit history profile [Favorable Influence]")
            
        return {
            'default_probability': proba,
            'default_probability_pct': round(proba * 100, 2),
            'non_default_probability_pct': round((1.0 - proba) * 100, 2),
            'risk_category': risk_cat,
            'risk_color': color,
            'risk_description': desc,
            'risk_factors': risk_factors,
            'model_used': self.best_model_name
        }

    def get_feature_importances_df(self) -> pd.DataFrame:
        """Returns feature importance / coefficient magnitudes for model explainability."""
        if not self.is_trained or self.best_model is None:
            return pd.DataFrame()
            
        feat_names = self.preprocessor.feature_names_
        
        if hasattr(self.best_model, 'feature_importances_'):
            imps = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            imps = np.abs(self.best_model.coef_[0])
        else:
            imps = np.ones(len(feat_names)) / len(feat_names)
            
        df_imp = pd.DataFrame({
            'Feature': feat_names,
            'Importance': imps
        }).sort_values(by='Importance', ascending=False)
        
        return df_imp
