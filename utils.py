"""
Utility Functions for Banking Credit Default Risk & Cross-Sell Engine
Student ID: IBMQ2DST1296
"""

import pandas as pd

def compute_dataset_kpis(df: pd.DataFrame, model_evaluations: dict, best_model_name: str) -> dict:
    """Computes actual, non-fabricated KPIs from dataset and model evaluations."""
    total_records = len(df)
    total_features = df.shape[1] - 1 # excluding target loan_status
    default_count = int(df['loan_status'].sum())
    default_rate_pct = round((default_count / max(total_records, 1)) * 100, 2)
    avg_income = float(df['person_income'].mean()) if 'person_income' in df.columns else 0.0
    avg_loan = float(df['loan_amnt'].mean()) if 'loan_amnt' in df.columns else 0.0
    
    best_eval = model_evaluations.get(best_model_name, {})
    best_auc = round(best_eval.get('ROC-AUC', 0.0), 4)
    best_acc = round(best_eval.get('Accuracy', 0.0) * 100, 2)
    
    return {
        'total_records': total_records,
        'total_features': total_features,
        'default_rate_pct': default_rate_pct,
        'avg_income': avg_income,
        'avg_loan': avg_loan,
        'best_model_name': best_model_name,
        'best_auc': best_auc,
        'best_accuracy_pct': best_acc,
        'customers_analyzed': total_records
    }

def get_data_intelligence_summary(df: pd.DataFrame) -> dict:
    """Generates dataset profiling summary for Data Intelligence view."""
    total_rows = len(df)
    total_cols = df.shape[1]
    missing_dict = df.isna().sum().to_dict()
    total_missing = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return {
        'total_rows': total_rows,
        'total_cols': total_cols,
        'missing_dict': missing_dict,
        'total_missing': total_missing,
        'duplicate_count': duplicate_count,
        'numerical_features': num_cols,
        'categorical_features': cat_cols
    }

def format_currency(val: float) -> str:
    """Formats float to currency string."""
    return f"${val:,.0f}"

def format_percent(val: float) -> str:
    """Formats float to percentage string."""
    return f"{val:.1f}%"
