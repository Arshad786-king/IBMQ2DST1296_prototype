"""
Configuration File for Banking Credit Default Risk & Cross-Sell Engine
Student ID: IBMQ2DST1296
Case Study: Banking Credit Default Risk & Cross-Sell Engine
"""

import os

STUDENT_ID = "IBMQ2DST1296"
CASE_STUDY_TITLE = "Banking Credit Default Risk & Cross-Sell Engine"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "credit_risk_dataset.csv")

# Feature Column Specifications
NUMERICAL_FEATURES = [
    'person_age', 'person_income', 'person_emp_length',
    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_cred_hist_length'
]

CATEGORICAL_FEATURES = [
    'person_home_ownership', 'loan_intent', 'loan_grade',
    'cb_person_default_on_file'
]

TARGET_COLUMN = 'loan_status'

# Risk Category Thresholds (As per prompt requirements)
# 0%–30%: LOW RISK
# 30%–60%: MEDIUM RISK
# 60%–100%: HIGH RISK
RISK_THRESHOLDS = {
    "LOW_RISK_MAX": 0.30,
    "MEDIUM_RISK_MAX": 0.60,
}

# Banking Products Catalog for Cross-Sell Engine
BANKING_PRODUCTS = {
    "Credit Card": {
        "category": "Credit",
        "min_income": 30000,
        "max_risk": 0.60,
        "min_emp_length": 1,
        "description": "Premium rewards credit card with tailored credit limits.",
        "icon": "💳"
    },
    "Fixed Deposit": {
        "category": "Investment",
        "min_income": 20000,
        "max_risk": 1.00, # Safe for all risk levels
        "min_emp_length": 0,
        "description": "High-yield guaranteed fixed deposit investment plan.",
        "icon": "🏦"
    },
    "Personal Loan": {
        "category": "Credit",
        "min_income": 40000,
        "max_risk": 0.40,
        "min_emp_length": 2,
        "description": "Unsecured personal loan for financial flexibility.",
        "icon": "💰"
    },
    "Auto Loan": {
        "category": "Credit",
        "min_income": 35000,
        "max_risk": 0.50,
        "min_emp_length": 1,
        "description": "Competitive auto financing with low interest rates.",
        "icon": "🚗"
    },
    "Investment Plan": {
        "category": "Investment",
        "min_income": 50000,
        "max_risk": 0.35,
        "min_emp_length": 2,
        "description": "Diversified wealth accumulation and mutual fund portfolios.",
        "icon": "📈"
    },
    "Insurance": {
        "category": "Protection",
        "min_income": 15000,
        "max_risk": 1.00, # Recommended for high risk protection
        "min_emp_length": 0,
        "description": "Comprehensive health, life, and credit protection insurance.",
        "icon": "🛡️"
    },
    "Savings Product": {
        "category": "Savings",
        "min_income": 10000,
        "max_risk": 1.00, # Recommended for all customer profiles
        "min_emp_length": 0,
        "description": "High-interest savings account with digital management tools.",
        "icon": "🐖"
    }
}
