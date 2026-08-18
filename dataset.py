"""
Dataset Module for Banking Credit Default Risk & Cross-Sell Engine
Student ID: IBMQ2DST1296
"""

import os
import pandas as pd
import numpy as np
from config import DATASET_PATH, DATA_DIR

def generate_benchmark_credit_dataset(num_records: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generates a realistic benchmark Credit Risk Dataset matching Kaggle/IBM Credit Risk standard schema."""
    np.random.seed(seed)
    
    ages = np.random.randint(20, 68, size=num_records)
    incomes = np.random.lognormal(mean=10.8, sigma=0.6, size=num_records).astype(int)
    incomes = np.clip(incomes, 15000, 250000)
    
    home_ownership_options = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
    home_ownership_p = [0.50, 0.38, 0.10, 0.02]
    home_ownership = np.random.choice(home_ownership_options, size=num_records, p=home_ownership_p)
    
    emp_lengths = np.random.exponential(scale=5.0, size=num_records).astype(float)
    emp_lengths = np.round(np.clip(emp_lengths, 0, 40), 1)
    
    # Intentionally insert a few realistic missing values for testing preprocessing pipeline
    missing_emp_idx = np.random.choice(num_records, size=int(num_records * 0.03), replace=False)
    emp_lengths[missing_emp_idx] = np.nan
    
    loan_intents = np.random.choice(
        ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
        size=num_records
    )
    
    loan_grades = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], size=num_records, p=[0.30, 0.30, 0.20, 0.10, 0.05, 0.03, 0.02])
    
    loan_amounts = np.random.randint(1000, 35000, size=num_records)
    
    # Interest rates correlated with loan grade
    grade_int_map = {'A': 7.5, 'B': 10.5, 'C': 13.5, 'D': 16.0, 'E': 18.5, 'F': 21.0, 'G': 23.5}
    base_int_rates = np.array([grade_int_map[g] for g in loan_grades])
    int_rates = base_int_rates + np.random.normal(0, 1.2, size=num_records)
    int_rates = np.round(np.clip(int_rates, 5.0, 25.0), 2)
    
    # Missing interest rates
    missing_int_idx = np.random.choice(num_records, size=int(num_records * 0.04), replace=False)
    int_rates[missing_int_idx] = np.nan
    
    loan_percent_income = np.round(loan_amounts / incomes, 3)
    
    cb_defaults = np.random.choice(['N', 'Y'], size=num_records, p=[0.82, 0.18])
    
    cred_hist_lens = (ages - 18 - np.random.randint(0, 5, size=num_records)).astype(float)
    cred_hist_lens = np.clip(cred_hist_lens, 1, 30)
    
    # Logit calculation for realistic default probability target
    # Higher risk drivers: low income, high loan percent income, high interest rate, past default, poor grade
    logit = (
        -1.8
        - 0.000015 * incomes
        + 3.5 * loan_percent_income
        + 0.12 * (np.nan_to_num(int_rates, nan=12.0) - 10)
        + 1.2 * (cb_defaults == 'Y').astype(int)
        + 0.8 * (home_ownership == 'RENT').astype(int)
        + 0.6 * np.isin(loan_grades, ['D', 'E', 'F', 'G']).astype(int)
        - 0.03 * np.nan_to_num(emp_lengths, nan=2.0)
    )
    
    prob_default = 1 / (1 + np.exp(-logit))
    loan_status = (np.random.rand(num_records) < prob_default).astype(int)
    
    df = pd.DataFrame({
        'person_age': ages,
        'person_income': incomes,
        'person_home_ownership': home_ownership,
        'person_emp_length': emp_lengths,
        'loan_intent': loan_intents,
        'loan_grade': loan_grades,
        'loan_amnt': loan_amounts,
        'loan_int_rate': int_rates,
        'loan_status': loan_status,
        'loan_percent_income': loan_percent_income,
        'cb_person_default_on_file': cb_defaults,
        'cb_person_cred_hist_length': cred_hist_lens
    })
    
    return df

def load_credit_dataset() -> pd.DataFrame:
    """Loads dataset from file or generates benchmark dataset if absent."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        
    if not os.path.exists(DATASET_PATH):
        df = generate_benchmark_credit_dataset()
        df.to_csv(DATASET_PATH, index=False)
    else:
        df = pd.read_csv(DATASET_PATH)
        
    return df
