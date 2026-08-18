"""
Streamlit Web Application for Banking Credit Intelligence Platform
Case Study: Banking Credit Default Risk & Cross-Sell Engine
Student ID: IBMQ2DST1296
Visual Theme: Blue Horizon Banking Intelligence (Light Palette)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

from config import STUDENT_ID, CASE_STUDY_TITLE
from dataset import load_credit_dataset
from model import CreditRiskModelEngine
from recommendation import RiskAwareRecommendationEngine
from utils import compute_dataset_kpis, get_data_intelligence_summary, format_currency, format_percent

# Streamlit Page Configuration
st.set_page_config(
    page_title="Banking Intelligence | Credit Risk & Cross-Sell Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Blue Horizon Banking Intelligence" Light Theme
st.markdown("""
<style>
    /* Global Background & Typography */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Top Header Bar */
    .top-header-bar {
        background: #FFFFFF;
        border-bottom: 1px solid #E2E8F0;
        padding: 16px 24px;
        margin-bottom: 24px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .header-brand {
        font-size: 24px;
        font-weight: 800;
        color: #1E3A8A;
        letter-spacing: -0.5px;
        margin: 0;
    }
    
    .header-sub {
        font-size: 14px;
        color: #475569;
        margin-top: 2px;
    }

    .status-badge-active {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Cards & Containers */
    .bank-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02), 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    
    .bank-card-title {
        font-size: 18px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Metric Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        text-align: left;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        border-top: 4px solid #1D4ED8;
    }
    
    .kpi-label {
        font-size: 12px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        color: #0F172A;
        margin-top: 6px;
    }
    
    .kpi-context {
        font-size: 12px;
        color: #0D9488;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Risk Status Badges */
    .badge-risk-low {
        background: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 15px;
        font-weight: 700;
        display: inline-block;
    }
    
    .badge-risk-medium {
        background: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 15px;
        font-weight: 700;
        display: inline-block;
    }

    .badge-risk-high {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FECACA;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 15px;
        font-weight: 700;
        display: inline-block;
    }

    /* Decision Summary Box */
    .decision-summary-box {
        background: #F0F9FF;
        border: 1px solid #BAE6FD;
        border-radius: 8px;
        padding: 16px;
        margin-top: 16px;
        margin-bottom: 16px;
    }

    /* Process Flow Step Box */
    .process-flow-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 16px;
        margin-bottom: 24px;
    }
    
    .process-step {
        text-align: center;
        flex: 1;
    }
    
    .step-title {
        font-size: 12px;
        font-weight: 700;
        color: #1E3A8A;
        text-transform: uppercase;
    }
    
    .step-desc {
        font-size: 11px;
        color: #64748B;
        margin-top: 2px;
    }

    .step-arrow {
        color: #94A3B8;
        font-size: 18px;
        font-weight: bold;
    }

    /* Warning & Disclaimer Notices */
    .notice-warning {
        background: #FEF2F2;
        border-left: 4px solid #EF4444;
        color: #991B1B;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 14px;
        margin-top: 12px;
        margin-bottom: 16px;
    }

    .notice-disclaimer {
        background: #F0FDFA;
        border-left: 4px solid #0D9488;
        color: #115E59;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 13px;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_initialized_engine():
    """Cache dataset loading and model training pipeline."""
    df = load_credit_dataset()
    engine = CreditRiskModelEngine()
    evals = engine.train_and_evaluate_all(df)
    recommender = RiskAwareRecommendationEngine()
    return df, engine, evals, recommender

# Load cached pipeline
df_data, ml_engine, model_evals, recommender_engine = get_initialized_engine()
kpis = compute_dataset_kpis(df_data, model_evals, ml_engine.best_model_name)

# Sidebar Navigation Shell
st.sidebar.markdown("""
<div style="text-align: left; padding: 8px 0 16px 0;">
    <div style="font-size: 20px; font-weight: 800; color: #1E3A8A; letter-spacing: -0.5px;">BANKING</div>
    <div style="font-size: 14px; font-weight: 700; color: #0D9488; letter-spacing: 1px;">INTELLIGENCE</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

navigation_option = st.sidebar.radio(
    "NAVIGATION",
    [
        "🌐 Overview",
        "📊 Data Intelligence",
        "🛡️ Credit Risk",
        "🔍 Risk Drivers",
        "🎛️ What-If Simulator",
        "🛍️ Cross-Sell Intelligence",
        "👤 Customer 360°",
        "📈 Portfolio Risk",
        "🏆 Model Performance",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="font-size: 12px; color: #64748B;">
    <strong>ID:</strong> <code>{STUDENT_ID}</code><br>
    <strong>Status:</strong> <span style="color:#059669;">● System Ready</span><br><br>
    <em>Academic Prototype</em>
</div>
""", unsafe_allow_html=True)


# Top Header Render Function
def render_top_header():
    st.markdown(f"""
    <div class="top-header-bar">
        <div>
            <div class="header-brand">Banking Intelligence</div>
            <div class="header-sub">{CASE_STUDY_TITLE}</div>
        </div>
        <div style="text-align: right;">
            <span class="status-badge-active">● System Ready</span>
            <div style="font-size: 12px; color: #64748B; margin-top: 4px;">Student ID: <code>{STUDENT_ID}</code></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 1. OVERVIEW PAGE
# ==========================================
if navigation_option == "🌐 Overview":
    render_top_header()
    
    # Hero Banner Card
    st.markdown("""
    <div class="bank-card" style="background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 100%); color: #FFFFFF; border: none;">
        <div style="font-size: 14px; font-weight: 700; color: #93C5FD; text-transform: uppercase; letter-spacing: 1px;">BANKING INTELLIGENCE</div>
        <div style="font-size: 32px; font-weight: 800; margin-top: 6px;">Smarter credit decisions.<br>More relevant customer recommendations.</div>
        <div style="font-size: 15px; color: #DBEAFE; margin-top: 12px; max-width: 850px;">
            An AI-powered decision-support prototype that estimates credit default risk and identifies potentially suitable banking products using customer and credit-risk information.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Action Cards
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown("""
        <div class="bank-card">
            <div class="bank-card-title">🛡️ CREDIT RISK</div>
            <div style="font-size: 13px; color: #475569;">Predict probability of loan default using machine learning.</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="bank-card">
            <div class="bank-card-title">🛍️ CROSS-SELL</div>
            <div style="font-size: 13px; color: #475569;">Identify suitable banking products using customer attributes.</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="bank-card">
            <div class="bank-card-title">📈 PORTFOLIO RISK</div>
            <div style="font-size: 13px; color: #475569;">Analyze portfolio-wide risk analytics and distributions.</div>
        </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown("""
        <div class="bank-card">
            <div class="bank-card-title">🏆 PERFORMANCE</div>
            <div style="font-size: 13px; color: #475569;">Evaluate candidate ML models with empirical metrics.</div>
        </div>
        """, unsafe_allow_html=True)
            
    # KPI Grid
    st.markdown("### 📊 Portfolio Metrics & Intelligence Overview")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Portfolio Records</div>
            <div class="kpi-value">{kpis['total_records']:,}</div>
            <div class="kpi-context">Observed in dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Default Rate</div>
            <div class="kpi-value">{kpis['default_rate_pct']:.2f}%</div>
            <div class="kpi-context">Observed in dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Model Accuracy</div>
            <div class="kpi-value">{kpis['best_accuracy_pct']:.1f}%</div>
            <div class="kpi-context">Evaluated on test set</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Model ROC-AUC</div>
            <div class="kpi-value">{kpis['best_auc']:.3f}</div>
            <div class="kpi-context">Best model: {kpis['best_model_name']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Process Flow Panel
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="bank-card">
        <div class="bank-card-title">⚙️ Risk Intelligence Decision Support Architecture</div>
        <div class="process-flow-container">
            <div class="process-step">
                <div class="step-title">Customer Profile</div>
                <div class="step-desc">Financial & Credit Data</div>
            </div>
            <div class="step-arrow">➔</div>
            <div class="process-step">
                <div class="step-title">Credit Analysis</div>
                <div class="step-desc">Feature Processing</div>
            </div>
            <div class="step-arrow">➔</div>
            <div class="process-step">
                <div class="step-title">Default Prob</div>
                <div class="step-desc">ML Risk Estimation</div>
            </div>
            <div class="step-arrow">➔</div>
            <div class="process-step">
                <div class="step-title">Risk Category</div>
                <div class="step-desc">Low / Med / High</div>
            </div>
            <div class="step-arrow">➔</div>
            <div class="process-step">
                <div class="step-title">Business Decision</div>
                <div class="step-desc">Opportunity Layer</div>
            </div>
            <div class="step-arrow">➔</div>
            <div class="process-step">
                <div class="step-title">Recommendation</div>
                <div class="step-desc">Risk-Aware Cross-Sell</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 2. DATA INTELLIGENCE PAGE
# ==========================================
elif navigation_option == "📊 Data Intelligence":
    render_top_header()
    
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="color:#1E3A8A; font-weight:800; margin:0;">Data Intelligence</h2>
        <div style="color:#475569; font-size:14px; margin-top:2px;">Automated inspection of the Credit Risk Dataset structure, features, and missing value distribution.</div>
    </div>
    """, unsafe_allow_html=True)
    
    summary = get_data_intelligence_summary(df_data)
    
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Dataset Records</div><div class="kpi-value">{summary['total_rows']:,}</div></div>""", unsafe_allow_html=True)
    with d2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Total Features</div><div class="kpi-value">{summary['total_cols']-1}</div></div>""", unsafe_allow_html=True)
    with d3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Missing Values</div><div class="kpi-value">{summary['total_missing']}</div></div>""", unsafe_allow_html=True)
    with d4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Duplicate Rows</div><div class="kpi-value">{summary['duplicate_count']}</div></div>""", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("""<div class="bank-card">""", unsafe_allow_html=True)
        st.markdown('<div class="bank-card-title">📌 Dataset Schema & Data Types</div>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            'Column Name': df_data.columns,
            'Data Type': [str(dtype) for dtype in df_data.dtypes],
            'Missing Count': [df_data[col].isna().sum() for col in df_data.columns]
        })
        st.dataframe(dtype_df, use_container_width=True)
        st.markdown("""</div>""", unsafe_allow_html=True)
        
    with col_s2:
        st.markdown("""<div class="bank-card">""", unsafe_allow_html=True)
        st.markdown('<div class="bank-card-title">🎯 Target Column Distribution (loan_status)</div>', unsafe_allow_html=True)
        target_counts = df_data['loan_status'].value_counts().reset_index()
        target_counts.columns = ['Status_Code', 'Count']
        target_counts['Label'] = target_counts['Status_Code'].map({0: 'Non-Default (0)', 1: 'Default (1)'})
        
        fig_tgt = px.pie(target_counts, names='Label', values='Count', color='Label', color_discrete_map={'Non-Default (0)': '#059669', 'Default (1)': '#DC2626'}, hole=0.4)
        fig_tgt.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#0F172A')
        st.plotly_chart(fig_tgt, use_container_width=True)
        st.markdown("""</div>""", unsafe_allow_html=True)


# ==========================================
# 3. CREDIT RISK PAGE
# ==========================================
elif navigation_option == "🛡️ Credit Risk":
    render_top_header()
    
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="color:#1E3A8A; font-weight:800; margin:0;">Credit Risk Assessment</h2>
        <div style="color:#475569; font-size:14px; margin-top:2px;">Estimate the customer's probability of loan default using machine learning.</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_form, col_output = st.columns([1, 1])
    
    with col_form:
        st.markdown("""<div class="bank-card">""", unsafe_allow_html=True)
        st.markdown('<div class="bank-card-title">📋 Customer Information</div>', unsafe_allow_html=True)
        
        # Demo Mode: Sample Customer Selector
        st.markdown("##### 🚀 Demo Mode: Auto-Fill Sample Dataset Record")
        sample_idx = st.selectbox("Select Sample Customer from Dataset", options=list(range(min(50, len(df_data)))), format_func=lambda i: f"Customer Record #{i} - {df_data.iloc[i]['person_home_ownership']}, ${df_data.iloc[i]['person_income']:,} Inc, ${df_data.iloc[i]['loan_amnt']:,} Loan (Status: {df_data.iloc[i]['loan_status']})")
        sample_row = df_data.iloc[sample_idx]
        
        with st.form(key="credit_risk_input_form"):
            st.markdown("##### 👤 Personal Profile")
            p1, p2 = st.columns(2)
            with p1:
                age = st.number_input("Age", min_value=18, max_value=100, value=int(sample_row['person_age']))
            with p2:
                emp_val = sample_row['person_emp_length'] if pd.notna(sample_row['person_emp_length']) else 0.0
                emp_len = st.number_input("Employment History (Years)", min_value=0.0, max_value=50.0, value=float(emp_val), step=0.5)
            
            home_opts = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
            home_idx = home_opts.index(sample_row['person_home_ownership']) if sample_row['person_home_ownership'] in home_opts else 0
            home = st.selectbox("Home Ownership Status", home_opts, index=home_idx)
            
            st.markdown("##### 💵 Financial Profile")
            income = st.number_input("Annual Income ($)", min_value=5000, max_value=1000000, value=int(sample_row['person_income']), step=5000)
            
            st.markdown("##### 💳 Credit Profile")
            c1, c2 = st.columns(2)
            with c1:
                cb_def = st.selectbox("Prior Default Record?", ['N', 'Y'], index=0 if sample_row['cb_person_default_on_file']=='N' else 1)
            with c2:
                cred_hist = st.number_input("Credit History Length (Years)", min_value=1.0, max_value=40.0, value=float(sample_row['cb_person_cred_hist_length']), step=1.0)
                
            st.markdown("##### 📝 Loan Profile")
            l1, l2 = st.columns(2)
            with l1:
                loan_amt = st.number_input("Requested Loan Amount ($)", min_value=500, max_value=100000, value=int(sample_row['loan_amnt']), step=1000)
            with l2:
                rate_val = sample_row['loan_int_rate'] if pd.notna(sample_row['loan_int_rate']) else 10.0
                loan_rate = st.number_input("Interest Rate (%)", min_value=4.0, max_value=30.0, value=float(rate_val), step=0.1)
                
            l3, l4 = st.columns(2)
            with l3:
                intent_opts = ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
                intent_idx = intent_opts.index(sample_row['loan_intent']) if sample_row['loan_intent'] in intent_opts else 0
                intent = st.selectbox("Loan Intent", intent_opts, index=intent_idx)
            with l4:
                grade_opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
                grade_idx = grade_opts.index(sample_row['loan_grade']) if sample_row['loan_grade'] in grade_opts else 0
                grade = st.selectbox("Loan Risk Grade", grade_opts, index=grade_idx)
                
            submit_risk_btn = st.form_submit_button("🛡️ Run Risk Assessment", use_container_width=True)
        st.markdown("""</div>""", unsafe_allow_html=True)
        
    with col_output:
        st.markdown("""<div class="bank-card">""", unsafe_allow_html=True)
        st.markdown('<div class="bank-card-title">🎯 Risk Intelligence Assessment</div>', unsafe_allow_html=True)
        
        customer_payload = {
            'person_age': age,
            'person_income': income,
            'person_home_ownership': home,
            'person_emp_length': emp_len,
            'loan_intent': intent,
            'loan_grade': grade,
            'loan_amnt': loan_amt,
            'loan_int_rate': loan_rate,
            'loan_percent_income': round(loan_amt / max(income, 1), 3),
            'cb_person_default_on_file': cb_def,
            'cb_person_cred_hist_length': cred_hist
        }
        
        risk_output = ml_engine.predict_credit_risk(customer_payload)
        prob_pct = risk_output['default_probability_pct']
        non_prob_pct = risk_output['non_default_probability_pct']
        category = risk_output['risk_category']
        
        badge_style = "badge-risk-low" if category == "LOW RISK" else ("badge-risk-medium" if category == "MEDIUM RISK" else "badge-risk-high")
        pred_status = "Likely Non-Default" if prob_pct <= 50.0 else "Likely Default"
        
        # Risk Card Header
        st.markdown(f"""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:20px; text-align:center; margin-bottom:20px;">
            <div style="font-size:13px; font-weight:700; color:#64748B; text-transform:uppercase;">DEFAULT PROBABILITY</div>
            <div style="font-size:42px; font-weight:800; color:#1E3A8A; margin:4px 0;">{prob_pct:.1f}%</div>
            <div class="{badge_style}">{category}</div>
            <div style="font-size:13px; color:#475569; margin-top:8px;">Prediction: <strong>{pred_status}</strong> (Non-Default Prob: {non_prob_pct:.1f}%)</div>
            <div style="font-size:11px; color:#94A3B8; margin-top:4px;"><em>Prototype Risk Classification Thresholds: LOW (0-30%), MEDIUM (30-60%), HIGH (60-100%)</em></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Horizontal Risk Scale Marker
        st.markdown("##### 📏 Risk Scale Index Position")
        fig_scale = go.Figure()
        fig_scale.add_trace(go.Scatter(
            x=[0, 30, 60, 100],
            y=[0, 0, 0, 0],
            mode='lines',
            line=dict(color='#CBD5E1', width=6)
        ))
        fig_scale.add_trace(go.Scatter(
            x=[prob_pct],
            y=[0],
            mode='markers+text',
            marker=dict(size=18, color='#059669' if category == "LOW RISK" else ('#D97706' if category == "MEDIUM RISK" else '#DC2626')),
            text=[f"{prob_pct:.1f}%"],
            textposition="top center",
            name="Applicant"
        ))
        fig_scale.update_layout(
            xaxis=dict(range=[0, 100], title="LOW ───────── MEDIUM ───────── HIGH", showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
            height=120,
            margin=dict(l=20, r=20, t=30, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_scale, use_container_width=True)
        
        # Business Decision Intelligence Summary Layer
        rec_res = recommender_engine.recommend_products(customer_payload, risk_output['default_probability'], category)
        biz_dec = recommender_engine.get_business_decision_summary(customer_payload, risk_output['default_probability'], category, rec_res)
        
        st.markdown(f"""
        <div class="decision-summary-box">
            <div style="font-size:14px; font-weight:800; color:#1E3A8A; margin-bottom:8px;">💼 CUSTOMER DECISION SUMMARY LAYER</div>
            <div style="font-size:13px; color:#0F172A;">
                • <strong>Credit Risk:</strong> {biz_dec['credit_risk_level']}<br>
                • <strong>Default Prob:</strong> {biz_dec['default_probability_pct']:.1f}%<br>
                • <strong>Cross-Sell Opportunity:</strong> <span style="color:#0D9488; font-weight:700;">{biz_dec['cross_sell_opportunity']}</span><br>
                • <strong>Top Product Recommendation:</strong> <strong>{biz_dec['top_recommendation']}</strong> (Affinity Score: {biz_dec['top_affinity_score']}%)<br>
                • <strong>Recommendation Rationale:</strong> <em>{biz_dec['recommendation_rationale']}</em>
            </div>
            <div style="font-size:11px; color:#64748B; margin-top:8px; font-style:italic;">{biz_dec['disclaimer_notice']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("##### 🔍 Features Contributing Strongly to Prediction")
        for factor in risk_output['risk_factors']:
            st.markdown(f"• {factor}")
            
        st.markdown("""</div>""", unsafe_allow_html=True)


# ==========================================
# 4. RISK DRIVERS PAGE
# ==========================================
elif navigation_option == "🔍 Risk Drivers":
    render_top_header()
    
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="color:#1E3A8A; font-weight:800; margin:0;">Risk Drivers & Model Explainability</h2>
        <div style="color:#475569; font-size:14px; margin-top:2px;">Identify which features exert greater model influence on credit default risk estimations.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""<div class="bank-card">""", unsafe_allow_html=True)
    st.markdown('<div class="bank-card-title">📈 Feature Importance & Model Influence</div>', unsafe_allow_html=True)
    st.markdown("This analysis displays features with greater model influence on predictions. *(Note: Displays statistical model influence; does not assert causal real-world claims).*")
    
    df_imp = ml_engine.get_feature_importances_df()
    if not df_imp.empty:
        fig_imp = px.bar(
            df_imp.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Top 10 Feature Importances ({ml_engine.best_model_name})",
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#0F172A')
        st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown("""</div>""", unsafe_allow_html=True)


# ==========================================
# 5. WHAT-IF SIMULATOR PAGE
# ==========================================
elif navigation_option == "🎛️ What-If Simulator":
    render_top_header()
    
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="color:#1E3A8A; font-weight:800; margin:0;">What-If Risk Simulator</h2>
        <div style="color:#475569; font-size:14px; margin-top:2px;">Model-based scenario analysis: Adjust customer financial inputs to observe how the predicted probability changes.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""<div class="bank-card">""", unsafe_allow_html=True)
    st.markdown('<div class="bank-card-title">⚙️ Scenario Parameter Controls</div>', unsafe_allow_html=True)
    
    w1, w2, w3 = st.columns(3)
    with w1:
        sim_income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=50000, step=5000)
        sim_emp = st.number_input("Employment History (Years)", min_value=0.0, max_value=40.0, value=3.0, step=1.0)
    with w2:
        orig_loan = st.number_input("Original Loan Amount ($)", min_value=1000, max_value=100000, value=20000, step=1000)
        sim_loan = st.number_input("Scenario Loan Amount ($)", min_value=1000, max_value=100000, value=12000, step=1000)
    with w3:
        orig_rate = st.number_input("Original Interest Rate (%)", min_value=5.0, max_value=25.0, value=14.0, step=0.5)
        sim_rate = st.number_input("Scenario Interest Rate (%)", min_value=5.0, max_value=25.0, value=9.5, step=0.5)
        
    orig_payload = {
        'person_age': 32, 'person_income': sim_income, 'person_home_ownership': 'RENT', 'person_emp_length': sim_emp,
        'loan_intent': 'PERSONAL', 'loan_grade': 'C', 'loan_amnt': orig_loan, 'loan_int_rate': orig_rate,
        'loan_percent_income': round(orig_loan/max(sim_income,1),3), 'cb_person_default_on_file': 'N', 'cb_person_cred_hist_length': 5.0
    }
    
    sim_payload = {
        'person_age': 32, 'person_income': sim_income, 'person_home_ownership': 'RENT', 'person_emp_length': sim_emp,
        'loan_intent': 'PERSONAL', 'loan_grade': 'A', 'loan_amnt': sim_loan, 'loan_int_rate': sim_rate,
        'loan_percent_income': round(sim_loan/max(sim_income,1),3), 'cb_person_default_on_file': 'N', 'cb_person_cred_hist_length': 5.0
    }
    
    orig_res = ml_engine.predict_credit_risk(orig_payload)
    sim_res = ml_engine.predict_credit_risk(sim_payload)
    
    delta_prob = sim_res['default_probability_pct'] - orig_res['default_probability_pct']
    
    res1, res2, res3 = st.columns(3)
    with res1:
        st.markdown(f"""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:16px; border-radius:8px; text-align:center;">
            <div style="font-size:12px; color:#64748B; font-weight:700;">BEFORE SCENARIO</div>
            <div style="font-size:32px; font-weight:800; color:#0F172A;">{orig_res['default_probability_pct']:.1f}%</div>
            <div style="font-size:12px; color:#475569;">Risk: {orig_res['risk_category']}</div>
        </div>
        """, unsafe_allow_html=True)
    with res2:
        st.markdown(f"""
        <div style="background:#EFF6FF; border:1px solid #BFDBFE; padding:16px; border-radius:8px; text-align:center;">
            <div style="font-size:12px; color:#1E3A8A; font-weight:700;">AFTER SCENARIO</div>
            <div style="font-size:32px; font-weight:800; color:#1D4ED8;">{sim_res['default_probability_pct']:.1f}%</div>
            <div style="font-size:12px; color:#1E3A8A;">Risk: {sim_res['risk_category']}</div>
        </div>
        """, unsafe_allow_html=True)
    with res3:
        st.markdown(f"""
        <div style="background:#ECFDF5; border:1px solid #A7F3D0; padding:16px; border-radius:8px; text-align:center;">
            <div style="font-size:12px; color:#047857; font-weight:700;">PROBABILITY DIFFERENCE</div>
            <div style="font-size:32px; font-weight:800; color:#059669;">{delta_prob:+.1f}% pts</div>
            <div style="font-size:12px; color:#047857;">Model-based scenario analysis</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""</div>""", unsafe_allow_html=True)


# ==========================================
# 6. CROSS-SELL INTELLIGENCE PAGE
# ==========================================
elif navigation_option == "🛍️ Cross-Sell Intelligence":
    render_top_header()
    
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="color:#1E3A8A; font-weight:800; margin:0;">Cross-Sell Intelligence</h2>
        <div style="color:#475569; font-size:14px; margin-top:2px;">Discover banking products that may be relevant to this customer based on attributes and predicted risk.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""<div class="bank-card">""", unsafe_allow_html=True)
    st.markdown('<div class="bank-card-title">👤 Customer Profile Context</div>', unsafe_allow_html=True)
    
    cs1, cs2, cs3 = st.columns(3)
    with cs1:
        cs_inc = st.number_input("Customer Annual Income ($)", value=65000, step=5000)
    with cs2:
        cs_amt = st.number_input("Loan Amount ($)", value=15000, step=1000)
    with cs3:
        cs_def = st.selectbox("Past Credit Default Record", ['N', 'Y'], index=0)
        
    cs_dict = {
        'person_age': 34, 'person_income': cs_inc, 'person_home_ownership': 'MORTGAGE', 'person_emp_length': 5.0,
        'loan_intent': 'PERSONAL', 'loan_grade': 'B', 'loan_amnt': cs_amt, 'loan_int_rate': 10.5,
        'loan_percent_income': round(cs_amt/max(cs_inc, 1), 3), 'cb_person_default_on_file': cs_def, 'cb_person_cred_hist_length': 7.0
    }
    
    cs_risk = ml_engine.predict_credit_risk(cs_dict)
    cs_recs = recommender_engine.recommend_products(cs_dict, cs_risk['default_probability'], cs_risk['risk_category'])
    cs_biz = recommender_engine.get_business_decision_summary(cs_dict, cs_risk['default_probability'], cs_risk['risk_category'], cs_recs)
    
    st.markdown(f"""
    <div class="decision-summary-box">
        <div style="font-size:14px; font-weight:800; color:#1E3A8A; margin-bottom:8px;">💼 CUSTOMER DECISION SUMMARY LAYER</div>
        <div style="font-size:13px; color:#0F172A;">
            • <strong>Credit Risk:</strong> {cs_biz['credit_risk_level']}<br>
            • <strong>Default Prob:</strong> {cs_biz['default_probability_pct']:.1f}%<br>
            • <strong>Cross-Sell Opportunity:</strong> <span style="color:#0D9488; font-weight:700;">{cs_biz['cross_sell_opportunity']}</span><br>
            • <strong>Top Recommendation:</strong> <strong>{cs_biz['top_recommendation']}</strong> (Affinity: {cs_biz['top_affinity_score']}%)
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""</div>""", unsafe_allow_html=True)
    
    if cs_recs['warning_message']:
        st.markdown(f"""
        <div class="notice-warning">
            ⚠️ <strong>HIGH RISK CREDIT RESTRICTION:</strong><br>
            {cs_recs['warning_message']}
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 🛍️ Ranked Product Recommendations")
    rc1, rc2, rc3 = st.columns(3)
    available_recs = cs_recs['recommendations']
    cols = [rc1, rc2, rc3]
    
    for idx, col in enumerate(cols):
        with col:
            if idx < len(available_recs):
                p = available_recs[idx]
                st.markdown(f"""
                <div class="bank-card" style="height:100%;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <span style="font-weight:800; color:#1E3A8A; font-size:14px;">RANK #{idx+1}</span>
                        <span style="background:#EFF6FF; color:#1D4ED8; font-weight:800; padding:4px 10px; border-radius:12px; font-size:12px;">
                            Affinity: {p['affinity_score']}%
                        </span>
                    </div>
                    <div style="font-size:20px; font-weight:800; color:#0F172A;">{p['icon']} {p['product_name']}</div>
                    <div style="font-size:12px; font-weight:600; color:#0D9488; margin-bottom:8px;">Category: {p['category']}</div>
                    <div style="font-size:13px; color:#475569; margin-bottom:12px;">{p['description']}</div>
                    <div style="font-size:12px; color:#1E293B; background:#F1F5F9; padding:8px; border-radius:6px;">
                        <strong>Why Recommended:</strong><br>{p['recommendation_reason']}<br><br>
                        <strong>Risk Compatibility:</strong> Suitable under prototype risk rules.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""<div class="bank-card" style="opacity:0.5;">No additional products match eligibility criteria.</div>""", unsafe_allow_html=True)
                
    st.markdown("""<div class="bank-card">""", unsafe_allow_html=True)
    st.markdown('<div class="bank-card-title">🛡️ Prototype Risk-Aware Recommendation Policy</div>', unsafe_allow_html=True)
    st.markdown("""
    The recommendation engine considers predicted credit default risk before ranking financial products:
    
    - **LOW RISK (0% – 30%):** Broader product eligibility including high-value rewards credit cards, auto loans, and investment plans.
    - **MEDIUM RISK (30% – 60%):** Conservative product recommendations prioritizing guaranteed fixed deposits, liquid savings, and credit protection insurance.
    - **HIGH RISK (60% – 100%):** Restricts additional credit-focused recommendations (Personal Loans, Credit Cards, Auto Loans) and prioritizes capital preservation and protection coverage.
    """)
    st.markdown(f"""
    <div class="notice-disclaimer">
        ℹ️ <strong>{cs_recs['limitation_disclaimer']}</strong>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""</div>""", unsafe_allow_html=True)


# ==========================================
# 7. CUSTOMER 360° PAGE
# ==========================================
elif navigation_option == "👤 Customer 360°":
    render_top_header()
    
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="color:#1E3A8A; font-weight:800; margin:0;">Customer 360° Intelligence</h2>
        <div style="color:#475569; font-size:14px; margin-top:2px;">Unified Customer Intelligence View combining profile, credit details, risk score, risk drivers, and product affinity recommendations.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""<div class="bank-card">""", unsafe_allow_html=True)
    c360_idx = st.number_input("Select Customer Index from Dataset", min_value=0, max_value=len(df_data)-1, value=0)
    c360_row = df_data.iloc[c360_idx].to_dict()
    c360_risk = ml_engine.predict_credit_risk(c360_row)
    c360_recs = recommender_engine.recommend_products(c360_row, c360_risk['default_probability'], c360_risk['risk_category'])
    c360_biz = recommender_engine.get_business_decision_summary(c360_row, c360_risk['default_probability'], c360_risk['risk_category'], c360_recs)
    
    u1, u2 = st.columns(2)
    with u1:
        st.markdown("##### 👤 Customer & Financial Profile")
        st.write(f"• **Age:** {c360_row['person_age']} years")
        st.write(f"• **Annual Income:** ${c360_row['person_income']:,}")
        st.write(f"• **Employment Duration:** {c360_row['person_emp_length']} years")
        st.write(f"• **Home Ownership:** {c360_row['person_home_ownership']}")
        st.write(f"• **Loan Requested:** ${c360_row['loan_amnt']:,} ({c360_row['loan_intent']})")
        st.write(f"• **Interest Rate / Grade:** {c360_row['loan_int_rate']}% / Grade {c360_row['loan_grade']}")
        
    with u2:
        st.markdown("##### 🎯 Risk & Product Affinity Assessment")
        b_st = "badge-risk-low" if c360_risk['risk_category'] == "LOW RISK" else ("badge-risk-medium" if c360_risk['risk_category'] == "MEDIUM RISK" else "badge-risk-high")
        st.markdown(f'<div class="{b_st}">{c360_risk["risk_category"]} ({c360_risk["default_probability_pct"]:.1f}%)</div>', unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:8px; font-size:13px; color:#475569;'>{c360_risk['risk_description']}</div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="decision-summary-box">
            <strong>Decision Summary:</strong> {c360_biz['decision_label']}<br>
            <strong>Top Product:</strong> {c360_biz['top_recommendation']} (Affinity: {c360_biz['top_affinity_score']}%)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("###### Recommended Products:")
        for r in c360_recs['recommendations'][:3]:
            st.markdown(f"• **{r['icon']} {r['product_name']}** (Score: {r['affinity_score']}%)")
    st.markdown("""</div>""", unsafe_allow_html=True)


# ==========================================
# 8. PORTFOLIO RISK PAGE
# ==========================================
elif navigation_option == "📈 Portfolio Risk":
    render_top_header()
    
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="color:#1E3A8A; font-weight:800; margin:0;">Portfolio Risk Analytics</h2>
        <div style="color:#475569; font-size:14px; margin-top:2px;">Portfolio-wide risk distribution and credit risk analysis across financial attributes.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Portfolio KPIs
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Total Records</div><div class="kpi-value">{kpis['total_records']:,}</div></div>""", unsafe_allow_html=True)
    with p2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Default Rate</div><div class="kpi-value">{kpis['default_rate_pct']:.2f}%</div></div>""", unsafe_allow_html=True)
    with p3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Non-Default Rate</div><div class="kpi-value">{100.0-kpis['default_rate_pct']:.2f}%</div></div>""", unsafe_allow_html=True)
    with p4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Avg Loan Amount</div><div class="kpi-value">${kpis['avg_loan']:,.0f}</div></div>""", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("""<div class="bank-card">""", unsafe_allow_html=True)
        st.markdown('<div class="bank-card-title">📊 Default Rate by Loan Grade</div>', unsafe_allow_html=True)
        grade_risk = df_data.groupby('loan_grade')['loan_status'].mean().reset_index()
        grade_risk['Default Rate %'] = grade_risk['loan_status'] * 100
        fig_grd = px.bar(grade_risk, x='loan_grade', y='Default Rate %', color='Default Rate %', color_continuous_scale='Reds')
        fig_grd.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#0F172A')
        st.plotly_chart(fig_grd, use_container_width=True)
        st.markdown("""</div>""", unsafe_allow_html=True)
        
    with col_chart2:
        st.markdown("""<div class="bank-card">""", unsafe_allow_html=True)
        st.markdown('<div class="bank-card-title">🏠 Default Rate by Home Ownership</div>', unsafe_allow_html=True)
        home_risk = df_data.groupby('person_home_ownership')['loan_status'].mean().reset_index()
        home_risk['Default Rate %'] = home_risk['loan_status'] * 100
        fig_hm = px.bar(home_risk, x='person_home_ownership', y='Default Rate %', color='Default Rate %', color_continuous_scale='Blues')
        fig_hm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#0F172A')
        st.plotly_chart(fig_hm, use_container_width=True)
        st.markdown("""</div>""", unsafe_allow_html=True)


# ==========================================
# 9. MODEL PERFORMANCE PAGE
# ==========================================
elif navigation_option == "🏆 Model Performance":
    render_top_header()
    
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="color:#1E3A8A; font-weight:800; margin:0;">Model Performance Benchmark</h2>
        <div style="color:#475569; font-size:14px; margin-top:2px;">Evaluation of candidate machine-learning models on the Credit Risk Dataset.</div>
    </div>
    """, unsafe_allow_html=True)
    
    best_metrics = model_evals[ml_engine.best_model_name]
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Accuracy</div><div class="kpi-value">{best_metrics['Accuracy']*100:.1f}%</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Precision</div><div class="kpi-value">{best_metrics['Precision']*100:.1f}%</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Recall</div><div class="kpi-value">{best_metrics['Recall']*100:.1f}%</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">F1 Score</div><div class="kpi-value">{best_metrics['F1 Score']:.3f}</div></div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">ROC-AUC</div><div class="kpi-value">{best_metrics['ROC-AUC']:.3f}</div></div>""", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Comparison Table
    st.markdown("""<div class="bank-card">""", unsafe_allow_html=True)
    st.markdown('<div class="bank-card-title">📊 Candidate ML Models Comparison Table</div>', unsafe_allow_html=True)
    
    comp_data = []
    for name, m in model_evals.items():
        comp_data.append({
            'Model Name': name,
            'Accuracy': f"{m['Accuracy']*100:.2f}%",
            'Precision': f"{m['Precision']*100:.2f}%",
            'Recall': f"{m['Recall']*100:.2f}%",
            'F1 Score': f"{m['F1 Score']:.4f}",
            'ROC-AUC': f"{m['ROC-AUC']:.4f}",
            'Status': "⭐ Best Model" if name == ml_engine.best_model_name else "Candidate"
        })
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
    st.markdown("""</div>""", unsafe_allow_html=True)
    
    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        st.markdown("""<div class="bank-card">""", unsafe_allow_html=True)
        st.markdown('<div class="bank-card-title">🧩 Confusion Matrix Inspection</div>', unsafe_allow_html=True)
        sel_mod = st.selectbox("Select Model to Inspect Confusion Matrix", list(model_evals.keys()), index=list(model_evals.keys()).index(ml_engine.best_model_name))
        cm = model_evals[sel_mod]['Confusion Matrix']
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
            x=['Non-Default (0)', 'Default (1)'],
            y=['Non-Default (0)', 'Default (1)'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#0F172A')
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("""</div>""", unsafe_allow_html=True)
        
    with col_eval2:
        st.markdown("""<div class="bank-card">""", unsafe_allow_html=True)
        st.markdown('<div class="bank-card-title">📈 Receiver Operating Characteristic (ROC Curve)</div>', unsafe_allow_html=True)
        y_test = model_evals[sel_mod]['y_test']
        y_proba = model_evals[sel_mod]['y_proba']
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{sel_mod} (AUC={model_evals[sel_mod]['ROC-AUC']:.3f})", line=dict(color='#1D4ED8', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Baseline', line=dict(color='#94A3B8', dash='dash')))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#0F172A')
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown("""</div>""", unsafe_allow_html=True)


# ==========================================
# 10. ABOUT PROJECT PAGE
# ==========================================
elif navigation_option == "ℹ️ About Project":
    render_top_header()
    
    st.markdown("""
    <div class="bank-card">
        <div class="bank-card-title">📌 Academic Project Details</div>
        <table style="width:100%; font-size:14px; color:#0F172A;">
            <tr><td style="width:200px; font-weight:700;">Project Title:</td><td>Banking Credit Default Risk & Cross-Sell Engine</td></tr>
            <tr><td style="font-weight:700;">Student ID:</td><td><code>IBMQ2DST1296</code></td></tr>
            <tr><td style="font-weight:700;">Dataset:</td><td>Credit Risk Dataset</td></tr>
            <tr><td style="font-weight:700;">Technologies:</td><td>Python, Pandas, NumPy, Scikit-learn, Streamlit, Plotly</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="bank-card">
        <div class="bank-card-title">🎯 Case Study Objective</div>
        <div style="font-size:14px; color:#475569; line-height:1.6;">
            Commercial banks face a dual challenge: minimizing credit default losses on loan portfolios while 
            maximizing revenue through cross-selling appropriate financial products to eligible customers. 
            Traditional cross-sell systems often recommend high-yield credit products indiscriminately without accounting 
            for applicant risk, leading to elevated default rates. This system integrates credit default risk prediction with 
            product recommendation into a risk-aware decision support engine.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="bank-card">
        <div class="bank-card-title">⚙️ Methodology & Architecture Flow</div>
        <div style="font-size:13px; color:#1E293B; background:#F8FAFC; padding:16px; border-radius:8px; font-family:monospace;">
            CSV Dataset ➔ dataset.py ➔ preprocessing.py ➔ model.py ➔ Risk Prediction ➔ recommendation.py ➔ Business Decision Layer ➔ app.py (Streamlit UI)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="notice-disclaimer">
        ℹ️ <strong>Academic Prototype Disclaimer & Dataset Limitation:</strong><br>
        The current prototype uses the Credit Risk Dataset. Cross-sell recommendations are generated from available customer and credit-risk attributes using prototype recommendation logic. Transaction-level product affinity requires additional transaction and product-history data. This is an academic decision-support prototype. Predictions and product recommendations are not financial advice, loan approval decisions, or official banking policy.
    </div>
    """, unsafe_allow_html=True)
