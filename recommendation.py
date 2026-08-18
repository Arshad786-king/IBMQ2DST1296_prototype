"""
Cross-Sell Recommendation Engine & Business Decision Intelligence Layer
Student ID: IBMQ2DST1296
"""

import pandas as pd
from config import BANKING_PRODUCTS, RISK_THRESHOLDS

class RiskAwareRecommendationEngine:
    """Central Recommendation Implementation for Risk-Aware Cross-Selling."""
    def __init__(self):
        self.products = BANKING_PRODUCTS

    def recommend_products(self, customer_dict: dict, default_probability: float, risk_category: str) -> dict:
        """Generates risk-adjusted product recommendations based on customer profile and predicted default risk."""
        if customer_dict is None:
            customer_dict = {}
            
        income = customer_dict.get('person_income', 0)
        emp_length = customer_dict.get('person_emp_length', 0)
        if emp_length is None or (isinstance(emp_length, float) and (emp_length != emp_length)):
            emp_length = 0.0
            
        recommendations = []
        restricted_products = []
        
        # Normalize default probability
        try:
            prob_float = float(default_probability) if default_probability is not None else 0.0
        except (ValueError, TypeError):
            prob_float = 0.0
            
        if prob_float > 1.0:
            prob_float = prob_float / 100.0
            
        is_high_risk = risk_category == "HIGH RISK" or prob_float > RISK_THRESHOLDS["MEDIUM_RISK_MAX"]
        
        for prod_name, prod_info in self.products.items():
            category = prod_info['category']
            min_inc = prod_info['min_income']
            max_risk = prod_info['max_risk']
            min_emp = prod_info['min_emp_length']
            
            # Check if customer is restricted due to high risk on credit products
            if is_high_risk and category == "Credit":
                restricted_products.append({
                    'product_name': prod_name,
                    'category': category,
                    'icon': prod_info['icon'],
                    'restriction_reason': "Restricted due to high predicted default risk."
                })
                continue
                
            # Eligibility check
            if income < min_inc:
                continue
            if emp_length < min_emp:
                continue
            if prob_float > max_risk:
                continue
                
            # Calculate affinity score
            base_score = 50.0
            
            # Income factor
            if income >= 75000:
                base_score += 20.0
            elif income >= 45000:
                base_score += 10.0
                
            # Risk penalty / bonus
            if risk_category == "LOW RISK":
                if category in ["Credit", "Investment"]:
                    base_score += 25.0
                else:
                    base_score += 10.0
            elif risk_category == "MEDIUM RISK":
                if category in ["Savings", "Investment", "Protection"]:
                    base_score += 25.0
                elif category == "Credit":
                    base_score += 5.0
            else: # HIGH RISK
                if category in ["Savings", "Protection"]:
                    base_score += 35.0
                elif prod_name == "Fixed Deposit":
                    base_score += 20.0
                    
            affinity_score = round(min(base_score, 98.5), 1)
            
            # Determine rationale
            if category == "Credit":
                reason = f"Eligible for credit line based on stable income (${income:,}) and low default risk ({prob_float*100:.1f}%)."
            elif category == "Investment":
                reason = f"Ideal wealth generation product for income profile (${income:,}) and credit status."
            elif category == "Protection":
                reason = "Recommended risk mitigation and credit protection coverage."
            else: # Savings
                reason = "Secure liquid savings build-up recommended for capital preservation."
                
            recommendations.append({
                'product_name': prod_name,
                'category': category,
                'icon': prod_info['icon'],
                'description': prod_info['description'],
                'affinity_score': affinity_score,
                'recommendation_reason': reason
            })
            
        # Sort recommendations by affinity score descending
        recommendations.sort(key=lambda x: x['affinity_score'], reverse=True)
        
        # Prototype warning notice if high risk
        warning_message = None
        if is_high_risk:
            warning_message = "Additional credit products are restricted in this prototype because the predicted credit risk is high."
            
        limitation_disclaimer = (
            "Prototype Product Recommendation: Recommendations are generated from available customer and credit attributes. "
            "Real transaction-history based cross-selling would require historical transaction and product-ownership data."
        )
        
        return {
            'recommendations': recommendations,
            'restricted_products': restricted_products,
            'warning_message': warning_message,
            'limitation_disclaimer': limitation_disclaimer
        }

    def get_business_decision_summary(self, customer_data: dict, default_probability: float, risk_category: str, recommendations) -> dict:
        """Generates unified Customer Business Decision Summary layer.
        
        Accepts:
        - customer_data: dictionary or Series of customer features
        - default_probability: float (0..1 or 0..100), None, string, or NaN
        - risk_category: str ("LOW", "LOW RISK", "MEDIUM", "HIGH", etc.)
        - recommendations: dictionary (containing 'recommendations' key), list of product dicts, or None
        
        Returns robust dictionary matching all UI requirements.
        """
        # 1. Normalize default probability
        if default_probability is None or (isinstance(default_probability, float) and (default_probability != default_probability)):
            prob_float = 0.0
        else:
            try:
                prob_float = float(default_probability)
            except (ValueError, TypeError):
                prob_float = 0.0
                
        if 0.0 <= prob_float <= 1.0:
            prob_pct = round(prob_float * 100.0, 2)
            prob_norm = round(prob_float, 4)
        elif prob_float > 1.0:
            prob_pct = round(min(prob_float, 100.0), 2)
            prob_norm = round(prob_pct / 100.0, 4)
        else:
            prob_pct = 0.0
            prob_norm = 0.0

        # 2. Normalize risk category
        if not risk_category or not isinstance(risk_category, str):
            if prob_pct <= 30.0:
                risk_norm = "LOW RISK"
            elif prob_pct <= 60.0:
                risk_norm = "MEDIUM RISK"
            else:
                risk_norm = "HIGH RISK"
        else:
            rc_upper = risk_category.strip().upper()
            if "LOW" in rc_upper:
                risk_norm = "LOW RISK"
            elif "MED" in rc_upper:
                risk_norm = "MEDIUM RISK"
            elif "HIGH" in rc_upper:
                risk_norm = "HIGH RISK"
            else:
                if prob_pct <= 30.0:
                    risk_norm = "LOW RISK"
                elif prob_pct <= 60.0:
                    risk_norm = "MEDIUM RISK"
                else:
                    risk_norm = "HIGH RISK"

        # 3. Normalize recommendations structure
        if isinstance(recommendations, dict):
            rec_list = recommendations.get('recommendations', [])
        elif isinstance(recommendations, (list, tuple)):
            rec_list = list(recommendations)
        else:
            rec_list = []
            
        rec_count = len(rec_list)

        # 4. Determine Top Recommendation & Rationale
        if rec_count > 0:
            top_rec = rec_list[0]
            if isinstance(top_rec, dict):
                top_prod_name = top_rec.get('product_name') or top_rec.get('product') or top_rec.get('name') or "Savings Product"
                top_affinity = float(top_rec.get('affinity_score') or top_rec.get('score') or top_rec.get('affinity') or 50.0)
                top_rationale = str(top_rec.get('recommendation_reason') or top_rec.get('reason') or top_rec.get('rationale') or "Matches customer profile criteria.")
            else:
                top_prod_name = str(top_rec)
                top_affinity = 50.0
                top_rationale = "Matches customer profile criteria."
        else:
            top_prod_name = "No suitable product identified"
            top_affinity = 0.0
            top_rationale = "No eligible product matching current risk and income criteria."

        # 5. Determine Decision Label & Cross-Sell Opportunity
        if risk_norm == "LOW RISK":
            cross_sell_opp = "HIGH" if rec_count > 0 else "LOW"
            decision_label = "Standard Approval Eligible with High Product Affinity"
        elif risk_norm == "MEDIUM RISK":
            cross_sell_opp = "MEDIUM" if rec_count > 0 else "LOW"
            decision_label = "Structured Approval / Conservative Product Alignment"
        else: # HIGH RISK
            cross_sell_opp = "LOW"
            decision_label = "Unsecured Credit Restricted / Capital Preservation Focus"

        return {
            "credit_risk": risk_norm,
            "credit_risk_level": risk_norm,
            "default_probability": prob_norm,
            "default_probability_pct": prob_pct,
            "cross_sell_opportunity": cross_sell_opp,
            "top_recommendation": top_prod_name,
            "top_affinity_score": top_affinity,
            "recommendation_count": rec_count,
            "decision": decision_label,
            "decision_label": decision_label,
            "rationale": top_rationale,
            "recommendation_rationale": top_rationale,
            "risk_compatibility": "Suitable under prototype risk-aware recommendation policy.",
            "disclaimer_notice": "Business Decision Layer: Prototype decision support recommendation based on customer attributes and risk rules. Not an official loan approval or banking policy."
        }
