# Credit Risk Prediction System with SHAP Explainability

Predicts loan default risk using XGBoost, with SHAP explainability to interpret individual predictions — built for transparent, auditable credit decisions.
🔗 **[Try the live demo](https://credit-risk-shap-htxzmz3iaduncqbjaahp8c.streamlit.app)**

## Problem
Lenders need to predict which applicants are likely to default, while also being able to explain *why* a decision was made (required in regulated credit environments).

## Dataset
"Give Me Some Credit" (Kaggle) — 150,000 loan applicants, 10 features (credit utilization, payment history, income, debt ratio, etc.). Default rate: 6.68%.

## Approach
1. EDA — explored feature distributions, found default risk decreases sharply with age
2. Preprocessing — handled missing values, capped outliers, stratified train/test split, scaling
3. Model comparison:

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.8118 |
| Random Forest | 0.8478 |
| **XGBoost** | **0.8614** |

4. Explainability — SHAP TreeExplainer to identify top risk drivers
5. Dashboard — Power BI visuals for risk segmentation and model performance

## Key Findings
- Top risk drivers (SHAP): credit utilization, payment history (30-59 and 90+ day lateness), age, debt ratio
- Default risk drops steadily with age (~13% under 30 → ~2% over 70)
- Model catches ~21% of actual defaulters at a low false-positive rate — reflects the real difficulty of predicting rare events

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, MySQL, Power BI

## Live Demo

An interactive Streamlit app lets you enter applicant details and get:
- A default risk score
- An approve/reject decision based on an adjustable bank risk cutoff
- A plain-English explanation of what drove the prediction, powered by SHAP

Try it here: https://credit-risk-shap-htxzmz3iaduncqbjaahp8c.streamlit.app

To run locally:
\`\`\`
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Screenshots
<img width="1490" height="716" alt="shap_screenshot" src="https://github.com/user-attachments/assets/513e726b-4ec3-44fe-8858-d58ad20661a3" />
<img width="1363" height="728" alt="power_bi_screenshot" src="https://github.com/user-attachments/assets/b6908982-d1ba-4f1b-82d8-045517c4d91d" />


