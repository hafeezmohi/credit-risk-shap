import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")

model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
explainer = shap.TreeExplainer(model)

FEATURE_ORDER = [
    "RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"
]

FRIENDLY_NAMES = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit Utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59 Days Late (count)",
    "DebtRatio": "Debt Ratio",
    "MonthlyIncome": "Monthly Income",
    "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines",
    "NumberOfTimes90DaysLate": "90+ Days Late (count)",
    "NumberRealEstateLoansOrLines": "Real Estate Loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89 Days Late (count)",
    "NumberOfDependents": "Dependents",
}

st.title("💳 Credit Risk Prediction")
st.caption("XGBoost model with SHAP explainability — trained on the Give Me Some Credit dataset (150k applicants, 0.86 ROC-AUC)")

st.divider()

# ---------- Sidebar inputs ----------
st.sidebar.header("Applicant Details")

age = st.sidebar.slider("Age", 18, 100, 35)
income = st.sidebar.number_input("Monthly Income", min_value=0, value=5000, step=500)
utilization = st.sidebar.slider("Credit Utilization", 0.0, 2.0, 0.3, help="Fraction of available credit currently used")
debt_ratio = st.sidebar.slider("Debt Ratio", 0.0, 2.0, 0.3)

st.sidebar.subheader("Payment History")
late_30_59 = st.sidebar.number_input("Times 30-59 Days Late", min_value=0, max_value=20, value=0)
late_60_89 = st.sidebar.number_input("Times 60-89 Days Late", min_value=0, max_value=20, value=0)
late_90 = st.sidebar.number_input("Times 90+ Days Late", min_value=0, max_value=20, value=0)

st.sidebar.subheader("Other")
open_credit_lines = st.sidebar.number_input("Open Credit Lines and Loans", min_value=0, max_value=30, value=5)
real_estate_loans = st.sidebar.number_input("Real Estate Loans", min_value=0, max_value=10, value=1)
dependents = st.sidebar.number_input("Number of Dependents", min_value=0, max_value=10, value=0)

st.sidebar.divider()
st.sidebar.subheader("Bank Policy")
st.sidebar.caption("Different lenders accept different levels of risk. Set your cutoff below.")
approval_cutoff = st.sidebar.slider(
    "Maximum risk % to approve",
    min_value=1, max_value=50, value=15,
    help="Applicants at or below this risk % get approved."
) / 100

predict_clicked = st.sidebar.button("Predict Risk", type="primary", use_container_width=True)

# ---------- Main panel ----------
if predict_clicked:
    input_data = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": utilization,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
        "NumberOfDependents": dependents
    }])[FEATURE_ORDER]

    input_scaled = scaler.transform(input_data)
    probability = float(model.predict_proba(input_scaled)[:, 1][0])

    if probability <= approval_cutoff:
        risk_label, color = "Within Policy", "green"
        decision, decision_icon = "Credit Approved", "✅"
    elif probability <= approval_cutoff + 0.10:
        risk_label, color = "Above Policy (borderline)", "orange"
        decision, decision_icon = "Approved with Conditions", "⚠️"
    else:
        risk_label, color = "Above Policy", "red"
        decision, decision_icon = "Not Approved", "❌"

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Default Probability", f"{probability:.1%}")
        st.markdown(f"### :{color}[{risk_label}]")
        st.progress(min(probability, 1.0))
        st.markdown(f"## {decision_icon} {decision}")

    with col2:
        st.subheader("Why this prediction?")
        st.caption("Each bar shows how much a factor pushed the risk score up (red) or down (blue) for this specific applicant.")

        shap_values = explainer.shap_values(input_scaled)

        contributions = pd.Series(shap_values[0], index=[FRIENDLY_NAMES[c] for c in FEATURE_ORDER])
        contributions = contributions.sort_values(key=abs, ascending=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in contributions.values]
        ax.barh(contributions.index, contributions.values, color=colors)
        ax.set_xlabel("Impact on risk score")
        ax.axvline(0, color="black", linewidth=0.8)
        fig.tight_layout()
        st.pyplot(fig)

    # ---------- Plain-English explanation ----------
    st.subheader("In plain terms")

    EXPLANATIONS = {
        "Credit Utilization": {
            "up": "you are using a lot of your available credit",
            "down": "you are using very little of your available credit",
        },
        "Age": {
            "up": "younger applicants are riskier in this data",
            "down": "older applicants are safer in this data",
        },
        "30-59 Days Late (count)": {
            "up": "you have paid late (30-59 days) before",
            "down": "you have never paid late by 30-59 days",
        },
        "60-89 Days Late (count)": {
            "up": "you have paid late (60-89 days) before",
            "down": "you have never paid late by 60-89 days",
        },
        "90+ Days Late (count)": {
            "up": "you have paid very late (90+ days) before, which is a serious warning sign",
            "down": "you have never paid very late (90+ days)",
        },
        "Debt Ratio": {
            "up": "your monthly bills are high compared to your income",
            "down": "your monthly bills are low compared to your income",
        },
        "Monthly Income": {
            "up": "your income is on the lower side",
            "down": "your income is on the higher side",
        },
        "Open Credit Lines": {
            "up": "you have many open credit lines and loans",
            "down": "you have very few open credit lines and loans",
        },
        "Real Estate Loans": {
            "up": "you have several real estate loans",
            "down": "you have few or no real estate loans",
        },
        "Dependents": {
            "up": "you support several dependents",
            "down": "you support few or no dependents",
        },
    }

    top_factors = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(3)

    st.write(f"**{decision}** — your estimated default risk is {probability:.1%}. This bank's policy approves applicants at or below {approval_cutoff:.0%} risk.")
    st.write("Here's why the model gave this score:")

    for feature, value in top_factors.items():
        direction = "up" if value > 0 else "down"
        reason = EXPLANATIONS.get(feature, {}).get(direction)
        if reason:
            bullet = reason[0].upper() + reason[1:]
            st.markdown(f"- {bullet}.")

    st.divider()
    with st.expander("See input values used"):
        st.dataframe(input_data.rename(columns=FRIENDLY_NAMES), use_container_width=True)

else:
    st.info("Fill in the applicant details in the sidebar and click **Predict Risk** to see the result.")