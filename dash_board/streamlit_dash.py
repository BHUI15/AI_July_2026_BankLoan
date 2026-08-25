# import datetime
# import pandas as pd

import requests
import streamlit as st
from advice import get_loan_advice

# Note: GEMINI.md recommends port 5001 for model serving
MLFLOW_ENDPOINT = "http://127.0.0.1:5001/invocations"

# The exact columns expected by the MLflow signature
REQUIRED_COLUMNS = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file"
]

st.set_page_config(page_title="Loan Application Predictor", page_icon=":money_with_wings:", layout="wide")
st.title("🎈Loan Application Predictor")
st.write("Enter loan application details to get an instant preliminary loan decision.")

st.subheader("Applicant Profile & Loan Request Details")

with st.form(key="loan_application_form"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        person_age = st.number_input("Applicant Age", min_value=18, max_value=100, value=25)
    with col2:
        person_gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    with col3:
        person_education = st.selectbox("Education Level", options=["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    with col4:
        person_income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0)

    st.divider()
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        person_emp_exp = st.number_input("Employment Experience (Years)", min_value=0, max_value=60, value=3, step=1)
    with col6:
        person_home_ownership = st.selectbox("Home Ownership Status", options=["RENT", "MORTGAGE", "OWN", "OTHER"])
    with col7:
        loan_amnt = st.number_input("Loan Amount Requested ($)", min_value=500.0, value=10000.0)
    with col8:
        loan_intent = st.selectbox("Loan Purpose", options=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])

    st.divider()
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=11.5)
    with col10:
        # Calculate percent income automatically, but allow override
        default_pct = round(loan_amnt / person_income, 2) if person_income > 0 else 0.0
        loan_percent_income = st.number_input("Loan as % of Income (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=min(default_pct, 1.0))
    with col11:
        cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0.0, max_value=50.0, value=5.0)
    with col12:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=680, step=1)

    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File?", options=["Y", "N"], index=1)
    
    # create a submit button for the form
    submit_button = st.form_submit_button("Submit")

    if submit_button:
        # Construct payload with exact strict types matched to signature
        payload = {
            "dataframe_split": {
                "columns": REQUIRED_COLUMNS,
                "data": [
                    [
                        float(person_age),                  # double
                        str(person_gender),                 # string
                        str(person_education),              # string
                        float(person_income),               # double
                        int(person_emp_exp),                # long
                        str(person_home_ownership),         # string
                        float(loan_amnt),                   # double
                        str(loan_intent),                   # string
                        float(loan_int_rate),               # double
                        float(loan_percent_income),         # double
                        float(cb_person_cred_hist_length),  # double
                        int(credit_score),                  # long
                        str(previous_loan_defaults_on_file) # string
                    ]
                ],
            }
        }

        with st.spinner("Analyzing loan application..."):
            try:
                response = requests.post(
                    MLFLOW_ENDPOINT, 
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=20
                )
                response.raise_for_status()
                
                # The model outputs a tensor dictionary array format according to your outputs spec
                # Usually returns: {"predictions":} or {"predictions":}
                raw_prediction = float(response.json()["predictions"][0])

                print(f"Raw prediction from model server: {raw_prediction}")
                # Assuming 1 = Approved, 0 = Rejected (adjust to match your label encoding)
                decision = "Approved" if raw_prediction >= 0.5 else "Rejected"

                # Display the prediction result
                if decision == "Approved":
                    st.balloons()
                    st.success(f"Loan Application Decision: {decision}")
                else:
                    st.error(f"Loan Application Decision: {decision}")
                    advice = get_loan_advice(application_rejected=True)
                    if advice:
                        st.warning("💡 Loan Application Advice")
                        st.write(f"**Summary:** {advice.loan_summary}")
                        st.write("**Recommended Actions:**")
                        for action in advice.action:
                            st.write(f"- {action}")
                        st.write(f"**Reason:** {advice.reason}")
                        st.write(f"**Next Steps:** {advice.next_steps}")

            except requests.exceptions.RequestException as e:
                st.error(f"Error communicating with the model server: {e}")
            except (ValueError, KeyError, IndexError) as e:
                st.error(f"Invalid response format from the model server: {e}")