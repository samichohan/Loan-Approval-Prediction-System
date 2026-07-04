import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide"
)

# ---------------- LOAD MODEL & DATA ---------------- #
@st.cache_resource
def load_model():
    return joblib.load("model/loan_pipeline.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/clean_loan.csv")

try:
    model = load_model()
    df = load_data()
except FileNotFoundError as e:
    st.error(f"❌ Required file not found: {e}")
    st.info("Make sure 'model/loan_pipeline.pkl' and 'data/clean_loan.csv' exist in your project.")
    st.stop()

# Education mapping (must match the mapping used during training)
education_order = {
    "High School": 0,
    "Associate": 1,
    "Bachelor": 2,
    "Master": 3,
    "Doctorate": 4
}

# ---------------- TITLE ---------------- #
st.title("💰 Loan Approval Prediction")
st.markdown("Predict whether a loan application will be **Approved** or **Rejected** using Machine Learning.")
st.divider()

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("Enter Applicant Details")

person_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=25)

person_gender = st.sidebar.selectbox(
    "Gender",
    sorted(df["person_gender"].unique())
)

person_education = st.sidebar.selectbox(
    "Education Level",
    ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
)

person_income = st.sidebar.number_input(
    "Annual Income ($)", min_value=0, value=50000, step=1000
)

person_emp_exp = st.sidebar.number_input(
    "Years of Employment Experience", min_value=0, max_value=60, value=2
)

person_home_ownership = st.sidebar.selectbox(
    "Home Ownership",
    sorted(df["person_home_ownership"].unique())
)

loan_amnt = st.sidebar.number_input(
    "Loan Amount ($)", min_value=0, value=10000, step=500
)

loan_intent = st.sidebar.selectbox(
    "Loan Purpose",
    sorted(df["loan_intent"].unique())
)

loan_int_rate = st.sidebar.number_input(
    "Loan Interest Rate (%)", min_value=0.0, max_value=40.0, value=12.0, step=0.1
)

loan_percent_income = st.sidebar.slider(
    "Loan as % of Income", min_value=0.0, max_value=1.0, value=0.2, step=0.01
)

cb_person_cred_hist_length = st.sidebar.number_input(
    "Credit History Length (years)", min_value=0, max_value=40, value=4
)

credit_score = st.sidebar.number_input(
    "Credit Score", min_value=300, max_value=850, value=650
)

previous_loan_defaults_on_file = st.sidebar.selectbox(
    "Previous Loan Defaults?",
    sorted(df["previous_loan_defaults_on_file"].unique())
)

predict = st.sidebar.button("🚀 Predict Loan Status")

with st.sidebar.expander("Debug Info"):
    import sklearn
    st.write("Scikit-learn:", sklearn.__version__)
    st.write("Joblib:", joblib.__version__)

# ---------------- MAIN AREA ---------------- #
col1, col2 = st.columns(2)

with col1:
    st.subheader("Applicant Summary")
    st.write("**Age:**", person_age)
    st.write("**Gender:**", person_gender)
    st.write("**Education:**", person_education)
    st.write("**Income:**", f"${person_income:,}")
    st.write("**Employment Experience:**", f"{person_emp_exp} years")
    st.write("**Home Ownership:**", person_home_ownership)
    st.write("**Loan Amount:**", f"${loan_amnt:,}")
    st.write("**Loan Purpose:**", loan_intent)
    st.write("**Interest Rate:**", f"{loan_int_rate}%")
    st.write("**Loan % of Income:**", loan_percent_income)
    st.write("**Credit History Length:**", f"{cb_person_cred_hist_length} years")
    st.write("**Credit Score:**", credit_score)
    st.write("**Previous Defaults:**", previous_loan_defaults_on_file)

with col2:
    st.subheader("Prediction Result")
    if predict:
        try:
            input_df = pd.DataFrame([{
                "person_age": person_age,
                "person_gender": person_gender,
                "person_education": education_order[person_education],
                "person_income": person_income,
                "person_emp_exp": person_emp_exp,
                "person_home_ownership": person_home_ownership,
                "loan_amnt": loan_amnt,
                "loan_intent": loan_intent,
                "loan_int_rate": loan_int_rate,
                "loan_percent_income": loan_percent_income,
                "cb_person_cred_hist_length": cb_person_cred_hist_length,
                "credit_score": credit_score,
                "previous_loan_defaults_on_file": previous_loan_defaults_on_file
            }])

            prediction = model.predict(input_df)[0]

            # Try to get probability/confidence if the model supports it
            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                confidence = proba[prediction]

            if prediction == 1:
                st.success("✅ Loan Approved")
            else:
                st.error("❌ Loan Rejected")

            if confidence is not None:
                st.metric("Model Confidence", f"{confidence*100:.1f}%")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Fill in the applicant details and click **Predict Loan Status** to see the result.")

# ---------------- DATASET ---------------- #
st.divider()
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- STATS ---------------- #
st.subheader("Dataset Statistics")
c1, c2, c3 = st.columns(3)
c1.metric("Total Applications", len(df))
c2.metric("Approved", int((df["loan_status"] == 1).sum()))
c3.metric("Rejected", int((df["loan_status"] == 0).sum()))

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("Made with ❤️ using Streamlit + Scikit-Learn")