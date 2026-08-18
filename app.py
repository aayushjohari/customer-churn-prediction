import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load("models/Customer_churn_model.pkl")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter the customer's information below to predict "
    "whether the customer is likely to churn."
)

st.divider()


# ==================================================
# CUSTOMER INFORMATION
# ==================================================

st.header("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )


col1, col2 = st.columns(2)

with col1:
    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

with col2:
    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


# ==================================================
# SERVICE INFORMATION
# ==================================================

st.header("📱 Service Information")

col1, col2 = st.columns(2)

with col1:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

with col2:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )


col1, col2 = st.columns(2)

with col1:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )


col1, col2 = st.columns(2)

with col1:
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )


col1, col2 = st.columns(2)

with col1:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

with col2:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )


col1, col2 = st.columns(2)

with col1:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

with col2:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )


# ==================================================
# BILLING INFORMATION
# ==================================================

st.header("💳 Billing Information")

col1, col2 = st.columns(2)

with col1:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )


total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0
)


st.divider()


# ==================================================
# CREATE INPUT DATAFRAME
# ==================================================

input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})


# ==================================================
# PREDICTION
# ==================================================

st.header("🔮 Prediction")

if st.button("Predict Churn", type="primary", use_container_width=True):

    prediction = model.predict(input_data)[0]

    churn_probability = model.predict_proba(input_data)[0][1]

    stay_probability = 1 - churn_probability

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")

    else:
        st.success("✅ Customer is likely to stay")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{churn_probability:.2%}"
        )

    with col2:
        st.metric(
            "Stay Probability",
            f"{stay_probability:.2%}"
        )

    st.write("Churn Risk")
    st.progress(churn_probability)