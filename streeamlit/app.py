import os
import joblib
import numpy as np
import streamlit as st

# ================================
# ✅ LOAD TRAINED MODEL
# ================================
MODEL_PATH = "/output/ieee_fraud_model/model.pkl"

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Real-Time Fraud Detection System")
st.write("Enter transaction details and predict fraud probability.")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found at: {MODEL_PATH}")
    st.stop()
else:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully")

# ================================
# ✅ INPUT FIELDS
# ================================
st.subheader("🔢 Enter Transaction Features")

TransactionAmt = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
dist1 = st.number_input("Distance (dist1)", min_value=0.0, value=1.0)

C1 = st.number_input("C1", min_value=0.0, value=1.0)
C2 = st.number_input("C2", min_value=0.0, value=1.0)
C3 = st.number_input("C3", min_value=0.0, value=1.0)

D1 = st.number_input("D1", min_value=0.0, value=1.0)
D2 = st.number_input("D2", min_value=0.0, value=1.0)

V1 = st.number_input("V1", min_value=0.0, value=1.0)
V2 = st.number_input("V2", min_value=0.0, value=1.0)
V3 = st.number_input("V3", min_value=0.0, value=1.0)

# ✅ MUST MATCH MODEL FEATURE ORDER EXACTLY
input_data = np.array([[ 
    TransactionAmt,
    dist1,
    C1, C2, C3,
    D1, D2,
    V1, V2, V3
]])

# ================================
# ✅ PREDICTION
# ================================
if st.button("🚨 Predict Fraud"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("🧾 Prediction Result")

    if prediction == 1:
        st.error("⚠️ FRAUD DETECTED")
    else:
        st.success("✅ Transaction is SAFE")

    st.write(f"### 📊 Fraud Probability: **{round(probability * 100, 2)}%**")
