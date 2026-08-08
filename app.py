import streamlit as st
import numpy as np
import joblib

model  = joblib.load("heartDisease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("❤️ Heart Disease Prediction")

age      = st.slider("Age", 20, 80, 50)
sex      = st.selectbox("Sex", [0, 1],
                         format_func=lambda x: "Female" if x==0 else "Male")

cp       = st.selectbox("Chest Pain Type", [1, 2, 3, 4],
                         format_func=lambda x: {
                             1:"Typical Angina",
                             2:"Atypical Angina",
                             3:"Non-anginal Pain",
                             4:"Asymptomatic"}[x])

trestbps = st.slider("Resting BP",      80, 200, 130)
chol     = st.slider("Cholesterol",    100, 400, 200)
thalach  = st.slider("Max Heart Rate",  70, 210, 150)

exang    = st.selectbox("Exercise Angina", [0, 1],
                         format_func=lambda x: "No" if x==0 else "Yes")

oldpeak  = st.slider("ST Depression", 0.0, 6.0, 1.0)

slope    = st.selectbox("ST Slope", [1, 2, 3],
                         format_func=lambda x: {
                             1:"Upsloping",
                             2:"Flat",
                             3:"Downsloping"}[x])

ca       = st.selectbox("Major Vessels", [0, 1, 2, 3])

thal     = st.selectbox("Thalassemia", [3, 6, 7],
                         format_func=lambda x: {
                             3:"Normal",
                             6:"Fixed Defect",
                             7:"Reversible Defect"}[x])

if st.button("Predict"):
    data = np.array([[age, sex, cp, trestbps, chol,
                      thalach, exang, oldpeak,
                      slope, ca, thal]])

    data_scaled = scaler.transform(data)
    result      = model.predict(data_scaled)[0]
    prob        = model.predict_proba(data_scaled)[0][1] * 100

    st.markdown("---")
    if result == 1:
        st.error(f"⚠️ High Risk of Heart Disease — {prob:.1f}%")
    else:
        st.success(f"✅ Low Risk of Heart Disease — {prob:.1f}%")