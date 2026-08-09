import streamlit as st #importing streamlit for user inkterface and web app development.
import numpy as np #importing numpy for numerical operations.
import joblib #importing joblib to load the trained model and scaler.

model  = joblib.load("heartDisease_model.pkl") #loading the trained model to use it for predictions.
scaler = joblib.load("scaler.pkl") #loading the fitted scaler to scale the inpput data before making a prediction.

st.title(" Heart Disease Prediction") #sets title of the web app.

age      = st.slider("Age", 20, 80, 50) # add slider for age input.

#selectbox for sex input with options 0 for Female and 1 for Male
sex      = st.selectbox("Sex", ["--Select--",0, 1], format_func=lambda x:"--Select--" if x=="--Select--" else ("Female" if x==0 else "Male"))

#selectbox for chest pain type input with options 1 for Typical Angina, 2 for Atypical Angina, 3 for Non-anginal Pain, and 4 for Asymptomatic.
cp       = st.selectbox("Chest Pain Type", [1, 2, 3, 4],format_func=lambda x: {1:"Typical Angina", 2:"Atypical Angina",3:"Non-anginal Pain",4:"Asymptomatic"}[x])

trestbps = st.slider("Resting BP",      80, 200, 130) # add slider for resting blood pressure  where min vlaue is 80, max value is 200 and default value is set as 130.
chol     = st.slider("Cholesterol",    100, 400, 200) # add slider for cholesterol  where min value is 100, max value is 400 and default value is set as 200.
thalach  = st.slider("Max Heart Rate",  70, 210, 150) # add slider for maximum heart rate  where min value is 70, max value is 210 and default value is set as 150.

#selectbox for exercise induced angina input with options 0 for No and 1 for Yes.
exang    = st.selectbox("Exercise Angina", [0, 1],format_func=lambda x: "No" if x==0 else "Yes")

oldpeak  = st.slider("ST Depression", 0.0, 6.0, 1.0) # adding slider for ST depression  where min value is 0.0, max value is 6.0 and default value is set as 1.0.

# add selectbox for slope with options 1 =unsloping, 2= flat, 3= downsloping.
slope    = st.selectbox("ST Slope", [1, 2, 3],format_func=lambda x: {1:"Upsloping",2:"Flat",3:"Downsloping"}[x])

# add selectbox for major vessels with options 0, 1, 2, 3.
ca       = st.selectbox("Major Vessels", [0, 1, 2, 3])

# add selectbox for thalassemia with options 3 = Normal, 6=Fixed Defect, 7=Reversible Defect.
thal     = st.selectbox("Thalassemia", [3, 6, 7],format_func=lambda x: {3:"Normal",6:"Fixed Defect",7:"Reversible Defect"}[x])


#create a predict button to display the result on the web app. 
if st.button("Predict"):

    #numpy array containing the input for predictions.
    data = np.array([[age, sex, cp, trestbps, chol,thalach, exang, oldpeak,slope, ca, thal]])

    data_scaled = scaler.transform(data) #Scaling the input data using the fitted scaler used during model training.
    result      = model.predict(data_scaled)[0] # Using th trained model to predict whether the patient has heart diesease.
    prob        = model.predict_proba(data_scaled)[0][1] * 100 #Probability of the patient having heart disease.

    # checking if the model predicts heart disease or not
    if result == 1: # displays a red warning message if th model predicts that the patient has heart disease.
        st.error(f"⚠️ High Risk of Heart Disease — {prob:.1f}%")
    else: # else green success message is displayed.
        st.success(f"✅ Low Risk of Heart Disease — {prob:.1f}%")