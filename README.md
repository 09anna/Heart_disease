# 🫀Heart Disease Prediction

## Project Overview

This project aims to to predict whether a patient has heart disease or not.



                                                         ---------- *** ----------

## Dataset

The Dataset used in this project is the Cleaveland's UCI Heart Disease Dataset.

Dataset Source: https://archive.ics.uci.edu/dataset/45/heart+disease

The original database contains 76 attributes collected from patient medical records. However,  most Machine Learning researchers use only 14 important attributes.

The dataset contains patient medical information such as:
- Age
- Sex
- cp: Chest pain type
- trestbps: Resting blood pressure
- chol: Serum cholesterol
- fbs: Fasting blood sugar
- restecg: resting ECG results
- thalach: Maximum heart rate achieved
- exang: Exercise induced angina
- oldpeak: ST depression induced by exercise relative to rest
- slope: slope of peak exercise ST segment
- ca: number of major vessels colored by flourosopy
- thal: thalassemia (blood condition information)
- Target: Diagnosis of heart disease
    - 0 -> No heart disease
    - 1 -> Heart disease

---

## Project Workflow

This project follows the following workflow:
- Data Collection and Exploration
- Data Cleaning and Transformation
- Exploratory Data Analysis
- Feature Selection
- Model Development
- Hypermeter tuning
- Model Deployment

---

## Technologies Used
- Jupyter Notebook
- Python
- Pandas
- Numpy
- Seaborn
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib

---

## How to Run

1. Install dependencies
2. Run the streamlit app
3. Open in browser
4. Use the App
- Fill in the patient details using the sliders and dropdowns
- Click the Prredict button
- View the result - High Risk or Low Risk with probability

## Key Insights
- Max Heart Rate is the strongest predictor
![alt text](image-1.png)
- 
