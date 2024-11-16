#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

# Load the model
model = joblib.load('CatBoost.pkl')

# Define feature options
RALR_options = { 
    0: 'No',
    1: 'Yes'
}

# Define feature names
feature_names = ["BMI","CHE", "Age", "RALR", "Blood_loss", "Surgical_duration", "PT", "AST"]

# Streamlit user interface
st.title("Predictive Model for Complications after Anatomical Liver Resection")

# RALR: categorical selection
RALR = st.selectbox("Robotic-Assisted Anatomic Liver Resection (RALR) (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Blood loss: numerical input
Blood_loss = st.number_input("Blood loss:")

# Surgical duration: numerical input
Surgical_duration = st.number_input("Surgical duration:")

# CHE: numerical input
CHE = st.number_input("Cholinesterase (CHE):")

# BMI: numerical input
BMI = st.number_input("Body Mass Index (BMI):")

# PT: numerical input
PT = st.number_input("Prothrombin time (PT):")

# AST: numerical input
AST = st.number_input("Aspartate Aminotransferase (AST):")

# age: numerical input
Age = st.number_input("Age:")


# Process inputs and make predictions
feature_values = [BMI, CHE, Age, RALR, Blood_loss, Surgical_duration, PT, AST]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
        
    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100
    
    # Display prediction results
    st.write(f"**Based on feature values, predicted probability of complications after ALR is:** {probability:.2f}%")
    
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of postoperative complications following anatomical liver resection. "
            f"The model predicts that your probability of having postoperative complications is {probability:.2f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
        )
    else:
        advice = (
            f"According to our model, you have a low risk of postoperative complications following anatomical liver resection. "
            f"The model predicts that your probability of not having postoperative complications is {probability:.2f}%. "
            "However, enhancing perioperative management is still very important. "
        )
    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")

