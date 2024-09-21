import numpy as np
import pandas as pd
import streamlit as st
import pickle
import joblib
import os

# Define the artifacts path
artifacts_path = 'artifacts/'

# Load the trained model and encoder dictionary from the artifacts folder
model = joblib.load(os.path.join(artifacts_path, 'adult_income_random_forest_model.pkl'))
encoder_dict = pickle.load(open(os.path.join(artifacts_path, 'adult_income_encoder.pkl'), 'rb'))

# Load the feature names from the saved text file
with open(os.path.join(artifacts_path, 'adult_income_feature_names.txt'), 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

# Streamlit app definition
def main():
    st.title("Income Predictor")
    
    html_temp = """
    <div style="background-color:#025246;padding:10px">
    <h2 style="color:white;text-align:center;">Income Prediction App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Input fields for each feature, dynamically based on feature_names
    age = st.text_input("Age", "0")
    workclass = st.selectbox("Workclass", encoder_dict['workclass'].keys())
    education = st.selectbox("Education", encoder_dict['education'].keys())
    marital_status = st.selectbox("Marital Status", encoder_dict['marital-status'].keys())
    occupation = st.selectbox("Occupation", encoder_dict['occupation'].keys())
    relationship = st.selectbox("Relationship", encoder_dict['relationship'].keys())
    race = st.selectbox("Race", encoder_dict['race'].keys())
    gender = st.selectbox("Gender", encoder_dict['gender'].keys())
    capital_gain = st.text_input("Capital Gain", "0")
    capital_loss = st.text_input("Capital Loss", "0")
    hours_per_week = st.text_input("Hours per week", "0")
    native_country = st.selectbox("Native Country", encoder_dict['native-country'].keys())

    if st.button("Predict"):
        # Prepare data for prediction
        data = {
            'age': int(age),
            'workclass': encoder_dict['workclass'][workclass],
            'education': encoder_dict['education'][education],
            'marital-status': encoder_dict['marital-status'][marital_status],
            'occupation': encoder_dict['occupation'][occupation],
            'relationship': encoder_dict['relationship'][relationship],
            'race': encoder_dict['race'][race],
            'gender': encoder_dict['gender'][gender],
            'capital-gain': int(capital_gain),
            'capital-loss': int(capital_loss),
            'hours-per-week': int(hours_per_week),
            'native-country': encoder_dict['native-country'][native_country]
        }

        # Convert input data into DataFrame using the dynamic feature names
        df = pd.DataFrame([data])

        # Predict using the loaded model
        prediction = model.predict(df)

        # Display the result
        output = int(prediction[0])
        if output == 1:
            st.success('Predicted Income: >50K')
        else:
            st.success('Predicted Income: <=50K')

if __name__ == '__main__':
    main()