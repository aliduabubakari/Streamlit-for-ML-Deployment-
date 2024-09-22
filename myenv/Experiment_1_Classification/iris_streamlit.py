import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# ==============================
# Load the Trained Model
# ==============================
# The pre-trained Random Forest model is saved in a directory called 'artifacts'. 
# The model was trained earlier to classify iris species based on flower measurements (sepal and petal dimensions).
# 'joblib.load()' is used to load the saved model from the file.
artifact_path = 'artifacts'
model = joblib.load(os.path.join(artifact_path, 'iris_random_forest_model.pkl'))

# ==============================
# Streamlit App Layout
# ==============================

# Streamlit allows us to create a web interface for interacting with our model.
# 'st.title()' sets the title of the app that will be displayed at the top of the page.
st.title("Iris Flower Species Classifier")

# ==============================
# Sidebar Inputs for Flower Features
# ==============================
# We use the Streamlit sidebar to create input widgets where users can provide values for sepal length, 
# sepal width, petal length, and petal width (the features used for classification).

# Sidebar section for user inputs
st.sidebar.header('Input Features')

# Four sliders allow users to input values for the four flower features. 
# The default values are chosen to represent typical flower measurements, but users can adjust these as they wish.
# Each slider has a minimum value of 0 and a maximum value of 10, though real iris flowers tend to fall within a narrower range.
sepal_length = st.sidebar.slider('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.sidebar.slider('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.sidebar.slider('Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.sidebar.slider('Petal Width (cm)', min_value=0.0, max_value=10.0, value=1.3)

# Collect the input values into a list.
# This list represents the input features (sepal and petal measurements) for a single flower instance.
feature_values = [sepal_length, sepal_width, petal_length, petal_width]

# ==============================
# Prepare the Input for Prediction
# ==============================
# The machine learning model expects the input to be in the form of a 2D array (even if it’s just one flower),
# so we use np.array() to create a NumPy array with the input features.
input_data = np.array([feature_values])

# ==============================
# Make Predictions Using the Model
# ==============================
# The model predicts the species of the iris flower based on the input values.
# 'model.predict()' returns the predicted class (species), and 'model.predict_proba()' gives the probabilities
# of each species (setosa, versicolor, virginica).
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# List of species names corresponding to the prediction classes
# (0 = setosa, 1 = versicolor, 2 = virginica).
species_names = ['setosa', 'versicolor', 'virginica']

# ==============================
# Display the Prediction Result
# ==============================
# 'st.subheader()' is used to create a section title for displaying the prediction result.
# We use 'st.write()' to show which species was predicted by the model.
st.subheader('Prediction')
st.write(f"Predicted species: {species_names[prediction[0]]}")

# Use argmax() to find the index of the highest probability.
# This gives the index of the species with the highest predicted probability.
#predicted_index = np.argmax(prediction_proba)

# Get the corresponding species name using the index.
#predicted_species = species_names[predicted_index]

# ==============================
# Display the Prediction Result
# ==============================
# Show the predicted species using argmax to identify the species with the highest probability.
#st.subheader('Prediction')
#st.write(f"Predicted species: {predicted_species}")

# ==============================
# Display Prediction Probabilities
# ==============================
# The model not only predicts a species but also gives probabilities for each species.
# These probabilities indicate the model’s confidence in its prediction for each species.
# We'll visualize this using a bar chart.

# Create a new section to show the prediction probabilities
st.subheader('Prediction Probabilities')

# Create a bar chart using Matplotlib to show the probability distribution for each species.
# The 'prediction_proba' array contains the probabilities of the flower belonging to each species.
fig, ax = plt.subplots()

# 'ax.bar()' creates a bar chart where each species name is on the x-axis and the corresponding probability on the y-axis.
ax.bar(species_names, prediction_proba[0])
ax.set_ylabel('Probability')
ax.set_title('Species Prediction Probabilities')

# Customize the chart to improve readability.
# Set the y-axis limit from 0 to 1 because probabilities range between 0 and 1.
plt.ylim(0, 1)

# Annotate each bar with its corresponding probability value.
# 'ax.text()' places text on the bars, displaying the actual probability value above each bar.
for i, v in enumerate(prediction_proba[0]):
    ax.text(i, v + 0.01, f'{v:.2f}', ha='center')

# 'st.pyplot()' renders the Matplotlib figure in the Streamlit app.
st.pyplot(fig)

# ==============================
# Display Numerical Probabilities
# ==============================
# For further clarity, we can also show the numerical values of the probabilities.
# 'st.write()' displays each species and its corresponding probability in a readable format.
st.write("Numerical probabilities:")
for species, prob in zip(species_names, prediction_proba[0]):
    st.write(f"{species}: {prob:.2f}")