import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# ==============================
# Loading the Pre-trained Model
# ==============================

# Load the pre-trained Linear Regression model saved earlier using joblib
# The model was trained on the California Housing dataset
# Define the path to the artifacts
artifact_path = 'artifacts'

# Load the trained model
model = joblib.load(os.path.join(artifact_path, 'california_linear_regression_model.pkl'))

#model = joblib.load('california_linear_regression_model.pkl')

# ==============================
# Load Dataset for Reference
# ==============================

# Fetch the California Housing dataset again for reference
# This is just for getting feature ranges and descriptions
california = fetch_california_housing()

# Create a DataFrame for easier data handling
df = pd.DataFrame(data=california.data, columns=california.feature_names)

# ==============================
# Streamlit App Interface
# ==============================

# Set the title for the web app
st.title("California Housing Price Predictor")

# Sidebar for user input
# We use sliders so the user can select different values for each feature of the dataset
st.sidebar.header('Input Features')

# Sliders for the user to input values for each of the features (Median Income, House Age, etc.)
# These values will be used as inputs for the model to predict the house price

# Median Income: Slider allows the user to input the median income in $10,000 increments
median_income = st.sidebar.slider('Median Income (10k USD)', 
                                  float(df['MedInc'].min()),  # minimum value from dataset
                                  float(df['MedInc'].max()),  # maximum value from dataset
                                  float(df['MedInc'].mean())) # default value (mean)

# House Age: Slider allows the user to input the age of the house
house_age = st.sidebar.slider('House Age (years)', 
                              float(df['HouseAge'].min()), 
                              float(df['HouseAge'].max()), 
                              float(df['HouseAge'].mean()))

# Average Rooms per Household: Slider for the average number of rooms per household
avg_rooms = st.sidebar.slider('Average Rooms per Household', 
                              float(df['AveRooms'].min()), 
                              float(df['AveRooms'].max()), 
                              float(df['AveRooms'].mean()))

# Average Bedrooms per Household: Slider for the average number of bedrooms per household
avg_bedrooms = st.sidebar.slider('Average Bedrooms per Household', 
                                 float(df['AveBedrms'].min()), 
                                 float(df['AveBedrms'].max()), 
                                 float(df['AveBedrms'].mean()))

# Population: Slider for the population in the area
population = st.sidebar.slider('Population', 
                               float(df['Population'].min()), 
                               float(df['Population'].max()), 
                               float(df['Population'].mean()))

# Average Occupancy: Slider for the average number of people per household
avg_occupancy = st.sidebar.slider('Average Occupancy', 
                                  float(df['AveOccup'].min()), 
                                  float(df['AveOccup'].max()), 
                                  float(df['AveOccup'].mean()))

# Latitude: Slider for the latitude (geographical location)
latitude = st.sidebar.slider('Latitude', 
                             float(df['Latitude'].min()), 
                             float(df['Latitude'].max()), 
                             float(df['Latitude'].mean()))

# Longitude: Slider for the longitude (geographical location)
longitude = st.sidebar.slider('Longitude', 
                              float(df['Longitude'].min()), 
                              float(df['Longitude'].max()), 
                              float(df['Longitude'].mean()))

# ==============================
# Prepare Input Data for Prediction
# ==============================

# We create an array with the input values provided by the user
# This input will be passed to the model for prediction
input_data = np.array([[median_income, house_age, avg_rooms, avg_bedrooms, population, avg_occupancy, latitude, longitude]])

# ==============================
# Make Prediction
# ==============================

# The trained model takes the input data and predicts the median house value
# The prediction result is returned in terms of 100,000s of USD, so we multiply it by 100,000 to display it in dollars
prediction = model.predict(input_data)

# Display the predicted price to the user
st.subheader('Predicted Median House Value')
st.write(f"Predicted Price: ${prediction[0]*100000:.2f}")  # Display in full dollar amounts

# ==============================
# Visualize Regression Line (Median Income vs House Price)
# ==============================

# This section will create a visual plot that shows how the model predicts house prices based on median income
# The red line shows the predictions made by the model, while the scatter points represent actual house prices in the dataset

# Subheader for the regression plot
st.subheader('Regression Plot: Median Income vs House Price')

# We generate a range of values for the median income from its minimum to its maximum
median_income_range = np.linspace(df['MedInc'].min(), df['MedInc'].max(), 100).reshape(-1, 1)

# For each value of median income, we keep the other input features constant (using the user's values for those)
# This allows us to isolate the effect of median income on the predicted house price
other_features = np.tile([house_age, avg_rooms, avg_bedrooms, population, avg_occupancy, latitude, longitude], (100, 1))
plot_input = np.hstack([median_income_range, other_features])  # Combine median income with the other fixed features

# Get the predicted house prices for the median income range using the model
predicted_prices = model.predict(plot_input)

# Plot the actual median house prices vs median income (scatter plot)
# The red line shows the predicted house prices based on median income
fig, ax = plt.subplots()
ax.scatter(df['MedInc'], california.target, label="Actual Prices", alpha=0.3)  # Actual data points
ax.plot(median_income_range, predicted_prices, color='red', label="Regression Line")  # Regression line (model predictions)

# Labeling the axes
ax.set_xlabel("Median Income (10k USD)")
ax.set_ylabel("Median House Price (in $100k)")
ax.legend()  # Add legend to distinguish between actual prices and regression line

# Show the plot in the Streamlit app
st.pyplot(fig)