import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Split data
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Streamlit app
st.title("Iris Flower Species Classifier")

# Input sliders
st.sidebar.header('Input Features')
sepal_length = st.sidebar.slider('Sepal Length (cm)', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider('Sepal Width (cm)', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider('Petal Length (cm)', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
petal_width = st.sidebar.slider('Petal Width (cm)', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))

# Make prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]]).reshape(1, -1)  # Ensure correct shape
prediction = clf.predict(input_data)
prediction_proba = clf.predict_proba(input_data)

# Display prediction
st.subheader('Prediction')
st.write(f"Predicted species: {iris.target_names[prediction][0]}")

# Display prediction probabilities
st.subheader('Prediction Probability')
st.write(f"Setosa: {prediction_proba[0][0]:.2f}, Versicolor: {prediction_proba[0][1]:.2f}, Virginica: {prediction_proba[0][2]:.2f}")

# Scatter plot
st.subheader('Scatter Plot of Sepal Length vs Sepal Width')

fig, ax = plt.subplots()
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
colors = ['red', 'green', 'blue']

for i, species in species_map.items():
    subset = df[df['target'] == i]
    ax.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], label=species, color=colors[i])

ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.legend()

st.pyplot(fig)  # Ensure the figure is passed correctly