import streamlit as st
import time
import numpy as np
import pandas as pd

# ==========================
# Title and Welcome Message
# ==========================
st.title("Hello, Streamlit!")
st.write("Welcome to your interactive Streamlit app.")

# ==========================
# Name Input and Greeting
# ==========================
name = st.text_input("What's your name?")
if name:
    st.write(f"Hello, {name}! We're glad to have you here.")

# ==========================
# Simple Counter with Stateful Clicks
# ==========================
st.write("### Counter Button")

if 'counter' not in st.session_state:
    st.session_state.counter = 0

if st.button("Click me!"):
    st.session_state.counter += 1
st.write(f"Button clicked: {st.session_state.counter} times")

# ==========================
# Dynamic Slider Interaction
# ==========================
st.write("### Dynamic Slider Example")
age = st.slider("How old are you?", min_value=0, max_value=100, value=25)
st.write(f"You're {age} years old.")

# ==========================
# Selectbox for Favorite Color
# ==========================
st.write("### Choose Your Favorite Color")
color = st.selectbox("Pick a color:", ["Red", "Blue", "Green", "Yellow", "Purple"])
st.write(f"Your favorite color is {color}.")

# ==========================
# Progress Bar Example
# ==========================
st.write("### Progress Bar Example")
progress_bar = st.progress(0)
for i in range(100):
    time.sleep(0.02)
    progress_bar.progress(i + 1)
st.write("Progress complete!")

# ==========================
# Interactive Charts with Random Data
# ==========================
st.write("### Simple Data Visualization")
data = pd.DataFrame(np.random.randn(20, 3), columns=["A", "B", "C"])

st.line_chart(data)
st.area_chart(data)
st.bar_chart(data)

# ==========================
# Final Goodbye Message
# ==========================
st.write("Thanks for exploring Streamlit with us!")