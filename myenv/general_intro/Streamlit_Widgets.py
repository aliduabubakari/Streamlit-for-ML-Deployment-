import streamlit as st

# Text input
text = st.text_input("Enter some text")
st.write("You entered:", text)

# Number input
number = st.number_input("Enter a number", min_value=0, max_value=100, value=50)
st.write("Number entered:", number)

# Slider
slider_value = st.slider("Choose a value", 0, 100, 50)
st.write("Slider value:", slider_value)

# Checkbox
checkbox = st.checkbox("Check me!")
st.write("Checkbox checked:", checkbox)

# Selectbox
option = st.selectbox("Choose an option", ["Option 1", "Option 2", "Option 3"])
st.write("Selected option:", option)

# Radio buttons
radio = st.radio("Select one", ["A", "B", "C"])
st.write("Radio selection:", radio)

# Date input
date = st.date_input("Select a date")
st.write("Selected date:", date)

# File uploader
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.write("Uploaded file name:", uploaded_file.name)
else:
    st.write("No file uploaded yet")