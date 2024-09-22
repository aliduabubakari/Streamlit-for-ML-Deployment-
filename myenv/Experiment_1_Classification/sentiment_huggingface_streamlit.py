import streamlit as st
from transformers import pipeline, AutoTokenizer
import torch

# ==============================
# Title and Description
# ==============================
st.title("Sentiment Analysis with Hugging Face")
st.write("This app uses a pre-trained DistilBERT model from Hugging Face for sentiment analysis. "
         "Enter some text, and click 'Predict' to analyze the sentiment.")

# ==============================
# Check for GPU Availability
# ==============================
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    st.write("✅ GPU detected! Using GPU for faster processing.")
else:
    st.write("⚠️ No GPU detected. Using CPU for processing.")

# ==============================
# Load the Pre-trained Model
# ==============================
# Using the DistilBERT model fine-tuned for sentiment analysis (SST-2)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

# Load the tokenizer with clean_up_tokenization_spaces set to True (to avoid warnings)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", clean_up_tokenization_spaces=True)

# ==============================
# Input for User Text
# ==============================
user_input = st.text_area("Enter some text to analyze the sentiment:")

# ==============================
# Predict Button
# ==============================
if st.button("Predict"):
    if user_input:
        with st.spinner('Analyzing sentiment...'):
            # Perform the sentiment analysis when the button is clicked
            result = sentiment_pipeline(user_input)[0]

        # Display the result
        st.subheader("Sentiment Prediction")
        sentiment = result['label']
        confidence = result['score']
        
        if sentiment == "POSITIVE":
            st.success(f"Predicted Sentiment: {sentiment} 😊")
        else:
            st.error(f"Predicted Sentiment: {sentiment} 😞")
        
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.warning("Please enter some text to analyze the sentiment.")

# ==============================
# GPU Status Display
# ==============================
if device == 0:
    st.write("Model is running on GPU for fast predictions.")
else:
    st.write("Model is running on CPU.")