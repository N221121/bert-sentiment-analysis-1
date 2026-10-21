
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./bert_sentiment_final"

st.set_page_config(
    page_title="BERT Sentiment Analyzer",
    page_icon="🎬"
)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review to predict its sentiment.")

review = st.text_area(
    "Your movie review",
    placeholder="The movie was amazing and the acting was excellent."
)

if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        inputs = tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        predicted_id = probabilities.argmax().item()
        sentiment = model.config.id2label[predicted_id]
        confidence = probabilities[predicted_id].item() * 100

        if sentiment.upper() == "POSITIVE":
            st.success(f"Sentiment: {sentiment}")
        else:
            st.error(f"Sentiment: {sentiment}")

        st.metric("Prediction confidence", f"{confidence:.2f}%")
