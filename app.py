import streamlit as st
import numpy as np
import pickle, json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ---------- Load assets (cached for performance) ----------
@st.cache_resource
def load_keras_model():
    # If mismatch errors occur, try: load_model("sentiment_model.h5", compile=False)
    m = load_model("sentiment_model.h5")
    return m

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

model = load_keras_model()
tokenizer = load_tokenizer()
le = load_label_encoder()
cfg = load_config()
MAXLEN = int(cfg.get("maxlen", 65))  # fallback if not present

# ---------- UI ----------
st.set_page_config(page_title="Sentiment Analysis", page_icon="🗣️", layout="centered")
st.title("🗣️ Sentiment Analysis ")
st.write("Type a product review,  click **Predict** to see the sentiment.")

# Use a form so pressing Enter submits
with st.form("predict_form", clear_on_submit=False):
    user_text = st.text_area("Your review:", height=140, placeholder="Type here...")
    submitted = st.form_submit_button("Predict")

def predict(text: str):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAXLEN, padding='post')
    probs = model.predict(pad)
    idx = int(np.argmax(probs, axis=1)[0])
    label = le.inverse_transform([idx])[0]
    conf = float(np.max(probs))  # confidence of top class
    return label, conf

if submitted:
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        label, conf = predict(user_text)
        # Colorize result
        color = {"positive": "green", "neutral": "orange", "negative": "red"}.get(label.lower(), "blue")
        st.markdown(f"**Predicted Sentiment:** <span style='color:{color}'>{label}</span>  \n**Confidence:** {conf:.3f}", unsafe_allow_html=True)
