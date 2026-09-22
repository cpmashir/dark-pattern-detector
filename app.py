import streamlit as st
import joblib
import numpy as np
from PIL import Image
import pytesseract
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set local Windows Tesseract OCR path if running on laptop
tess_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tess_path):
    pytesseract.pytesseract.tesseract_cmd = tess_path
elif os.path.exists('/usr/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

st.set_page_config(
    page_title="Dark Pattern Detector", 
    page_icon="🛡️", 
    layout="centered"
)

# --- MODEL LOADING & CACHING ---
@st.cache_resource
def load_classical():
    model = joblib.load('best_model.pkl')
    vec = joblib.load('tfidf_vectorizer.pkl')
    return model, vec

@st.cache_resource
def load_distilbert():
    # Load directly from your live Hugging Face repository
    model_repo = "cpmashir/dark-pattern-distilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = AutoModelForSequenceClassification.from_pretrained(model_repo)
    model.eval()
    return tokenizer, model

# Load models
classical_model, classical_vectorizer = load_classical()
distil_tokenizer, distil_model = load_distilbert()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Model Selector")
model_choice = st.sidebar.selectbox(
    "Select Active Architecture:",
    ("Classical ML (LinearSVC - Fast)", "Deep Learning (Fine-Tuned DistilBERT)")
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Benchmarks:**
* **LinearSVC:** Macro F1: `0.9385` (<15ms latency)
* **DistilBERT:** Macro F1: `0.9745` (Contextual self-attention)
""")

st.title("🛡️ Deceptive UI & Dark Pattern Detector")
st.caption(f"Active Engine: **{model_choice}**")

tab1, tab2 = st.tabs(["📝 Text Copy Audit", "🖼️ UI Screenshot OCR"])

def run_inference(text_input):
    st.markdown("---")
    st.subheader("Audit Verdict")

    if "LinearSVC" in model_choice:
        # 1. Classical Prediction via TF-IDF
        vec = classical_vectorizer.transform([text_input])
        pred_raw = classical_model.predict(vec)[0]
        label = "Dark Pattern" if str(pred_raw) in ["1", "Dark Pattern"] else "Not Dark Pattern"

        # Confidence approximation via decision boundary distance
        if hasattr(classical_model, "decision_function"):
            dec = classical_model.decision_function(vec)
            margin = abs(dec[0]) if dec.ndim == 1 else np.max(dec)
            confidence = float((1 / (1 + np.exp(-margin))) * 100)
        else:
            confidence = 92.0

    else:
        # 2. Deep Learning Prediction via DistilBERT
        inputs = distil_tokenizer(
            text_input, 
            return_tensors="pt", 
            truncation=True, 
            max_length=64
        )
        # Prevent token_type_ids crash in DistilBERT
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            outputs = distil_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = float(probs[0][pred_idx].item() * 100)
            label = distil_model.config.id2label[pred_idx]

    # Render results
    if str(label).lower() in ["not dark pattern", "not a dark pattern", "0"]:
        st.success(f"✅ **Safe / Informational Copy** (Class: `{label}`)")
    else:
        st.error(f"⚠️ **Deceptive Pattern Detected!** (Class: `{label}`)")

    conf_clamped = min(max(int(confidence), 10), 100)
    st.progress(conf_clamped)
    st.caption(f"Confidence: **{confidence:.2f}%** | Analyzed by: {model_choice}")

# TAB 1: Text Copy Input
with tab1:
    user_text = st.text_area(
        "Paste website banner or button text:", 
        height=100,
        placeholder="e.g. Hurry! Only 2 items left at this price!"
    )
    if st.button("Audit Text", key="btn_text", width="stretch"):
        if user_text.strip():
            run_inference(user_text)
        else:
            st.warning("Please type or paste some text first.")

# TAB 2: Screenshot OCR Input
with tab2:
    img_upload = st.file_uploader(
        "Upload interface screenshot (.png, .jpg, .jpeg)", 
        type=["png", "jpg", "jpeg"]
    )
    if img_upload:
        image = Image.open(img_upload)
        image.thumbnail((800, 800))
        st.image(image, caption="Uploaded Interface Sample", width="stretch")

        if st.button("Extract Text & Audit", key="btn_ocr", width="stretch"):
            with st.spinner("Extracting text via Tesseract OCR..."):
                try:
                    extracted = pytesseract.image_to_string(image).strip()
                except Exception as err:
                    st.error(f"OCR Error: {err}")
                    extracted = ""

            if extracted:
                st.markdown(f"**Extracted Text:** *\"{extracted}\"*")
                run_inference(extracted)
            else:
                st.error("No legible text detected from image.")