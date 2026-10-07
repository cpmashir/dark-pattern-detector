import streamlit as st
import joblib
import numpy as np
from PIL import Image
import pytesseract
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Platform-independent Tesseract path
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
    cats = joblib.load('categories.pkl')
    return model, vec, cats

@st.cache_resource
def load_distilbert():
    model_repo = "cpmashir/dark-pattern-distilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = AutoModelForSequenceClassification.from_pretrained(model_repo)
    model.eval()
    return tokenizer, model

classical_model, classical_vectorizer, categories = load_classical()
distil_tokenizer, distil_model = load_distilbert()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Model Selector")
model_choice = st.sidebar.selectbox(
    "Select Active Architecture:",
    ("Classical ML (LinearSVC - Fast)", "Deep Learning (Fine-Tuned DistilBERT)")
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Trained Categories (8 Classes):**")
for cat in categories:
    st.sidebar.markdown(f"- {cat}")

st.title("🛡️ Deceptive UI & Dark Pattern Detector")
st.caption(f"Active Engine: **{model_choice}**")

tab1, tab2 = st.tabs(["📝 Text Copy Audit", "🖼️ UI Screenshot OCR"])

def run_inference(text_input):
    st.markdown("---")
    st.subheader("Audit Verdict")

    pred_category = ""
    confidence = 0.0

    if "LinearSVC" in model_choice:
        vec = classical_vectorizer.transform([text_input])
        pred_idx = int(classical_model.predict(vec)[0])
        pred_category = categories[pred_idx]

        # Multi-class confidence score via softmax on decision boundaries
        if hasattr(classical_model, "decision_function"):
            decision = classical_model.decision_function(vec)
            scores = decision[0] if decision.ndim > 1 else decision
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / np.sum(exp_scores)
            confidence = float(probs[pred_idx] * 100) if decision.ndim > 1 else 92.0
        else:
            confidence = 90.0

    else:
        inputs = distil_tokenizer(
            text_input, 
            return_tensors="pt", 
            truncation=True, 
            max_length=64
        )
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            outputs = distil_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = float(probs[0][pred_idx].item() * 100)

            if hasattr(distil_model.config, "id2label") and distil_model.config.id2label:
                pred_category = str(distil_model.config.id2label.get(pred_idx, pred_idx))
            else:
                pred_category = categories[pred_idx]

    # --- RENDER RESULTS ---
    if pred_category.strip().lower() in ["not dark pattern", "safe"]:
        st.success(f"✅ **Safe / Informational Copy**")
        st.info(f"**Classification:** `{pred_category}`")
    else:
        st.error(f"⚠️ **Deceptive Pattern Detected!**")
        st.markdown(f"**Identified Category:** `{pred_category}`")

    conf_clamped = min(max(int(confidence), 10), 100)
    st.progress(conf_clamped)
    st.caption(f"Confidence: **{confidence:.2f}%** | Architecture: **{model_choice}**")

# TAB 1: Direct Text Input
with tab1:
    user_text = st.text_area(
        "Paste website banner or button text:", 
        height=100,
        placeholder="e.g. Hurry! Only 2 items left at this price!"
    )
    if st.button("Audit Text", key="btn_text", use_container_width=True):
        if user_text.strip():
            run_inference(user_text)
        else:
            st.warning("Please type or paste some text first.")

# TAB 2: Image OCR Input
with tab2:
    img_upload = st.file_uploader(
        "Upload interface screenshot (.png, .jpg, .jpeg)", 
        type=["png", "jpg", "jpeg"]
    )
    if img_upload:
        image = Image.open(img_upload)
        image.thumbnail((800, 800))
        st.image(image, caption="Uploaded Interface Sample", use_container_width=True)

        if st.button("Extract Text & Audit", key="btn_ocr", use_container_width=True):
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