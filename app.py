# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
import re

MODEL_PATH = "model/best_mobilenetv2_tomato.keras"
CLASS_INDEX_PATH = "class_indices.json"

# -----------------------------------------------
# Helper functions
# -----------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def normalize_label(s: str) -> str:
    """Normalize label strings for fuzzy matching."""
    if s is None:
        return ""
    s = s.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def prettify_label(s: str) -> str:
    """Make a label human-readable."""
    return s.replace("_", " ").replace("-", " ").title()

def load_labels_and_mapping(path=CLASS_INDEX_PATH):
    """Load class names from saved JSON."""
    with open(path, "r") as f:
        class_indices = json.load(f)
    labels = [k for k, _ in sorted(class_indices.items(), key=lambda x: x[1])]
    norm_to_orig = {normalize_label(k): k for k in labels}
    return labels, norm_to_orig

def preprocess_image(img: Image.Image, img_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def predict(img, model, labels):
    x = preprocess_image(img)
    probs = model.predict(x)[0]
    return dict(zip(labels, probs.tolist()))

# -----------------------------------------------
# Disease Information
# -----------------------------------------------
DISEASE_INFO = {
    "Bacterial spot": {
        "description": "A bacterial disease causing small, dark, water-soaked spots on leaves and fruit.",
        "prevention": [
            "Avoid overhead watering.",
            "Use disease-free seeds and transplants.",
            "Apply copper-based bactericides early."
        ]
    },
    "Early blight": {
        "description": "Caused by the fungus *Alternaria solani*, it leads to dark concentric spots on older leaves.",
        "prevention": [
            "Rotate crops annually.",
            "Avoid overhead irrigation.",
            "Remove infected plant debris."
        ]
    },
    "Late blight": {
        "description": "Caused by *Phytophthora infestans*, producing irregular gray-green lesions on leaves and fruit.",
        "prevention": [
            "Use resistant tomato varieties.",
            "Destroy infected plants immediately.",
            "Use fungicide treatments as needed."
        ]
    },
    "Leaf Mold": {
        "description": "Fungal disease that creates yellow spots on upper leaves and fuzzy mold on undersides.",
        "prevention": [
            "Ensure good air circulation.",
            "Avoid excessive humidity in greenhouses.",
            "Apply sulfur-based fungicides."
        ]
    },
    "Septoria leaf spot": {
        "description": "Caused by *Septoria lycopersici*, producing small circular spots with dark borders.",
        "prevention": [
            "Use clean seeds and tools.",
            "Water plants at the base.",
            "Remove infected leaves promptly."
        ]
    },
    "Spider mites Two-spotted spider mite": {
        "description": "Tiny pests causing stippling and yellowing of leaves, often forming webbing.",
        "prevention": [
            "Spray leaves with water to reduce mites.",
            "Introduce natural predators like ladybugs.",
            "Avoid drought stress."
        ]
    },
    "Target Spot": {
        "description": "Fungal disease causing concentric ring spots on leaves and fruit.",
        "prevention": [
            "Apply preventive fungicides.",
            "Avoid overhead irrigation.",
            "Rotate crops regularly."
        ]
    },
    "Tomato Yellow Leaf Curl Virus": {
        "description": "Viral disease spread by whiteflies, causing yellowing and curling of leaves.",
        "prevention": [
            "Control whitefly populations.",
            "Use virus-resistant tomato varieties.",
            "Remove infected plants immediately."
        ]
    },
    "Tomato mosaic virus": {
        "description": "A virus causing mottling and distortion of leaves and fruits.",
        "prevention": [
            "Wash hands and tools after handling plants.",
            "Avoid tobacco near tomatoes.",
            "Use resistant varieties."
        ]
    },
    "healthy": {
        "description": "Your tomato plant looks healthy! Keep up the good care.",
        "prevention": [
            "Continue balanced watering and sunlight.",
            "Inspect regularly for early signs of disease.",
            "Rotate crops each growing season."
        ]
    },
    "powdery mildew": {
        "description": "Fungal infection causing white, powdery growth on leaves and stems.",
        "prevention": [
            "Avoid overcrowding plants.",
            "Ensure good air circulation.",
            "Apply fungicidal sprays if necessary."
        ]
    }
}

DISEASE_INFO_LOOKUP = {normalize_label(k): v for k, v in DISEASE_INFO.items()}

# -----------------------------------------------
# Streamlit UI
# -----------------------------------------------
st.set_page_config(page_title="Tomato Disease Detector", page_icon="🍅", layout="centered")

# Add GitHub and LinkedIn links in the sidebar
st.sidebar.markdown(
    "### 🔗 Links\n"
    "[🍅 GitHub Repository](https://github.com/imdwipayana/Tomato_Disease_Detector)  \n"
    "[🔗 LinkedIn Profile](https://www.linkedin.com/in/yourprofile/)"
)

st.title("🍅 Tomato Disease Detector")
st.write("Upload a tomato leaf image to predict its health status and get prevention tips.")

uploaded = st.file_uploader("📤 Choose a tomato leaf image", type=["jpg", "jpeg", "png"])
model = load_model()
labels, norm_label_map = load_labels_and_mapping()

if uploaded:
    img = Image.open(io.BytesIO(uploaded.getvalue()))
    st.image(img, caption="Uploaded Image")

    with st.spinner("🔍 Analyzing image..."):
        preds = predict(img, model, labels)

    # Sort and keep top 3
    sorted_preds = sorted(preds.items(), key=lambda x: -x[1])[:3]

    st.subheader("📊 Prediction Results (Top 3)")
    for k, v in sorted_preds:
        pretty_name = prettify_label(k)
        st.write(f"**{pretty_name}** — {v:.2%}")
        st.progress(float(v))

    # Get top prediction
    top_label = sorted_preds[0][0]
    norm_top = normalize_label(top_label)
    info = DISEASE_INFO_LOOKUP.get(norm_top)

    st.markdown("---")
    if info:
        if norm_top == "healthy":
            st.success("✅ Your tomato plant appears healthy!")
        else:
            st.warning(f"⚠️ Possible **{prettify_label(top_label)}** detected.")

        st.markdown(f"### 🩺 About {prettify_label(top_label)}")
        st.write(info["description"])

        with st.expander("🌱 Prevention Tips", expanded=True):
            for tip in info["prevention"]:
                st.markdown(f"- {tip}")
    else:
        st.warning(f"No detailed information available for **{prettify_label(top_label)}**.")