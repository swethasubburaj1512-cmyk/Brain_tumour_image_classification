%%writefile /content/app.py

# ⚙️  STEP 2:  Import libraries
import os, json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from pyngrok import ngrok

# ------------------------------------------------------------
# 🔧 CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")

MODEL_PATH = "/content/models_outputs/MobileNetV2_best.h5"  # <-- adjust if needed
IMG_SIZE   = (224, 224)
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

st.title("🧠 Brain Tumor Classification")
st.write("Upload a **brain MRI image** below to classify the tumor type.")

# ------------------------------------------------------------
# 🧩 LOAD TRAINED MODEL
# ------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    return tf.keras.models.load_model(path)

with st.sidebar:
    st.header("Model")
    st.caption("Ensure the model path exists below 👇")
    st.code(MODEL_PATH, language="text")

try:
    assert os.path.exists(MODEL_PATH), f"Model not found at:\n{MODEL_PATH}"
    model = load_model(MODEL_PATH)
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Model load failed:\n{e}")
    st.stop()

# ------------------------------------------------------------
# 🔧 HELPERS
# ------------------------------------------------------------
def preprocess_image(pil_img: Image.Image, target_hw=(224, 224)) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    arr = tf.image.resize(arr, target_hw)
    arr = tf.cast(arr, tf.float32) / 255.0
    arr = tf.expand_dims(arr, 0)
    return arr.numpy()

def safe_softmax(vec: np.ndarray) -> np.ndarray:
    x = vec - np.max(vec, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

# ------------------------------------------------------------
# 📤 IMAGE UPLOAD SECTION
# ------------------------------------------------------------
uploaded_file = st.file_uploader("Upload an MRI Image (jpg/png/jpeg)",
                                 type=["jpg","jpeg","png"])

if not uploaded_file:
    st.info("👆 Please upload a brain MRI image to begin prediction.")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception as e:
    st.error(f"Could not read the image file: {e}")
    st.stop()

st.image(image, caption="🩺 Uploaded MRI Image", use_container_width=True)

# ------------------------------------------------------------
# 🔮 PREDICT
# ------------------------------------------------------------
img_batch = preprocess_image(image, IMG_SIZE)

try:
    preds = model.predict(img_batch)
    preds = np.array(preds)
    if preds.ndim == 2 and (preds.min() < 0 or preds.max() > 1):
        preds = safe_softmax(preds)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

if preds.ndim == 1:
    probs = preds
else:
    probs = preds[0]

pred_class = int(np.argmax(probs))
pred_label = CLASS_NAMES[pred_class] if len(CLASS_NAMES)==len(probs) else f"class_{pred_class}"
confidence = float(probs[pred_class])

# ------------------------------------------------------------
# 🎯 DISPLAY PREDICTION RESULTS
# ------------------------------------------------------------
st.subheader("🧾 Prediction Results")
st.success(f"**Predicted Tumor Type:** {pred_label}")
st.write(f"**Confidence:** {confidence*100:.2f}%")

st.markdown("### 🔍 Model Confidence per Class")
conf_data = { (CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}") : float(probs[i])
              for i in range(len(probs)) }
st.bar_chart(conf_data)

result = {
    "predicted_label": pred_label,
    "confidence": confidence,
    "per_class": conf_data,
}
st.download_button("⬇️ Download result (JSON)",
                   data=json.dumps(result, indent=2),
                   file_name="prediction_result.json",
                   mime="application/json")

# ------------------------------------------------------------
# 📚 ABOUT SECTION
# ------------------------------------------------------------
with st.expander("ℹ️ About this App"):
    st.write("""
This app uses a **deep learning model** trained on brain MRI images
to classify tumors into four categories:
- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

**Framework:** TensorFlow + Streamlit
**Model file:** MobileNetV2_best.h5
""")
