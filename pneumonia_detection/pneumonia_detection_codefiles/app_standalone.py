import os
import numpy as np
import joblib
import cv2
import pydicom
import streamlit as st
from PIL import Image
import io

st.set_page_config(page_title="Pneumonia Detection Lab", layout="wide")

# Direct Local Model Loading
MODEL_PATH = "pneumonia_detection/deployment/pneumonia_detection_model_v1_0.joblib"
if not os.path.exists(MODEL_PATH):
    # Fallback to local root
    MODEL_PATH = "pneumonia_detection_model_v1_0.joblib"

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading serialized model: {e}")
        return None

model = load_model()
IMG_SIZE = 256

def preprocess_image_bytes(image_bytes, is_dcm=False):
    if is_dcm:
        dcm = pydicom.dcmread(io.BytesIO(image_bytes))
        img = dcm.pixel_array.astype(np.float32)
    else:
        img_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_min, img_max = np.min(img), np.max(img)
    if img_max - img_min != 0:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

st.title("AI driven Pneumonia Detection Lab")
st.write("Upload a standard chest X-ray image file (PNG, JPG, JPEG or a medical DICOM (.dcm)) to get an analysis")

# Adjusted column layouts to pull image preview columns tightly to the left side of the screen
col1, col2 = st.columns([0.7, 0.3])

with col1:
    uploaded_file = st.file_uploader("Choose a Chest X-ray file...", type=["png", "jpg", "jpeg", "dcm"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    is_dcm = uploaded_file.name.lower().endswith('.dcm')

    with col2:
        st.markdown("#### Uploaded Image Preview")
        try:
            if is_dcm:
                dcm = pydicom.dcmread(io.BytesIO(file_bytes))
                img_array = dcm.pixel_array
                img_array = ((img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array)) * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
            else:
                img = Image.open(io.BytesIO(file_bytes))

            # Reduced the size of the preview image by using a fixed width instead of full container scaling
            st.image(img, caption="Chest X-Ray Preview",width=260)
        except Exception as e:
            st.error(f"Could not preview image: {e}")

    with col1:
        if st.button("Run Pneumonia Analysis"):
            if model is None:
                st.error("Model is not loaded. Cannot run analysis.")
            else:
                with st.spinner("Processing image & executing model inference locally... "):
                    try:
                        processed_img = preprocess_image_bytes(file_bytes, is_dcm=is_dcm)
                        prediction_prob = float(model.predict(processed_img)[0][0])
                        prediction_class = 1 if prediction_prob > 0.5 else 0
                        prediction_label = "Pneumonia (Lung Opacity)" if prediction_class == 1 else "No Pneumonia (Normal/Other)"

                        st.success("Analysis Complete!")
                        st.metric(label="Diagnostic Result", value=prediction_label)
                        st.metric(label="Probability Confidence", value=f"{prediction_prob * 100:.2f}%")
                    except Exception as e:
                        st.error(f"Inference error: {e}")
