import streamlit as st
from PIL import Image
import os
import joblib
from transformers import BeitFeatureExtractor, BeitForImageClassification, ViTFeatureExtractor, ViTModel
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor
)
import cv2
import torch
import numpy as np

# --- Configuration ---
# Define paths for the image classification model
BEIT_CLASSIFIER_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'final_model_finetuned')

# Define paths for the Joblib models
TRADITIONAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'traditional_model', 'traditional_model.joblib')
TRADITIONAL_SCALER_PATH = os.path.join(os.path.dirname(__file__), 'traditional_model', 'traditional_scaler.joblib')

PREDICTION_LABELS = {
    0: 'Bacteria',
    1: 'Fungi',
    2: 'Healthy',
    3: 'Pest',
    4: 'Phytopthora',
    5: 'Virus',
}

# --- Page Functions ---
def beit_classification_page():
    st.title("BEiT Potato Disease Classifier Image App")

    @st.cache_resource
    def load_image_classifier_model(model_path):
        try:
            print(f"Attempting to load image classifier from: {model_path}")
            feature_extractor = BeitFeatureExtractor.from_pretrained(model_path)
            model = BeitForImageClassification.from_pretrained(model_path)
            model.config.id2label = PREDICTION_LABELS
            return feature_extractor, model
        except Exception as e:
            st.error(f"Error loading image classifier model: {e}. Is the model too large or files missing?")
            return None, None

    feature_extractor, model = load_image_classifier_model(BEIT_CLASSIFIER_MODEL_DIR)

    if not feature_extractor or not model:
        st.warning("Image classifier model could not be loaded. Please check logs for details.")
        return
    
    st.write("Upload an image to classify potato diseases.")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        return
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_container_width=True)
    st.write("")
    st.write("Classifying...")

    try:
        # Perform inference with your local model
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        prediction = model.config.id2label[predicted_class_idx]
        st.success(f"Prediction: **{prediction}**")
    except Exception as e:
        st.error(f"Error during image classification inference: {e}")

def traditional_classification_page():
    @st.cache_resource
    def load_model_and_scaler():
        try:
            model = joblib.load(TRADITIONAL_MODEL_PATH)
            scaler = joblib.load(TRADITIONAL_SCALER_PATH)
            return scaler, model
        except Exception as e:
            st.error(f"Error: {e}.")
            return None, None
    
    @st.cache_resource
    def load_vit_and_feature():
        try:
            model_id = 'google/vit-base-patch16-224-in21k'
            vit = ViTModel.from_pretrained(model_id)
            vit.eval()

            feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)
            return vit, feature_extractor
        except Exception as e:
            st.error(f"Error: {e}.")
            return None, None
        
    st.title("Traditional Potato Disease Classifier Image App")
    scaler, best_estimator = load_model_and_scaler()
    vit, feature_extractor = load_vit_and_feature()

    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    size = (feature_extractor.size["height"], feature_extractor.size["width"])

    inference_transforms = Compose([
        Resize(size),
        ToTensor(),
        normalize
    ])

    def extract_features(img_original, transforms):
        # Note that OpenCV reads images with BGR by default
        # Thus, we need to convert it to RGB since it may cause issues otherwise
        img_np_rgb = np.array(img_original)
        img_rgb = cv2.cvtColor(img_np_rgb, cv2.COLOR_BGR2RGB)

        # Get numpy array of image
        img = Image.fromarray(img_rgb)
        img_tensor = transforms(img).unsqueeze(0)

        # Disable gradient calculation, use inference mode
        # The goal isn't to train, but to extract features
        with torch.no_grad():
            outputs = vit(img_tensor)
            features = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

        # Returns the feature vector
        return features

    # test_samples = [extract_features(img, inference_transforms) for img in test_images]
    # test_samples = scaler.transform(test_samples)
    # test_pred = best_estimator.predict(test_samples)
    
    st.write("Upload an image to classify potato diseases.")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        return
    
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Uploaded Image.", use_container_width=True)
    st.write("")
    st.write("Classifying...")

    try:
        features = extract_features(image_pil, inference_transforms)
        features_reshaped = features.reshape(1, -1)
        scaled_features = scaler.transform(features_reshaped)
        predicted_class = best_estimator.predict(scaled_features)
        prediction = PREDICTION_LABELS.get(predicted_class[0], "Unknown")
        st.success(f"Prediction: **{prediction}**")
    except Exception as e:
        st.error(f"Error during image classification inference: {e}")
# --- Main App Logic (Sidebar Navigation) ---

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["BEiT", "Traditional"])

if page == "BEiT":
    beit_classification_page()
elif page == "Traditional":
    traditional_classification_page()