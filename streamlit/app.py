import streamlit as st
#import cv2
#import numpy as np
from PIL import Image
import os
#import requests
from transformers import BeitImageProcessor, BeitForImageClassification # Adjust based on your model

# To locally host the streamlit website, do the ff:
# cd streamlit
# streamlit run app.py

# st.write(
#     "Streamlit is also great for more traditional ML use cases like computer vision or NLP. Here's an example of edge detection using OpenCV."
# )

# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
# if uploaded_file:
#     image = Image.open(uploaded_file)
# else:
#     image = Image.open(requests.get("https://picsum.photos/200/120", stream=True).raw)

# edges = cv2.Canny(np.array(image), 100, 200)
# tab1, tab2 = st.tabs(["Detected edges", "Original"])
# tab1.image(edges, use_container_width=True)
# tab2.image(image, use_container_width=True)

# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model(model_path):
    # The path should be relative to your Streamlit app's root
    try:
        feature_extractor = BeitImageProcessor.from_pretrained(model_path)
        model = BeitForImageClassification.from_pretrained(model_path)
        model.config.id2label = {
            0: 'Bacteria', 
            1: 'Fungi',
            2: 'Healthy',
            3: 'Pest',
            4: 'Phytopthora',
            5: 'Virus',
        }
        return feature_extractor, model
    except Exception as e:
        st.error(f"Error loading model locally: {e}. Is the model too large or files missing?")
        return None, None

current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, 'final_model_finetuned')
print(model_dir)
feature_extractor, model = load_model(model_dir)

if feature_extractor and model:
    st.title("BEiT Potato Disease Classifier Image App (Local Model)")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        from PIL import Image
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
            st.write("Prediction:", prediction)
        except Exception as e:
            st.error(f"Error during local inference: {e}")
else:
    st.warning("Model could not be loaded. Please check logs for details.")