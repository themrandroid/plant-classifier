import streamlit as st
import json
import os
import numpy as np
import random
from PIL import Image
import tensorflow as tf #type: ignore
from tensorflow.keras.models import load_model #type: ignore


# Load Data + Model
with open("plant_data.json", "r", encoding="utf-8") as f:
    plant_data = json.load(f)

tflite_model_path = "colab/best_finetune_60.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (224, 224)
CLASS_NAMES = list(plant_data.keys())

REP_IMG_DIR = "images/representatives"

# Helper Functions
def preprocess_image(image_file):
    """Load and preprocess uploaded image for model prediction"""
    image = Image.open(image_file).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_from_image(image_file, top_k=3):
    """Predict top-k plants from uploaded image using TFLite model"""
    img_tensor = preprocess_image(image_file)
    # set input
    interpreter.set_tensor(input_details[0]['index'], img_tensor)
    # run inference
    interpreter.invoke()
    # get predictions
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    # top-k
    top_indices = preds.argsort()[-top_k:][::-1]
    results = [(CLASS_NAMES[i], preds[i]) for i in top_indices]
    return results

def get_image_path(scientific_name):
    # normalize: lowercase + underscores
    # filename = scientific_name.lower().replace(" ", "_") + ".jpg"
    for f in os.listdir(REP_IMG_DIR):
        if f.lower().startswith(scientific_name.lower().replace(" ", "_").split("_")[0]):  
            return os.path.join(REP_IMG_DIR, f)
    return None

def search_by_name(query):
    """Return plants whose names contain the query string"""
    query = query.lower()
    matches = {k: v for k, v in plant_data.items() if query in k.lower() or query in v["common_name"].lower()}
    return matches

def display_plant(scientific_name, info, prob=None):
    """Show plant card with image + details"""
    img_path = get_image_path(scientific_name)
    if img_path:
        st.image(img_path, width=200)

    st.markdown(f"### {info['common_name']}  *({scientific_name})*")
    st.write(f"**Description:** {info['description']}")
    st.write(f"**Did You Know:** {info['fun_fact']}")
    if prob:
        st.write(f"**Prediction Confidence:** {prob*100:.2f}%")

# --- Title ---
st.title("🌿 Little Botanist")

# --- Search Bars (side by side) ---
col1, col2 = st.columns(2)

search_name = None
uploaded_file = None

with col1:
    search_name = st.text_input(" Search by Name", placeholder="Enter plant name...")
    if search_name:
        # Search both scientific & common names
        matches = [sci for sci, info in plant_data.items()
                   if search_name.lower() in sci.lower() or search_name.lower() in info["common_name"].lower()]
        if matches:
            selected = st.selectbox("Select a plant:", matches)
        else:
            selected = None
    else:
        selected = None

with col2:
    st.subheader("Search by Image")
    uploaded_file = st.file_uploader("Upload an image of a plant", type=["jpg", "png", "jpeg"])

# --- Results Section ---
if selected:
    details = plant_data[selected]
    image_path = get_image_path(selected)

    col1, col2 = st.columns([1, 3])
    with col1:
        if image_path:
            st.image(Image.open(image_path), width=200)

    with col2:
        st.subheader(details["common_name"])
        st.text(f"Scientific Name: {selected}")
        st.write(details["description"])
        st.info(f"Did You Know: {details['fun_fact']}")

elif uploaded_file:
    predictions = predict_from_image(uploaded_file, top_k=3)
    st.success("Top Predictions:")

    for sci_name, prob in predictions:
        details = plant_data[sci_name]
        image_path = get_image_path(sci_name)

        col1, col2 = st.columns([1, 3])
        with col1:
            if image_path:
                st.image(Image.open(image_path), width=200)

        with col2:
            st.markdown(f"### {details['common_name']} *({sci_name})*")
            # st.write(f"**Prediction Confidence:** {prob*100:.2f}%")
            st.write(details["description"])
            st.info(f"Did You Know: {details['fun_fact']}")

# --- Divider ---
st.markdown("---")
st.header("Featured Plants by Category")

# Category ranges (index-based from CLASS_NAMES)
categories = {
    "Flowers": (0, 25),
    "Fruits & Fruit Plants": (25, 50),
    "Common Trees & Shrubs": (50, 73),
    "Weeds & Medicinal Plants": (73, 97)
}

# Show plants per category
for category, (start, end) in categories.items():
    with st.expander(category, expanded=True):
        # Plants in this category
        plants_in_cat = CLASS_NAMES[start:end]

        # Pick 5 random ones
        sample_plants = random.sample(plants_in_cat, min(5, len(plants_in_cat)))

        st.subheader("Highlights")
        for plant in sample_plants:
            details = plant_data[plant]
            image_path = get_image_path(plant)

            col1, col2 = st.columns([1, 3])
            with col1:
                if image_path:
                    st.image(Image.open(image_path), width=150)

            with col2:
                st.markdown(f"### {details['common_name']}")
                st.write(f"*{plant}*")  # italic scientific name

                st.write(details["description"])
                st.info(f"**Did You Know:**  *{details['fun_fact']}*")

        # --- Show More button ---
        if st.button(f"Show All in {category}", key=category):
            st.subheader("Full List")
            for plant in plants_in_cat:
                details = plant_data[plant]
                image_path = get_image_path(plant)

                col1, col2 = st.columns([1, 3])
                with col1:
                    if image_path:
                        st.image(Image.open(image_path), width=120)

                with col2:
                    st.markdown(f"**{details['common_name']}** *({plant})*")
                    st.write(details["description"])