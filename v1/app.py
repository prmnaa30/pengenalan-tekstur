
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Assuming img_size is (128, 128) and is consistent with training
img_size = (128, 128)

# Load the trained model
@st.cache_resource
def load_model():
    # Adjust the path if your model file is in a different location
    model_path = 'texture_model.keras'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()

# Assuming class_names are ['batu', 'kain', 'kayu'] based on previous output
# Make sure the order matches the model's output classes
class_names = ['batu', 'kain', 'kayu'] # Replace with your actual class names in the correct order

st.title("Aplikasi Pengenalan Tekstur")

uploaded_file = st.file_uploader("Unggah gambar tekstur...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang Diunggah', use_column_width=True)

    if model is not None:
        # Preprocess the image and make prediction
        img_array = np.array(image.resize(img_size)) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)

        if 0 <= predicted_class_index < len(class_names):
             predicted_class_name = class_names[predicted_class_index]
             st.write(f"Prediksi Tekstur: **{predicted_class_name}**")
        else:
             st.warning("Model output out of expected range.")
             st.write(f"Predicted class index: {predicted_class_index}")

    else:
        st.warning("Model could not be loaded.")

