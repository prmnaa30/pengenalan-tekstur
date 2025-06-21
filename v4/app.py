
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
    try:
        model = tf.keras.models.load_model(model_path)
        # Ensure the base model layers are not trainable for prediction
        for layer in model.layers[0].layers: # Assuming base_model is the first layer in Sequential
             layer.trainable = False
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model = load_model()

# Assuming class_names are ['batu', 'kain', 'kayu'] based on previous output
# Make sure the order matches the model's output classes
class_names = ['batu', 'kain', 'kayu'] # Replace with your actual class names in the correct order

# Define a confidence threshold for OOD detection
# Images with a highest predicted probability below this threshold will be considered OOD
ood_threshold = 0.7 # You can adjust this value based on your validation results

st.title("Aplikasi Pengenalan Tekstur dengan Deteksi OOD")

uploaded_file = st.file_uploader("Unggah gambar tekstur...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang Diunggah', use_container_width=True)

    if model is not None:
        # Preprocess the image and make prediction
        img_array = np.array(image.resize(img_size)) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        confidence = np.max(predictions) # Get the highest probability

        st.write(f"Keyakinan Prediksi Tertinggi: {confidence:.4f}") # Display confidence

        # Check for OOD based on the threshold
        if confidence < ood_threshold:
            st.write("Status: **Out-of-Distribution (OOD)**")
            st.write("Gambar ini mungkin bukan salah satu tekstur yang dikenali (batu, kain, kayu).")
        else:
            st.write("Status: **In-Distribution (ID)**")
            if 0 <= predicted_class_index < len(class_names):
                 predicted_class_name = class_names[predicted_class_index]
                 st.write(f"Prediksi Tekstur: **{predicted_class_name}**")
            else:
                 st.warning("Model output out of expected range.")
                 st.write(f"Predicted class index: {predicted_class_index}")

    else:
        st.warning("Model tidak dapat dimuat.")

