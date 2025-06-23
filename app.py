import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Deteksi Tekstur & OOD",
    page_icon="🤖",
    layout="centered"
)

# --- Judul dan Deskripsi ---
st.title("🤖 Aplikasi Deteksi Tekstur")
st.write(
    "Unggah gambar tekstur (batu, kain, atau kayu) untuk diklasifikasikan. "
    "Aplikasi ini juga dapat mendeteksi jika gambar yang diunggah berada di luar kategori tersebut (Out-of-Distribution)."
)

# --- Variabel Global ---
MODEL_PATH = 'texture_model.keras'
CLASSES = ['batu', 'kain', 'kayu']
IMG_SIZE = (512, 512)
CONFIDENCE_THRESHOLD = 0.80 # Ambang batas kepercayaan 80%

# --- Fungsi untuk Memuat Model (dengan Caching) ---
# @st.cache_resource akan memastikan model hanya dimuat sekali
@st.cache_resource
def load_model():
    """Memuat model klasifikasi dari file .keras"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

# Panggil fungsi untuk memuat model
model = load_model()

# --- Preprocessing Input Gambar ---
def preprocess_image(image):
    """Melakukan preprocessing pada gambar sebelum dimasukkan ke model"""
    # 1. Resize gambar
    img = image.resize(IMG_SIZE)
    # 2. Convert gambar ke RGB (Untuk handle PNG)
    img = img.convert('RGB')
    # 3. Konversi ke array numpy
    img_array = np.array(img)
    # 4. Tambahkan dimensi batch
    img_array = np.expand_dims(img_array, axis=0)
    # 5. Gunakan fungsi preprocess_input dari EfficientNet
    processed_img = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return processed_img

# --- UI untuk Unggah Gambar ---
uploaded_file = st.file_uploader(
    "Pilih sebuah gambar...",
    type=["jpg", "jpeg", "png"]
)

st.markdown("---")

# --- Logika Utama Aplikasi ---
if model is not None and uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

    # Lakukan prediksi saat tombol ditekan
    if st.button("✨ Klasifikasikan Gambar"):
        with st.spinner("Model sedang berpikir... 🤔"):
            # 1. Preprocess gambar
            processed_image = preprocess_image(image)
            
            # 2. Lakukan prediksi
            prediction = model.predict(processed_image)
            
            # 3. Dapatkan skor kepercayaan dan kelas
            confidence = np.max(prediction)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = CLASSES[predicted_class_index]
            
            # 4. Terapkan logika OOD
            with col2:
                st.subheader("Hasil Prediksi:")
                if confidence >= CONFIDENCE_THRESHOLD:
                    st.success(f"✅ Terdeteksi sebagai: **{predicted_class_name.capitalize()}**")
                    st.write(f"Tingkat Kepercayaan: **{confidence:.2%}**")
                    st.progress(float(confidence))
                else:
                    st.error("❌ Gambar Tidak Dikenali (Out-of-Distribution)")
                    st.write(
                        "Model tidak cukup yakin untuk mengklasifikasikan gambar ini. "
                        "Kemungkinan gambar ini bukan batu, kain, atau kayu."
                    )
                    st.write(f"Prediksi terdekat: {predicted_class_name.capitalize()} (Kepercayaan: {confidence:.2%})")

# --- Sidebar Informasi ---
with st.sidebar:
    st.header("Bagaimana Cara Kerjanya?")
    st.write(
        "1. Anda mengunggah gambar.\n"
        "2. Gambar diproses agar sesuai dengan input model.\n"
        "3. Model (EfficientNetB0) memprediksi probabilitas untuk setiap kelas.\n"
        "4. **Logika OOD:** Jika probabilitas tertinggi (kepercayaan) **di atas 80%**, hasilnya akan ditampilkan. Jika tidak, gambar dianggap 'Tidak Dikenali'."
    )
    st.header("Tentang Model")
    st.write(
        "Model ini dilatih untuk mengenali 3 kelas tekstur:\n"
        "- **Batu**\n"
        "- **Kain**\n"
        "- **Kayu**"
    )