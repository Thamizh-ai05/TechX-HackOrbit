import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load Models with Caching
@st.cache_resource
def load_underwater_model():
    return load_model("underwater_plastic_model.h5")

# Sidebar Menu
st.sidebar.title("🌊 Ocean Plastic Monitoring System")
option = st.sidebar.radio("Select Module:", [
    "Plastic Flow Prediction",
    "Underwater Plastic Detection (AI)"
])

# Module 1: Plastic Flow Prediction
def plastic_flow_prediction():
    st.header("🌍 Plastic Flow Prediction")
    river = st.selectbox("Select River", ["Ganges", "Yangtze", "Amazon"])
    speed = st.slider("Ocean Current Speed (km/h)", 0, 20, 5)
    direction = st.slider("Current Direction (degrees)", 0, 360, 90)
    area = np.random.uniform(50, 200)
    st.success(f"Plastic from {river} expected to gather in an area of {area:.2f} sq km.")

# Module 2: Underwater Plastic Detection (AI)
def underwater_plastic_detection():
    st.header("🤖 Underwater Plastic Detection")
    uploaded = st.file_uploader("Upload an underwater image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img_resized = img.resize((224, 224))
        img_array = img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Load model and predict
        model = load_underwater_model()
        pred = model.predict(img_array)[0][0]

        if pred > 0.5:
            st.success(f"🟢 Plastic Detected ({pred*100:.1f}% confidence)")
        else:
            st.info(f"⚪ No Plastic Detected ({(1-pred)*100:.1f}% confidence)")

# Run Selected Module
if option == "Plastic Flow Prediction":
    plastic_flow_prediction()
elif option == "Underwater Plastic Detection (AI)":
    underwater_plastic_detection()
else:
    print("invalid")
