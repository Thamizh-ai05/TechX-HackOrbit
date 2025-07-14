# AI- Powered Ocean Plastic Forecast and Detection System
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import networkx as nx
import branca

@st.cache_resource
def load_underwater_model():
    return load_model("underwater_plastic_model.h5")

@st.cache_resource
def load_type_model():
    return load_model("plastic_type_model.keras")

st.sidebar.title("🌊 Ocean Plastic Monitoring System")
option = st.sidebar.radio("Select Module:", [
    "Plastic Flow Prediction",
    "Underwater Plastic Detection (AI)",
    "Seasonal Microplastic Heatmap",
    "Satellite & Sampling Data Fusion",
    "Ocean Plastic Cleanup Simulation",
    "AI-Based Plastic Type Classification"
])

def plastic_flow_prediction():
    st.header("🌍 Plastic Flow Prediction")
    river = st.selectbox("Select River", ["Ganges", "Yangtze", "Amazon"])
    speed = st.slider("Ocean Current Speed (km/h)", 0, 20, 5)
    direction = st.slider("Current Direction (degrees)", 0, 360, 90)
    area = np.random.uniform(50, 200)
    st.success(f"Plastic from {river} expected to gather in an area of {area:.2f} sq km.")

def underwater_plastic_detection():
    st.header("🤖 Underwater Plastic Detection")
    uploaded = st.file_uploader("Upload image...", type=["jpg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption='Uploaded Image', use_column_width=True)
        model = load_underwater_model()
        img_array = img_to_array(img.resize((224, 224))) / 255.0
        pred = model.predict(np.expand_dims(img_array, axis=0))[0][0]
        st.success(f"🟢 Plastic detected ({pred*100:.1f}% confidence)" if pred > 0.5 else f"⚪ No plastic detected ({(1-pred)*100:.1f}%)")

def seasonal_heatmap():
    st.header("🗓 Microplastic Heatmap by Month")
    dates = pd.date_range("2024-01-01", "2024-12-31", freq='M')
    selected = st.select_slider("Select Month", options=dates.strftime("%Y-%m"))
    month_seed = int(selected.split('-')[1])
    np.random.seed(month_seed)
    df = pd.DataFrame({
        "Latitude": np.random.uniform(-30, 30, 300),
        "Longitude": np.random.uniform(-180, 180, 300),
        "Concentration": np.random.uniform(10, 100, 300)
    })
    m = folium.Map(location=[0, 0], zoom_start=2)
    HeatMap([[r["Latitude"], r["Longitude"], r["Concentration"]] for _, r in df.iterrows()],
            radius=15, blur=20, max_zoom=5).add_to(m)
    colormap = branca.colormap.LinearColormap(['blue', 'green', 'yellow', 'red'], vmin=10, vmax=100)
    colormap.caption = f'Microplastic - {selected}'
    m.add_child(colormap)
    st_folium(m, width=700, height=450)

def satellite_sampling_fusion():
    st.header("🛰 Satellite + Sampling Fusion")
    m = folium.Map(location=[12, 22], zoom_start=5)
    for pt in [(10, 20), (15, 25), (12, 22)]:
        folium.CircleMarker(location=pt, radius=8, color='blue', fill_opacity=0.4, popup="Satellite Roughness").add_to(m)
    for pt in [(11, 21), (14, 24), (13, 23)]:
        folium.CircleMarker(location=pt, radius=8, color='green', fill_opacity=0.4, popup="Sampling Point").add_to(m)
    for pt in [(10.5, 20.5), (14.5, 24.5)]:
        folium.CircleMarker(location=pt, radius=8, color='red', fill_opacity=0.6, popup="AI Detection").add_to(m)
    legend = '''
     <div style="position: fixed; bottom: 50px; left: 50px; width: 180px; height: 110px; 
                 background-color: white; border:2px solid grey; z-index:9999; font-size:14px;">
     &nbsp;<b>Legend</b><br>
     &nbsp;<i class="fa fa-circle" style="color:blue"></i>&nbsp;Satellite Roughness<br>
     &nbsp;<i class="fa fa-circle" style="color:green"></i>&nbsp;Sampling Point<br>
     &nbsp;<i class="fa fa-circle" style="color:red"></i>&nbsp;AI Detection
     </div>'''
    m.get_root().html.add_child(branca.element.Element(legend))
    st_folium(m, width=700, height=450)

def cleanup_simulation():
    st.header("♻ Ocean Cleanup Simulation")
    G = nx.grid_2d_graph(10, 10)
    for (u, v) in G.edges():
        G[u][v]['weight'] = np.random.randint(1, 10)

    start, end = (0, 0), (9, 9)
    path = nx.dijkstra_path(G, start, end)
    st.write("Optimized cleanup route (shortest path):")
    st.code(str(path))
    st.success(f"Total route points: {len(path)}")

def plastic_type_classification():
    st.header("🧠 Plastic Type Classification (Bag, Bottle, Net)")
    uploaded = st.file_uploader("Upload image of plastic type...", type=["jpg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        model = load_type_model()
        img_array = img_to_array(img.resize((224, 224))) / 255.0
        pred = model.predict(np.expand_dims(img_array, axis=0))[0]
        classes = ['bag', 'bottle', 'net']
        label = classes[np.argmax(pred)]
        confidence = np.max(pred)
        st.success(f"🔍 Detected: *{label.upper()}* ({confidence*100:.2f}% confidence)")

if option == "Plastic Flow Prediction":
    plastic_flow_prediction()
elif option == "Underwater Plastic Detection (AI)":
    underwater_plastic_detection()
elif option == "Seasonal Microplastic Heatmap":
    seasonal_heatmap()
elif option == "Satellite & Sampling Data Fusion":
    satellite_sampling_fusion()
elif option == "Ocean Plastic Cleanup Simulation":
    cleanup_simulation()
elif option == "AI-Based Plastic Type Classification":
    plastic_type_classification()
