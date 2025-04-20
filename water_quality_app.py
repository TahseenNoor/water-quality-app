import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os

# ----------------------------
# Neural Network Class
# ----------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        return self.fc(x)

# ----------------------------
# Try to Load the Model
# ----------------------------
model = SimpleNN()
try:
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    model_status = "Model loaded successfully ✅"
except Exception as e:
    model = None
    model_status = f"Error loading model ❌: {e}"

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="FlowPredict", layout="wide")
st.markdown("""
    <h1 style='text-align: center;'>💧 FlowPredict: Real-Time Water Quality Prediction</h1>
    <p style='text-align: center;'>Predict water quality for multiple use cases like pollution monitoring, fish farming, and more.</p>
""", unsafe_allow_html=True)

# Display model status
st.sidebar.info(model_status)

# ----------------------------
# Sidebar: Use Case Selection
# ----------------------------
use_case = st.sidebar.selectbox("Select Use Case", ["General Quality Monitoring", "Fish Farming", "Drinking Water"])
st.sidebar.markdown("---")

# ----------------------------
# Main: Image + Map
# ----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4149/4149648.png", width=150)  # Cute water icon

with col2:
    st.markdown("#### 🌍 Location-based Sampling (Demo Map)")
    st.map()  # Basic demo map with random coordinates

st.markdown("---")

# ----------------------------
# Water Quality Input Section
# ----------------------------
st.subheader("💡 Water Quality Analyzer")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1)

with col2:
    temp = st.slider("Temperature (°C)", 0.0, 40.0, 25.0, 0.1)

with col3:
    turbidity = st.slider("Turbidity (NTU)", 0.0, 100.0, 5.0, 0.1)

# ----------------------------
# Prediction Logic
# ----------------------------
st.markdown("### 📊 Water Quality Status")

if model:
    input_tensor = torch.tensor([[ph, temp, turbidity]], dtype=torch.float32)
    prediction = model(input_tensor).item()

    if prediction > 0.7:
        status = "✅ Good Quality Water"
        color = "green"
    elif prediction > 0.4:
        status = "⚠️ Moderate Quality"
        color = "orange"
    else:
        status = "❌ Poor Quality Water"
        color = "red"

    st.markdown(f"<h2 style='color:{color};'>{status}</h2>", unsafe_allow_html=True)
else:
    st.error("Model not available. Please check if 'model.pth' exists and is compatible.")

# ----------------------------
# Use Case Info Box
# ----------------------------
st.markdown("### 🧠 Use Case Info")
if use_case == "General Quality Monitoring":
    st.info("Used for general pollution and quality monitoring in rivers, lakes, and industrial areas.")
elif use_case == "Fish Farming":
    st.info("Focuses on optimal pH, temperature, and turbidity ranges for aquatic health.")
else:
    st.info("Strict parameters suited for human consumption safety.")
