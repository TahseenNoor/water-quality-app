import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# ---------- App Styling ----------
st.set_page_config(layout="wide")

page_bg_color = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #F4F9F9;
}
</style>
"""
st.markdown(page_bg_color, unsafe_allow_html=True)

# ---------- Title & Header ----------
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("https://github.com/TahseenNoor/water-quality-app/raw/main/Screenshot%202025-04-20%20231221.png", width=60)
with col2:
    st.markdown("<h1 style='color:#0077b6;'>FlowPredict: Real-Time Water Quality Prediction</h1>", unsafe_allow_html=True)

st.markdown("Predict water quality for multiple use cases like pollution monitoring, fish farming, and more.")

# ---------- Sidebar ----------
with st.sidebar:
    st.title("Select Use Case")
    use_case = st.radio("Choose a mode:", ["General Quality Monitoring", "Fish Farming", "Drinking Water"])

# ---------- Sample Map ----------
st.subheader("Location-based Sampling (Demo Map)")
map_data = pd.DataFrame({
    'lat': np.random.uniform(12.90, 12.95, 10),
    'lon': np.random.uniform(77.50, 77.55, 10)
})
st.map(map_data)

# ---------- Model Loading ----------
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = SimpleNN()
try:
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    model_ready = True
except Exception as e:
    st.error("Error loading model. Please ensure 'model.pth' is present.")
    model_ready = False

# ---------- Water Quality Analysis ----------
st.subheader("Water Quality Analyzer")

ph_val = st.slider("pH Level", 0.0, 14.0, 7.0)
temp_val = st.slider("Temperature (°C)", 0.0, 40.0, 25.0)
turb_val = st.slider("Turbidity (NTU)", 0.0, 100.0, 5.0)

def check_water_quality(ph, temp, turb):
    if not model_ready:
        return "Model not available"
    input_tensor = torch.tensor([[ph, temp, turb]], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    return "Good Quality ✅" if predicted_class == 1 else "Poor Quality ❌"

if st.button("Analyze Quality"):
    result = check_water_quality(ph_val, temp_val, turb_val)
    st.success(f"Water Quality Status: {result}")

# ---------- Footer ----------
st.markdown("---")
st.markdown("🔬 Built with ❤️ using Streamlit | Powered by AI & Data Science")
