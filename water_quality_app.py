import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ----------------------------
# 🔧 CONFIGURATION + STYLING
# ----------------------------

# Page config
st.set_page_config(page_title="Water Quality Analyzer", page_icon="💧", layout="wide")

# CSS styling
st.markdown("""
    <style>
        html, body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f1f8ff;
        }
        .title {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .block-container {
            padding: 2rem 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# 🔹 IMAGE + HEADER
# ----------------------------

image = Image.open("Screenshot 2025-04-20 231221.png")

st.markdown('<div class="title">', unsafe_allow_html=True)
st.image(image, width=60)
st.markdown("## Water Quality Analyzer")
st.markdown('</div>', unsafe_allow_html=True)

st.write("Monitor water parameters, analyze pollution levels, and explore different use cases like fish farming, agriculture, or drinking water safety.")

# ----------------------------
# 📤 UPLOAD DATASET (OPTIONAL)
# ----------------------------

st.sidebar.header("Upload Your Dataset (optional)")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom dataset loaded!")
else:
    df = pd.read_csv("https://github.com/TahseenNoor/water-quality-app/raw/refs/heads/main/realfishdataset.csv")

# ----------------------------
# 🎯 USE CASE SELECTOR
# ----------------------------

use_case = st.sidebar.selectbox("Select Use Case", ["Fish Farming", "Agriculture", "Drinking Water"])

# ----------------------------
# 📊 SHOW DATA + CHARTS
# ----------------------------

st.subheader("📊 Water Sample Data")
st.dataframe(df)

st.subheader("📈 Water Quality Parameter Trends")
st.bar_chart(df[["ph", "temperature", "turbidity"]])

# ----------------------------
# 🧠 LOAD MODEL
# ----------------------------

# Define model
class ETDFNNBinary(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ETDFNNBinary, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.output(x)

# Initialize + load
quality_classes = ['Suitable for Aquaculture', 'Needs Treatment', 'Not Recommended']
model = ETDFNNBinary(input_size=3, hidden_size=64)

try:
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
except:
    st.warning("⚠️ Trained model not found. Predictions will not be available.")

# ----------------------------
# 🔍 INDIVIDUAL ANALYSIS
# ----------------------------

st.subheader("🔍 Check Water Suitability")
col1, col2, col3 = st.columns(3)
with col1:
    ph_val = st.slider("pH", 5.0, 9.0, 7.0)
with col2:
    temp_val = st.slider("Temperature (°C)", 20.0, 35.0, 27.0)
with col3:
    turb_val = st.slider("Turbidity (NTU)", 1.0, 10.0, 5.0)

if st.button("Analyze Quality"):
    input_tensor = torch.tensor([[ph_val, temp_val, turb_val]], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_idx = torch.argmax(outputs, dim=1).item()
        quality = quality_classes[predicted_idx] if predicted_idx < len(quality_classes) else "Unknown"

    # Styled result card
    color_map = {
        "Suitable for Aquaculture": "#d4edda",
        "Needs Treatment": "#fff3cd",
        "Not Recommended": "#f8d7da"
    }
    st.markdown(f"""
    <div style="background-color:{color_map.get(quality, '#e2e3e5')}; padding:20px; border-radius:10px">
        <h4>💧 Water Quality Status: <strong>{quality}</strong></h4>
        <p><em>Use case:</em> {use_case}</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# 📡 BATCH PREDICTION
# ----------------------------

def predict_quality(ph, temp, turb):
    input_tensor = torch.tensor([[ph, temp, turb]], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_idx = torch.argmax(outputs, dim=1).item()
    return quality_classes[predicted_idx] if predicted_idx < len(quality_classes) else "Unknown"

if st.checkbox("🧪 Run Predictions on Entire Dataset"):
    predictions = []
    for _, row in df.iterrows():
        q = predict_quality(row['ph'], row['temperature'], row['turbidity'])
        predictions.append(q)
    df['Predicted Quality'] = predictions
    st.dataframe(df)

    st.subheader("📊 Prediction Distribution")
    quality_counts = df['Predicted Quality'].value_counts()
    st.bar_chart(quality_counts)
