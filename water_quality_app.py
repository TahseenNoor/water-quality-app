import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os

# ----------------------------
# 🔧 CONFIGURATION + STYLING
# ----------------------------

st.set_page_config(page_title="Water Quality Analyzer", page_icon="💧", layout="wide")

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
# 📤 LOAD DATASET
# ----------------------------

df = pd.read_csv("https://github.com/TahseenNoor/water-quality-app/raw/refs/heads/main/realfishdataset.csv")

# ----------------------------
# 📊 DISPLAY DATA & TRENDS
# ----------------------------

st.subheader("📊 Water Sample Data")
st.dataframe(df)

st.subheader("📈 Water Quality Parameter Trends")
st.bar_chart(df[["ph", "temperature", "turbidity"]])

# ----------------------------
# 🧠 MODEL DEFINITION
# ----------------------------

class ETDFNNBinary(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ETDFNNBinary, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # ✅ Replaced BatchNorm1d
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x = torch.tanh(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.output(x)

# Use case labels
quality_classes = ['Suitable for Aquaculture', 'Needs Treatment', 'Not Recommended']

# ----------------------------
# 📥 LOAD MODEL
# ----------------------------

model = ETDFNNBinary(input_size=3, hidden_size=64)
model_path = "/mnt/data/model.pth"

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        model_ready = True
    except Exception as e:
        st.error("❌ Error loading model.")
        st.text(f"Details: {str(e)}")
        model_ready = False
else:
    st.warning("⚠️ Trained model not found. Predictions will not be available.")
    model_ready = False

# ----------------------------
# 🔍 SINGLE INPUT PREDICTION
# ----------------------------

st.subheader("🔍 Check Water Suitability")
ph_val = st.slider("pH", 5.0, 9.0, 7.0)
temp_val = st.slider("Temperature (°C)", 20.0, 35.0, 27.0)
turb_val = st.slider("Turbidity (NTU)", 1.0, 10.0, 5.0)

if st.button("Analyze Quality"):
    if model_ready:
        input_tensor = torch.tensor([[ph_val, temp_val, turb_val]], dtype=torch.float32)
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_idx = torch.argmax(outputs, dim=1).item()
            quality = quality_classes[predicted_idx] if predicted_idx < len(quality_classes) else "Unknown"

        st.success(f"💡 Water Quality Status: **{quality}**")
    else:
        st.warning("⚠️ Model not ready. Please upload a valid model.")

# ----------------------------
# 🧪 OPTIONAL: PREDICT WHOLE DATASET
# ----------------------------

if model_ready and st.checkbox("Run Prediction on Full Dataset"):
    def predict_quality(row):
        input_tensor = torch.tensor([[row['ph'], row['temperature'], row['turbidity']]], dtype=torch.float32)
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_idx = torch.argmax(outputs, dim=1).item()
        return quality_classes[predicted_idx] if predicted_idx < len(quality_classes) else "Unknown"

    df['Predicted Quality'] = df.apply(predict_quality, axis=1)
    st.dataframe(df)

    st.subheader("📊 Prediction Distribution")
    st.bar_chart(df['Predicted Quality'].value_counts())
