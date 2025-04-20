import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Load image
image = Image.open("294d91ac-fe0d-47c3-8413-f5d6ce10bdec.png")

# Set page config
st.set_page_config(page_title="Water Quality Analyzer", page_icon="💧", layout="wide")

# Light pastel background
st.markdown("""
    <style>
        body {
            background-color: #f1f8ff;
        }
        .title {
            display: flex;
            align-items: center;
            gap: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Title with image
st.markdown('<div class="title">', unsafe_allow_html=True)
st.image(image, width=60)
st.markdown("## Water Quality Analyzer")
st.markdown('</div>', unsafe_allow_html=True)

st.write("Monitor water parameters, analyze pollution levels, and explore different use cases like fish farming, agriculture, or drinking water safety.")

# Load dataset
df = pd.read_csv("https://github.com/TahseenNoor/water-quality-app/raw/refs/heads/main/realfishdataset.csv")


# Show data
st.subheader("📊 Water Sample Data")
st.dataframe(df)

# Show charts
st.subheader("📈 Water Quality Parameter Trends")
st.bar_chart(df[["ph", "temperature", "turbidity"]])

# Define model
class ETDFNNBinary(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ETDFNNBinary, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 3)  # You can change this based on your use case

    def forward(self, x):
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.output(x)

# Use case classes (can be extended: suitable for drinking, agriculture, etc.)
quality_classes = ['Suitable for Aquaculture', 'Needs Treatment', 'Not Recommended']

model = ETDFNNBinary(input_size=3, hidden_size=64)

# Load model
try:
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
except:
    st.warning("⚠️ Trained model not found. Predictions will not be available.")

# Prediction input
st.subheader("🔍 Check Water Suitability")
ph_val = st.slider("pH", 5.0, 9.0, 7.0)
temp_val = st.slider("Temperature (°C)", 20.0, 35.0, 27.0)
turb_val = st.slider("Turbidity (NTU)", 1.0, 10.0, 5.0)

if st.button("Analyze Quality"):
    input_tensor = torch.tensor([[ph_val, temp_val, turb_val]], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_idx = torch.argmax(outputs, dim=1).item()
        quality = quality_classes[predicted_idx] if predicted_idx < len(quality_classes) else "Unknown"
    st.success(f"💡 Water Quality Status: **{quality}**")
