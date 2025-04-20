import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os

# ---------------------
# 🎨 Styling + Header
# ---------------------
st.set_page_config(page_title="Water Safety Checker", page_icon="🐟", layout="wide")

st.markdown("""
    <style>
        html, body {
            background-color: #f1f8ff;
        }
        .title {
            display: flex;
            align-items: center;
            gap: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Logo + Title
try:
    image = Image.open("Screenshot 2025-04-20 231221.png")
    st.markdown('<div class="title">', unsafe_allow_html=True)
    st.image(image, width=60)
    st.markdown("## Fish Water Safety Checker")
    st.markdown('</div>', unsafe_allow_html=True)
except:
    st.title("Fish Water Safety Checker")

st.write("Check if water conditions are safe for common aquaculture species like Tilapia, Rohu, or Catla.")

# ---------------------
# 📊 Load Data
# ---------------------
df = pd.read_csv("https://github.com/TahseenNoor/water-quality-app/raw/refs/heads/main/realfishdataset.csv")
st.subheader("📊 Sample Water Dataset")
st.dataframe(df)

st.subheader("📈 Parameter Trends")
st.bar_chart(df[["ph", "temperature", "turbidity"]])

# ---------------------
# 🧠 Binary Model (Safe / Unsafe)
# ---------------------
class ETDFNNBinary(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ETDFNNBinary, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.output(x))

model = ETDFNNBinary(input_size=3, hidden_size=64)
model_path = "/mnt/data/model.pth"

# Try loading model
model_ready = False
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        model_ready = True
    except Exception as e:
        st.error("❌ Error loading model.")
        st.text(f"Details: {str(e)}")

else:
    st.warning("⚠️ No model found. Please upload the model.pth file.")

# ---------------------
# 🧪 Prediction Input
# ---------------------
st.subheader("🔍 Test Your Water Sample")
ph_val = st.slider("pH", 5.0, 9.0, 7.0)
temp_val = st.slider("Temperature (°C)", 20.0, 35.0, 27.0)
turb_val = st.slider("Turbidity (NTU)", 1.0, 10.0, 5.0)

if st.button("Analyze Quality"):
    if model_ready:
        input_tensor = torch.tensor([[ph_val, temp_val, turb_val]], dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = 1 if output.item() >= 0.5 else 0
            label = "✅ Safe for Fish" if prediction == 1 else "⚠️ Unsafe for Fish"
            st.success(f"**Water Quality Result:** {label}")
    else:
        st.warning("Model not loaded. Please upload a trained model.")

# ---------------------
# 📤 Predict on All Rows
# ---------------------
if model_ready and st.checkbox("🔁 Run on Full Dataset"):
    def predict_row(row):
        x = torch.tensor([[row['ph'], row['temperature'], row['turbidity']]], dtype=torch.float32)
        with torch.no_grad():
            out = model(x)
            return 1 if out.item() >= 0.5 else 0

    df['Prediction'] = df.apply(predict_row, axis=1)
    df['Prediction Label'] = df['Prediction'].map({1: 'Safe', 0: 'Unsafe'})
    st.dataframe(df[['ph', 'temperature', 'turbidity', 'fish', 'Prediction Label']])

    st.subheader("📊 Prediction Breakdown")
    st.bar_chart(df['Prediction Label'].value_counts())
