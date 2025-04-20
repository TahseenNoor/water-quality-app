
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- MODEL CLASS ---
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

# --- LOAD THE TRAINED MODEL ---
model = ETDFNNBinary(input_size=3, hidden_size=64)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

scaler = StandardScaler()

# --- STREAMLIT UI ---
st.title("💧 Water Quality Prediction App")
st.write("Check if your water condition is safe for selected fish species")

ph = st.slider("pH Level", 4.0, 10.0, 7.0)
temp = st.slider("Temperature (°C)", 10.0, 40.0, 25.0)
turbidity = st.slider("Turbidity (NTU)", 0.0, 100.0, 10.0)

input_data = np.array([[ph, temp, turbidity]])

# Just for demonstration: use dummy scaler if no fit data
input_scaled = scaler.fit_transform(input_data)

input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
with torch.no_grad():
    prediction = model(input_tensor).item()

st.write("### 🔍 Prediction:")
if prediction > 0.5:
    st.success("✅ Safe water for Tilapia / Rohu / Catla")
else:
    st.error("❌ Unsafe water for selected fish species")
