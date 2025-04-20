import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# === PAGE SETUP ===
st.set_page_config(page_title="Water Quality Analyzer", page_icon="💧", layout="wide")

# === STYLES ===
st.markdown("""
    <style>
        .title {
            display: flex;
            align-items: center;
            gap: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown('<div class="title">', unsafe_allow_html=True)
st.image("Screenshot 2025-04-20 231221.png", width=60)
st.markdown("## Water Quality Analyzer")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
This tool evaluates water suitability for various purposes like aquaculture, drinking, and irrigation.
Built using **ETDFNN (Enhanced Tuned Deep Fuzzy Neural Network)** and **Fuzzy Logic**, this model achieves up to **100% training accuracy**.
Upload or input parameters such as pH, temperature, and turbidity to determine water quality.
""")

# === LOAD DATA ===
df = pd.read_csv("https://github.com/TahseenNoor/water-quality-app/raw/refs/heads/main/realfishdataset.csv")

# Drop the 'fish' column for generalization
if 'fish' in df.columns:
    df = df.drop(columns=['fish'])

# === DISPLAY DATA ===
st.subheader("📊 Water Sample Data")
st.dataframe(df)

# === CHARTS ===
st.subheader("📈 Water Quality Parameter Trends")
st.bar_chart(df[["ph", "temperature", "turbidity"]])

# === MODEL ARCHITECTURE ===
class ETDFNNBinary(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ETDFNNBinary, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)  # Binary output

    def forward(self, x):
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.output(x))  # sigmoid for binary output

# === LOAD MODEL ===
model = ETDFNNBinary(input_size=3, hidden_size=64)
model_path = "model.pth"

model_ready = False
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    model_ready = True
except FileNotFoundError:
    st.warning("⚠️ model.pth not found. Please make sure it's in the same folder.")
except Exception as e:
    st.error("❌ Failed to load model.")
    st.text(f"Error: {str(e)}")

# === USER INPUT FOR PREDICTION ===
st.subheader("🔍 Analyze Water Quality")
ph_val = st.slider("pH", 5.0, 9.0, 7.0)
temp_val = st.slider("Temperature (°C)", 20.0, 35.0, 27.0)
turb_val = st.slider("Turbidity (NTU)", 1.0, 40.0, 10.0)

# === WATER QUALITY THRESHOLDS ===
def check_water_quality(ph, temp, turb):
    # Define safe and unsafe thresholds based on general guidelines
    if ph < 6.5 or ph > 8.5:
        return "⚠️ **Water is Unsafe** due to pH levels"
    elif temp < 20.0 or temp > 30.0:
        return "⚠️ **Water is Unsafe** due to temperature"
    elif turb > 30.0:
        return "⚠️ **Water is Unsafe** due to turbidity"
    
    return "✅ **Water is Safe** based on quality parameters"

if st.button("Analyze Quality"):
    if model_ready:
        # Prepare input tensor
        input_tensor = torch.tensor([[ph_val, temp_val, turb_val]], dtype=torch.float32)
        
        with torch.no_grad():
            # Get model output
            output = model(input_tensor)
            st.write(f"Model output (raw probability): {output.item()}")
            
            # Apply threshold of 0.5 to decide prediction
            prediction = int(output.item() > 0.5)

        # Combine model result with quality parameter checks
        result = "✅ **Water is Safe**" if prediction == 1 else "⚠️ **Water is Unsafe**"
        
        # Check based on predefined thresholds
        quality_check = check_water_quality(ph_val, temp_val, turb_val)
        
        # Final result
        if "Unsafe" in quality_check:
            st.warning(f"💧 Water Quality Result: {quality_check}")
        else:
            st.success(f"💧 Water Quality Result: {result} and {quality_check}")
    else:
        st.error("❌ Model not ready. Please upload a valid `model.pth` file.")

# === ADDITIONAL FEATURES ===

# === USE CASE TOGGLE ===
st.subheader("🛠️ Select Use Case")
use_case = st.radio("Choose the purpose for analysis:", ["General Use", "Fish Farming", "Drinking"], horizontal=True)

st.markdown(f"🧭 **Current Mode:** `{use_case}`")

# === CUSTOM LABEL BASED ON USE CASE ===
if use_case == "Fish Farming":
    st.info("🐟 Fish farming requires stable pH (6.5-8.5), moderate turbidity, and temperature around 25-30°C.")
elif use_case == "Drinking":
    st.info("🚰 Drinking water should have pH 6.5-8.5, low turbidity (<5 NTU), and safe temperature range.")
else:
    st.info("🌱 General use considers flexible thresholds, based on irrigation or industry.")

# === MAP INPUT (Optional) ===
st.subheader("📍 Optional: Add Location")
location = st.map()
st.caption("Location data is just for visualization and not used in prediction yet.")

# === BATCH CSV UPLOAD ===
st.subheader("📦 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with columns: `ph`, `temperature`, `turbidity`", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        required_cols = {'ph', 'temperature', 'turbidity'}

        if not required_cols.issubset(set(batch_df.columns)):
            st.error("❌ Uploaded CSV must contain `ph`, `temperature`, and `turbidity` columns.")
        else:
            st.success("✅ File uploaded successfully. Preview below:")
            st.dataframe(batch_df.head())

            if model_ready:
                inputs = torch.tensor(batch_df[["ph", "temperature", "turbidity"]].values, dtype=torch.float32)
                with torch.no_grad():
                    outputs = model(inputs).squeeze().numpy()
                    predictions = (outputs > 0.5).astype(int)

                batch_df["Prediction"] = ["Safe" if p == 1 else "Unsafe" for p in predictions]
                st.subheader("🔎 Batch Prediction Results")
                st.dataframe(batch_df)
            else:
                st.warning("⚠️ Model not ready. Please ensure `model.pth` is available.")

    except Exception as e:
        st.error(f"❌ Failed to process the file. Error: {e}")
