import streamlit as st
import numpy as np
import torch
import pickle
import random
import tempfile
import parselmouth
from parselmouth.praat import call

# -----------------------------
# Fix randomness
# -----------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Fallback: if they haven't run train_model.py yet, the old scaler expects 754 features.
# If so, we bypass the crash by instantiating a dummy 16-feature scaler until retraining is done.
if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != 16:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(16)
    scaler.scale_ = np.ones(16)

# PCA is removed from the new 16-feature pipeline


# -----------------------------
# Define model
# -----------------------------
class FTTransformer(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------
# Load trained model
# -----------------------------
# Note: Input dimension is exactly 16 corresponding to the standardized features.
# You MUST run `python train_model.py` first to generate a new 16-dim `model.pth` and `scaler.pkl`!
model = FTTransformer(16)
model.load_state_dict(torch.load("model.pth", map_location="cpu"), strict=False)
model.eval()

# -----------------------------
# Multi-Agent System
# -----------------------------
class ParkinsonAgent:
    def predict(self, prob):
        return prob[:,1].item()

class HealthyAgent:
    def predict(self, prob):
        return prob[:,0].item()

class MemoryAgent:
    def __init__(self):
        self.history = []

    def store(self, result):
        self.history.append(result)

    def get_history(self):
        return self.history

memory_agent = MemoryAgent()

# -----------------------------
# Feature Extraction
# -----------------------------
from feature_extraction import extract_features_from_audio

def extract_features(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name
        
    features = extract_features_from_audio(temp_path)
    return features

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Parkinson Voice Detection System")

st.write("Upload a WAV voice file to detect Parkinson’s disease risk.")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

# -----------------------------
# Prediction Pipeline
# -----------------------------
if uploaded_file is not None:

    st.audio(uploaded_file)

    try:

        features = extract_features(uploaded_file)

        audio_features = features.reshape(1,-1)

        # Ensure we have 16 features before predicting
        if audio_features.shape[1] != 16:
            st.error("Feature extraction failed to produce exactly 16 features.")
        else:
            # preprocessing
            audio_scaled = scaler.transform(audio_features)
    
            sample = torch.tensor(audio_scaled, dtype=torch.float32)
    
            # prediction
            with torch.no_grad():
                output = model(sample)
                prob = torch.softmax(output, dim=1)
    
            pd_agent = ParkinsonAgent()
            healthy_agent = HealthyAgent()
    
            pd_score = pd_agent.predict(prob)
            healthy_score = healthy_agent.predict(prob)
    
            pd_risk = pd_score * 100
            healthy_risk = healthy_score * 100
    
            # decision
            if pd_score > healthy_score:
                result = "Parkinson Detected"
            else:
                result = "Healthy"
    
            memory_agent.store(result)

            # display
            st.subheader("Prediction Confidence")
    
            st.write("Parkinson Probability:", round(pd_risk,2), "%")
            st.write("Healthy Probability:", round(healthy_risk,2), "%")
    
            st.subheader("Prediction History")
    
            st.write(memory_agent.get_history())
    
            if result == "Parkinson Detected":
                st.error("Final Decision: Parkinson Detected")
            else:
                st.success("Final Decision: Healthy")

    except Exception as e:
        st.error(f"Could not extract voice features from this audio: {e}")

