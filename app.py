import streamlit as st
import numpy as np
import torch
import pickle
import random
import parselmouth
from parselmouth.praat import call

# -----------------------------
# Fix randomness
# -----------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------
# Load preprocessing models
# -----------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

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
model = FTTransformer(100)
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
# Feature Extraction (Praat)
# -----------------------------
def extract_features(audio_file):

    sound = parselmouth.Sound(audio_file)

    pitch = call(sound, "To Pitch", 0.0, 75, 500)

    pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 500)

    jitter = call(pointProcess, "Get jitter (local)", 0,0,0.0001,0.02,1.3)

    shimmer = call([sound, pointProcess],
                   "Get shimmer (local)",0,0,0.0001,0.02,1.3,1.6)

    harmonicity = call(sound, "To Harmonicity (cc)",0.01,75,0.1,1.0)

    hnr = call(harmonicity, "Get mean",0,0)

    return np.array([jitter, shimmer, hnr])


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

        # preprocessing
        audio_scaled = scaler.transform(audio_features)

        audio_pca = pca.transform(audio_scaled)

        sample = torch.tensor(audio_pca, dtype=torch.float32)

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

    except:
        st.error("Could not extract voice features from this audio.")
