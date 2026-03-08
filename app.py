import streamlit as st
import librosa
import numpy as np
import torch
import pickle
import random

# -----------------------------
# Fix randomness (stable results)
# -----------------------------

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

# -----------------------------
# Load preprocessing models
# -----------------------------

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

# -----------------------------
# Define FT Transformer model
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

    # Load audio
    y, sr = librosa.load(uploaded_file, sr=16000, mono=True)

    # Normalize
    y = librosa.util.normalize(y)

    # Remove silence
    y, _ = librosa.effects.trim(y)

    # -----------------------------
    # Feature extraction
    # -----------------------------

    features = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(bandwidth))

    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    audio_features = np.array(features).reshape(1, -1)

    # -----------------------------
    # Match dataset feature size
    # -----------------------------

    target_size = 754

    audio_features = np.pad(
        audio_features,
        ((0,0),(0,max(0,target_size-audio_features.shape[1]))),
        mode='constant'
    )

    audio_features = audio_features[:, :target_size]

    # -----------------------------
    # Apply preprocessing
    # -----------------------------

    audio_scaled = scaler.transform(audio_features)

    audio_pca = pca.transform(audio_scaled)

    sample = torch.tensor(audio_pca, dtype=torch.float32)

    # -----------------------------
    # Model prediction
    # -----------------------------

    with torch.no_grad():
        output = model(sample)
        prob = torch.softmax(output, dim=1)

    # -----------------------------
    # Multi-Agent Decision
    # -----------------------------

    pd_agent = ParkinsonAgent()
    healthy_agent = HealthyAgent()

    pd_score = pd_agent.predict(prob)
    healthy_score = healthy_agent.predict(prob)

    pd_risk = pd_score * 100
    healthy_risk = healthy_score * 100

    if pd_score >= 0.60:
        result = "Parkinson Detected"

    elif healthy_score >= 0.60:
        result = "Healthy"

    else:
        result = "Uncertain (Low Confidence)"

    memory_agent.store(result)

    # -----------------------------
    # Display results
    # -----------------------------

    st.subheader("Prediction Confidence")

st.write("Parkinson Probability:", round(pd_score,3))
st.write("Healthy Probability:", round(healthy_score,3))

    st.subheader("Prediction History")

    st.write(memory_agent.get_history())

    if pd_score > healthy_score + 0.15:
    result = "Parkinson Detected"

elif healthy_score > pd_score + 0.15:
    result = "Healthy"

else:
    result = "Uncertain (Voice not clear)"

