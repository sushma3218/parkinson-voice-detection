import streamlit as st
import librosa
import numpy as np
import torch
import pickle

# Load scaler and PCA
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

# Define model structure
class FTTransformer(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load model
model = FTTransformer(100)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

st.title("Parkinson Voice Detection System")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:

    st.audio(uploaded_file)

    y, sr = librosa.load(uploaded_file, sr=16000)

    y = librosa.util.normalize(y)

    y, _ = librosa.effects.trim(y)

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

    target_size = 754

    if audio_features.shape[1] < target_size:
        padding = np.zeros((1, target_size - audio_features.shape[1]))
        audio_features = np.concatenate((audio_features, padding), axis=1)

    audio_scaled = scaler.transform(audio_features)

    audio_pca = pca.transform(audio_scaled)

    sample = torch.tensor(audio_pca, dtype=torch.float32)

    with torch.no_grad():
        output = model(sample)
        prob = torch.softmax(output, dim=1)

    risk_score = prob[:,1].item()

    st.write("Parkinson Risk Score:", risk_score)

    if risk_score > 0.5:
        st.error("Prediction: Parkinson Detected")
    else:
        st.success("Prediction: Healthy")