import parselmouth
from parselmouth.praat import call
import librosa
import numpy as np

def extract_features_from_audio(audio_path):
    """
    Extracts a robust standardized set of biomedical and acoustic features from a WAV file.
    Features:
      1. Jitter (local)
      2. Shimmer (local)
      3. HNR (Harmonics-to-Noise Ratio)
      4-16. MFCCs (13 elements, representing vocal tract spectral shape)
    
    Total features: 16
    """
    try:
        # --- Parselmouth (Biomedical Features) ---
        sound = parselmouth.Sound(audio_path)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        # --- Librosa (Spectral Features: MFCC) ---
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0) # take mean across time
        
        # Combine all features into a single NumPy array
        # Replace NaNs with 0 if any extraction failed
        features = np.hstack(([jitter, shimmer, hnr], mfccs_mean))
        features = np.nan_to_num(features)
        
        return features

    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        # Return a zero vector of size 16 if extraction completely fails
        return np.zeros(16)
