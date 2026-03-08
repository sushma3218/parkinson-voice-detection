import os
import glob
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from feature_extraction import extract_features_from_audio

# Define Model Architecture
class FTTransformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_new_model(dataset_dir):
    """
    Reads healthy and parkinson audio files, extracts standard features,
    trains a new FTTransformer, and saves the new model and scaler.
    """
    print("Extracting features from dataset...")
    
    features_list = []
    labels_list = []

    # Assuming directory structure:
    # dataset/healthy/*.wav
    # dataset/parkinson/*.wav
    
    healthy_files = glob.glob(os.path.join(dataset_dir, "healthy", "*.wav"))
    pd_files = glob.glob(os.path.join(dataset_dir, "parkinson", "*.wav"))
    
    for f in healthy_files:
        feats = extract_features_from_audio(f)
        features_list.append(feats)
        labels_list.append(0) # 0 for healthy
        
    for f in pd_files:
        feats = extract_features_from_audio(f)
        features_list.append(feats)
        labels_list.append(1) # 1 for parkinson

    X = np.array(features_list)
    y = np.array(labels_list)
    
    if len(X) == 0:
        print("Error: No audio files found to train on. Please check the dataset directory.")
        return

    print(f"Extracted dataset shape: {X.shape}")

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the new scaler (replacing the old one)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Saved new scaler to scaler.pkl")
    
    # We no longer need PCA since we reduced the feature count natively down to 16.
    # To keep app.py from breaking, we save a mock PCA or omit it. 
    # For now, we will simply not use PCA in the new app.py

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Create the model
    input_dim = X.shape[1]
    model = FTTransformer(input_dim)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training model...")
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Save the new model
    torch.save(model.state_dict(), "model.pth")
    print("Saved new model weights to model.pth")
    print(f"Update app.py input_dim to {input_dim}")

if __name__ == "__main__":
    # Example usage: Run this when you have your raw dataset ready.
    # train_new_model("path/to/my_audio_dataset")
    print("To retrain, provide the path to your dataset containing 'healthy' and 'parkinson' folders.")
