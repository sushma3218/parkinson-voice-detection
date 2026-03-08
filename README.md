# Parkinson Voice Detection System

This project implements an AI system for detecting Parkinson’s disease from voice recordings. It uses an **FT-Transformer** model to analyze a robust set of 16 biomedical and spectral features extracted straight from `.wav` files.

## Architecture Highlights
- **`feature_extraction.py`**: A unified script extracting Jitter, Shimmer, HNR (via Parselmouth/Praat) and MFCCs (via Librosa). This guarantees that the features we train on are **exactly identical** to those used in the Streamlit web application.
- **`train_model.py`**: A dedicated retraining script that converts raw `.wav` datasets into the 16-dimensional feature arrays and trains the PyTorch model (`FTTransformer`).
- **`app.py`**: The Streamlit user interface that handles live audio uploads, invokes `feature_extraction.py`, and outputs a predicted healthy vs. Parkinson confidence score.

## Installation

Ensure you have installed the core dependencies inside your Python environment:

```bash
pip install -r requirements.txt
```

## How to use

### 1. Retrain the model (CRITICAL FIRST STEP)
Currently, the pipeline has been upgraded from an initial 754 raw feature set to a more robust, standardized 16 feature set directly extractable during inference. Therefore, **you must execute the training script** on your dataset so that a new `model.pth` and `scaler.pkl` are generated with the correct 16-feature input size.

1. Ensure your original dataset `.wav` files are arranged in two folders (e.g., `data/healthy/` and `data/parkinson/`).
2. Open `train_model.py` and set the path referencing your dataset directory at the bottom of the script.
3. Run the script:
   ```bash
   python train_model.py
   ```
4. Confirm `model.pth` and `scaler.pkl` update successfully in the repository root.

### 2. Launch the Streamlit App
Once successfully retrained on the 16 audio features, you can launch the live web application:

```bash
streamlit run app.py
```
