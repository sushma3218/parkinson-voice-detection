# Parkinson Voice Detection AI Agent Training

## Agent Name
Parkinson Voice Detection Assistant

---

# Project Overview

This project implements an AI system for detecting Parkinson’s disease using voice recordings.

Parkinson’s disease is a progressive neurological disorder that affects motor control and speech production. Research has shown that subtle abnormalities in speech signals can act as early biomarkers for Parkinson’s disease.

The goal of this project is to build an intelligent system that analyzes voice features and predicts the likelihood of Parkinson’s disease.

---

# System Architecture

The project pipeline consists of the following stages:

Voice Input  
↓  
Audio Preprocessing  
↓  
Feature Extraction  
↓  
Feature Scaling  
↓  
Dimensionality Reduction (PCA)  
↓  
FT Transformer Model  
↓  
Multi-Agent Decision System  
↓  
Prediction Output

---

# Dataset Information

The model is trained on a Parkinson voice dataset containing acoustic features extracted from speech signals.

Important biomedical voice features include:

- Jitter
- Shimmer
- Harmonic-to-Noise Ratio (HNR)
- Recurrence Period Density Entropy (RPDE)
- Detrended Fluctuation Analysis (DFA)
- Pitch Period Entropy (PPE)

These features represent speech instability patterns associated with Parkinson’s disease.

---

# Data Processing Steps

## 1. Data Cleaning
The dataset is cleaned and formatted to ensure correct feature representation.

## 2. Feature Scaling
StandardScaler is applied to normalize all features.

Example:

StandardScaler → Normalize feature values

---

# Handling Class Imbalance

Medical datasets often contain imbalanced classes.

To address this issue the project uses:

Synthetic Minority Oversampling Technique (SMOTE)

This technique generates synthetic samples of the minority class to balance the dataset.

---

# Synthetic Data Generation

Gaussian Copula modeling is used to generate synthetic samples that improve dataset diversity.

Benefits:

- increases dataset size
- improves model generalization
- reduces overfitting

---

# Dimensionality Reduction

Principal Component Analysis (PCA) is applied to reduce high dimensional feature space.

Example:

754 features  
↓  
PCA  
↓  
100 principal components

Benefits:

- removes redundant features
- improves model efficiency
- reduces noise

---

# Machine Learning Model

The system uses an FT Transformer model designed for tabular datasets.

Model architecture:

Input Layer  
↓  
Fully Connected Layer (64 neurons)  
↓  
ReLU Activation  
↓  
Output Layer (2 classes)

Output classes:

0 → Healthy  
1 → Parkinson

---

# Quantum Machine Learning Component

A Quantum Support Vector Machine (QSVM) experiment is performed to compare classical deep learning with quantum-inspired classification.

Purpose:

- explore quantum machine learning techniques
- evaluate classification performance

---

# Model Evaluation

The model is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Example evaluation results:

Accuracy ≈ 95%

Confusion Matrix Example:

True Healthy predicted Healthy: 110  
True Healthy predicted Parkinson: 3  
True Parkinson predicted Healthy: 10  
True Parkinson predicted Parkinson: 133

---

# Multi-Agent Decision Framework

The system uses a multi-agent architecture to improve prediction reliability.

Agents include:

## Parkinson Agent
Calculates probability of Parkinson’s disease.

## Healthy Agent
Calculates probability of healthy speech.

## Memory Agent
Stores prediction history and previous outputs.

Final decision is determined using agent confidence scores.

---

# Web Application

The trained model is deployed using Streamlit.

Web application features:

- Upload WAV voice file
- Play uploaded audio
- Extract acoustic features
- Run prediction using trained model
- Display Parkinson probability
- Display Healthy probability
- Store prediction history

Deployment pipeline:

Google Colab  
↓  
GitHub Repository  
↓  
Streamlit Cloud Deployment

---

# Current Limitations

The training dataset contains engineered biomedical features, while the deployed web application extracts raw spectral features from audio. This feature mismatch can lead to prediction variations.

Future improvements include integrating biomedical acoustic feature extraction using Praat-based analysis.

---

# Future Improvements

1. Extract full biomedical voice features from audio using Praat.
2. Retrain model using the same feature extraction pipeline.
3. Improve prediction stability and reliability.
4. Add waveform and spectrogram visualization in the web application.
5. Expand dataset with more voice recordings.

---

# Final Outcome

The project successfully combines:

- Voice signal processing
- Deep learning
- Quantum-inspired machine learning
- Multi-agent decision systems
- Web application deployment

to create an AI system capable of detecting Parkinson’s disease using voice biomarkers.

---
