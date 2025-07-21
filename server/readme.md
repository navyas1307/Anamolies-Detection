# Fraud Detection System with Real-Time Scoring API

This project implements a fraud detection pipeline that combines Autoencoder neural networks and Isolation Forests to detect anomalies in financial transactions. It also features a Flask-based REST API for real-time risk scoring and a transaction stream simulator to emulate production behavior.

---

## Project Structure

.
├── app.py # Flask REST API for scoring and streaming
├── train.py # Full model training and evaluation pipeline
├── EDA.py # Exploratory Data Analysis and data cleaning script
├── data_simulator.py # Simulator for streaming transaction data
├── ae.pt # Trained Autoencoder model
├── iforest.pkl # Trained Isolation Forest model
├── scaler.pkl # Scaler used during training
├── cat_encoder.pkl # Categorical encoder (TargetEncoding)
├── metadata.json # Metadata for feature ordering, threshold, etc.
├── cleaned_fraud_data.csv # Cleaned training dataset
└── frontend/ # Frontend (optional)


## Features

- Anomaly Detection using ensemble of Autoencoder and Isolation Forest
- Robust time-based feature engineering with rolling windows and velocity metrics
- Categorical encoding using target encoding
- Real-time scoring via Flask REST API
- Streaming capability using a custom simulator
- Precision/Recall-optimized thresholding with evaluation reporting

---

## Tech Stack

- Python 3.10+
- Flask (for API)
- PyTorch (for Autoencoder)
- Scikit-learn (for Isolation Forest and scaling)
- Pandas, NumPy (for data processing)
- Category Encoders
- Matplotlib, Seaborn (for EDA)

---

