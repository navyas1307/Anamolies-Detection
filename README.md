# Banking Fraud Detection System

This is an end-to-end fraud detection system developed during a banking internship. It identifies suspicious financial transactions using a hybrid unsupervised learning approach, combining **Autoencoders** and **Isolation Forests**. The system includes data preprocessing, feature engineering, model training, and a real-time scoring API with dynamic threshold optimization.

---

## Project Highlights

- **Autoencoder + Isolation Forest ensemble** for improved anomaly detection.
- **80% Recall on actual fraud cases** with optimized precision-recall balance.
- **Robust Feature Engineering**: Velocity, deviation, rolling stats, merchant behavior.
- **Flask API** for real-time transaction scoring and fraud risk prediction.
- **Dynamic data simulation** for frontend/backend testing.

---

## Performance Summary

| Metric        | Value   |
|---------------|---------|
| Accuracy      | 98.66%  |
| Recall        | 80.18%  |
| ROC-AUC       | 0.9896  |
| Anomalies Detected | 1,681 of 80,000 |

---

## Model Architecture

### 🔹 Autoencoder
- 5-layer encoder-decoder with ReLU and Dropout
- Trained to minimize MSE reconstruction loss
- Learns typical transaction patterns

### 🔹 Isolation Forest
- Trained on scaled feature vectors
- Detects rare isolation patterns in data
- Boosts ensemble’s ability to generalize

### 🔹 Ensemble Scoring
- Final Score = `0.6 * AE Error + 0.4 * (1 - IF Score)`
- Threshold optimized via precision-recall F1 balance

---

## Code Structure

```bash
Anamolies-Detector/
│
├── frontend/                    # Frontend files (HTML, CSS, JS)
│   ├── index.html              # Main web interface
│   ├── style.css               # Styling for the UI
│   └── script.js               # Frontend logic and API calls
│
├── server/                     # Backend and ML logic
│   ├── app.py                  # Flask app to handle API routes
│   ├──data_simulator.py
|   |── model.py                # Anomaly detection logic (Isolation Forest, LOF)
│   └── EDA.py                  # Exploratory Data Analysis scripts
│
├── README.md                   # Project overview and instructions
├── requirements.txt            # Python dependencies
└── .gitignore                  # Files/folders to ignore in version control

