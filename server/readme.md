# Real-Time Fraud Detection using Enhanced Anomaly Detection Models

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-red.svg)
![Flask](https://img.shields.io/badge/Flask-green.svg)


A production-ready fraud detection engine that combines **Isolation Forest** and **Autoencoder** models to flag suspicious financial transactions in real-time. Designed and deployed during internship at **Punjab National Bank**, this system supports **real-time API scoring**, **synthetic transaction simulation**, and an interactive **dashboard interface**.

---

## 🔍 Project Highlights

- ✅ **Hybrid Ensemble**: Weighted fusion (60% IF + 40% AE) for precision-recall optimized scoring.
- ⚡ **Real-Time Detection**: Sub-100ms response via Flask REST API.
- 🧠 **Unsupervised Models**: Works well even with limited labeled data.
- 📊 **Advanced Features**: 20+ behavior-based, temporal, and statistical features.
- 🧪 **Simulator**: Synthetic transaction stream for testing system robustness.
- 📈 **Dashboard**: Real-time fraud monitoring interface with interactive controls.

---

## 🚀 Performance Summary

| Metric       | Hybrid Model | Isolation Forest | Autoencoder |
|--------------|--------------|------------------|-------------|
| **Precision** | 75.20%       | 70.07%           | 63.28%      |
| **Recall**    | 84.95%       | 69.91%           | 82.18%      |
| **F1-Score**  | 79.78%       | 69.99%           | 71.50%      |
| **ROC-AUC**   | 98.12%       | 99.18%           | 99.13%      |
| **Specificity** | 99.49%     | 99.45%           | 99.13%      |
| **Latency**   | < 100ms      | ✓                | ✓           |

---

---

## 🧩 Key Components

### 🔍 Feature Engineering
- **Customer Metrics**: mean/std/iqr spending, CV, z-scores
- **Amount Analysis**: log/sqrt transformations, binary outlier flags
- **Temporal Features**: simulated weekends/month ends
- **Merchant Profiling**: frequency, diversity, fraud risk encoding

### 🤖 Model Pipeline
- **Isolation Forest**:
  - Tree-based unsupervised model
  - Tuned contamination: 0.012
- **Autoencoder**:
  - Deep neural network (PyTorch)
  - Trained on normal transactions (MSE loss)
- **Hybrid Score**:
  - Weighted score: `0.6 * IF + 0.4 * AE`
  - Threshold: 0.5127 (F1 optimized)

---

## 🛠️ Tech Stack

| Category          | Stack                           |
|-------------------|----------------------------------|
| **Language**      | Python 3.10+                    |
| **Backend**       | Flask                           |
| **ML Frameworks** | Scikit-learn, PyTorch           |
| **Visualization** | Matplotlib, Seaborn             |
| **Web UI**        | HTML, CSS, JS (app.js)          |
| **Simulator**     | Python + Randomized Streams     |

---

## ⚙️ How to Run

### Setup
```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt


