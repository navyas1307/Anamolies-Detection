# Fraud Detection System with Real-Time Scoring API

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-red.svg)
![Flask](https://img.shields.io/badge/Flask-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A high-performance fraud detection pipeline that combines **Autoencoder neural networks** and **Isolation Forests** for real-time anomaly detection in financial transactions. Features a Flask-based REST API for instant risk scoring and a transaction stream simulator for production testing.

## Key Performance Metrics

- **Precision**: 70.06%
- **Recall**: 70.87%
- **ROC-AUC**: 99.18%
- **Real-time Processing**: < 100ms per transaction

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Raw Transaction│───▶│  Feature Engine  │───▶│ Ensemble Models │
│      Data       │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
                    ┌──────────────────┐    ┌─────────────────┐
                    │ Time-based       │    │ • Autoencoder   │
                    │ Rolling Windows  │    │ • Isolation     │
                    │ Velocity Metrics │    │   Forest        │
                    └──────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                            ┌─────────────────┐
                                            │ Risk Score API  │
                                            │ (0.0 - 1.0)     │
                                            └─────────────────┘
```

## Features

### Core Capabilities
- **Dual Model Ensemble**: Combines Autoencoder and Isolation Forest for robust detection
- **Real-time Scoring**: Sub-100ms response times via Flask REST API
- **Advanced Feature Engineering**: Time-based rolling windows and velocity metrics
- **Precision-Optimized**: Threshold optimization balancing precision and recall
- **Streaming Support**: Built-in transaction simulator for testing

### Technical Highlights
- **Smart Categorical Encoding**: Target encoding for categorical variables
- **Robust Preprocessing**: Automated scaling and feature standardization
- **Model Persistence**: Serialized models for instant deployment
- **Comprehensive Evaluation**: Detailed performance metrics and confusion matrices

## Tech Stack

| Category | Technology |
|----------|------------|
| **Backend** | Python 3.10+, Flask |
| **ML Framework** | PyTorch, Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Encoding** | Category Encoders |
| **Visualization** | Matplotlib, Seaborn |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system

# Create virtual environment
python -m venv fraud_env
source fraud_env/bin/activate  # On Windows: fraud_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation & Training

```bash
# Run exploratory data analysis
python EDA.py

# Train the models
python train.py
```

### Start the API Server

```bash
# Launch the Flask API
python app.py
```

The API will be available at `http://localhost:5000`

### Test with Simulator

```bash
# Start transaction stream simulator
python data_simulator.py
```

## API Usage

### Score Individual Transaction

**POST** `/score`

```bash
curl -X POST http://localhost:5000/score \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 1500.00,
    "merchant_category": "grocery",
    "transaction_time": "2024-01-15T14:30:00",
    "user_id": "user_12345",
    "card_type": "credit"
  }'
```

**Response:**
```json
{
  "risk_score": 0.23,
  "risk_level": "low",
  "model_scores": {
    "autoencoder": 0.21,
    "isolation_forest": 0.25
  },
  "processing_time_ms": 45
}
```

### Batch Scoring

**POST** `/batch_score`

```bash
curl -X POST http://localhost:5000/batch_score \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"amount": 1500.00, "merchant_category": "grocery", ...},
      {"amount": 5000.00, "merchant_category": "electronics", ...}
    ]
  }'
```

### Health Check

**GET** `/health`

```bash
curl http://localhost:5000/health
```

## Model Performance

### Training Results

Based on the latest training run on 49,934 transactions with 611 fraud cases:

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| **Isolation Forest** | 70.06% | 70.87% | 70.46% | 99.18% |
| **Autoencoder** | 70.03% | 69.23% | 69.63% | 99.13% |

### Confusion Matrix (Best Model - Isolation Forest)

|                | Predicted Normal | Predicted Fraud |
|----------------|------------------|-----------------|
| **Actual Normal** | 49,138 (TN) | 185 (FP) |
| **Actual Fraud** | 178 (FN) | 433 (TP) |

**Key Metrics:**
- **Accuracy**: 99.27%
- **Specificity**: 99.62%
- **False Positive Rate**: 0.38%

## Feature Engineering

The system automatically generates advanced features including:

- **Temporal Features**: Hour, day of week, month patterns
- **Rolling Statistics**: Transaction amount statistics over various windows
- **Velocity Metrics**: Transaction frequency and amount velocity
- **User Behavior**: Historical spending patterns and deviations
- **Merchant Analysis**: Category-based risk profiling

## Model Architecture

### Isolation Forest
- **Contamination**: 1.84% (automatically tuned)
- **Estimators**: 100 trees
- **Max Samples**: Auto-scaled based on dataset size
- **Threshold**: 0.5044 (precision-recall optimized)

### Autoencoder
- **Architecture**: [15 → 8 → 4 → 8 → 15]
- **Activation**: ReLU hidden layers, Linear output
- **Loss Function**: Mean Squared Error
- **Training**: 20 epochs with early stopping
- **Threshold**: 0.0058 (precision-recall optimized)

## Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t fraud-detection .

# Run container
docker run -p 5000:5000 fraud-detection
```

### Production Considerations

- **Load Balancing**: Use nginx or similar for high-traffic scenarios
- **Monitoring**: Implement logging and metrics collection
- **Model Updates**: Schedule periodic retraining with new data
- **Security**: Add authentication and rate limiting for production APIs

## Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## Requirements

Create a `requirements.txt` file with:

```txt
torch>=1.9.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
flask>=2.0.0
category-encoders>=2.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch and Scikit-learn
- Inspired by modern fraud detection techniques
- Optimized for production deployment

---

**Note**: This system is designed for educational and research purposes. For production fraud detection, ensure compliance with relevant financial regulations and data privacy laws.
