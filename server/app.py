from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import joblib
import json
from data_simulator import DataSimulator
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, send_from_directory
import os
from flask_cors import CORS

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Global variables for models
autoencoder = None
iforest = None
scaler = None
cat_encoder = None
metadata = None
data_sim = None

def load_models():
    global autoencoder, iforest, scaler, cat_encoder, metadata, data_sim
    
    print("Loading models...")
    
    # Load metadata
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Metadata loaded: {metadata}")
    
    # Load autoencoder with correct architecture
    autoencoder = Autoencoder(metadata['input_dim'])
    autoencoder.load_state_dict(torch.load('ae.pt', map_location='cpu'))
    autoencoder.eval()
    
    # Load other models
    iforest = joblib.load('iforest.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Load categorical encoder if it exists
    if metadata['has_cat_encoder']:
        cat_encoder = joblib.load('cat_encoder.pkl')
        print("Categorical encoder loaded")
    else:
        cat_encoder = None
        print("No categorical encoder found")
    
    # Initialize data simulator
    try:
        data_sim = DataSimulator('bs140513_032310.csv')
        print("Data simulator initialized")
    except Exception as e:
        print(f"Warning: Could not initialize data simulator: {e}")
        data_sim = None
    
    print("Models loaded successfully!")

def preprocess_transaction_fixed(txn_data):
    """Improved preprocessing with realistic feature generation"""
    if isinstance(txn_data, dict):
        txn_data = [txn_data]
    
    df = pd.DataFrame(txn_data)
    
    # Convert step to proper day-based datetime
    base_date = datetime(2011, 1, 1)
    df['datetime'] = df['step'].apply(lambda x: base_date + timedelta(days=x))
    
    # Create proper time-based features
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['month'] = df['datetime'].dt.month
    df['quarter'] = df['datetime'].dt.quarter
    df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Create cyclical features
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # REALISTIC rolling features for single transactions
    for idx, row in df.iterrows():
        amount = row['amount']
        
        # More realistic base values for normal behavior
        base_count_7d = np.random.randint(1, 4)  # 1-3 transactions in 7 days
        base_count_30d = np.random.randint(5, 16)  # 5-15 transactions in 30 days
        
        # 7-day rolling features
        df.loc[idx, 'txn_count_7d'] = base_count_7d
        df.loc[idx, 'amt_sum_7d'] = amount * base_count_7d * 0.8  # Simulate lower spending
        df.loc[idx, 'amt_mean_7d'] = amount * 0.7  # Simulate lower average
        df.loc[idx, 'amt_std_7d'] = amount * 0.1  # Small standard deviation
        df.loc[idx, 'amt_max_7d'] = amount * 1.2
        df.loc[idx, 'amt_min_7d'] = amount * 0.5
        
        # 30-day rolling features
        df.loc[idx, 'txn_count_30d'] = base_count_30d
        df.loc[idx, 'amt_sum_30d'] = amount * base_count_30d * 0.9
        df.loc[idx, 'amt_mean_30d'] = amount * 0.8
        df.loc[idx, 'amt_std_30d'] = amount * 0.15
        
        # Velocity features
        df.loc[idx, 'txn_velocity_7d'] = base_count_7d / 7
        df.loc[idx, 'txn_velocity_30d'] = base_count_30d / 30
        
        # Deviation features
        df.loc[idx, 'amt_deviation_7d'] = abs(amount - df.loc[idx, 'amt_mean_7d'])
        df.loc[idx, 'amt_deviation_30d'] = abs(amount - df.loc[idx, 'amt_mean_30d'])
        
        # Enhanced features for fraud detection
        df.loc[idx, 'amt_diff_ratio'] = abs(amount - df.loc[idx, 'amt_mean_7d']) / (df.loc[idx, 'amt_mean_7d'] + 1e-8)
        df.loc[idx, 'amt_range_7d'] = df.loc[idx, 'amt_max_7d'] - df.loc[idx, 'amt_min_7d']
        df.loc[idx, 'amt_cv_7d'] = df.loc[idx, 'amt_std_7d'] / (df.loc[idx, 'amt_mean_7d'] + 1e-8)
        
        # Unique merchants
        df.loc[idx, 'uniq_merchants_7d'] = max(1, int(base_count_7d * 0.8))
        df.loc[idx, 'uniq_merchants_30d'] = max(1, int(base_count_30d * 0.6))
        
        # Merchant risk and frequency (simulated normal values)
        df.loc[idx, 'merchant_risk_score'] = amount * 0.5
        df.loc[idx, 'merchant_frequency'] = np.random.randint(5, 15)
        
        # Time-based features
        df.loc[idx, 'days_since_last_txn'] = np.random.uniform(1.0, 3.0)  # 1-3 days
        df.loc[idx, 'hours_since_last_txn'] = df.loc[idx, 'days_since_last_txn'] * 24
        
        # Cumulative features
        cumulative_count = np.random.randint(30, 100)  # 30-100 transactions
        df.loc[idx, 'cumulative_txn_count'] = cumulative_count
        df.loc[idx, 'cumulative_amount'] = amount * cumulative_count * 0.9
        df.loc[idx, 'avg_amount_to_date'] = amount * 0.7
        
        # Same day transactions
        same_day_count = np.random.randint(1, 3)  # 1-2 transactions per day
        df.loc[idx, 'same_day_txn_count'] = same_day_count
        df.loc[idx, 'same_day_total_amount'] = amount * same_day_count
        
        # Behavioral trend (normal behavior)
        df.loc[idx, 'amount_trend_7d'] = np.random.uniform(-0.1, 0.1)
    
    # Ensure required columns exist
    if 'merchant' not in df.columns:
        df['merchant'] = 'M' + str(np.random.randint(1000000, 9999999))
    if 'category' not in df.columns:
        categories = ['es_transportation', 'es_health', 'es_food', 'es_shopping', 'es_wellnessandbeauty']
        df['category'] = np.random.choice(categories)
    
    # Build feature matrix matching training
    base_features = [
        'amount', 'step', 'amt_diff_ratio', 'merchant_risk_score', 'merchant_frequency',
        'txn_count_7d', 'amt_sum_7d', 'amt_mean_7d', 'amt_std_7d', 'amt_max_7d', 'amt_min_7d',
        'txn_count_30d', 'amt_sum_30d', 'amt_mean_30d', 'amt_std_30d',
        'txn_velocity_7d', 'txn_velocity_30d', 'amt_range_7d', 'amt_cv_7d',
        'amt_deviation_7d', 'amt_deviation_30d', 'amount_trend_7d',
        'uniq_merchants_7d', 'uniq_merchants_30d',
        'days_since_last_txn', 'hours_since_last_txn', 'cumulative_txn_count', 'avg_amount_to_date',
        'day_of_year_sin', 'day_of_year_cos', 'month_sin', 'month_cos',
        'dow_sin', 'dow_cos', 'is_weekend', 'quarter', 
        'same_day_txn_count', 'same_day_total_amount'
    ]
    
    X = df[base_features].copy()
    
    # Add derived features (matching training)
    X['amount_zscore'] = (X['amount'] - X['amount'].mean()) / (X['amount'].std() + 1e-8)
    X['amount_percentile'] = X['amount'].rank(pct=True)
    X['amount_log'] = np.log1p(X['amount'])
    X['is_very_high_amount'] = (X['amount'] > X['amount'].quantile(0.99)).astype(int)
    X['is_high_amount'] = (X['amount'] > X['amount'].quantile(0.95)).astype(int)
    X['is_medium_high_amount'] = (X['amount'] > X['amount'].quantile(0.90)).astype(int)
    X['is_round_amount'] = (X['amount'] % 1 == 0).astype(int)
    X['is_very_round_amount'] = (X['amount'] % 100 == 0).astype(int)
    X['txn_freq_ratio'] = X['txn_velocity_7d'] / (X['txn_velocity_30d'] + 1e-8)
    X['amount_consistency'] = 1 / (X['amt_std_7d'] + 1)
    X['merchant_diversity'] = X['uniq_merchants_7d'] / (X['txn_count_7d'] + 1e-8)
    X['unusual_time'] = (X['dow_sin'] < -0.7).astype(int)
    X['very_frequent_txns'] = (X['same_day_txn_count'] > 5).astype(int)
    X['burst_activity'] = (X['hours_since_last_txn'] < 1).astype(int)
    X['amount_spike_2x'] = (X['amount'] > 2 * X['amt_mean_7d']).astype(int)
    X['amount_spike_3x'] = (X['amount'] > 3 * X['amt_mean_7d']).astype(int)
    X['amount_spike_5x'] = (X['amount'] > 5 * X['amt_mean_7d']).astype(int)
    X['freq_anomaly'] = (X['txn_velocity_7d'] > 1.5 * X['txn_velocity_30d']).astype(int)
    X['extreme_freq_anomaly'] = (X['txn_velocity_7d'] > 3 * X['txn_velocity_30d']).astype(int)
    X['merchant_change_rate'] = X['uniq_merchants_7d'] / (X['uniq_merchants_30d'] + 1e-8)
    X['new_merchant_indicator'] = (X['merchant_frequency'] == 1).astype(int)
    X['amount_vs_historical'] = X['amount'] / (X['avg_amount_to_date'] + 1e-8)
    X['amount_deviation_score'] = X['amt_deviation_7d'] / (X['amt_std_7d'] + 1e-8)
    
    # Handle categorical features
    if cat_encoder is not None:
        print("Applying categorical encoding...")
        try:
            cat_cols = ['merchant', 'category']
            cat_features = cat_encoder.transform(df[cat_cols])
            
            # Convert to DataFrame with proper column names
            if hasattr(cat_features, 'columns'):
                cat_features = cat_features.astype(float)
            else:
                cat_features = pd.DataFrame(cat_features, columns=cat_cols, index=df.index)
                cat_features = cat_features.astype(float)
            
            # Combine with numeric features
            X = pd.concat([X, cat_features], axis=1)
            print(f"After categorical encoding: {X.shape}")
            
        except Exception as e:
            print(f"Error in categorical encoding: {e}")
            # If categorical encoding fails, add zeros
            for col in cat_cols:
                X[col] = 0.0
    
    # Ensure we have exactly the right features in the right order
    expected_features = metadata['feature_cols']
    
    # Add missing features with zeros
    for feature in expected_features:
        if feature not in X.columns:
            X[feature] = 0.0
    
    # Reorder columns to match training order
    X = X[expected_features]
    
    # Clean up data
    X = X.fillna(0.0)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0.0)
    
    print(f"Final feature matrix shape: {X.shape}")
    return X, df

def calculate_risk_score_fixed(X, debug=False):
    """Calculate risk score with EXACT same normalization as training"""
    if debug:
        print(f"Input features shape: {X.shape}")
        print(f"Input features (first row): {X.iloc[0].to_dict()}")
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    if debug:
        print(f"Scaled features min: {X_scaled.min():.4f}, max: {X_scaled.max():.4f}")
    
    # Autoencoder reconstruction error
    X_tensor = torch.FloatTensor(X_scaled)
    with torch.no_grad():
        reconstructed = autoencoder(X_tensor)
        ae_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
    
    # Isolation Forest score
    if_scores = iforest.decision_function(X_scaled)
    
    if debug:
        print(f"Raw autoencoder errors: min={ae_errors.min():.6f}, max={ae_errors.max():.6f}")
        print(f"Raw isolation forest scores: min={if_scores.min():.4f}, max={if_scores.max():.4f}")
    
    # Get normalization parameters from training
    norm_params = metadata['normalization_params']
    
    # Normalize autoencoder errors EXACTLY like training
    ae_range = norm_params['ae_max'] - norm_params['ae_min']
    if ae_range > 0:
        ae_errors_norm = (ae_errors - norm_params['ae_min']) / ae_range
    else:
        ae_errors_norm = np.zeros_like(ae_errors)
    
    # Normalize isolation forest scores EXACTLY like training
    if_range = norm_params['if_max'] - norm_params['if_min']
    if if_range > 0:
        if_scores_norm = (if_scores - norm_params['if_min']) / if_range
    else:
        if_scores_norm = np.zeros_like(if_scores)
    
    # Invert IF scores for consistency (EXACTLY like training)
    if_scores_norm = 1 - if_scores_norm
    
    # Ensure non-negative scores (EXACTLY like training)
    ae_errors_norm = np.maximum(ae_errors_norm, 0)
    if_scores_norm = np.maximum(if_scores_norm, 0)
    
    # FIXED: Correct blending weights to match training (0.4 AE + 0.6 IF)
    blended_scores = 0.4 * ae_errors_norm + 0.6 * if_scores_norm
    
    if debug:
        print(f"Normalized AE errors: min={ae_errors_norm.min():.4f}, max={ae_errors_norm.max():.4f}, mean={ae_errors_norm.mean():.4f}")
        print(f"Normalized IF scores: min={if_scores_norm.min():.4f}, max={if_scores_norm.max():.4f}, mean={if_scores_norm.mean():.4f}")
        print(f"Final blended scores: min={blended_scores.min():.4f}, max={blended_scores.max():.4f}, mean={blended_scores.mean():.4f}")
        print(f"Threshold: {metadata['threshold']:.4f}")
        print(f"Score vs threshold: {blended_scores[0]:.4f} {'>' if blended_scores[0] > metadata['threshold'] else '<='} {metadata['threshold']:.4f}")
    
    return blended_scores

# FIXED: Updated risk level thresholds
def get_risk_level(score, threshold):
    """Convert score to risk level with better distribution"""
    if score > threshold * 1.3:  # High risk
        return "High"
    elif score > threshold * 1.0:  # Medium risk
        return "Medium"
    elif score > threshold * 0.7:  # Medium-low risk
        return "Medium-Low"
    else:                         # Low risk
        return "Low"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

@app.route('/score', methods=['POST'])
def score_transaction():
    try:
        data = request.json
        print(f"Received data: {data}")
        
        # Preprocess transaction(s)
        X, df_orig = preprocess_transaction_fixed(data)
        
        # Calculate risk scores
        scores = calculate_risk_score_fixed(X, debug=True)
        
        # Prepare response
        if isinstance(data, dict):
            # Single transaction
            score = float(scores[0])
            risk_level = get_risk_level(score, metadata['threshold'])
            is_anomaly = score > metadata['threshold']
            
            print(f"Final score: {score:.4f}, Risk level: {risk_level}, Is anomaly: {is_anomaly}")
            
            return jsonify({
                'risk_score': score,
                'risk_level': risk_level,
                'is_anomaly': is_anomaly,
                'threshold': metadata['threshold'],
                'debug_info': {
                    'feature_shape': X.shape,
                    'feature_columns': X.columns.tolist(),
                    'sample_features': X.iloc[0].to_dict() if len(X) > 0 else {}
                }
            })
        else:
            # Multiple transactions
            results = []
            for i, score in enumerate(scores):
                score = float(score)
                risk_level = get_risk_level(score, metadata['threshold'])
                is_anomaly = score > metadata['threshold']
                
                results.append({
                    'risk_score': score,
                    'risk_level': risk_level,
                    'is_anomaly': is_anomaly,
                    'threshold': metadata['threshold']
                })
            
            return jsonify(results)
            
    except Exception as e:
        import traceback
        print(f"Error in score_transaction: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/debug', methods=['POST'])
def debug_transaction():
    """Debug endpoint to understand scoring issues"""
    try:
        data = request.json
        print("=== DEBUGGING TRANSACTION SCORING ===")
        
        # Preprocess transaction
        X, df_orig = preprocess_transaction_fixed(data)
        print(f"Preprocessed features shape: {X.shape}")
        print(f"Feature columns: {X.columns.tolist()}")
        print(f"Expected columns: {metadata['feature_cols']}")
        
        if len(X) > 0:
            print(f"Feature values (first row):\n{X.iloc[0].to_dict()}")
            
            # Calculate scores with full debugging
            scores = calculate_risk_score_fixed(X, debug=True)
            
            return jsonify({
                'debug_complete': True,
                'score': float(scores[0]) if len(scores) > 0 else 0,
                'feature_info': {
                    'shape': X.shape,
                    'columns': X.columns.tolist(),
                    'values': X.iloc[0].to_dict()
                },
                'metadata': metadata,
                'normalization_params': metadata['normalization_params']
            })
        else:
            return jsonify({'error': 'No features generated'}), 400
            
    except Exception as e:
        import traceback
        print(f"Error in debug_transaction: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test_with_different_amounts():
    """Test endpoint with different transaction amounts"""
    try:
        # Test with different amounts to see score variation
        test_transactions = [
            {'step': 100, 'amount': 10, 'merchant': 'grocery_store', 'category': 'food_dining'},
            {'step': 100, 'amount': 50, 'merchant': 'gas_station', 'category': 'gas_transport'},
            {'step': 100, 'amount': 200, 'merchant': 'department_store', 'category': 'shopping'},
            {'step': 100, 'amount': 1000, 'merchant': 'electronics_store', 'category': 'shopping'},
            {'step': 100, 'amount': 5000, 'merchant': 'unknown_merchant', 'category': 'misc_net'}
        ]
        
        results = []
        for txn in test_transactions:
            print(f"Testing with transaction: {txn}")
            
            # Process it
            X, df_orig = preprocess_transaction_fixed(txn)
            scores = calculate_risk_score_fixed(X, debug=True)
            
            score = float(scores[0])
            risk_level = get_risk_level(score, metadata['threshold'])
            is_anomaly = score > metadata['threshold']
            
            results.append({
                'transaction': txn,
                'risk_score': score,
                'risk_level': risk_level,
                'is_anomaly': is_anomaly
            })
        
        return jsonify({
            'test_results': results,
            'threshold': metadata['threshold'],
            'summary': {
                'total_tested': len(test_transactions),
                'flagged_as_anomaly': sum(1 for r in results if r['is_anomaly']),
                'score_range': {
                    'min': min(r['risk_score'] for r in results),
                    'max': max(r['risk_score'] for r in results)
                }
            }
        })
        
    except Exception as e:
        import traceback
        print(f"Error in test_with_different_amounts: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/stream', methods=['GET'])
def stream_data():
    try:
        if data_sim is None:
            return jsonify({'error': 'Data simulator not available'}), 500
            
        # Get next batch of transactions
        batch = data_sim.get_next_batch(5)
        
        # Convert to list of dictionaries
        transactions = batch.to_dict('records')
        
        return jsonify(transactions)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize models when app starts
try:
    load_models()
    print("App initialized successfully!")
except Exception as e:
    print(f"Error initializing app: {e}")
    import traceback
    traceback.print_exc()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')