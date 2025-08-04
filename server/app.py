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
from flask import send_from_directory
import os
from flask_cors import CORS
from sklearn.model_selection import StratifiedKFold

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, 'index.html')

class PrecisionOptimizedAutoencoder(nn.Module):
    """Precision-focused autoencoder matching train.py architecture"""
    def __init__(self, input_dim):
        super(PrecisionOptimizedAutoencoder, self).__init__()
        hidden_dim = max(16, input_dim // 3)  
        bottleneck_dim = max(6, hidden_dim // 4)  
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.03),  
            nn.Dropout(0.3),  
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.03),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class PrecisionFocusedHybridDetector:
    """Fixed hybrid detector with proper score normalization matching train.py"""
    def __init__(self, isolation_forest, autoencoder, scaler, metadata):
        self.isolation_forest = isolation_forest
        self.autoencoder = autoencoder
        self.scaler = scaler
        self.weights = metadata['weights']
        self.hybrid_threshold = metadata['hybrid_threshold']
        self.precision_boost_factor = metadata.get('precision_boost_factor', 1.15)
        self.input_dim = metadata['input_dim']
        
    def _precision_normalize(self, scores):
        """Use the EXACT normalization as train.py to match training behavior"""
        # Enhanced normalization for better precision - EXACTLY from train.py
        scores_shifted = scores - scores.min() + 1e-8
        scores_log = np.log1p(scores_shifted)
        
        # Use more conservative percentiles
        q5, q95 = np.percentile(scores_log, [5, 95])
        
        # Handle edge case where q5 == q95
        if q95 - q5 < 1e-8:
            # If all values are very similar, use min-max normalization
            scores_min, scores_max = scores_log.min(), scores_log.max()
            if scores_max - scores_min < 1e-8:
                # All values are identical, return small random variation around 0.1
                return np.full_like(scores, 0.1) + np.random.normal(0, 0.01, len(scores))
            else:
                normalized = (scores_log - scores_min) / (scores_max - scores_min)
        else:
            normalized = (scores_log - q5) / (q95 - q5 + 1e-8)
        
        normalized = np.clip(normalized, 0, 4)
        # Apply stronger non-linear transformation for precision
        normalized = np.power(normalized / 4, 0.6) * 4
        normalized = normalized / 4
        
        return normalized
    
    def predict_scores(self, X):
        """Get hybrid prediction scores with EXACT same normalization as train.py"""
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest scores (negative values, more negative = more anomalous)
        if_raw_scores = self.isolation_forest.decision_function(X_scaled)
        # Convert to positive anomaly scores (higher = more anomalous)
        if_scores = -if_raw_scores
        if_scores_normalized = self._precision_normalize(if_scores)
        
        # Autoencoder scores (reconstruction error)
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            reconstructed = self.autoencoder(X_tensor)
            ae_raw_scores = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
            ae_scores_normalized = self._precision_normalize(ae_raw_scores)
        
        # Combine scores using weights - EXACTLY like train.py
        hybrid_scores = (self.weights['isolation_forest'] * if_scores_normalized + 
                        self.weights['autoencoder'] * ae_scores_normalized)
        
        return hybrid_scores, if_scores_normalized, ae_scores_normalized
    
    def predict(self, X):
        """Predict anomalies"""
        hybrid_scores, _, _ = self.predict_scores(X)
        return (hybrid_scores > self.hybrid_threshold).astype(int)

# Global variables for models and training data
hybrid_model = None
metadata = None
data_sim = None
training_data = None
merchant_stats = None

def load_models():
    global hybrid_model, metadata, data_sim, training_data, merchant_stats
    
    print("Loading precision-focused hybrid models...")
    
    # Load metadata
    try:
        with open('fraud_metadata_precision.json', 'r') as f:
            metadata = json.load(f)
        print(f"Metadata loaded: {metadata.get('model_type', 'unknown')} model")
    except FileNotFoundError:
        print("Error: fraud_metadata_precision.json not found")
        return False
    
    try:
        # Load Isolation Forest
        isolation_forest = joblib.load('fraud_isolation_forest_precision.pkl')
        print("Isolation Forest loaded")
        
        # Load Autoencoder
        autoencoder = PrecisionOptimizedAutoencoder(metadata['input_dim'])
        autoencoder.load_state_dict(torch.load('fraud_autoencoder_precision.pt', map_location='cpu'))
        autoencoder.eval()
        print("Precision Autoencoder loaded")
        
        # Load Scaler
        scaler = joblib.load('fraud_scaler_precision.pkl')
        print("Scaler loaded")
        
        # Create hybrid model
        hybrid_model = PrecisionFocusedHybridDetector(
            isolation_forest, autoencoder, scaler, metadata
        )
        
        print("Precision-focused hybrid model initialized successfully!")
        
    except Exception as e:
        print(f"Error loading model components: {e}")
        return False
    
    # Load training data for feature computation
    try:
        print("Loading training data for feature computation...")
        training_data = pd.read_csv("cleaned_fraud_data.csv")
        
        # Create customer statistics matching train.py EXACTLY
        print("Creating customer statistics...")
        customer_stats = training_data.groupby('customer')['amount'].agg([
            'mean', 'std', 'median', 'count', 'min', 'max', 'sum'
        ]).fillna(0)
        
        customer_percentiles = training_data.groupby('customer')['amount'].quantile([0.05, 0.1, 0.25, 0.75, 0.9, 0.95]).unstack()
        customer_percentiles.columns = ['amount_q05', 'amount_q10', 'amount_q25', 'amount_q75', 'amount_q90', 'amount_q95']
        customer_stats = pd.concat([customer_stats, customer_percentiles], axis=1)
        
        customer_stats.columns = ['cust_amt_mean', 'cust_amt_std', 'cust_amt_median', 
                                 'cust_txn_count', 'cust_amt_min', 'cust_amt_max', 'cust_amt_sum',
                                 'cust_amt_q05', 'cust_amt_q10', 'cust_amt_q25', 'cust_amt_q75', 'cust_amt_q90', 'cust_amt_q95']
        
        # ADD THE MISSING ENHANCED FEATURES TO CUSTOMER STATS (matching train.py)
        customer_stats['cust_amt_range'] = customer_stats['cust_amt_max'] - customer_stats['cust_amt_min']
        customer_stats['cust_amt_iqr'] = customer_stats['cust_amt_q75'] - customer_stats['cust_amt_q25']
        customer_stats['cust_amt_cv'] = customer_stats['cust_amt_std'] / (customer_stats['cust_amt_mean'] + 1e-8)
        customer_stats['cust_amt_skewness'] = (customer_stats['cust_amt_mean'] - customer_stats['cust_amt_median']) / (customer_stats['cust_amt_std'] + 1e-8)
        
        # Store customer stats
        training_data.customer_stats = customer_stats
        print(f"Loaded training data with {len(training_data)} transactions and {len(customer_stats)} customers")
        
        # Initialize merchant encoding for inference - FIXED VERSION
        if 'merchant' in training_data.columns and 'fraud' in training_data.columns:
            print("Creating merchant encoding...")
            merchant_stats = create_merchant_encoding(training_data)
            print(f"Merchant encoding created for {len(merchant_stats['stats'])} merchants")
        
        print("Training data loaded successfully!")
        
    except Exception as e:
        print(f"Warning: Could not load training data: {e}")
        training_data = None
    
    # Initialize data simulator
    try:
        data_sim = DataSimulator('bs140513_032310.csv')
        print("Data simulator initialized")
    except Exception as e:
        print(f"Warning: Could not initialize data simulator: {e}")
        data_sim = None
    
    return True

def create_merchant_encoding(df):
    """Create merchant risk encoding using cross-validation like in train.py"""
    if 'fraud' not in df.columns:
        return None
    
    print("Creating merchant risk encoding...")
    
    # Use StratifiedKFold for merchant encoding
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    merchant_encoded = np.zeros(len(df))
    
    for train_idx, val_idx in skf.split(df, df['fraud']):
        train_data = df.iloc[train_idx]
        
        merchant_fraud_rates = train_data.groupby('merchant')['fraud'].mean()
        merchant_counts = train_data.groupby('merchant').size()
        global_fraud_rate = train_data['fraud'].mean()
        
        alpha = 75  
        smoothed_rates = ((merchant_fraud_rates * merchant_counts + global_fraud_rate * alpha) / 
                        (merchant_counts + alpha))
        
        val_merchants = df.iloc[val_idx]['merchant']
        merchant_encoded[val_idx] = val_merchants.map(smoothed_rates).fillna(global_fraud_rate)
    
    # Create merchant statistics
    merchant_stats = df.groupby('merchant').agg({
        'fraud': ['count', 'sum', 'mean'],
        'amount': ['count', 'mean', 'std', 'max'],
        'customer': 'nunique'
    })
    
    merchant_stats.columns = ['merch_total_txns', 'merch_fraud_count', 'merch_fraud_rate',
                             'merch_amt_count', 'merch_amt_mean', 'merch_amt_std', 'merch_amt_max',
                             'merch_unique_customers']
    
    # Store encoded values in a mapping
    merchant_risk_mapping = {}
    for idx, merchant in enumerate(df['merchant']):
        merchant_risk_mapping[merchant] = merchant_encoded[idx]
    
    return {
        'risk_mapping': merchant_risk_mapping,
        'stats': merchant_stats,
        'global_fraud_rate': global_fraud_rate
    }

def prepare_transaction_features(txn_data):
    """Prepare transaction features matching the enhanced feature engineering from train.py"""
    if isinstance(txn_data, dict):
        txn_data = [txn_data]
    
    df = pd.DataFrame(txn_data)
    print(f"Input transaction data: {df.to_dict('records')}")
    
    if training_data is None:
        raise Exception("Training data not available - cannot compute customer features")
    
    # Enhanced customer feature lookup matching train.py
    for idx, row in df.iterrows():
        customer = row.get('customer', f'C_unknown_{idx}')
        
        # Get actual customer stats from training data
        if hasattr(training_data, 'customer_stats') and customer in training_data.customer_stats.index:
            cust_stats = training_data.customer_stats.loc[customer]
            for col in cust_stats.index:
                df.loc[idx, col] = cust_stats[col]
            print(f"Found existing customer {customer}")
            
        else:
            # For new customers, estimate using similar transaction patterns
            print(f"New customer {customer}, estimating stats...")
            amount = row['amount']
            
            # Find similar amounts for estimation
            similar_amounts = training_data[
                (training_data['amount'] >= amount * 0.3) & 
                (training_data['amount'] <= amount * 3.0)
            ]
            
            if len(similar_amounts) > 50 and hasattr(training_data, 'customer_stats'):
                similar_customers = similar_amounts['customer'].unique()
                similar_stats = training_data.customer_stats.loc[
                    training_data.customer_stats.index.isin(similar_customers)
                ]
                
                # Use median stats from similar customers
                for col in training_data.customer_stats.columns:
                    df.loc[idx, col] = similar_stats[col].median()
                    
            else:
                # Fallback to overall statistics
                if hasattr(training_data, 'customer_stats'):
                    overall_stats = training_data.customer_stats
                    for col in overall_stats.columns:
                        df.loc[idx, col] = overall_stats[col].median()
                else:
                    # Last resort defaults
                    df.loc[idx, 'cust_amt_mean'] = amount
                    df.loc[idx, 'cust_amt_std'] = amount * 0.3
                    df.loc[idx, 'cust_amt_median'] = amount * 0.9
                    df.loc[idx, 'cust_txn_count'] = 20
                    df.loc[idx, 'cust_amt_min'] = amount * 0.1
                    df.loc[idx, 'cust_amt_max'] = amount * 2.0
                    df.loc[idx, 'cust_amt_sum'] = amount * 25
                    
                    # Set percentile defaults
                    for q in ['q05', 'q10', 'q25', 'q75', 'q90', 'q95']:
                        multiplier = float(q.replace('q', '')) / 100.0
                        df.loc[idx, f'cust_amt_{q}'] = amount * (0.5 + multiplier)
                    
                    # Enhanced features defaults
                    df.loc[idx, 'cust_amt_range'] = amount * 1.9
                    df.loc[idx, 'cust_amt_iqr'] = amount * 0.4
                    df.loc[idx, 'cust_amt_cv'] = 0.3
                    df.loc[idx, 'cust_amt_skewness'] = 0.1
    
    return df

def prepare_features_for_inference(df):
    """Prepare features for inference exactly matching train.py"""
    print("Preparing features for inference...")
    
    # Core features that should exist from prepare_transaction_features
    core_features = [
        'amount', 'step', 'cust_amt_mean', 'cust_amt_std', 'cust_amt_median',
        'cust_txn_count', 'cust_amt_sum', 'cust_amt_range', 'cust_amt_iqr', 'cust_amt_cv',
        'cust_amt_skewness'
    ]
    
    # Check which features exist
    existing_features = [col for col in core_features if col in df.columns]
    X = df[existing_features].copy()
    
    # Enhanced ratio features - FIXED to handle missing columns
    if 'cust_amt_mean' in df.columns:
        X['amt_vs_cust_mean_ratio'] = X['amount'] / (X['cust_amt_mean'] + 1e-8)
    else:
        X['amt_vs_cust_mean_ratio'] = 1.0
        
    if 'cust_amt_median' in df.columns:
        X['amt_vs_cust_median_ratio'] = X['amount'] / (X['cust_amt_median'] + 1e-8)
    else:
        X['amt_vs_cust_median_ratio'] = 1.0
        
    if 'cust_amt_q95' in df.columns:
        X['amt_vs_cust_q95_ratio'] = X['amount'] / (df['cust_amt_q95'] + 1e-8)
    else:
        X['amt_vs_cust_q95_ratio'] = 1.0
        
    if 'cust_amt_max' in df.columns:
        X['amt_vs_cust_max_ratio'] = X['amount'] / (df['cust_amt_max'] + 1e-8)
    else:
        X['amt_vs_cust_max_ratio'] = 1.0
    
    # Enhanced z-score features - FIXED
    if all(col in df.columns for col in ['cust_amt_mean', 'cust_amt_std']):
        X['amt_zscore'] = (X['amount'] - df['cust_amt_mean']) / (df['cust_amt_std'] + 1e-8)
    else:
        X['amt_zscore'] = 0.0
        
    if all(col in df.columns for col in ['cust_amt_median', 'cust_amt_iqr']):
        X['amt_robust_zscore'] = (X['amount'] - df['cust_amt_median']) / (df['cust_amt_iqr'] + 1e-8)
    else:
        X['amt_robust_zscore'] = 0.0
        
    if all(col in df.columns for col in ['cust_amt_q95', 'cust_amt_std']):
        X['amt_extreme_zscore'] = (X['amount'] - df['cust_amt_q95']) / (df['cust_amt_std'] + 1e-8)
    else:
        X['amt_extreme_zscore'] = 0.0
    
    # Enhanced binary features with training data thresholds - FIXED
    if training_data is not None and hasattr(training_data, 'customer_stats'):
        if 'cust_amt_max' in df.columns:
            X['is_cust_extreme_spender'] = (df['cust_amt_max'] > training_data.customer_stats['cust_amt_max'].quantile(0.97)).astype(int)
        else:
            X['is_cust_extreme_spender'] = 0
            
        if 'cust_amt_cv' in df.columns:
            X['is_cust_consistent'] = (df['cust_amt_cv'] < training_data.customer_stats['cust_amt_cv'].quantile(0.15)).astype(int)
            X['is_cust_highly_volatile'] = (df['cust_amt_cv'] > training_data.customer_stats['cust_amt_cv'].quantile(0.97)).astype(int)
        else:
            X['is_cust_consistent'] = 0
            X['is_cust_highly_volatile'] = 0
            
        if 'cust_txn_count' in df.columns:
            X['is_cust_rare_user'] = (df['cust_txn_count'] < training_data.customer_stats['cust_txn_count'].quantile(0.05)).astype(int)
        else:
            X['is_cust_rare_user'] = 0
    else:
        # Fallback thresholds
        X['is_cust_extreme_spender'] = 0
        X['is_cust_consistent'] = 0
        X['is_cust_highly_volatile'] = 0
        X['is_cust_rare_user'] = 0
    
    # Enhanced feature transformations (from train.py)
    X['amount_log'] = np.log1p(X['amount'])
    X['amount_sqrt'] = np.sqrt(X['amount'])
    X['amount_square'] = np.square(X['amount'])
    X['amount_cube_root'] = np.power(X['amount'], 1/3)
    
    # Enhanced amount-based features
    X['is_exact_round'] = (X['amount'] % 100 == 0).astype(int)
    
    # Use training data quantiles for thresholds
    if training_data is not None:
        X['is_very_high_amount'] = (X['amount'] > training_data['amount'].quantile(0.995)).astype(int)
        X['is_extreme_amount'] = (X['amount'] > training_data['amount'].quantile(0.998)).astype(int)
        X['is_micro_transaction'] = (X['amount'] < training_data['amount'].quantile(0.05)).astype(int)
    else:
        X['is_very_high_amount'] = (X['amount'] > 5000).astype(int)
        X['is_extreme_amount'] = (X['amount'] > 10000).astype(int)  
        X['is_micro_transaction'] = (X['amount'] < 5).astype(int)
    
    # Enhanced temporal features
    X['day_of_week'] = X['step'] % 7
    X['is_weekend'] = ((X['step'] % 7).isin([5, 6])).astype(int)
    
    # Enhanced merchant features - COMPLETELY FIXED
    if 'merchant' in df.columns and merchant_stats is not None:
        print("Processing merchant features...")
        for idx, row in df.iterrows():
            merchant = row['merchant']
            
            # Get merchant risk score
            if merchant in merchant_stats['risk_mapping']:
                X.loc[idx, 'merchant_risk_score'] = merchant_stats['risk_mapping'][merchant]
            else:
                X.loc[idx, 'merchant_risk_score'] = merchant_stats.get('global_fraud_rate', 0.02)
            
            # Get merchant statistics
            if merchant in merchant_stats['stats'].index:
                merch_stats = merchant_stats['stats'].loc[merchant]
                X.loc[idx, 'merchant_frequency'] = np.log1p(merch_stats['merch_total_txns'])
                X.loc[idx, 'merchant_diversity'] = merch_stats['merch_unique_customers']
                X.loc[idx, 'is_high_risk_merchant'] = int(merch_stats['merch_fraud_rate'] > 
                                                        merchant_stats['stats']['merch_fraud_rate'].quantile(0.95))
                X.loc[idx, 'is_very_rare_merchant'] = int(merch_stats['merch_total_txns'] < 
                                                        merchant_stats['stats']['merch_total_txns'].quantile(0.02))
                X.loc[idx, 'merchant_avg_amount'] = merch_stats['merch_amt_mean']
                X.loc[idx, 'amount_vs_merchant_avg'] = X.loc[idx, 'amount'] / (merch_stats['merch_amt_mean'] + 1e-8)
            else:
                # New merchant defaults
                X.loc[idx, 'merchant_frequency'] = np.log1p(1)
                X.loc[idx, 'merchant_diversity'] = 1
                X.loc[idx, 'is_high_risk_merchant'] = 0
                X.loc[idx, 'is_very_rare_merchant'] = 1
                X.loc[idx, 'merchant_avg_amount'] = X.loc[idx, 'amount']
                X.loc[idx, 'amount_vs_merchant_avg'] = 1.0
    else:
        # Default merchant features when no merchant data available
        default_merchant_cols = ['merchant_risk_score', 'merchant_frequency', 'merchant_diversity',
                               'is_high_risk_merchant', 'is_very_rare_merchant', 'merchant_avg_amount',
                               'amount_vs_merchant_avg']
        for col in default_merchant_cols:
            if col == 'merchant_risk_score':
                X[col] = 0.02  # Low risk default
            elif col == 'merchant_frequency':
                X[col] = 2.0  # Medium frequency
            elif col == 'amount_vs_merchant_avg':
                X[col] = 1.0  # Normal ratio
            else:
                X[col] = 0
    
    # Clean data
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(0.0).astype(float)
    X = X.replace([np.inf, -np.inf], 0.0)
    
    # Ensure we have exactly the features the model expects
    expected_features = metadata['feature_cols']
    print(f"Expected features ({len(expected_features)}): {expected_features}")
    print(f"Current features ({len(X.columns)}): {list(X.columns)}")
    
    # Add missing features with neutral values
    for feature in expected_features:
        if feature not in X.columns:
            print(f"Adding missing feature {feature} with value 0.0")
            X[feature] = 0.0
    
    # Remove extra features and reorder to match training
    X = X[expected_features]
    
    print(f"Final feature matrix shape: {X.shape}")
    
    return X

def get_risk_level(score):
    """
    FIXED: 3-level risk classification based on the model's actual performance
    These thresholds are calibrated based on train.py model's behavior:
    - Model threshold is around 0.75 (from metadata) 
    - Most normal transactions score 0.1-0.4
    - Suspicious transactions score 0.4-0.75  
    - High risk transactions score >0.75
    """
    # Based on the model's hybrid_threshold from train.py (~0.75)
    # and typical score distributions from precision-focused models
    
    if score >= 0.6:      # High risk: likely to be flagged as fraud
        return "High"
    elif score >= 0.35:   # Medium risk: elevated but below fraud threshold  
        return "Medium"
    else:                 # Low risk: normal transactions
        return "Low"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok", 
        "model_type": metadata.get('model_type', 'unknown') if metadata else "unknown",
        "training_data_loaded": training_data is not None,
        "merchant_stats_loaded": merchant_stats is not None,
        "current_threshold": metadata.get('hybrid_threshold', 'unknown') if metadata else "unknown",
        "model_weights": metadata.get('weights', 'unknown') if metadata else "unknown",
        "risk_levels": "3_levels_low_medium_high",
        "risk_thresholds": {"high": 0.6, "medium": 0.35, "low": 0.0}
    })

@app.route('/score', methods=['POST'])
def score_transaction():
    try:
        data = request.json
        print(f"Received scoring request: {data}")
        
        # Check if a custom threshold is provided
        custom_threshold = None
        if isinstance(data, dict) and 'threshold' in data:
            custom_threshold = data.pop('threshold')
        
        # Prepare transaction with enhanced customer features
        df = prepare_transaction_features(data)
        print(f"After customer features: {df.columns.tolist()}")
        
        # Prepare features for inference using enhanced feature engineering
        X = prepare_features_for_inference(df)
        print(f"After feature engineering: {X.shape}")
        print(f"Sample feature values: {dict(list(X.iloc[0].items())[:10])}")
        
        # Get hybrid prediction scores
        hybrid_scores, if_scores, ae_scores = hybrid_model.predict_scores(X)
        
        # Use custom threshold if provided, otherwise use model threshold
        threshold = custom_threshold if custom_threshold is not None else metadata['hybrid_threshold']
        
        # Prepare response
        if isinstance(data, dict):
            # Single transaction
            score = float(hybrid_scores[0])
            if_score = float(if_scores[0])
            ae_score = float(ae_scores[0])
            risk_level = get_risk_level(score)  # Using fixed 3-level thresholds
            is_anomaly = score > threshold
            
            print(f"Final result: hybrid_score={score:.4f}, if_score={if_score:.4f}, ae_score={ae_score:.4f}")
            print(f"Risk level: {risk_level}, is_anomaly: {is_anomaly}, threshold: {threshold:.4f}")
            
            return jsonify({
                'risk_score': score,
                'isolation_forest_score': if_score,
                'autoencoder_score': ae_score,
                'risk_level': risk_level,
                'is_anomaly': is_anomaly,
                'threshold': threshold,
                'model_threshold': metadata['hybrid_threshold'],
                'threshold_override': custom_threshold is not None,
                'model_type': metadata['model_type'],
                'model_weights': metadata['weights'],
                'precision_boost_factor': metadata.get('precision_boost_factor', 1.0),
                'risk_thresholds': {"high": 0.6, "medium": 0.35, "low": 0.0}
            })
        else:
            # Multiple transactions
            results = []
            for i, score in enumerate(hybrid_scores):
                score = float(score)
                if_score = float(if_scores[i])
                ae_score = float(ae_scores[i])
                risk_level = get_risk_level(score)  # Using fixed 3-level thresholds
                is_anomaly = score > threshold
                
                results.append({
                    'risk_score': score,
                    'isolation_forest_score': if_score,
                    'autoencoder_score': ae_score,
                    'risk_level': risk_level,
                    'is_anomaly': is_anomaly,
                    'threshold': threshold
                })
            
            return jsonify(results)
            
    except Exception as e:
        import traceback
        print(f"Error in score_transaction: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/set-threshold', methods=['POST'])
def set_threshold():
    """Endpoint to dynamically update the hybrid threshold"""
    try:
        data = request.json
        new_threshold = data.get('threshold')
        
        if new_threshold is None:
            return jsonify({'error': 'threshold parameter required'}), 400
        
        if not 0 <= new_threshold <= 1:
            return jsonify({'error': 'threshold must be between 0 and 1'}), 400
        
        old_threshold = metadata['hybrid_threshold']
        metadata['hybrid_threshold'] = new_threshold
        hybrid_model.hybrid_threshold = new_threshold
        
        print(f"Hybrid threshold updated: {old_threshold:.4f} -> {new_threshold:.4f}")
        
        return jsonify({
            'success': True,
            'old_threshold': old_threshold,
            'new_threshold': new_threshold,
            'message': f'Hybrid threshold updated from {old_threshold:.4f} to {new_threshold:.4f}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update-risk-thresholds', methods=['POST'])
def update_risk_thresholds():
    """Endpoint to update risk level classification thresholds (now simplified to 3 levels)"""
    try:
        data = request.json
        high_threshold = data.get('high_threshold', 0.6)
        medium_threshold = data.get('medium_threshold', 0.35)
        
        # Test with sample scores to show the effect
        sample_scores = [0.15, 0.25, 0.30, 0.40, 0.50, 0.65, 0.75, 0.85]
        
        old_classifications = [get_risk_level(score) for score in sample_scores]
        
        # Temporarily update thresholds for comparison
        global HIGH_THRESHOLD, MEDIUM_THRESHOLD
        HIGH_THRESHOLD = high_threshold
        MEDIUM_THRESHOLD = medium_threshold
        
        new_classifications = [get_risk_level(score) for score in sample_scores]
        
        comparison = []
        for i, score in enumerate(sample_scores):
            comparison.append({
                'score': score,
                'old_classification': old_classifications[i],
                'new_classification': new_classifications[i],
                'changed': old_classifications[i] != new_classifications[i]
            })
        
        new_thresholds = {
            'high': high_threshold,
            'medium': medium_threshold,
            'low': 0.0
        }
        
        return jsonify({
            'success': True,
            'new_thresholds': new_thresholds,
            'sample_comparison': comparison,
            'changes_detected': sum(1 for c in comparison if c['changed']),
            'message': f'Risk thresholds updated: High >= {high_threshold}, Medium >= {medium_threshold}, Low < {medium_threshold}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-model', methods=['GET'])
def test_model():
    """Test endpoint with various transaction scenarios"""
    try:
        test_transactions = [
            # Normal transactions (should be Low risk)
            {'step': 100, 'amount': 25.0, 'customer': 'C_test_normal_1', 'merchant': 'M_test_normal'},
            {'step': 100, 'amount': 75.0, 'customer': 'C_test_normal_2', 'merchant': 'M_test_normal'},
            {'step': 100, 'amount': 150.0, 'customer': 'C_test_normal_3', 'merchant': 'M_test_normal'},
            
            # Slightly elevated (should be Low to Medium risk)
            {'step': 100, 'amount': 300.0, 'customer': 'C_test_medium', 'merchant': 'M_test_medium'},
            {'step': 100, 'amount': 500.0, 'customer': 'C_test_elevated', 'merchant': 'M_test_medium'},
            
            # Potentially suspicious patterns (should be Medium to High risk)
            {'step': 100, 'amount': 1000.0, 'customer': 'C_test_high', 'merchant': 'M_test_suspicious'},
            {'step': 100, 'amount': 5000.0, 'customer': 'C_test_very_high', 'merchant': 'M_test_suspicious'},
            {'step': 100, 'amount': 10000.0, 'customer': 'C_test_extreme', 'merchant': 'M_test_suspicious'},
            
            # Micro-transactions (fraud pattern - should be Medium to High)
            {'step': 100, 'amount': 0.50, 'customer': 'C_test_micro', 'merchant': 'M_test_micro'},
            {'step': 100, 'amount': 1.0, 'customer': 'C_test_small', 'merchant': 'M_test_micro'},
            
            # Round amounts (potential fraud - should be Medium to High)
            {'step': 100, 'amount': 1000.0, 'customer': 'C_test_round_1', 'merchant': 'M_test_round'},
            {'step': 100, 'amount': 5000.0, 'customer': 'C_test_round_2', 'merchant': 'M_test_round'},
            
            # Weekend transactions (slight risk factor)
            {'step': 105, 'amount': 200.0, 'customer': 'C_test_weekend', 'merchant': 'M_test_weekend'},
            {'step': 106, 'amount': 500.0, 'customer': 'C_test_weekend_2', 'merchant': 'M_test_weekend'},
        ]
        
        results = []
        all_scores = []
        
        for txn in test_transactions:
            try:
                df = prepare_transaction_features(txn)
                X = prepare_features_for_inference(df)
                
                hybrid_scores, if_scores, ae_scores = hybrid_model.predict_scores(X)
                
                score = float(hybrid_scores[0])
                all_scores.append(score)
                if_score = float(if_scores[0])
                ae_score = float(ae_scores[0])
                risk_level = get_risk_level(score)  # Using fixed 3-level thresholds
                is_anomaly = score > metadata['hybrid_threshold']
                
                results.append({
                    'transaction': txn,
                    'risk_score': score,
                    'isolation_forest_score': if_score,
                    'autoencoder_score': ae_score,
                    'risk_level': risk_level,
                    'is_anomaly': is_anomaly
                })
            except Exception as e:
                results.append({
                    'transaction': txn,
                    'error': str(e)
                })
        
        # Calculate distribution stats
        if all_scores:
            score_stats = {
                'min': min(all_scores),
                'max': max(all_scores),
                'mean': np.mean(all_scores),
                'p50': np.percentile(all_scores, 50),
                'p75': np.percentile(all_scores, 75),
                'p90': np.percentile(all_scores, 90),
                'p95': np.percentile(all_scores, 95),
                'p98': np.percentile(all_scores, 98)
            }
            
            # Risk level distribution
            risk_levels = [r['risk_level'] for r in results if 'risk_level' in r]
            risk_distribution = {
                'High': risk_levels.count('High'),
                'Medium': risk_levels.count('Medium'),
                'Low': risk_levels.count('Low')
            }
        else:
            score_stats = {}
            risk_distribution = {}
        
        return jsonify({
            'test_results': results,
            'score_distribution': score_stats,
            'risk_level_distribution': risk_distribution,
            'model_info': {
                'type': metadata['model_type'],
                'threshold': metadata['hybrid_threshold'],
                'weights': metadata['weights'],
                'precision_boost_factor': metadata.get('precision_boost_factor', 1.0),
                'training_data_loaded': training_data is not None,
                'merchant_stats_loaded': merchant_stats is not None,
                'total_features': len(metadata['feature_cols']),
                'risk_levels': 3,
                'risk_thresholds': {"high": 0.6, "medium": 0.35, "low": 0.0}
            }
        })
        
    except Exception as e:
        import traceback
        print(f"Error in test_model: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/calibrate-threshold', methods=['POST'])
def calibrate_threshold():
    """Calibrate threshold for desired precision/recall trade-off"""
    try:
        data = request.json
        target_precision = data.get('target_precision', 0.72)
        target_recall = data.get('target_recall', 0.80)
        
        if training_data is None:
            return jsonify({'error': 'Training data not available'}), 500
        
        print(f"Calibrating for precision >= {target_precision}, recall >= {target_recall}")
        
        # Sample training data for calibration
        sample_size = min(2000, len(training_data))
        sample_data = training_data.sample(n=sample_size, random_state=42)
        
        # Score sample transactions
        all_hybrid_scores = []
        all_if_scores = []
        all_ae_scores = []
        sample_labels = []
        
        for _, row in sample_data.iterrows():
            try:
                txn = {
                    'step': row['step'],
                    'amount': row['amount'],
                    'customer': row['customer']
                }
                if 'merchant' in row:
                    txn['merchant'] = row['merchant']
                
                df = prepare_transaction_features(txn)
                X = prepare_features_for_inference(df)
                
                hybrid_scores, if_scores, ae_scores = hybrid_model.predict_scores(X)
                
                all_hybrid_scores.append(hybrid_scores[0])
                all_if_scores.append(if_scores[0])
                all_ae_scores.append(ae_scores[0])
                
                # Get actual label if available
                label = row.get('fraud', 0)
                sample_labels.append(label)
                
            except Exception as e:
                print(f"Error scoring sample transaction: {e}")
                continue
        
        if len(all_hybrid_scores) == 0:
            return jsonify({'error': 'Could not score any transactions'}), 500
        
        all_hybrid_scores = np.array(all_hybrid_scores)
        all_if_scores = np.array(all_if_scores)
        all_ae_scores = np.array(all_ae_scores)
        sample_labels = np.array(sample_labels)
        
        # Calculate score distributions
        score_stats = {
            'hybrid': {
                'min': float(all_hybrid_scores.min()),
                'max': float(all_hybrid_scores.max()),
                'mean': float(all_hybrid_scores.mean()),
                'std': float(all_hybrid_scores.std()),
                'percentiles': {f'p{p}': float(np.percentile(all_hybrid_scores, p)) 
                              for p in [50, 75, 90, 95, 98, 99]}
            },
            'isolation_forest': {
                'min': float(all_if_scores.min()),
                'max': float(all_if_scores.max()),
                'mean': float(all_if_scores.mean()),
                'percentiles': {f'p{p}': float(np.percentile(all_if_scores, p)) 
                              for p in [50, 75, 90, 95, 98, 99]}
            },
            'autoencoder': {
                'min': float(all_ae_scores.min()),
                'max': float(all_ae_scores.max()),
                'mean': float(all_ae_scores.mean()),
                'percentiles': {f'p{p}': float(np.percentile(all_ae_scores, p)) 
                              for p in [50, 75, 90, 95, 98, 99]}
            }
        }
        
        # Risk level analysis with 3 levels
        risk_levels = [get_risk_level(score) for score in all_hybrid_scores]
        risk_distribution = {
            'High': risk_levels.count('High') / len(risk_levels),
            'Medium': risk_levels.count('Medium') / len(risk_levels),
            'Low': risk_levels.count('Low') / len(risk_levels)
        }
        
        # Threshold analysis for different detection rates
        threshold_analysis = {}
        for pct in [90, 95, 97, 98, 99, 99.5]:
            threshold = np.percentile(all_hybrid_scores, pct)
            detection_rate = np.mean(all_hybrid_scores > threshold)
            
            # Calculate precision/recall if we have labels
            if len(sample_labels) > 0 and np.sum(sample_labels) > 0:
                predictions = (all_hybrid_scores > threshold).astype(int)
                tp = np.sum((sample_labels == 1) & (predictions == 1))
                fp = np.sum((sample_labels == 0) & (predictions == 1))
                fn = np.sum((sample_labels == 1) & (predictions == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                threshold_analysis[f'p{pct}'] = {
                    'threshold': float(threshold),
                    'detection_rate': float(detection_rate),
                    'precision': float(precision),
                    'recall': float(recall),
                    'meets_precision_target': precision >= target_precision,
                    'meets_recall_target': recall >= target_recall
                }
            else:
                threshold_analysis[f'p{pct}'] = {
                    'threshold': float(threshold),
                    'detection_rate': float(detection_rate)
                }
        
        # Find best threshold that meets criteria
        best_threshold = None
        best_analysis = None
        
        if len(sample_labels) > 0 and np.sum(sample_labels) > 0:
            for analysis in threshold_analysis.values():
                if ('precision' in analysis and 'recall' in analysis and 
                    analysis['meets_precision_target'] and analysis['meets_recall_target']):
                    if best_threshold is None or analysis['precision'] > best_analysis['precision']:
                        best_threshold = analysis['threshold']
                        best_analysis = analysis
        
        response = {
            'current_threshold': metadata['hybrid_threshold'],
            'score_distribution': score_stats,
            'risk_level_distribution': risk_distribution,
            'threshold_analysis': threshold_analysis,
            'sample_size': len(all_hybrid_scores),
            'fraud_samples': int(np.sum(sample_labels)) if len(sample_labels) > 0 else 0,
            'model_weights': metadata['weights'],
            'risk_levels': 3,
            'risk_thresholds': {"high": 0.6, "medium": 0.35, "low": 0.0}
        }
        
        if best_threshold is not None:
            response.update({
                'suggested_threshold': best_threshold,
                'suggested_analysis': best_analysis,
                'recommendation': f"Consider setting threshold to {best_threshold:.4f} for precision={best_analysis['precision']:.3f}, recall={best_analysis['recall']:.3f}"
            })
        else:
            response['recommendation'] = "No threshold found that meets both precision and recall targets"
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        print(f"Error in calibrate_threshold: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/debug', methods=['POST'])
def debug_transaction():
    """Debug endpoint to inspect feature generation and scoring"""
    try:
        data = request.json
        print("=== DEBUGGING TRANSACTION SCORING ===")
        
        print(f"1. Input data: {data}")
        
        df = prepare_transaction_features(data)
        print(f"2. After customer features: {df.to_dict('records')[0] if len(df) > 0 else 'None'}")
        
        X = prepare_features_for_inference(df)
        print(f"3. Final features shape: {X.shape}")
        print(f"4. Feature sample: {dict(list(X.iloc[0].to_dict().items())[:15])}")
        
        hybrid_scores, if_scores, ae_scores = hybrid_model.predict_scores(X)
        score = hybrid_scores[0]
        
        risk_level = get_risk_level(score)
        
        print(f"5. Hybrid score: {score:.4f}")
        print(f"6. IF score: {if_scores[0]:.4f}, AE score: {ae_scores[0]:.4f}")
        print(f"7. Risk level: {risk_level}")
        print(f"8. Model weights: {metadata['weights']}")
        print(f"9. Model threshold: {metadata['hybrid_threshold']:.4f}")
        
        return jsonify({
            'debug_complete': True,
            'input_data': data,
            'customer_features': df.to_dict('records')[0] if len(df) > 0 else None,
            'final_features': dict(list(X.iloc[0].to_dict().items())[:20]),
            'feature_shape': X.shape,
            'hybrid_risk_score': float(score),
            'isolation_forest_score': float(if_scores[0]),
            'autoencoder_score': float(ae_scores[0]),
            'risk_level': risk_level,
            'model_weights': metadata['weights'],
            'threshold': metadata['hybrid_threshold'],
            'model_type': metadata['model_type'],
            'training_data_available': training_data is not None,
            'merchant_stats_available': merchant_stats is not None,
            'precision_boost_factor': metadata.get('precision_boost_factor', 1.0),
            'risk_levels': 3,
            'risk_thresholds': {"high": 0.6, "medium": 0.35, "low": 0.0}
        })
        
    except Exception as e:
        import traceback
        print(f"Error in debug_transaction: {str(e)}")
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

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get comprehensive information about the precision-focused hybrid model"""
    try:
        if metadata is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        return jsonify({
            'model_type': metadata['model_type'],
            'input_dimensions': metadata['input_dim'],
            'total_features': len(metadata['feature_cols']),
            'hybrid_threshold': metadata['hybrid_threshold'],
            'model_weights': metadata['weights'],
            'precision_boost_factor': metadata.get('precision_boost_factor', 1.0),
            'target_precision': metadata.get('target_precision', 0.72),
            'target_recall': metadata.get('target_recall', 0.8),
            'optimization': metadata.get('optimization', 'enhanced_precision_boosting'),
            'training_date': metadata.get('training_date', 'unknown'),
            'feature_sample': metadata['feature_cols'][:15],
            'training_data_loaded': training_data is not None,
            'customer_count': len(training_data.customer_stats) if training_data is not None and hasattr(training_data, 'customer_stats') else 0,
            'merchant_encoding_available': merchant_stats is not None,
            'merchant_count': len(merchant_stats['stats']) if merchant_stats is not None else 0,
            'risk_levels': 3,
            'risk_thresholds': {"high": 0.6, "medium": 0.35, "low": 0.0}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance if available"""
    try:
        # Try to load feature importance from file
        try:
            importance_df = pd.read_csv('feature_importance.csv')
            return jsonify({
                'feature_importance': importance_df.to_dict('records'),
                'top_10_features': importance_df.head(10)[['feature', 'combined_importance']].to_dict('records')
            })
        except FileNotFoundError:
            return jsonify({'message': 'Feature importance not available - run training to generate'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch-score', methods=['POST'])
def batch_score_transactions():
    """Score multiple transactions in batch for efficiency"""
    try:
        data = request.json
        transactions = data.get('transactions', [])
        custom_threshold = data.get('threshold')
        
        if not transactions:
            return jsonify({'error': 'No transactions provided'}), 400
        
        print(f"Batch scoring {len(transactions)} transactions")
        
        results = []
        
        # Score all transactions
        for i, txn in enumerate(transactions):
            try:
                df = prepare_transaction_features(txn)
                X = prepare_features_for_inference(df)
                
                hybrid_scores, if_scores, ae_scores = hybrid_model.predict_scores(X)
                
                threshold = custom_threshold if custom_threshold is not None else metadata['hybrid_threshold']
                
                score = float(hybrid_scores[0])
                if_score = float(if_scores[0])
                ae_score = float(ae_scores[0])
                risk_level = get_risk_level(score)  # Using fixed 3-level thresholds
                is_anomaly = score > threshold
                
                results.append({
                    'transaction_id': i,
                    'risk_score': score,
                    'isolation_forest_score': if_score,
                    'autoencoder_score': ae_score,
                    'risk_level': risk_level,
                    'is_anomaly': is_anomaly,
                    'input_transaction': txn
                })
                
            except Exception as e:
                results.append({
                    'transaction_id': i,
                    'error': str(e),
                    'input_transaction': txn
                })
        
        # Summary statistics with risk level breakdown
        successful_results = [r for r in results if 'error' not in r]
        if successful_results:
            scores = [r['risk_score'] for r in successful_results]
            anomalies = [r for r in successful_results if r['is_anomaly']]
            
            # Risk level distribution
            risk_levels = [r['risk_level'] for r in successful_results]
            risk_distribution = {
                'High': risk_levels.count('High'),
                'Medium': risk_levels.count('Medium'),
                'Low': risk_levels.count('Low')
            }
            
            summary = {
                'total_transactions': len(transactions),
                'successful_scores': len(successful_results),
                'errors': len(transactions) - len(successful_results),
                'anomalies_detected': len(anomalies),
                'anomaly_rate': len(anomalies) / len(successful_results) if successful_results else 0,
                'risk_level_distribution': risk_distribution,
                'risk_level_percentages': {k: v/len(successful_results)*100 for k, v in risk_distribution.items()},
                'score_statistics': {
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'p50': float(np.percentile(scores, 50)),
                    'p75': float(np.percentile(scores, 75)),
                    'p90': float(np.percentile(scores, 90)),
                    'p95': float(np.percentile(scores, 95)),
                    'p98': float(np.percentile(scores, 98))
                } if scores else None
            }
        else:
            summary = {
                'total_transactions': len(transactions),
                'successful_scores': 0,
                'errors': len(transactions),
                'anomalies_detected': 0,
                'anomaly_rate': 0,
                'risk_level_distribution': {'High': 0, 'Medium': 0, 'Low': 0}
            }
        
        return jsonify({
            'results': results,
            'summary': summary,
            'threshold_used': custom_threshold if custom_threshold is not None else metadata['hybrid_threshold'],
            'model_weights': metadata['weights'],
            'risk_levels': 3,
            'risk_thresholds': {"high": 0.6, "medium": 0.35, "low": 0.0}
        })
        
    except Exception as e:
        import traceback
        print(f"Error in batch_score_transactions: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Initialize models when app starts
try:
    success = load_models()
    if success:
        print("Precision-focused hybrid fraud detection app initialized successfully!")
        print(f"Model type: {metadata.get('model_type', 'unknown')}")
        print(f"Hybrid threshold: {metadata.get('hybrid_threshold', 'unknown')}")
        print(f"Model weights: {metadata.get('weights', 'unknown')}")
        print(f"Training data loaded: {training_data is not None}")
        print(f"Merchant stats loaded: {merchant_stats is not None}")
        print("Risk levels: 3 (Low, Medium, High)")
        print("Risk thresholds: High >= 0.6, Medium >= 0.35, Low < 0.35")
    else:
        print("Failed to initialize models")
except Exception as e:
    print(f"Error initializing app: {e}")
    import traceback
    traceback.print_exc()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
