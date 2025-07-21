import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from category_encoders import TargetEncoder
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # More sophisticated architecture for better anomaly detection
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

def load_and_preprocess_data():
    print("Loading data...")
    df = pd.read_csv("cleaned_fraud_data.csv")
    
    # Convert step to proper day-based datetime
    base_date = datetime(2011, 1, 1)
    df['datetime'] = df['step'].apply(lambda x: base_date + timedelta(days=x))
    
    # Sort by datetime for proper time-series features
    df = df.sort_values(['customer', 'datetime']).reset_index(drop=True)
    
    # Create proper time-based features
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['month'] = df['datetime'].dt.month
    df['quarter'] = df['datetime'].dt.quarter
    df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Create cyclical features for better model performance
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def generate_rolling_features_optimized(df):
    """Generate proper rolling features based on day-based timeline"""
    print("Generating rolling features (optimized for day-based data)...")
    
    # Sort by customer and datetime
    df = df.sort_values(['customer', 'datetime']).reset_index(drop=True)
    
    rolling_features = []
    
    # Process each customer separately for memory efficiency
    for customer in df['customer'].unique():
        customer_data = df[df['customer'] == customer].copy()
        
        if len(customer_data) == 0:
            continue
            
        # Sort by datetime
        customer_data = customer_data.sort_values('datetime').reset_index(drop=True)
        
        # Create proper rolling windows for day-based data
        # 7-day rolling window for amount-based features
        customer_data['txn_count_7d'] = customer_data['amount'].rolling(window=7, min_periods=1).count()
        customer_data['amt_sum_7d'] = customer_data['amount'].rolling(window=7, min_periods=1).sum()
        customer_data['amt_mean_7d'] = customer_data['amount'].rolling(window=7, min_periods=1).mean()
        customer_data['amt_std_7d'] = customer_data['amount'].rolling(window=7, min_periods=1).std().fillna(0)
        customer_data['amt_max_7d'] = customer_data['amount'].rolling(window=7, min_periods=1).max()
        customer_data['amt_min_7d'] = customer_data['amount'].rolling(window=7, min_periods=1).min()
        
        # 30-day rolling window for amount-based features
        customer_data['txn_count_30d'] = customer_data['amount'].rolling(window=30, min_periods=1).count()
        customer_data['amt_sum_30d'] = customer_data['amount'].rolling(window=30, min_periods=1).sum()
        customer_data['amt_mean_30d'] = customer_data['amount'].rolling(window=30, min_periods=1).mean()
        customer_data['amt_std_30d'] = customer_data['amount'].rolling(window=30, min_periods=1).std().fillna(0)
        
        # Velocity features (transactions per day)
        customer_data['txn_velocity_7d'] = customer_data['txn_count_7d'] / 7
        customer_data['txn_velocity_30d'] = customer_data['txn_count_30d'] / 30
        
        # Amount deviation features
        customer_data['amt_deviation_7d'] = abs(customer_data['amount'] - customer_data['amt_mean_7d'])
        customer_data['amt_deviation_30d'] = abs(customer_data['amount'] - customer_data['amt_mean_30d'])
        
        # Enhanced features for better fraud detection
        customer_data['amt_diff_ratio'] = abs(customer_data['amount'] - customer_data['amt_mean_7d']) / (customer_data['amt_mean_7d'] + 1e-8)
        customer_data['amt_range_7d'] = customer_data['amt_max_7d'] - customer_data['amt_min_7d']
        customer_data['amt_cv_7d'] = customer_data['amt_std_7d'] / (customer_data['amt_mean_7d'] + 1e-8)  # Coefficient of variation
        
        # Enhanced merchant features
        if 'merchant' in customer_data.columns:
            def count_unique_merchants(series, window_size):
                unique_counts = []
                for i in range(len(series)):
                    start_idx = max(0, i - window_size + 1)
                    end_idx = i + 1
                    window_data = series.iloc[start_idx:end_idx]
                    unique_counts.append(len(set(window_data.dropna())))
                return unique_counts
            
            customer_data['uniq_merchants_7d'] = count_unique_merchants(customer_data['merchant'], 7)
            customer_data['uniq_merchants_30d'] = count_unique_merchants(customer_data['merchant'], 30)
            
            # Merchant risk score and frequency
            merchant_totals = customer_data.groupby('merchant')['amount'].sum()
            merchant_counts = customer_data.groupby('merchant').size()
            customer_data['merchant_risk_score'] = customer_data['merchant'].map(merchant_totals).fillna(0)
            customer_data['merchant_frequency'] = customer_data['merchant'].map(merchant_counts).fillna(0)
        else:
            customer_data['uniq_merchants_7d'] = 1
            customer_data['uniq_merchants_30d'] = 1
            customer_data['merchant_risk_score'] = 0
            customer_data['merchant_frequency'] = 0
        
        # Time-based features
        customer_data['days_since_last_txn'] = customer_data['step'].diff().fillna(0)
        customer_data['hours_since_last_txn'] = customer_data['days_since_last_txn'] * 24  # Convert to hours
        
        # Cumulative features
        customer_data['cumulative_txn_count'] = range(1, len(customer_data) + 1)
        customer_data['cumulative_amount'] = customer_data['amount'].cumsum()
        customer_data['avg_amount_to_date'] = customer_data['cumulative_amount'] / customer_data['cumulative_txn_count']
        
        # Same day transaction features
        customer_data['same_day_txn_count'] = customer_data.groupby(['customer', 'step'])['amount'].transform('count')
        customer_data['same_day_total_amount'] = customer_data.groupby(['customer', 'step'])['amount'].transform('sum')
        
        # Behavioral change indicators
        customer_data['amount_trend_7d'] = customer_data['amount'].rolling(window=7, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
        ).fillna(0)
        
        # New: High-frequency transaction flags
        customer_data['high_freq_flag_7d'] = (customer_data['txn_velocity_7d'] > customer_data['txn_velocity_30d'] * 2).astype(int)
        customer_data['high_freq_flag_30d'] = (customer_data['txn_velocity_30d'] > customer_data['txn_velocity_30d'].quantile(0.95)).astype(int)
        
        # New: Location change features
        if 'merchant_location' in customer_data.columns:
            customer_data['location_change_7d'] = customer_data['merchant_location'].rolling(window=7, min_periods=1).apply(
                lambda x: len(set(x)) / len(x) if len(x) > 0 else 0, raw=False
            )
        else:
            customer_data['location_change_7d'] = 0
            
        rolling_features.append(customer_data)
    
    # Combine all customer data
    df = pd.concat(rolling_features, ignore_index=True)
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df

def prepare_features(df):
    print("Preparing features...")
    
    # Sample data for faster training if dataset is very large
    if len(df) > 100000:
        print(f"Dataset has {len(df)} rows. Sampling 100000 rows for training...")
        df = df.sample(n=100000, random_state=42).reset_index(drop=True)
    
    # Check if fraud column exists
    if 'fraud' not in df.columns:
        print("Warning: No 'fraud' column found. Cannot evaluate model performance.")
        fraud_labels = None
    else:
        fraud_labels = df['fraud'].values
        print(f"Found fraud column with {fraud_labels.sum()} fraud cases out of {len(fraud_labels)} total transactions")
        print(f"Fraud rate: {fraud_labels.mean():.4f}")
    
    # Enhanced feature set for better recall and precision balance
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
        'same_day_txn_count', 'same_day_total_amount', 'high_freq_flag_7d', 
        'high_freq_flag_30d', 'location_change_7d'  # New features
    ]
    
    # Check which base features exist
    existing_features = [col for col in base_features if col in df.columns]
    missing_features = [col for col in base_features if col not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing features {missing_features}. Creating with default values.")
        for col in missing_features:
            if col == 'step':
                df[col] = df.index.values
            else:
                df[col] = 0.0
    
    # Start with numeric features
    X = df[existing_features].copy()
    
    # Add enhanced derived features for better anomaly detection
    X['amount_zscore'] = (X['amount'] - X['amount'].mean()) / (X['amount'].std() + 1e-8)
    X['amount_percentile'] = X['amount'].rank(pct=True)
    X['amount_log'] = np.log1p(X['amount'])  # Log transformation for skewed amounts
    
    # Multiple amount thresholds for better detection
    X['is_very_high_amount'] = (X['amount'] > X['amount'].quantile(0.99)).astype(int)
    X['is_high_amount'] = (X['amount'] > X['amount'].quantile(0.95)).astype(int)
    X['is_medium_high_amount'] = (X['amount'] > X['amount'].quantile(0.90)).astype(int)
    X['is_round_amount'] = (X['amount'] % 1 == 0).astype(int)
    X['is_very_round_amount'] = (X['amount'] % 100 == 0).astype(int)
    
    # Enhanced behavioral features
    X['txn_freq_ratio'] = X['txn_velocity_7d'] / (X['txn_velocity_30d'] + 1e-8)
    X['amount_consistency'] = 1 / (X['amt_std_7d'] + 1)
    X['merchant_diversity'] = X['uniq_merchants_7d'] / (X['txn_count_7d'] + 1e-8)
    
    # Time-based anomaly features
    X['unusual_time'] = (X['dow_sin'] < -0.7).astype(int)
    X['very_frequent_txns'] = (X['same_day_txn_count'] > 5).astype(int)
    X['burst_activity'] = (X['hours_since_last_txn'] < 1).astype(int)
    
    # Multiple spike detection levels
    X['amount_spike_2x'] = (X['amount'] > 2 * X['amt_mean_7d']).astype(int)
    X['amount_spike_3x'] = (X['amount'] > 3 * X['amt_mean_7d']).astype(int)
    X['amount_spike_5x'] = (X['amount'] > 5 * X['amt_mean_7d']).astype(int)
    
    # Frequency anomaly detection
    X['freq_anomaly'] = (X['txn_velocity_7d'] > 1.5 * X['txn_velocity_30d']).astype(int)
    X['extreme_freq_anomaly'] = (X['txn_velocity_7d'] > 3 * X['txn_velocity_30d']).astype(int)
    
    # Merchant behavior features
    X['merchant_change_rate'] = X['uniq_merchants_7d'] / (X['uniq_merchants_30d'] + 1e-8)
    X['new_merchant_indicator'] = (X['merchant_frequency'] == 1).astype(int)
    
    # Amount pattern features
    X['amount_vs_historical'] = X['amount'] / (X['avg_amount_to_date'] + 1e-8)
    X['amount_deviation_score'] = X['amt_deviation_7d'] / (X['amt_std_7d'] + 1e-8)
    
    # Encode categorical variables
    cat_cols = ['merchant', 'category']
    existing_cat_cols = [col for col in cat_cols if col in df.columns]
    
    if existing_cat_cols:
        print(f"Encoding categorical columns: {existing_cat_cols}")
        
        # Handle high cardinality categories
        for col in existing_cat_cols:
            unique_vals = df[col].nunique()
            if unique_vals > 500:
                print(f"Warning: {col} has {unique_vals} unique values. Keeping top 300 most frequent.")
                top_categories = df[col].value_counts().head(300).index
                df[col] = df[col].where(df[col].isin(top_categories), 'Other')
        
        # Use actual fraud labels if available, otherwise use sophisticated pseudo-target
        if fraud_labels is not None:
            target_for_encoding = fraud_labels
        else:
            high_amount = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
            high_freq = (df['txn_count_7d'] > df['txn_count_7d'].quantile(0.95)).astype(int)
            target_for_encoding = np.maximum(high_amount, high_freq)
        
        cat_encoder = TargetEncoder(cols=existing_cat_cols, smoothing=10)
        df_encoded = cat_encoder.fit_transform(df[existing_cat_cols], target_for_encoding)
        df_encoded = df_encoded.astype(float)
        
        # Add encoded categorical features
        X = pd.concat([X, df_encoded], axis=1)
    else:
        cat_encoder = None
    
    # Convert all columns to numeric
    print("Converting all features to numeric...")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill NaN values and ensure float type
    X = X.fillna(0.0).astype(float)
    
    feature_cols = list(X.columns)
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Feature columns: {len(feature_cols)}")
    
    return X, feature_cols, cat_encoder, fraud_labels

def train_models(X, fraud_labels=None):
    print("Training models...")
    
    X = X.fillna(0)
    print(f"Training data shape: {X.shape}")
    
    # Use RobustScaler to handle outliers better
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Autoencoder with improved architecture
    print("Training Autoencoder...")
    input_dim = X_scaled.shape[1]
    autoencoder = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)
    
    X_tensor = torch.FloatTensor(X_scaled)
    
    from torch.utils.data import DataLoader, TensorDataset
    
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    autoencoder.train()
    epochs = 40  # Increased epochs for better training
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = autoencoder(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f'Epoch [{epoch}/{epochs}], Average Loss: {total_loss/len(dataloader):.4f}')
    
    # Get reconstruction errors
    autoencoder.eval()
    with torch.no_grad():
        reconstructed = autoencoder(X_tensor)
        ae_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
    
    print(f"Autoencoder errors: min={ae_errors.min():.6f}, max={ae_errors.max():.6f}, mean={ae_errors.mean():.6f}")
    
    
    print("Training Isolation Forest...")
    # Set contamination based on fraud rate if available
    if fraud_labels is not None:
        fraud_rate = fraud_labels.mean()
        contamination_setting = min(0.1, max(0.01, fraud_rate * 1.5))
        print(f"Setting contamination to {contamination_setting:.4f} based on fraud rate {fraud_rate:.4f}")
    else:
        contamination_setting = 0.05
        print("Using default contamination: 0.05")
    
    iforest = IsolationForest(
        n_estimators=500,  # Increased estimators for better performance
        contamination=contamination_setting,
        random_state=42, 
        n_jobs=-1,
        max_features=0.8,
        max_samples=0.7,
        bootstrap=True
    )
    iforest.fit(X_scaled)
    if_scores = iforest.decision_function(X_scaled)
    
    print(f"Isolation Forest scores: min={if_scores.min():.4f}, max={if_scores.max():.4f}, mean={if_scores.mean():.4f}")
    
    # Store normalization parameters
    ae_min, ae_max = ae_errors.min(), ae_errors.max()
    if_min, if_max = if_scores.min(), if_scores.max()
    
    # Normalize scores
    ae_range = ae_max - ae_min
    if_range = if_max - if_min
    
    if ae_range > 0:
        ae_errors_norm = (ae_errors - ae_min) / ae_range
    else:
        ae_errors_norm = np.zeros_like(ae_errors)
    
    if if_range > 0:
        if_scores_norm = (if_scores - if_min) / if_range
    else:
        if_scores_norm = np.zeros_like(if_scores)
    
    if_scores_norm = 1 - if_scores_norm  # Invert for consistency
    
    # Ensure non-negative scores
    ae_errors_norm = np.maximum(ae_errors_norm, 0)
    if_scores_norm = np.maximum(if_scores_norm, 0)
    
    # Revised ensemble weights to prioritize recall
    blended_scores = 0.4 * ae_errors_norm + 0.6 * if_scores_norm
    
    print(f"Blended scores: min={blended_scores.min():.4f}, max={blended_scores.max():.4f}, mean={blended_scores.mean():.4f}")
    
    # Optimized threshold selection for recall >= 0.8
    if fraud_labels is not None:
        print("Optimizing threshold for recall >= 0.8...")
        
        # Use precision-recall curve to find optimal threshold
        precision_curve, recall_curve, thresholds = precision_recall_curve(fraud_labels, blended_scores)
        
        # Calculate F1 scores for all thresholds
        f1_scores = 2 * (precision_curve[:-1] * recall_curve[:-1]) / (precision_curve[:-1] + recall_curve[:-1] + 1e-8)
        
        # Strategy: Find thresholds meeting recall target of 0.8
        candidate_indices = []
        for i in range(len(thresholds)):
            if recall_curve[i] >= 0.8:
                candidate_indices.append(i)
        
        if candidate_indices:
            # Select threshold with highest F1 score among candidates
            best_idx = candidate_indices[np.argmax(f1_scores[candidate_indices])]
            threshold = thresholds[best_idx]
            best_precision = precision_curve[best_idx]
            best_recall = recall_curve[best_idx]
            best_f1 = f1_scores[best_idx]
            print(f"Optimal threshold: {threshold:.4f}")
            print(f"Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}")
        else:
            # Fallback: Use threshold that maximizes F1 with min 75% recall
            print("No threshold meets recall target. Finding best threshold with min 75% recall...")
            candidate_indices = []
            for i in range(len(thresholds)):
                if recall_curve[i] >= 0.75:
                    candidate_indices.append(i)
            
            if candidate_indices:
                best_idx = candidate_indices[np.argmax(f1_scores[candidate_indices])]
                threshold = thresholds[best_idx]
                best_precision = precision_curve[best_idx]
                best_recall = recall_curve[best_idx]
                best_f1 = f1_scores[best_idx]
                print(f"Alternative threshold: {threshold:.4f}")
                print(f"Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}")
            else:
                # Final fallback: Maximize F1 with min 60% recall
                print("No threshold meets min recall. Using best F1 with min 60% recall")
                min_recall = 0.6
                valid_indices = recall_curve[:-1] >= min_recall
                
                if np.any(valid_indices):
                    valid_f1_scores = f1_scores[valid_indices]
                    valid_thresholds = thresholds[valid_indices]
                    best_idx = np.argmax(valid_f1_scores)
                    threshold = valid_thresholds[best_idx]
                    best_precision = precision_curve[best_idx]
                    best_recall = recall_curve[best_idx]
                    best_f1 = f1_scores[best_idx]
                    print(f"Fallback threshold: {threshold:.4f}")
                    print(f"Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}")
                else:
                    threshold = np.percentile(blended_scores, 90)  # More conservative
                    print(f"Conservative threshold: {threshold:.4f}")
    else:
        # Conservative threshold for high recall
        threshold = np.percentile(blended_scores, 85)
        print(f"Threshold set at 85th percentile: {threshold:.4f}")
    
    print(f"Anomalies detected: {np.sum(blended_scores > threshold)}")
    
    normalization_params = {
        'ae_min': float(ae_min),
        'ae_max': float(ae_max),
        'if_min': float(if_min),
        'if_max': float(if_max)
    }
    
    return autoencoder, iforest, scaler, threshold, normalization_params, blended_scores

def evaluate_model_performance(blended_scores, fraud_labels, threshold):
    """Evaluate the unsupervised model against actual fraud labels"""
    print("\n" + "="*60)
    print("MODEL EVALUATION AGAINST ACTUAL FRAUD LABELS")
    print("="*60)
    
    if fraud_labels is None:
        print("Cannot evaluate: No fraud labels available")
        return
    
    # Create binary predictions
    predictions = (blended_scores > threshold).astype(int)
    
    # Basic statistics
    print(f"\nDataset Statistics:")
    print(f"Total transactions: {len(fraud_labels)}")
    print(f"Actual fraud cases: {fraud_labels.sum()}")
    print(f"Actual fraud rate: {fraud_labels.mean():.4f}")
    print(f"Predicted anomalies: {predictions.sum()}")
    print(f"Predicted anomaly rate: {predictions.mean():.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(fraud_labels, predictions)
    print(f"\nConfusion Matrix:")
    print(f"{'':>12} {'Predicted':>20}")
    print(f"{'Actual':>12} {'Normal':>10} {'Anomaly':>10}")
    print(f"{'Normal':>12} {cm[0,0]:>10} {cm[0,1]:>10}")
    print(f"{'Fraud':>12} {cm[1,0]:>10} {cm[1,1]:>10}")
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    print(f"\nPerformance Metrics:")
    print(f"Accuracy: {accuracy:.4f} (Overall correctness)")
    print(f"Recall: {recall:.4f} (What % of actual fraud cases were detected)")
    print(f"F1-Score: {f1_score:.4f} (Harmonic mean of precision and recall)")
    print(f"Specificity: {specificity:.4f} (What % of normal transactions were correctly identified)")
    
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(fraud_labels, predictions, target_names=['Normal', 'Fraud']))
    
    # ROC-AUC Score
    try:
        roc_auc = roc_auc_score(fraud_labels, blended_scores)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        print("(Higher is better, 0.5 = random, 1.0 = perfect)")
    except Exception as e:
        print(f"Could not calculate ROC-AUC: {e}")
    
    # Analyze score distributions
    print(f"\nScore Distribution Analysis:")
    fraud_scores = blended_scores[fraud_labels == 1]
    normal_scores = blended_scores[fraud_labels == 0]
    
    print(f"Normal transactions - Mean score: {normal_scores.mean():.4f}, Std: {normal_scores.std():.4f}")
    print(f"Fraud transactions - Mean score: {fraud_scores.mean():.4f}, Std: {fraud_scores.std():.4f}")
    
    # Threshold analysis
    print(f"\nThreshold Analysis:")
    print(f"Current threshold: {threshold:.4f}")
    print(f"% of fraud cases above threshold: {(fraud_scores > threshold).mean():.4f}")
    print(f"% of normal cases above threshold: {(normal_scores > threshold).mean():.4f}")

def save_evaluation_results(blended_scores, fraud_labels, threshold, feature_cols):
    """Save evaluation results to files"""
    if fraud_labels is None:
        return
    
    # Create evaluation results dataframe
    results_df = pd.DataFrame({
        'risk_score': blended_scores,
        'is_anomaly': (blended_scores > threshold).astype(int),
        'actual_fraud': fraud_labels,
        'correct_prediction': ((blended_scores > threshold).astype(int) == fraud_labels).astype(int)
    })
    
    # Save results
    results_df.to_csv('evaluation_results.csv', index=False)
    print(f"\nEvaluation results saved to 'evaluation_results.csv'")
    
    # Save summary statistics
    predictions = (blended_scores > threshold).astype(int)
    tp = np.sum((predictions == 1) & (fraud_labels == 1))
    fp = np.sum((predictions == 1) & (fraud_labels == 0))
    fn = np.sum((predictions == 0) & (fraud_labels == 1))
    tn = np.sum((predictions == 0) & (fraud_labels == 0))
    
    summary_stats = {
        'total_transactions': len(fraud_labels),
        'actual_fraud_cases': int(fraud_labels.sum()),
        'predicted_anomalies': int(predictions.sum()),
        'threshold': float(threshold),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
        'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0,
        'feature_columns': feature_cols
    }
    
    with open('evaluation_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"Evaluation summary saved to 'evaluation_summary.json'")

def save_models(autoencoder, iforest, scaler, threshold, feature_cols, cat_encoder, normalization_params):
    print("Saving models...")
    
    # Save PyTorch model
    torch.save(autoencoder.state_dict(), 'ae.pt')
    
    # Save sklearn models
    joblib.dump(iforest, 'iforest.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save categorical encoder if it exists
    if cat_encoder is not None:
        joblib.dump(cat_encoder, 'cat_encoder.pkl')
        print("Categorical encoder saved")
    
    # Save metadata including normalization parameters
    metadata = {
        'feature_cols': feature_cols,
        'threshold': threshold,
        'input_dim': len(feature_cols),
        'has_cat_encoder': cat_encoder is not None,
        'normalization_params': normalization_params
    }
    
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Models saved successfully!")

def main():
    print("Starting fraud detection training with recall optimization...")
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Generate rolling features
        df = generate_rolling_features_optimized(df)
        print(f"Rolling features generated. Shape: {df.shape}")
        
        # Prepare features
        X, feature_cols, cat_encoder, fraud_labels = prepare_features(df)
        print(f"Features prepared. Shape: {X.shape}")
        
        # Train models with fraud labels for threshold optimization
        autoencoder, iforest, scaler, threshold, normalization_params, blended_scores = train_models(X, fraud_labels)
        
        # Evaluate model performance
        evaluate_model_performance(blended_scores, fraud_labels, threshold)
        
        # Save evaluation results
        save_evaluation_results(blended_scores, fraud_labels, threshold, feature_cols)
        
        # Save models
        save_models(autoencoder, iforest, scaler, threshold, feature_cols, cat_encoder, normalization_params)
        
        print("\nTraining and evaluation completed successfully!")
        
    except FileNotFoundError:
        print("Error: Could not find 'cleaned_fraud_data.csv'. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()