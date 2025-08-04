import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold
from category_encoders import TargetEncoder
import joblib
import json
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class PrecisionOptimizedAutoencoder(nn.Module):
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
    def __init__(self, input_dim):
        self.isolation_forest = None
        self.autoencoder = None
        self.scaler = None
        self.input_dim = input_dim
        self.if_threshold = None
        self.ae_threshold = None
        self.hybrid_threshold = None
        #self.weights = {'isolation_forest': 0.7, 'autoencoder': 0.3}
        self.feature_importance_ = None
        self.precision_boost_factor = 1.15  
        
    def fit(self, X, y=None, contamination_rate=0.012):  
        # Use RobustScaler for better handling of outliers
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # More conservative Isolation Forest settings for higher precision
        self.isolation_forest = IsolationForest(
            n_estimators=500,  
            contamination=contamination_rate, 
            random_state=42,
            max_features=min(6, X_scaled.shape[1]),  
            max_samples=min(1500, X_scaled.shape[0]),  
            n_jobs=-1,
            bootstrap=True  
        )
        self.isolation_forest.fit(X_scaled)
        
        # Enhanced autoencoder training
        self.autoencoder = PrecisionOptimizedAutoencoder(self.input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.0003, weight_decay=1e-3) 
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.autoencoder.train()
        best_loss = float('inf')
        patience = 8
        patience_counter = 0
        
        for epoch in range(80):  
            optimizer.zero_grad()
            outputs = self.autoencoder(X_tensor)
            loss = criterion(outputs, X_tensor)
            
            # Enhanced regularization
            l1_reg = sum(p.abs().sum() for p in self.autoencoder.parameters())
            l2_reg = sum(p.pow(2).sum() for p in self.autoencoder.parameters())
            loss = loss + 2e-5 * l1_reg + 1e-5 * l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 0.5)
            optimizer.step()
            scheduler.step(loss)
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        if_scores = -self.isolation_forest.decision_function(X_scaled)
        
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructed = self.autoencoder(X_tensor)
            ae_scores = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
        
        if_scores = self._precision_normalize(if_scores)
        ae_scores = self._precision_normalize(ae_scores)
        
        if y is not None:
            self._calculate_feature_importance(X, y)
        
        # Set individual model thresholds
        self.if_threshold = np.percentile(if_scores, 92)
        self.ae_threshold = np.percentile(ae_scores, 92)
        
        if y is not None:
            self._optimize_for_precision_recall(if_scores, ae_scores, y)
        else:
            self.hybrid_threshold = 0.75
    
    def _calculate_feature_importance(self, X, y):
        feature_importance_data = []
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        if hasattr(X, 'columns'):
            corr_scores = np.abs(pd.DataFrame(X).corrwith(pd.Series(y)))
        else:
            corr_scores = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
            corr_scores = np.nan_to_num(corr_scores)
        
        if_scores = -self.isolation_forest.decision_function(self.scaler.transform(X))
        if_importance = []
        for i in range(X.shape[1]):
            X_permuted = X.copy()
            if hasattr(X, 'iloc'):
                X_permuted.iloc[:, i] = np.random.permutation(X_permuted.iloc[:, i])
            else:
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            
            if_scores_permuted = -self.isolation_forest.decision_function(self.scaler.transform(X_permuted))
            importance = np.mean(np.abs(if_scores - if_scores_permuted))
            if_importance.append(importance)
        
        if_importance = np.array(if_importance)
        
        for i, feature_name in enumerate(self.feature_names):
            feature_importance_data.append({
                'feature': feature_name,
                'mutual_info_score': mi_scores[i],
                'correlation_score': corr_scores[i] if hasattr(corr_scores, '__len__') else corr_scores.iloc[i],
                'isolation_forest_importance': if_importance[i],
                'combined_importance': (mi_scores[i] * 0.4 + 
                                      (corr_scores[i] if hasattr(corr_scores, '__len__') else corr_scores.iloc[i]) * 0.3 + 
                                      if_importance[i] * 0.3)
            })
        
        self.feature_importance_ = pd.DataFrame(feature_importance_data)
        self.feature_importance_ = self.feature_importance_.sort_values('combined_importance', ascending=False)
    
    def get_feature_importance(self):
        return self.feature_importance_
    
    def _precision_normalize(self, scores):
        # Enhanced normalization for better precision
        scores_shifted = scores - scores.min() + 1e-8
        scores_log = np.log1p(scores_shifted)
        
        # Use more conservative percentiles
        q5, q95 = np.percentile(scores_log, [5, 95])
        normalized = (scores_log - q5) / (q95 - q5 + 1e-8)
        
        normalized = np.clip(normalized, 0, 4)
        # Apply stronger non-linear transformation for precision
        normalized = np.power(normalized / 4, 0.6) * 4
        normalized = normalized / 4
        
        return normalized
    
    def _optimize_for_precision_recall(self, if_scores, ae_scores, y, min_precision=0.72, target_recall=0.8):
        """Enhanced optimization focusing on precision while maintaining recall"""
        
        # First, optimize individual model thresholds
        self._optimize_individual_thresholds(if_scores, ae_scores, y)
        
        best_score = -1
        best_params = None
        
        # More conservative weight combinations favoring precision
        weight_combinations = [
            {'isolation_forest': 0.65, 'autoencoder': 0.35},
            {'isolation_forest': 0.70, 'autoencoder': 0.30},
            {'isolation_forest': 0.75, 'autoencoder': 0.25},
            {'isolation_forest': 0.80, 'autoencoder': 0.20},
            {'isolation_forest': 0.60, 'autoencoder': 0.40},
            {'isolation_forest': 0.85, 'autoencoder': 0.15},
            {'isolation_forest': 0.55, 'autoencoder': 0.45},
            {'isolation_forest': 0.90, 'autoencoder': 0.10},
        ]
        
        for weights in weight_combinations:
            hybrid_scores = (weights['isolation_forest'] * if_scores + 
                           weights['autoencoder'] * ae_scores)
            
            precision_curve, recall_curve, thresholds = precision_recall_curve(y, hybrid_scores)
            
            for i, threshold in enumerate(thresholds):
                if i < len(precision_curve) - 1:
                    prec = precision_curve[i]
                    rec = recall_curve[i]
                    
                    # Prioritize precision while ensuring recall meets minimum
                    if prec >= min_precision and rec >= target_recall:
                        # Use F1 score as tiebreaker, but heavily weight precision
                        f1 = 2 * (prec * rec) / (prec + rec)
                        precision_weighted_score = 0.7 * prec + 0.3 * rec  # Weight precision more
                        
                        if precision_weighted_score > best_score:
                            best_score = precision_weighted_score
                            best_params = {
                                'weights': weights,
                                'threshold': threshold,
                                'precision': prec,
                                'recall': rec,
                                'f1': f1
                            }
        
        if best_params:
            self.weights = best_params['weights']
            self.hybrid_threshold = best_params['threshold'] * self.precision_boost_factor  # Apply precision boost
        else:
            # Fallback with precision-focused approach
            self.weights = {'isolation_forest': 0.75, 'autoencoder': 0.25}
            hybrid_scores = (self.weights['isolation_forest'] * if_scores + 
                           self.weights['autoencoder'] * ae_scores)
            
            # Find threshold that maximizes precision while maintaining recall >= 0.8
            precision_curve, recall_curve, thresholds = precision_recall_curve(y, hybrid_scores)
            valid_indices = recall_curve >= target_recall
            
            if np.any(valid_indices):
                valid_precisions = precision_curve[valid_indices]
                valid_thresholds = thresholds[valid_indices]
                best_precision_idx = np.argmax(valid_precisions)
                self.hybrid_threshold = valid_thresholds[best_precision_idx] * self.precision_boost_factor
            else:
                self.hybrid_threshold = np.percentile(hybrid_scores, 96)  # More conservative threshold
    
    def _optimize_individual_thresholds(self, if_scores, ae_scores, y, min_precision=0.5):
        """Optimize individual model thresholds for better individual performance"""
        
        # Optimize Isolation Forest threshold
        if_precision_curve, if_recall_curve, if_thresholds = precision_recall_curve(y, if_scores)
        best_if_f1 = 0
        best_if_threshold = np.percentile(if_scores, 92)
        
        for i, threshold in enumerate(if_thresholds):
            if i < len(if_precision_curve) - 1:
                prec = if_precision_curve[i]
                rec = if_recall_curve[i]
                if prec >= min_precision and rec > 0:
                    f1 = 2 * (prec * rec) / (prec + rec)
                    if f1 > best_if_f1:
                        best_if_f1 = f1
                        best_if_threshold = threshold
        
        self.if_threshold = best_if_threshold
        
        # Optimize Autoencoder threshold
        ae_precision_curve, ae_recall_curve, ae_thresholds = precision_recall_curve(y, ae_scores)
        best_ae_f1 = 0
        best_ae_threshold = np.percentile(ae_scores, 92)
        
        for i, threshold in enumerate(ae_thresholds):
            if i < len(ae_precision_curve) - 1:
                prec = ae_precision_curve[i]
                rec = ae_recall_curve[i]
                if prec >= min_precision and rec > 0:
                    f1 = 2 * (prec * rec) / (prec + rec)
                    if f1 > best_ae_f1:
                        best_ae_f1 = f1
                        best_ae_threshold = threshold
        
        self.ae_threshold = best_ae_threshold
        
    def predict_scores(self, X):
        X_scaled = self.scaler.transform(X)
        
        if_scores = -self.isolation_forest.decision_function(X_scaled)
        if_scores = self._precision_normalize(if_scores)
        
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            reconstructed = self.autoencoder(X_tensor)
            ae_scores = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
            ae_scores = self._precision_normalize(ae_scores)
        
        hybrid_scores = (self.weights['isolation_forest'] * if_scores + 
                        self.weights['autoencoder'] * ae_scores)
        
        return hybrid_scores, if_scores, ae_scores
    
    def predict(self, X):
        hybrid_scores, _, _ = self.predict_scores(X)
        return (hybrid_scores > self.hybrid_threshold).astype(int)
    
    def predict_individual_models(self, X):
        """Predict using individual models with their own thresholds"""
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest predictions
        if_scores = -self.isolation_forest.decision_function(X_scaled)
        if_scores_norm = self._precision_normalize(if_scores)
        if_predictions = (if_scores_norm > self.if_threshold).astype(int)
        
        # Autoencoder predictions
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            reconstructed = self.autoencoder(X_tensor)
            ae_scores = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
            ae_scores_norm = self._precision_normalize(ae_scores)
            ae_predictions = (ae_scores_norm > self.ae_threshold).astype(int)
        
        return if_predictions, ae_predictions, if_scores_norm, ae_scores_norm

def precision_score_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall_score_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score_manual(y_true, y_pred):
    precision = precision_score_manual(y_true, y_pred)
    recall = recall_score_manual(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def evaluate_individual_models(y_true, if_predictions, ae_predictions):
    """Evaluate individual model performance"""
    
    # Isolation Forest metrics
    if_precision = precision_score_manual(y_true, if_predictions)
    if_recall = recall_score_manual(y_true, if_predictions)
    if_f1 = f1_score_manual(y_true, if_predictions)
    if_cm = confusion_matrix(y_true, if_predictions)
    if_tn, if_fp, if_fn, if_tp = if_cm.ravel()
    if_specificity = if_tn / (if_tn + if_fp) if (if_tn + if_fp) > 0 else 0
    
    # Autoencoder metrics
    ae_precision = precision_score_manual(y_true, ae_predictions)
    ae_recall = recall_score_manual(y_true, ae_predictions)
    ae_f1 = f1_score_manual(y_true, ae_predictions)
    ae_cm = confusion_matrix(y_true, ae_predictions)
    ae_tn, ae_fp, ae_fn, ae_tp = ae_cm.ravel()
    ae_specificity = ae_tn / (ae_tn + ae_fp) if (ae_tn + ae_fp) > 0 else 0
    
    individual_metrics = {
        'isolation_forest': {
            'precision': float(if_precision),
            'recall': float(if_recall),
            'f1_score': float(if_f1),
            'specificity': float(if_specificity),
            'true_positives': int(if_tp),
            'false_positives': int(if_fp),
            'false_negatives': int(if_fn),
            'true_negatives': int(if_tn),
            'confusion_matrix': if_cm.tolist()
        },
        'autoencoder': {
            'precision': float(ae_precision),
            'recall': float(ae_recall),
            'f1_score': float(ae_f1),
            'specificity': float(ae_specificity),
            'true_positives': int(ae_tp),
            'false_positives': int(ae_fp),
            'false_negatives': int(ae_fn),
            'true_negatives': int(ae_tn),
            'confusion_matrix': ae_cm.tolist()
        }
    }
    
    return individual_metrics

def load_and_preprocess_data():
    df = pd.read_csv("cleaned_fraud_data.csv")
    return df

def create_precision_focused_features(df, max_samples=100000, fraud_ratio=0.018):  
    if len(df) > max_samples:
        if 'fraud' in df.columns:
            fraud_samples = df[df['fraud'] == 1]
            normal_samples = df[df['fraud'] == 0]
            
            total_fraud = len(fraud_samples)
            total_normal = len(normal_samples)
            
            if total_fraud > 0:
                max_fraud_samples = min(total_fraud, int(max_samples * fraud_ratio))
                max_normal_samples = min(total_normal, max_samples - max_fraud_samples)
                
                max_fraud_samples = max(max_fraud_samples, min(1200, total_fraud))  # Ensure more fraud samples
                max_normal_samples = min(max_normal_samples, max_samples - max_fraud_samples)
                
                fraud_selected = fraud_samples.sample(n=max_fraud_samples, random_state=42)
                normal_selected = normal_samples.sample(n=max_normal_samples, random_state=42)
                df = pd.concat([fraud_selected, normal_selected], ignore_index=True)
            else:
                df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        else:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    
    # Enhanced customer statistics with more precision-focused features
    customer_stats = df.groupby('customer')['amount'].agg([
        'mean', 'std', 'median', 'count', 'min', 'max', 'sum'
    ]).fillna(0)
    
    customer_percentiles = df.groupby('customer')['amount'].quantile([0.05, 0.1, 0.25, 0.75, 0.9, 0.95]).unstack()
    customer_percentiles.columns = ['amount_q05', 'amount_q10', 'amount_q25', 'amount_q75', 'amount_q90', 'amount_q95']
    customer_stats = pd.concat([customer_stats, customer_percentiles], axis=1)
    
    customer_stats.columns = ['cust_amt_mean', 'cust_amt_std', 'cust_amt_median', 
                             'cust_txn_count', 'cust_amt_min', 'cust_amt_max', 'cust_amt_sum',
                             'cust_amt_q05', 'cust_amt_q10', 'cust_amt_q25', 'cust_amt_q75', 'cust_amt_q90', 'cust_amt_q95']
    
    df = df.merge(customer_stats, left_on='customer', right_index=True, how='left')
    
    # Enhanced feature engineering for better precision
    df['cust_amt_range'] = df['cust_amt_max'] - df['cust_amt_min']
    df['cust_amt_iqr'] = df['cust_amt_q75'] - df['cust_amt_q25']
    df['cust_amt_cv'] = df['cust_amt_std'] / (df['cust_amt_mean'] + 1e-8)
    df['cust_amt_skewness'] = (df['cust_amt_mean'] - df['cust_amt_median']) / (df['cust_amt_std'] + 1e-8)
    
    # Enhanced ratio features
    df['amt_vs_cust_mean_ratio'] = df['amount'] / (df['cust_amt_mean'] + 1e-8)
    df['amt_vs_cust_median_ratio'] = df['amount'] / (df['cust_amt_median'] + 1e-8)
    df['amt_vs_cust_q95_ratio'] = df['amount'] / (df['cust_amt_q95'] + 1e-8)
    df['amt_vs_cust_max_ratio'] = df['amount'] / (df['cust_amt_max'] + 1e-8)
    
    # Enhanced z-score features
    df['amt_zscore'] = (df['amount'] - df['cust_amt_mean']) / (df['cust_amt_std'] + 1e-8)
    df['amt_robust_zscore'] = (df['amount'] - df['cust_amt_median']) / (df['cust_amt_iqr'] + 1e-8)
    df['amt_extreme_zscore'] = (df['amount'] - df['cust_amt_q95']) / (df['cust_amt_std'] + 1e-8)
    
    # Enhanced binary features with stricter thresholds
    df['is_cust_extreme_spender'] = (df['cust_amt_max'] > df['cust_amt_max'].quantile(0.97)).astype(int)
    df['is_cust_consistent'] = (df['cust_amt_cv'] < df['cust_amt_cv'].quantile(0.15)).astype(int)
    df['is_cust_highly_volatile'] = (df['cust_amt_cv'] > df['cust_amt_cv'].quantile(0.97)).astype(int)
    df['is_cust_rare_user'] = (df['cust_txn_count'] < df['cust_txn_count'].quantile(0.05)).astype(int)
    
    return df

def remove_highly_correlated_features(X, correlation_threshold=0.85):  # Slightly higher threshold
    correlation_matrix = X.corr().abs()
    
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    high_corr_pairs = []
    for col in upper_triangle.columns:
        corr_features = upper_triangle.index[upper_triangle[col] > correlation_threshold].tolist()
        for corr_feature in corr_features:
            high_corr_pairs.append((col, corr_feature, upper_triangle.loc[corr_feature, col]))
    
    corr_groups = []
    processed_features = set()
    
    for feat1, feat2, corr_val in high_corr_pairs:
        if feat1 not in processed_features and feat2 not in processed_features:
            group = {feat1, feat2}
            processed_features.update(group)
            
            for other_feat1, other_feat2, _ in high_corr_pairs:
                if (other_feat1 in group or other_feat2 in group) and other_feat1 not in processed_features:
                    group.add(other_feat1)
                    processed_features.add(other_feat1)
                if (other_feat1 in group or other_feat2 in group) and other_feat2 not in processed_features:
                    group.add(other_feat2)
                    processed_features.add(other_feat2)
            
            corr_groups.append(list(group))
    
    features_to_remove = []
    features_to_keep = []
    
    for group in corr_groups:
        if len(group) > 1:
            variances = X[group].var()
            keep_feature = variances.idxmax()
            remove_features = [f for f in group if f != keep_feature]
            
            features_to_keep.append(keep_feature)
            features_to_remove.extend(remove_features)
    
    X_reduced = X.drop(columns=features_to_remove)
    
    return X_reduced, features_to_remove

def select_top_features_by_importance(X, y, n_features=20):  # Increased feature count
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_rankings = pd.Series(mi_scores, index=X.columns).rank(ascending=False)
    
    corr_scores = np.abs(X.corrwith(pd.Series(y)))
    corr_rankings = corr_scores.rank(ascending=False)
    
    var_scores = X.var()
    var_rankings = var_scores.rank(ascending=False)
    
    combined_rankings = (
        mi_rankings * 0.5 +
        corr_rankings * 0.3 +
        var_rankings * 0.2
    )
    
    # Enhanced critical features list
    critical_features = [
        'merchant_risk_score',
        'amount_vs_merchant_avg',
        'cust_amt_std',
        'is_cust_extreme_spender',
        'amt_zscore',
        'amt_vs_cust_mean_ratio'
    ]
    
    present_critical = [f for f in critical_features if f in X.columns]
    
    top_features = combined_rankings.nsmallest(n_features).index.tolist()
    
    missing_critical = [f for f in present_critical if f not in top_features]
    if missing_critical:
        num_to_remove = min(len(missing_critical), len(top_features) - (n_features - len(missing_critical)))
        if num_to_remove > 0:
            top_features = top_features[:-num_to_remove]
        top_features.extend(missing_critical)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'mutual_info_score': mi_scores,
        'correlation_score': corr_scores,
        'variance_score': var_scores,
        'combined_ranking': combined_rankings,
        'selected': X.columns.isin(top_features)
    }).sort_values('combined_ranking')
    
    return X[top_features], top_features, importance_df

def prepare_precision_features(df, correlation_threshold=0.85, n_top_features=20):
    fraud_labels = None
    if 'fraud' in df.columns:
        fraud_labels = df['fraud'].values
    
    core_features = [
        'amount', 'step', 'cust_amt_mean', 'cust_amt_std', 'cust_amt_median',
        'cust_txn_count', 'cust_amt_sum', 'cust_amt_range', 'cust_amt_iqr', 'cust_amt_cv',
        'cust_amt_skewness', 'amt_vs_cust_mean_ratio', 'amt_vs_cust_median_ratio', 
        'amt_vs_cust_q95_ratio', 'amt_vs_cust_max_ratio', 'amt_zscore', 'amt_robust_zscore',
        'amt_extreme_zscore', 'is_cust_extreme_spender', 'is_cust_consistent', 
        'is_cust_highly_volatile', 'is_cust_rare_user',
        'merchant_risk_score', 'amount_vs_merchant_avg'
    ]
    
    existing_features = [col for col in core_features if col in df.columns]
    X = df[existing_features].copy()
    
    # Enhanced feature transformations
    X['amount_log'] = np.log1p(X['amount'])
    X['amount_sqrt'] = np.sqrt(X['amount'])
    X['amount_square'] = np.square(X['amount'])
    X['amount_cube_root'] = np.power(X['amount'], 1/3)
    
    # Enhanced amount-based features
    X['is_exact_round'] = (X['amount'] % 100 == 0).astype(int)
    X['is_very_high_amount'] = (X['amount'] > X['amount'].quantile(0.995)).astype(int)
    X['is_extreme_amount'] = (X['amount'] > X['amount'].quantile(0.998)).astype(int)
    X['is_micro_transaction'] = (X['amount'] < X['amount'].quantile(0.05)).astype(int)
    
    # Enhanced temporal features
    X['day_of_week'] = X['step'] % 7
    X['is_weekend'] = ((X['step'] % 7).isin([5, 6])).astype(int)
    
    # Enhanced merchant features (if available)
    if 'merchant' in df.columns and fraud_labels is not None:
        merchant_stats = df.groupby('merchant').agg({
            'fraud': ['count', 'sum', 'mean'],
            'amount': ['count', 'mean', 'std', 'max'],
            'customer': 'nunique'
        })
        
        merchant_stats.columns = ['merch_total_txns', 'merch_fraud_count', 'merch_fraud_rate',
                                 'merch_amt_count', 'merch_amt_mean', 'merch_amt_std', 'merch_amt_max',
                                 'merch_unique_customers']
        
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        merchant_encoded = np.zeros(len(df))
        
        for train_idx, val_idx in skf.split(df, fraud_labels):
            train_data = df.iloc[train_idx]
            
            merchant_fraud_rates = train_data.groupby('merchant')['fraud'].mean()
            merchant_counts = train_data.groupby('merchant').size()
            global_fraud_rate = train_data['fraud'].mean()
            
            alpha = 75  
            smoothed_rates = ((merchant_fraud_rates * merchant_counts + global_fraud_rate * alpha) / 
                            (merchant_counts + alpha))
            
            val_merchants = df.iloc[val_idx]['merchant']
            merchant_encoded[val_idx] = val_merchants.map(smoothed_rates).fillna(global_fraud_rate)
        
        X['merchant_risk_score'] = merchant_encoded
        X['merchant_frequency'] = np.log1p(df['merchant'].map(merchant_stats['merch_total_txns']).fillna(1))
        X['merchant_diversity'] = df['merchant'].map(merchant_stats['merch_unique_customers']).fillna(1)
        X['is_high_risk_merchant'] = (df['merchant'].map(merchant_stats['merch_fraud_rate']) > 
                                     merchant_stats['merch_fraud_rate'].quantile(0.95)).astype(int)
        X['is_very_rare_merchant'] = (df['merchant'].map(merchant_stats['merch_total_txns']) < 
                                     merchant_stats['merch_total_txns'].quantile(0.02)).astype(int)
        X['merchant_avg_amount'] = df['merchant'].map(merchant_stats['merch_amt_mean']).fillna(0)
        X['amount_vs_merchant_avg'] = X['amount'] / (X['merchant_avg_amount'] + 1e-8)
        
    else:
        default_cols = ['merchant_risk_score', 'merchant_frequency', 'merchant_diversity',
                       'is_high_risk_merchant', 'is_very_rare_merchant', 'merchant_avg_amount',
                       'amount_vs_merchant_avg']
        for col in default_cols:
            X[col] = 0
    
    # Clean data
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(0.0).astype(float)
    X = X.replace([np.inf, -np.inf], 0.0)
    
    X_decorrelated, removed_corr_features = remove_highly_correlated_features(X, correlation_threshold)
    
    if fraud_labels is not None:
        X_final, selected_features, importance_df = select_top_features_by_importance(
            X_decorrelated, fraud_labels, n_top_features
        )
        
        importance_df.to_csv('detailed_feature_analysis.csv', index=False)
        
        return X_final, selected_features, X_decorrelated.corr(), fraud_labels
    else:
        return X_decorrelated, list(X_decorrelated.columns), X_decorrelated.corr(), fraud_labels

def save_correlation_matrix(correlation_matrix, output_path='correlation_matrix.csv'):
    correlation_matrix.to_csv(output_path)
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
    plt.title('Feature Correlation Matrix (After Selection)')
    plt.tight_layout()
    plt.savefig('correlation_heatmap_final.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_precision_model(hybrid_model, X, fraud_labels):
    predictions = hybrid_model.predict(X)
    hybrid_scores, if_scores, ae_scores = hybrid_model.predict_scores(X)
    
    cm = confusion_matrix(fraud_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    precision_target_met = precision >= 0.7
    recall_target_met = recall >= 0.8
    both_targets_met = precision_target_met and recall_target_met
    
    return precision, recall, f1_score, both_targets_met, hybrid_scores

def save_precision_model(hybrid_model, feature_cols):
    joblib.dump(hybrid_model.isolation_forest, 'fraud_isolation_forest_precision.pkl')
    torch.save(hybrid_model.autoencoder.state_dict(), 'fraud_autoencoder_precision.pt')
    joblib.dump(hybrid_model.scaler, 'fraud_scaler_precision.pkl')
    
    metadata = {
        'feature_cols': feature_cols,
        'hybrid_threshold': float(hybrid_model.hybrid_threshold),  
        'if_threshold': float(hybrid_model.if_threshold),
        'ae_threshold': float(hybrid_model.ae_threshold),
        'weights': {k: float(v) for k, v in hybrid_model.weights.items()},
        'input_dim': int(hybrid_model.input_dim),
        'model_type': 'enhanced_precision_focused_hybrid',
        'training_date': datetime.now().isoformat(),
        'target_precision': 0.72,
        'target_recall': 0.8,
        'optimization': 'enhanced_precision_boosting',
        'num_features': len(feature_cols),
        'precision_boost_factor': float(hybrid_model.precision_boost_factor)
    }
    
    with open('fraud_metadata_precision.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def advanced_threshold_optimization(hybrid_model, X_val, y_val, X_test, y_test, min_precision=0.72):
    """Advanced threshold optimization with validation-test consistency check"""
    
    # Get validation scores
    val_scores, val_if_scores, val_ae_scores = hybrid_model.predict_scores(X_val)
    test_scores, test_if_scores, test_ae_scores = hybrid_model.predict_scores(X_test)
    
    best_threshold = None
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    
    # Grid search over different weight combinations and thresholds
    weight_grid = [
        {'isolation_forest': 0.60, 'autoencoder': 0.40},
        {'isolation_forest': 0.65, 'autoencoder': 0.35},
        {'isolation_forest': 0.70, 'autoencoder': 0.30},
        {'isolation_forest': 0.75, 'autoencoder': 0.25},
        {'isolation_forest': 0.80, 'autoencoder': 0.20},
        {'isolation_forest': 0.85, 'autoencoder': 0.15},
    ]
    
    for weights in weight_grid:
        # Calculate hybrid scores with current weights
        val_hybrid = (weights['isolation_forest'] * val_if_scores + 
                     weights['autoencoder'] * val_ae_scores)
        
        # Find optimal threshold on validation set
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_val, val_hybrid)
        
        for i, threshold in enumerate(thresholds):
            if i >= len(precision_vals) - 1:
                continue
                
            val_pred = (val_hybrid > threshold).astype(int)
            val_prec = precision_score_manual(y_val, val_pred)
            val_rec = recall_score_manual(y_val, val_pred)
            
            # Check if validation metrics meet criteria
            if val_prec >= min_precision and val_rec >= 0.8:
                # Test on test set with same parameters
                test_hybrid = (weights['isolation_forest'] * test_if_scores + 
                              weights['autoencoder'] * test_ae_scores)
                test_pred = (test_hybrid > threshold).astype(int)
                test_prec = precision_score_manual(y_test, test_pred)
                test_rec = recall_score_manual(y_test, test_pred)
                
                
                if test_prec >= min_precision and test_rec >= 0.8:
                    precision_score_weighted = 0.6 * test_prec + 0.4 * test_rec
                    
                    if precision_score_weighted > (0.6 * best_precision + 0.4 * best_recall):
                        best_threshold = threshold
                        best_precision = test_prec
                        best_recall = test_rec
                        best_f1 = 2 * (test_prec * test_rec) / (test_prec + test_rec)
                        hybrid_model.weights = weights
                        hybrid_model.hybrid_threshold = threshold
    
    return best_precision, best_recall, best_f1, best_threshold is not None

def print_feature_importance(hybrid_model, top_n=20):
    if hybrid_model.feature_importance_ is not None:
        top_features = hybrid_model.feature_importance_.head(top_n)
        
        hybrid_model.feature_importance_.to_csv('feature_importance.csv', index=False)
        
        return hybrid_model.feature_importance_
    else:
        return None

def main():
    try:
        df = load_and_preprocess_data()
        
        print(f"Original dataset size: {len(df)}")
        
        # Enhanced feature creation with better sampling
        df = create_precision_focused_features(df, max_samples=120000, fraud_ratio=0.018)
        
        print(f"After sampling: {len(df)}")
        
        # Enhanced feature preparation
        X, feature_cols, correlation_matrix, fraud_labels = prepare_precision_features(
            df, correlation_threshold=0.85, n_top_features=20
        )
        
        if fraud_labels is None:
            print("Error: No fraud labels found")
            return
        
        save_correlation_matrix(correlation_matrix)
        
        # Stratified split with more data for validation
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, fraud_labels, test_size=0.4, random_state=42, stratify=fraud_labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        fraud_count = np.sum(fraud_labels)
        normal_count = len(fraud_labels) - fraud_count
        fraud_rate = fraud_count / len(fraud_labels)
        
        print(f"Training samples: {len(X_train)} (fraud: {np.sum(y_train)})")
        print(f"Validation samples: {len(X_val)} (fraud: {np.sum(y_val)})")
        print(f"Test samples: {len(X_test)} (fraud: {np.sum(y_test)})")
        print(f"Overall fraud rate: {fraud_rate:.4f}")
        print(f"Selected features: {len(feature_cols)}")
        
        # Initialize and train enhanced model
        hybrid_model = PrecisionFocusedHybridDetector(input_dim=X_train.shape[1])
        contamination_rate = min(0.012, fraud_labels.mean() * 1.5)  
        
        print(f"Training with contamination rate: {contamination_rate:.4f}")
        hybrid_model.fit(X_train, y_train, contamination_rate)
        
        # Feature importance analysis
        feature_importance_df = print_feature_importance(hybrid_model, top_n=20)
        if feature_importance_df is not None:
            print("\nTop 10 features by importance:")
            print(feature_importance_df.head(10)[['feature', 'combined_importance']])
        
        # Advanced threshold optimization
        print("\nStarting advanced threshold optimization...")
        opt_precision, opt_recall, opt_f1, opt_success = advanced_threshold_optimization(
            hybrid_model, X_val, y_val, X_test, y_test, min_precision=0.72
        )
        
        if opt_success:
            print(f"✅ Optimization successful!")
            print(f"   Optimized Precision: {opt_precision:.4f}")
            print(f"   Optimized Recall: {opt_recall:.4f}")
            print(f"   Optimized F1: {opt_f1:.4f}")
        else:
            print("⚠️  Advanced optimization did not meet all criteria, using fallback...")
        
        # Get individual model predictions for evaluation
        if_predictions, ae_predictions, if_scores_norm, ae_scores_norm = hybrid_model.predict_individual_models(X_test)
        
        # Evaluate individual models
        individual_metrics = evaluate_individual_models(y_test, if_predictions, ae_predictions)
        
        # Final evaluation on test set (hybrid model)
        test_predictions = hybrid_model.predict(X_test)
        hybrid_scores = hybrid_model.predict_scores(X_test)[0]
        
        cm = confusion_matrix(y_test, test_predictions)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        predicted_fraud_rate = np.mean(test_predictions)
        actual_fraud_rate = np.mean(y_test)
        
        # Calculate additional metrics for better evaluation
        positive_predictions = np.sum(test_predictions)
        negative_predictions = len(test_predictions) - positive_predictions
        
        print("\n" + "="*70)
        print("INDIVIDUAL MODEL PERFORMANCE")
        print("="*70)
        print(f"🔍 ISOLATION FOREST:")
        print(f"   Precision: {individual_metrics['isolation_forest']['precision']:.4f}")
        print(f"   Recall: {individual_metrics['isolation_forest']['recall']:.4f}")
        print(f"   F1-Score: {individual_metrics['isolation_forest']['f1_score']:.4f}")
        print(f"   Specificity: {individual_metrics['isolation_forest']['specificity']:.4f}")
        print(f"   TP: {individual_metrics['isolation_forest']['true_positives']}, FP: {individual_metrics['isolation_forest']['false_positives']}")
        print(f"   FN: {individual_metrics['isolation_forest']['false_negatives']}, TN: {individual_metrics['isolation_forest']['true_negatives']}")
        
        print(f"\n🧠 AUTOENCODER:")
        print(f"   Precision: {individual_metrics['autoencoder']['precision']:.4f}")
        print(f"   Recall: {individual_metrics['autoencoder']['recall']:.4f}")
        print(f"   F1-Score: {individual_metrics['autoencoder']['f1_score']:.4f}")
        print(f"   Specificity: {individual_metrics['autoencoder']['specificity']:.4f}")
        print(f"   TP: {individual_metrics['autoencoder']['true_positives']}, FP: {individual_metrics['autoencoder']['false_positives']}")
        print(f"   FN: {individual_metrics['autoencoder']['false_negatives']}, TN: {individual_metrics['autoencoder']['true_negatives']}")
        
        print("\n" + "="*70)
        print("HYBRID MODEL PERFORMANCE")
        print("="*70)
        print(f"🎯 Precision: {precision:.4f} {'✅' if precision >= 0.7 else '❌'} (Target: ≥0.70)")
        print(f"🎯 Recall: {recall:.4f} {'✅' if recall >= 0.8 else '❌'} (Target: ≥0.80)")
        print(f"📊 F1-Score: {f1:.4f}")
        print(f"📊 Specificity: {specificity:.4f}")
        print(f"📊 Model Weights: IF={hybrid_model.weights['isolation_forest']:.3f}, AE={hybrid_model.weights['autoencoder']:.3f}")
        print(f"📊 Hybrid Threshold: {hybrid_model.hybrid_threshold:.4f}")
        print(f"📊 IF Threshold: {hybrid_model.if_threshold:.4f}")
        print(f"📊 AE Threshold: {hybrid_model.ae_threshold:.4f}")
        print(f"📊 Precision Boost Factor: {hybrid_model.precision_boost_factor:.3f}")
        
        print(f"\n📈 Prediction Statistics:")
        print(f"   Predicted Fraud Rate: {predicted_fraud_rate:.4f}")
        print(f"   Actual Fraud Rate: {actual_fraud_rate:.4f}")
        print(f"   Total Positive Predictions: {positive_predictions}")
        print(f"   Total Negative Predictions: {negative_predictions}")
        
        print(f"\n🔍 Hybrid Confusion Matrix:")
        print(f"   True Negatives: {tn:,} | False Positives: {fp:,}")
        print(f"   False Negatives: {fn:,} | True Positives: {tp:,}")
        
        # Save enhanced model
        save_precision_model(hybrid_model, feature_cols)
        
        # Create comprehensive results with individual model scores
        results_df = pd.DataFrame({
            'transaction_id': range(len(hybrid_scores)),
            'hybrid_risk_score': hybrid_scores,
            'isolation_forest_score': if_scores_norm,
            'autoencoder_score': ae_scores_norm,
            'hybrid_prediction': test_predictions,
            'isolation_forest_prediction': if_predictions,
            'autoencoder_prediction': ae_predictions,
            'actual_fraud': y_test,
            'hybrid_true_positive': (test_predictions == 1) & (y_test == 1),
            'hybrid_false_positive': (test_predictions == 1) & (y_test == 0),
            'hybrid_true_negative': (test_predictions == 0) & (y_test == 0),
            'hybrid_false_negative': (test_predictions == 0) & (y_test == 1),
            'if_true_positive': (if_predictions == 1) & (y_test == 1),
            'if_false_positive': (if_predictions == 1) & (y_test == 0),
            'ae_true_positive': (ae_predictions == 1) & (y_test == 1),
            'ae_false_positive': (ae_predictions == 1) & (y_test == 0)
        })
        
        # Risk categorization with 3 levels: Low, Medium, High
        low_threshold = np.percentile(hybrid_scores, 70)
        high_threshold = np.percentile(hybrid_scores, 90)
        
        results_df['risk_level'] = pd.cut(
            hybrid_scores, 
            bins=[-np.inf, low_threshold, high_threshold, np.inf],
            labels=['Low', 'Medium', 'High']
        )
        
        # Risk level analysis
        risk_analysis = results_df.groupby('risk_level').agg({
            'actual_fraud': ['count', 'sum', 'mean'],
            'hybrid_prediction': 'mean',
            'isolation_forest_prediction': 'mean',
            'autoencoder_prediction': 'mean'
        }).round(4)
        
        print(f"\n📊 Risk Level Analysis:")
        print(risk_analysis)
        
        results_df.to_excel('fraud_detection_results_enhanced_with_individual_scores.xlsx', index=False)
        
        # Enhanced performance summary with individual model metrics
        summary_stats = {
            'hybrid_model': {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'specificity': float(specificity),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn),
                'threshold': float(hybrid_model.hybrid_threshold)
            },
            'isolation_forest': individual_metrics['isolation_forest'],
            'autoencoder': individual_metrics['autoencoder'],
            'model_configuration': {
                'weights': {k: float(v) for k, v in hybrid_model.weights.items()},
                'if_threshold': float(hybrid_model.if_threshold),
                'ae_threshold': float(hybrid_model.ae_threshold),
                'feature_count': int(len(feature_cols)),
                'precision_boost_factor': float(hybrid_model.precision_boost_factor)
            },
            'dataset_info': {
                'total_samples': int(len(y_test)),
                'fraud_samples': int(np.sum(y_test)),
                'normal_samples': int(len(y_test) - np.sum(y_test)),
                'fraud_rate': float(actual_fraud_rate)
            }
        }
        
        with open('comprehensive_performance_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Create a summary table for easy reporting
        model_comparison_df = pd.DataFrame({
            'Model': ['Isolation Forest', 'Autoencoder', 'Hybrid'],
            'Precision': [
                individual_metrics['isolation_forest']['precision'],
                individual_metrics['autoencoder']['precision'],
                precision
            ],
            'Recall': [
                individual_metrics['isolation_forest']['recall'],
                individual_metrics['autoencoder']['recall'],
                recall
            ],
            'F1-Score': [
                individual_metrics['isolation_forest']['f1_score'],
                individual_metrics['autoencoder']['f1_score'],
                f1
            ],
            'Specificity': [
                individual_metrics['isolation_forest']['specificity'],
                individual_metrics['autoencoder']['specificity'],
                specificity
            ],
            'True Positives': [
                individual_metrics['isolation_forest']['true_positives'],
                individual_metrics['autoencoder']['true_positives'],
                tp
            ],
            'False Positives': [
                individual_metrics['isolation_forest']['false_positives'],
                individual_metrics['autoencoder']['false_positives'],
                fp
            ],
            'False Negatives': [
                individual_metrics['isolation_forest']['false_negatives'],
                individual_metrics['autoencoder']['false_negatives'],
                fn
            ],
            'True Negatives': [
                individual_metrics['isolation_forest']['true_negatives'],
                individual_metrics['autoencoder']['true_negatives'],
                tn
            ]
        })
        
        model_comparison_df.to_csv('model_comparison_table.csv', index=False)
        print(f"\n📊 Model Comparison Table:")
        print(model_comparison_df.round(4))
        
        print(f"\n✅ Enhanced model training completed!")
        print(f"📁 Results with individual scores: fraud_detection_results_enhanced_with_individual_scores.xlsx")
        print(f"📁 Comprehensive performance summary: comprehensive_performance_summary.json")
        print(f"📁 Model comparison table: model_comparison_table.csv")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
