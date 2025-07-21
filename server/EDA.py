import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("Set2")

def clean_data(filepath):
    """Simple data cleaning function"""
    print("Loading and cleaning data...")
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded data: {df.shape}")
        
        # 1. Remove quotes from all columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip("'\"")
        
        # 2. Fix numeric columns
        numeric_cols = ['step', 'age', 'amount', 'fraud']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 3. Remove rows with missing fraud values
        if 'fraud' in df.columns:
            df = df.dropna(subset=['fraud'])
        
        # 4. Remove duplicates
        df = df.drop_duplicates()
        
        # 5. Fill missing values
        if 'age' in df.columns:
            df['age'].fillna(df['age'].median(), inplace=True)
        if 'amount' in df.columns:
            df['amount'].fillna(df['amount'].median(), inplace=True)
        
        print(f"✓ Cleaned data: {df.shape}")
        print(f"✓ Missing values: {df.isnull().sum().sum()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def simple_eda(df):
    """Simple EDA analysis"""
    
    # Basic info
    print(f"Dataset size: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Fraud analysis
    if 'fraud' in df.columns:
        fraud_count = df['fraud'].sum()
        total_count = len(df)
        fraud_rate = fraud_count / total_count
        
        print(f"\nFRAUD SUMMARY:")
        print(f"Normal transactions: {total_count - fraud_count:,}")
        print(f"Fraudulent transactions: {fraud_count:,}")
        print(f"Fraud rate: {fraud_rate:.1%}")
    
    # Amount analysis
    if 'amount' in df.columns:
        print(f"\nAMOUNT SUMMARY:")
        print(f"Average amount: ${df['amount'].mean():.2f}")
        print(f"Median amount: ${df['amount'].median():.2f}")
        print(f"Max amount: ${df['amount'].max():.2f}")
        
        if 'fraud' in df.columns:
            normal_avg = df[df['fraud']==0]['amount'].mean()
            fraud_avg = df[df['fraud']==1]['amount'].mean()
            print(f"Normal avg: ${normal_avg:.2f}")
            print(f"Fraud avg: ${fraud_avg:.2f}")
    
    # Category analysis
    if 'category' in df.columns:
        print(f"\nCATEGORY SUMMARY:")
        print(df['category'].value_counts().head())
        
        if 'fraud' in df.columns:
            print(f"\nFRAUD BY CATEGORY:")
            fraud_by_cat = df.groupby('category')['fraud'].mean().sort_values(ascending=False)
            for cat, rate in fraud_by_cat.head().items():
                print(f"{cat}: {rate:.1%}")

def create_simple_plots(df):
    """Create simple visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Fraud pie chart
    if 'fraud' in df.columns:
        fraud_counts = df['fraud'].value_counts()
        axes[0,0].pie(fraud_counts.values, labels=['Normal', 'Fraud'], 
                     autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        axes[0,0].set_title('Fraud vs Normal')
    
    # 2. Amount histogram
    if 'amount' in df.columns:
        axes[0,1].hist(df['amount'], bins=30, alpha=0.7, color='skyblue')
        axes[0,1].set_title('Amount Distribution')
        axes[0,1].set_xlabel('Amount')
    
    # 3. Category bar chart
    if 'category' in df.columns:
        cat_counts = df['category'].value_counts()
        axes[1,0].bar(range(len(cat_counts)), cat_counts.values, color='lightblue')
        axes[1,0].set_title('Transactions by Category')
        axes[1,0].set_xticks(range(len(cat_counts)))
        axes[1,0].set_xticklabels(cat_counts.index, rotation=45)
    
    # 4. Fraud rate by category
    if 'category' in df.columns and 'fraud' in df.columns:
        fraud_rate = df.groupby('category')['fraud'].mean()
        axes[1,1].bar(range(len(fraud_rate)), fraud_rate.values, color='lightcoral')
        axes[1,1].set_title('Fraud Rate by Category')
        axes[1,1].set_xticks(range(len(fraud_rate)))
        axes[1,1].set_xticklabels(fraud_rate.index, rotation=45)
        axes[1,1].set_ylabel('Fraud Rate')
    
    plt.tight_layout()
    plt.show()

# Main execution
df = clean_data("bs140513_032310.csv")

if df is not None:
    # Run simple analysis
    simple_eda(df)
    
    # Create visualizations
    create_simple_plots(df)
    
    # Save cleaned data
    clean_filename = "cleaned_fraud_data.csv"
    try:
        df.to_csv(clean_filename, index=False)
        print(f"\n✓ Cleaned data saved as: {clean_filename}")
        print(f"  - Original file: bs140513_032310.csv")
        print(f"  - Clean file: {clean_filename}")
        print(f"  - Rows saved: {len(df):,}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
    
    
    
else:
    print("❌ Could not load data")