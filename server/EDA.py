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


# Main execution
df = clean_data("./server/bs140513_032310.csv")


if df is not None:
    # Run simple analysis
    simple_eda(df)
    
    
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
