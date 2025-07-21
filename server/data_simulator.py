import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataSimulator:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.current_index = 0
        self.total_rows = len(self.df)
        
        # Convert step to datetime for better display
        base_date = datetime(2011, 1, 1)
        self.df['datetime'] = self.df['step'].apply(lambda x: base_date + timedelta(minutes=x))
        
        print(f"Data simulator initialized with {self.total_rows} transactions")
    
    def get_next_batch(self, n=5):
        """Get the next n transactions from the dataset"""
        if self.current_index >= self.total_rows:
            # Reset to beginning if we've reached the end
            self.current_index = 0
        
        # Get the next n rows
        end_index = min(self.current_index + n, self.total_rows)
        batch = self.df.iloc[self.current_index:end_index].copy()
        
        # Update index
        self.current_index = end_index
        
        # Add formatted datetime string for display
        batch['time'] = batch['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return batch
    
    def reset(self):
        """Reset the simulator to the beginning"""
        self.current_index = 0
    
    def get_current_position(self):
        """Get current position in the dataset"""
        return self.current_index, self.total_rows