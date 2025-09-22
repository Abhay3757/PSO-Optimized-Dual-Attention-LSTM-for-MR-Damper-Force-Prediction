"""
Data Preprocessing Module for MR Damper Force Prediction

This module provides functions for loading, cleaning, normalizing, and preparing
time-series data for the Dual-Attention LSTM model.

Features:
- Data loading and cleaning
- Feature normalization
- Time-series window creation
- Time-aware data splitting
- Velocity calculation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class MRDamperDataset(Dataset):
    """Custom PyTorch Dataset for MR Damper time-series data"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Initialize the dataset
        
        Args:
            sequences: Input sequences of shape (num_samples, lookback, num_features)
            targets: Target values of shape (num_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class DataPreprocessor:
    """Main data preprocessing class for MR Damper data"""
    
    def __init__(self, 
                 normalize_method: str = 'standard',
                 lookback_window: int = 20):
        """
        Initialize the preprocessor
        
        Args:
            normalize_method: 'standard' or 'minmax' normalization
            lookback_window: Number of past timesteps to include in each sample
        """
        self.normalize_method = normalize_method
        self.lookback_window = lookback_window
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_columns = ['A', 'D', 'Y', 'V']  # Input features
        self.target_column = 'F'  # Output target
        
    def load_and_clean_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load and clean the MR Damper dataset
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Cleaned DataFrame
        """
        print(f"Loading data from {csv_path}")
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Remove unnecessary columns
        columns_to_keep = ['t', 'A', 'D', 'Y', 'V', 'F']
        df = df[columns_to_keep].copy()
        
        # Handle missing values
        print(f"Initial shape: {df.shape}")
        print(f"Missing values before cleaning:\n{df.isnull().sum()}")
        
        # Remove rows with missing target values
        df = df.dropna(subset=[self.target_column])
        
        # Forward fill missing values in features (if any)
        df[self.feature_columns] = df[self.feature_columns].fillna(method='ffill')
        
        # Remove any remaining NaN rows
        df = df.dropna()
        
        print(f"Final shape after cleaning: {df.shape}")
        print(f"Missing values after cleaning:\n{df.isnull().sum()}")
        
        # Sort by time to ensure temporal order
        df = df.sort_values('t').reset_index(drop=True)
        
        return df
    
    def calculate_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate velocity from displacement if not already present
        
        Args:
            df: DataFrame with displacement column 'D'
            
        Returns:
            DataFrame with calculated velocity
        """
        df = df.copy()
        
        # Calculate velocity as derivative of displacement
        if 'V' not in df.columns or df['V'].isnull().any():
            print("Calculating velocity from displacement...")
            dt = df['t'].diff().mean()  # Average time step
            df['V'] = df['D'].diff() / dt
            df['V'].fillna(0, inplace=True)  # Fill first value
        
        return df
    
    def normalize_features(self, 
                          train_df: pd.DataFrame, 
                          val_df: pd.DataFrame, 
                          test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Normalize features and targets using training data statistics
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame  
            test_df: Test DataFrame
            
        Returns:
            Tuple of normalized DataFrames (train, val, test)
        """
        print(f"Normalizing features using {self.normalize_method} normalization...")
        
        # Initialize scalers
        if self.normalize_method == 'standard':
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        else:
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        
        # Fit scalers on training data only
        self.feature_scaler.fit(train_df[self.feature_columns])
        self.target_scaler.fit(train_df[[self.target_column]])
        
        # Transform all datasets
        train_normalized = train_df.copy()
        val_normalized = val_df.copy()
        test_normalized = test_df.copy()
        
        # Transform features
        train_normalized[self.feature_columns] = self.feature_scaler.transform(train_df[self.feature_columns])
        val_normalized[self.feature_columns] = self.feature_scaler.transform(val_df[self.feature_columns])
        test_normalized[self.feature_columns] = self.feature_scaler.transform(test_df[self.feature_columns])
        
        # Transform targets
        train_normalized[self.target_column] = self.target_scaler.transform(train_df[[self.target_column]]).flatten()
        val_normalized[self.target_column] = self.target_scaler.transform(val_df[[self.target_column]]).flatten()
        test_normalized[self.target_column] = self.target_scaler.transform(test_df[[self.target_column]]).flatten()
        
        return train_normalized, val_normalized, test_normalized
    
    def create_time_series_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time-series windows for LSTM input
        
        Args:
            df: Normalized DataFrame
            
        Returns:
            Tuple of (sequences, targets) arrays
        """
        sequences = []
        targets = []
        
        # Create sliding windows
        for i in range(self.lookback_window, len(df)):
            # Input sequence: past lookback_window timesteps
            seq = df[self.feature_columns].iloc[i-self.lookback_window:i].values
            # Target: current timestep force
            target = df[self.target_column].iloc[i]
            
            sequences.append(seq)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        print(f"Created {len(sequences)} time-series windows")
        print(f"Sequence shape: {sequences.shape}")
        print(f"Target shape: {targets.shape}")
        
        return sequences, targets
    
    def time_aware_split(self, 
                        df: pd.DataFrame, 
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform time-aware data splitting (not random)
        
        Args:
            df: DataFrame to split
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        n_samples = len(df)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def inverse_transform_target(self, normalized_target: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized target values back to original scale
        
        Args:
            normalized_target: Normalized target values
            
        Returns:
            Original scale target values
        """
        if self.target_scaler is None:
            raise ValueError("Target scaler has not been fitted yet")
        
        # Ensure proper shape for inverse transform
        if normalized_target.ndim == 1:
            normalized_target = normalized_target.reshape(-1, 1)
        
        return self.target_scaler.inverse_transform(normalized_target).flatten()
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """
        Get feature statistics for analysis
        
        Returns:
            Dictionary with feature statistics
        """
        if self.feature_scaler is None:
            return {}
        
        stats = {}
        if hasattr(self.feature_scaler, 'mean_'):
            stats['feature_means'] = dict(zip(self.feature_columns, self.feature_scaler.mean_))
            stats['feature_stds'] = dict(zip(self.feature_columns, self.feature_scaler.scale_))
        elif hasattr(self.feature_scaler, 'data_min_'):
            stats['feature_mins'] = dict(zip(self.feature_columns, self.feature_scaler.data_min_))
            stats['feature_maxs'] = dict(zip(self.feature_columns, self.feature_scaler.data_max_))
        
        return stats
    
    def preprocess_full_pipeline(self, 
                                csv_path: str,
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.15,
                                test_ratio: float = 0.15,
                                batch_size: int = 64,
                                shuffle: bool = True) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline
        
        Args:
            csv_path: Path to CSV file
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
            batch_size: Batch size for DataLoaders
            shuffle: Whether to shuffle training data
            
        Returns:
            Dictionary containing all processed data and metadata
        """
        print("=" * 50)
        print("Starting MR Damper Data Preprocessing Pipeline")
        print("=" * 50)
        
        # Step 1: Load and clean data
        df = self.load_and_clean_data(csv_path)
        
        # Step 2: Calculate velocity if needed
        df = self.calculate_velocity(df)
        
        # Step 3: Time-aware split
        train_df, val_df, test_df = self.time_aware_split(df, train_ratio, val_ratio, test_ratio)
        
        # Step 4: Normalize features
        train_norm, val_norm, test_norm = self.normalize_features(train_df, val_df, test_df)
        
        # Step 5: Create time-series windows
        train_seq, train_targets = self.create_time_series_windows(train_norm)
        val_seq, val_targets = self.create_time_series_windows(val_norm)
        test_seq, test_targets = self.create_time_series_windows(test_norm)
        
        # Step 6: Create PyTorch datasets and dataloaders
        train_dataset = MRDamperDataset(train_seq, train_targets)
        val_dataset = MRDamperDataset(val_seq, val_targets)
        test_dataset = MRDamperDataset(test_seq, test_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Prepare results
        results = {
            'raw_data': {
                'train': train_df,
                'val': val_df,
                'test': test_df
            },
            'normalized_data': {
                'train': train_norm,
                'val': val_norm,
                'test': test_norm
            },
            'sequences': {
                'train': (train_seq, train_targets),
                'val': (val_seq, val_targets),
                'test': (test_seq, test_targets)
            },
            'datasets': {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
            },
            'dataloaders': {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            },
            'metadata': {
                'num_features': len(self.feature_columns),
                'feature_names': self.feature_columns,
                'target_name': self.target_column,
                'lookback_window': self.lookback_window,
                'normalize_method': self.normalize_method,
                'feature_stats': self.get_feature_stats()
            }
        }
        
        print("=" * 50)
        print("Preprocessing Pipeline Complete!")
        print(f"Features: {self.feature_columns}")
        print(f"Target: {self.target_column}")
        print(f"Lookback window: {self.lookback_window}")
        print(f"Normalization: {self.normalize_method}")
        print("=" * 50)
        
        return results


def main():
    """Test the preprocessing pipeline"""
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        normalize_method='standard',
        lookback_window=20
    )
    
    # Run preprocessing pipeline
    data_path = "Dataset/ranran_50_converted.csv"
    results = preprocessor.preprocess_full_pipeline(
        csv_path=data_path,
        batch_size=64,
        shuffle=True
    )
    
    # Display results summary
    print("\nDataLoader Shapes:")
    for phase in ['train', 'val', 'test']:
        loader = results['dataloaders'][phase]
        x_batch, y_batch = next(iter(loader))
        print(f"{phase.capitalize()}: X={x_batch.shape}, Y={y_batch.shape}")


if __name__ == "__main__":
    main()