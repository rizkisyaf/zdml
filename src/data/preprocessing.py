import os
import numpy as np
import pandas as pd
import argparse
import pickle
from typing import Tuple, Dict, List, Optional, Union
from sklearn.model_selection import train_test_split
import sys
import logging

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.features import (
    extract_features,
    create_sequences,
    normalize_features
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_orderbook_data(file_path: str) -> pd.DataFrame:
    """
    Load order book data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with order book data
    """
    logger.info(f"Loading order book data from {file_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Define all expected columns (10 levels of bids and asks)
        expected_columns = ['timestamp']
        
        # Add bid columns (price and quantity for 10 levels)
        for i in range(1, 11):
            expected_columns.extend([f'bid_price{i}', f'bid_qty{i}'])
        
        # Add ask columns (price and quantity for 10 levels)
        for i in range(1, 11):
            expected_columns.extend([f'ask_price{i}', f'ask_qty{i}'])
        
        # Check if the data has all expected columns
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing expected columns: {missing_columns}")
            raise ValueError(f"Missing expected columns: {missing_columns}")
        
        logger.info(f"Loaded {len(df)} rows of order book data with {len(df.columns)} columns")
        logger.info(f"Order book depth: 10 levels of bids and asks")
        return df
    
    except Exception as e:
        logger.error(f"Error loading order book data: {str(e)}")
        raise


def preprocess_data(df: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
    """
    Preprocess order book data.
    
    Args:
        df: DataFrame with order book data
        window_size: Window size for rolling calculations
        
    Returns:
        DataFrame with preprocessed data
    """
    logger.info("Preprocessing order book data")
    
    try:
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values in the data")
            # Fill missing values
            df = df.fillna(method='ffill')
            remaining_missing = df.isnull().sum().sum()
            if remaining_missing > 0:
                df = df.fillna(method='bfill')
                
        # Check for duplicate timestamps
        duplicate_timestamps = df['timestamp'].duplicated().sum()
        if duplicate_timestamps > 0:
            logger.warning(f"Found {duplicate_timestamps} duplicate timestamps")
            # Keep the last record for each timestamp
            df = df.drop_duplicates(subset='timestamp', keep='last')
        
        # Extract features
        features_df = extract_features(df, window_size)
        
        logger.info(f"Preprocessed data has {len(features_df)} rows and {features_df.shape[1]} columns")
        return features_df
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise


def prepare_training_data(
    features_df: pd.DataFrame,
    sequence_length: int = 100,
    prediction_horizon: int = 1,
    target_column: str = 'mid_price',
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Prepare training data for LSTM model.
    
    Args:
        features_df: DataFrame with extracted features
        sequence_length: Length of input sequences
        prediction_horizon: How many steps ahead to predict
        target_column: Column to predict
        test_size: Fraction of data to use for testing
        val_size: Fraction of data to use for validation
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with training, validation, and test data
    """
    logger.info(f"Preparing training data with sequence length {sequence_length}")
    
    try:
        # Create sequences
        X, y = create_sequences(
            features_df,
            sequence_length=sequence_length,
            target_column=target_column,
            prediction_horizon=prediction_horizon
        )
        
        # Create multi-target y if needed
        if target_column == 'mid_price':
            # Add direction prediction target
            direction = np.zeros((len(y), 1))
            for i in range(1, len(y)):
                direction[i] = 1.0 if y[i] > y[i-1] else 0.0
            
            # Add volatility prediction target
            volatility = np.zeros((len(y), 1))
            for i in range(sequence_length, len(features_df) - prediction_horizon):
                idx = i - sequence_length
                if idx >= 0 and idx < len(volatility):
                    volatility[idx] = features_df['volatility'].iloc[i]
            
            # Combine targets
            y_price = y.reshape(-1, 1)
            y = np.hstack((y_price, direction, volatility))
        
        # Split data into training, validation, and test sets
        # Use time-based split to preserve temporal order
        train_size = 1 - test_size - val_size
        train_idx = int(len(X) * train_size)
        val_idx = int(len(X) * (train_size + val_size))
        
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test, y_test = X[val_idx:], y[val_idx:]
        
        # Normalize features
        X_train_norm, X_val_norm, X_test_norm = normalize_features(X_train, X_val, X_test)
        
        logger.info(f"Training data: {X_train_norm.shape}, {y_train.shape}")
        logger.info(f"Validation data: {X_val_norm.shape}, {y_val.shape}")
        logger.info(f"Test data: {X_test_norm.shape}, {y_test.shape}")
        
        return {
            'X_train': X_train_norm,
            'y_train': y_train,
            'X_val': X_val_norm,
            'y_val': y_val,
            'X_test': X_test_norm,
            'y_test': y_test,
            'feature_names': features_df.columns.drop('timestamp').tolist()
        }
    
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        raise


def save_processed_data(data: Dict[str, np.ndarray], output_path: str) -> None:
    """
    Save processed data to a file.
    
    Args:
        data: Dictionary with processed data
        output_path: Path to save the data
    """
    logger.info(f"Saving processed data to {output_path}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save data
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info("Data saved successfully")
    
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        raise


def load_processed_data(input_path: str) -> Dict[str, np.ndarray]:
    """
    Load processed data from a file.
    
    Args:
        input_path: Path to the data file
        
    Returns:
        Dictionary with processed data
    """
    logger.info(f"Loading processed data from {input_path}")
    
    try:
        # Load data
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info("Data loaded successfully")
        return data
    
    except Exception as e:
        logger.error(f"Error loading processed data: {str(e)}")
        raise


def main():
    """
    Main function for data preprocessing.
    """
    parser = argparse.ArgumentParser(description='Preprocess order book data')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to output pickle file')
    parser.add_argument('--sequence_length', type=int, default=100, help='Sequence length for LSTM')
    parser.add_argument('--prediction_horizon', type=int, default=1, help='Prediction horizon')
    parser.add_argument('--window_size', type=int, default=20, help='Window size for rolling calculations')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size')
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = load_orderbook_data(args.input)
        
        # Preprocess data
        features_df = preprocess_data(df, args.window_size)
        
        # Prepare training data
        data = prepare_training_data(
            features_df,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
            test_size=args.test_size,
            val_size=args.val_size
        )
        
        # Save processed data
        save_processed_data(data, args.output)
        
        logger.info("Data preprocessing completed successfully")
    
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
