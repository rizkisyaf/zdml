#!/usr/bin/env python
"""
Bitcoin Perpetual Futures Prediction Example

This script demonstrates how to use the Bitcoin perpetual futures prediction system.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function.
    """
    # Check if data file exists
    data_path = 'futures_orderbook_data.csv'
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please download the data file and place it in the current directory.")
        sys.exit(1)
    
    # Step 1: Preprocess data
    logger.info("Step 1: Preprocessing data")
    
    from src.data.preprocessing import load_orderbook_data, preprocess_data, prepare_training_data, save_processed_data
    
    # Load data
    df = load_orderbook_data(data_path)
    logger.info(f"Loaded {len(df)} rows of order book data with {len(df.columns)} columns")
    logger.info(f"Using all 10 levels of order book data for enhanced feature extraction")
    
    # Display first few rows
    print("\nFirst few rows of order book data (showing only first level for brevity):")
    columns_to_show = ['timestamp', 'bid_price1', 'bid_qty1', 'ask_price1', 'ask_qty1']
    print(df[columns_to_show].head())
    
    # Display order book depth
    print("\nOrder book depth:")
    print(f"Number of bid levels: 10")
    print(f"Number of ask levels: 10")
    
    # Preprocess data
    window_size = 20
    features_df = preprocess_data(df, window_size)
    logger.info(f"Preprocessed data has {len(features_df)} rows and {features_df.shape[1]} columns")
    
    # Display first few rows of preprocessed data
    print("\nFirst few rows of preprocessed data (showing only key features):")
    key_features = ['mid_price', 'weighted_mid_price', 'spread', 'imbalance', 'liquidity_imbalance', 'price_range']
    print(features_df[key_features].head())
    
    # Display all extracted features
    print("\nAll extracted features:")
    for i, feature in enumerate(features_df.columns):
        if feature != 'timestamp':
            print(f"{i}: {feature}")
    
    # Prepare training data
    sequence_length = 100
    prediction_horizon = 1
    test_size = 0.2
    val_size = 0.1
    
    data = prepare_training_data(
        features_df,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        test_size=test_size,
        val_size=val_size
    )
    
    # Print data shapes
    print("\nData shapes:")
    print(f"X_train: {data['X_train'].shape}")
    print(f"y_train: {data['y_train'].shape}")
    print(f"X_val: {data['X_val'].shape}")
    print(f"y_val: {data['y_val'].shape}")
    print(f"X_test: {data['X_test'].shape}")
    print(f"y_test: {data['y_test'].shape}")
    
    # Save processed data
    processed_data_path = 'processed_data.pkl'
    save_processed_data(data, processed_data_path)
    logger.info(f"Processed data saved to {processed_data_path}")
    
    # Step 2: Train model (with reduced epochs for demonstration)
    logger.info("\nStep 2: Training model (with reduced epochs for demonstration)")
    
    from src.train import train_model
    
    # Create model directory
    model_dir = 'saved_models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Train model with reduced epochs for demonstration
    results = train_model(
        data_path=processed_data_path,
        model_save_dir=model_dir,
        batch_size=512,
        epochs=5,  # Reduced for demonstration
        learning_rate=1e-4,
        weight_decay=1e-5,
        patience=3,
        use_gpu=torch.cuda.is_available(),
        loss_type='hybrid',
        output_size=3,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    
    # Print model architecture
    print("\nModel Architecture:")
    print(f"Input size: {data['X_train'].shape[2]} features")
    print(f"Hidden size: {512 if data['X_train'].shape[2] > 100 else 256}")
    print(f"Projection size: 128")
    print(f"Number of LSTM layers: 3")
    print(f"Number of attention heads: 8")
    print(f"Output size: 3 (price, direction, volatility)")
    
    # Print test metrics
    print("\nTest metrics:")
    for k, v in results['test_metrics'].items():
        print(f"{k}: {v:.4f}")
    
    # Step 3: Make predictions
    logger.info("\nStep 3: Making predictions")
    
    from src.predict import run_prediction
    
    # Create prediction directory
    prediction_dir = 'prediction_results'
    os.makedirs(prediction_dir, exist_ok=True)
    
    # Run prediction
    prediction_results = run_prediction(
        model_path=results['model_path'],
        data_path=data_path,
        output_dir=prediction_dir,
        window_size=window_size,
        use_gpu=torch.cuda.is_available(),
        batch_size=512,
        visualize=True
    )
    
    # Print prediction metrics
    print("\nPrediction metrics:")
    for k, v in prediction_results['metrics'].items():
        print(f"{k}: {v:.4f}")
    
    # Print trading metrics
    print("\nTrading metrics:")
    for k, v in prediction_results['trading_metrics'].items():
        print(f"{k}: {v:.4f}")
    
    logger.info("\nExample completed successfully")
    logger.info(f"Model saved to {results['model_path']}")
    logger.info(f"Predictions saved to {prediction_dir}")
    
    # Display instructions for running the complete pipeline
    print("\n" + "="*80)
    print("To run the complete pipeline with more epochs, use the following command:")
    print("./run_pipeline.py --data futures_orderbook_data.csv --output_dir pipeline_results --epochs 100")
    print("="*80)
    print("\nThis implementation utilizes all 10 levels of order book data to extract rich features,")
    print("providing the model with comprehensive market microstructure information.")


if __name__ == '__main__':
    main() 