import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import json
from typing import Dict, List, Tuple, Optional, Union, Any
import time

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import load_orderbook_data, preprocess_data
from src.models.lstm_model import OrderBookLSTM
from src.utils.features import create_sequences, normalize_features
from src.utils.metrics import calculate_model_metrics, calculate_trading_metrics
from src.visualization.visualize import (
    plot_price_predictions, 
    plot_direction_accuracy, 
    plot_trading_performance,
    plot_order_book_snapshot
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device) -> Tuple[OrderBookLSTM, Dict[str, Any]]:
    """
    Load a trained model.
    
    Args:
        model_path: Path to the model file
        device: Device to load the model on
        
    Returns:
        Tuple of (model, model_info)
    """
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model configuration
        model_config = checkpoint['model_config']
        
        # Create model
        model = OrderBookLSTM(**model_config).to(device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set model to evaluation mode
        model.eval()
        
        # Extract model info
        model_info = {
            'feature_names': checkpoint['feature_names'],
            'sequence_length': checkpoint['sequence_length'],
            'prediction_horizon': checkpoint['prediction_horizon'],
            'output_size': checkpoint['output_size'],
            'loss_type': checkpoint['loss_type'],
            'test_metrics': checkpoint['test_metrics'],
            'training_time': checkpoint['training_time']
        }
        
        logger.info(f"Model loaded successfully")
        return model, model_info
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def predict_from_data(
    model: OrderBookLSTM,
    data: pd.DataFrame,
    feature_names: List[str],
    sequence_length: int,
    prediction_horizon: int,
    device: torch.device,
    batch_size: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions from preprocessed data.
    
    Args:
        model: Trained model
        data: Preprocessed data
        feature_names: Names of the features
        sequence_length: Length of input sequences
        prediction_horizon: How many steps ahead to predict
        device: Device to run the model on
        batch_size: Batch size for prediction
        
    Returns:
        Tuple of (predictions, targets)
    """
    logger.info(f"Preparing data for prediction")
    
    try:
        # Create sequences
        X, y = create_sequences(
            data,
            sequence_length=sequence_length,
            target_column='mid_price',
            prediction_horizon=prediction_horizon
        )
        
        # Create multi-target y if needed
        if model.output_size > 1:
            # Add direction prediction target
            direction = np.zeros((len(y), 1))
            for i in range(1, len(y)):
                direction[i] = 1.0 if y[i] > y[i-1] else 0.0
            
            # Add volatility prediction target
            volatility = np.zeros((len(y), 1))
            for i in range(sequence_length, len(data) - prediction_horizon):
                idx = i - sequence_length
                if idx >= 0 and idx < len(volatility):
                    volatility[idx] = data['volatility'].iloc[i]
            
            # Combine targets
            y_price = y.reshape(-1, 1)
            y = np.hstack((y_price, direction, volatility))
        
        # Normalize features
        X_norm, = normalize_features(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_norm)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Make predictions
        logger.info(f"Making predictions")
        predictions = []
        
        with torch.no_grad():
            for batch_X, in dataloader:
                # Move data to device
                batch_X = batch_X.to(device)
                
                # Forward pass
                outputs = model(batch_X)
                
                # Add to predictions
                predictions.append(outputs.cpu().numpy())
        
        # Concatenate predictions
        predictions = np.vstack(predictions)
        
        logger.info(f"Predictions shape: {predictions.shape}, Targets shape: {y.shape}")
        return predictions, y
    
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise


def run_prediction(
    model_path: str,
    data_path: str,
    output_dir: str,
    window_size: int = 20,
    use_gpu: bool = True,
    batch_size: int = 512,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    Run prediction on new data.
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the data file
        output_dir: Directory to save the results
        window_size: Window size for feature extraction
        use_gpu: Whether to use GPU
        batch_size: Batch size for prediction
        visualize: Whether to generate visualizations
        
    Returns:
        Dictionary with prediction results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model, model_info = load_model(model_path, device)
    
    # Extract model parameters
    feature_names = model_info['feature_names']
    sequence_length = model_info['sequence_length']
    prediction_horizon = model_info['prediction_horizon']
    
    # Load and preprocess data
    logger.info(f"Loading data from {data_path}")
    df = load_orderbook_data(data_path)
    
    # Preprocess data
    features_df = preprocess_data(df, window_size)
    
    # Make predictions
    predictions, targets = predict_from_data(
        model=model,
        data=features_df,
        feature_names=feature_names,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        device=device,
        batch_size=batch_size
    )
    
    # Calculate metrics
    metrics = calculate_model_metrics(targets, predictions)
    logger.info(f"Prediction metrics: {metrics}")
    
    # Calculate trading metrics
    mid_prices = features_df['mid_price'].values[sequence_length + prediction_horizon - 1:]
    if len(mid_prices) > len(predictions):
        mid_prices = mid_prices[:len(predictions)]
    elif len(mid_prices) < len(predictions):
        predictions = predictions[:len(mid_prices)]
        targets = targets[:len(mid_prices)]
    
    trading_metrics = calculate_trading_metrics(targets, predictions, mid_prices)
    logger.info(f"Trading metrics: {trading_metrics}")
    
    # Save predictions
    predictions_df = pd.DataFrame(predictions, columns=[f'pred_{i}' for i in range(predictions.shape[1])])
    targets_df = pd.DataFrame(targets, columns=[f'target_{i}' for i in range(targets.shape[1])])
    
    # Add timestamps if available
    if 'timestamp' in features_df.columns:
        timestamps = features_df['timestamp'].values[sequence_length + prediction_horizon - 1:]
        if len(timestamps) > len(predictions):
            timestamps = timestamps[:len(predictions)]
        predictions_df['timestamp'] = timestamps
        targets_df['timestamp'] = timestamps
    
    # Save to CSV
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")
    
    targets_path = os.path.join(output_dir, 'targets.csv')
    targets_df.to_csv(targets_path, index=False)
    logger.info(f"Targets saved to {targets_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'model_metrics': {k: float(v) for k, v in metrics.items()},
            'trading_metrics': {k: float(v) for k, v in trading_metrics.items()}
        }, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Generate visualizations
    if visualize:
        logger.info(f"Generating visualizations")
        
        # Plot price predictions
        plot_price_predictions(
            targets, predictions, 
            title='Price Predictions',
            save_path=os.path.join(output_dir, 'price_predictions.png')
        )
        
        # Plot direction accuracy
        plot_direction_accuracy(
            targets, predictions, 
            title='Direction Prediction Accuracy',
            save_path=os.path.join(output_dir, 'direction_accuracy.png')
        )
        
        # Plot trading performance
        # Extract positions and returns
        if predictions.shape[1] > 1:
            # Use direction predictions
            positions = (predictions[:, 1] > 0.5).astype(int) * 2 - 1  # Convert to -1/1
        else:
            # Calculate direction from consecutive predictions
            positions = np.zeros(len(predictions))
            for i in range(1, len(predictions)):
                positions[i] = 1 if predictions[i] > predictions[i-1] else -1
        
        # Calculate returns
        returns = np.zeros(len(positions))
        for i in range(1, len(positions)):
            price_return = (mid_prices[i] - mid_prices[i-1]) / mid_prices[i-1]
            returns[i] = positions[i-1] * price_return
        
        plot_trading_performance(
            returns, positions, mid_prices,
            title='Trading Performance',
            save_path=os.path.join(output_dir, 'trading_performance.png')
        )
        
        # Plot order book snapshot
        plot_order_book_snapshot(
            df, 0,  # First snapshot
            title='Order Book Snapshot (First)',
            save_path=os.path.join(output_dir, 'order_book_first.png')
        )
        
        plot_order_book_snapshot(
            df, len(df) // 2,  # Middle snapshot
            title='Order Book Snapshot (Middle)',
            save_path=os.path.join(output_dir, 'order_book_middle.png')
        )
        
        plot_order_book_snapshot(
            df, len(df) - 1,  # Last snapshot
            title='Order Book Snapshot (Last)',
            save_path=os.path.join(output_dir, 'order_book_last.png')
        )
    
    # Return results
    return {
        'predictions': predictions,
        'targets': targets,
        'metrics': metrics,
        'trading_metrics': trading_metrics
    }


def main():
    """
    Main function for prediction.
    """
    parser = argparse.ArgumentParser(description='Make predictions with trained LSTM model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--output_dir', type=str, default='prediction_results', help='Directory to save results')
    parser.add_argument('--window_size', type=int, default=20, help='Window size for feature extraction')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for prediction')
    parser.add_argument('--no_visualize', action='store_true', help='Disable visualization generation')
    
    args = parser.parse_args()
    
    try:
        # Run prediction
        run_prediction(
            model_path=args.model,
            data_path=args.data,
            output_dir=args.output_dir,
            window_size=args.window_size,
            use_gpu=not args.no_gpu,
            batch_size=args.batch_size,
            visualize=not args.no_visualize
        )
        
        logger.info("Prediction completed successfully")
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
