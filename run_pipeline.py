#!/usr/bin/env python
"""
Bitcoin Perpetual Futures Prediction Pipeline

This script runs the entire pipeline from data preprocessing to prediction.
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(
    data_path: str,
    output_dir: str,
    sequence_length: int = 100,
    prediction_horizon: int = 1,
    window_size: int = 20,
    batch_size: int = 512,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 10,
    use_gpu: bool = True,
    loss_type: str = 'hybrid',
    output_size: int = 3,
    test_size: float = 0.2,
    val_size: float = 0.1
) -> None:
    """
    Run the entire pipeline from data preprocessing to prediction.
    
    Args:
        data_path: Path to the order book data
        output_dir: Directory to save the results
        sequence_length: Length of input sequences
        prediction_horizon: How many steps ahead to predict
        window_size: Window size for feature extraction
        batch_size: Batch size for training and prediction
        epochs: Number of epochs for training
        learning_rate: Learning rate for training
        weight_decay: Weight decay for training
        patience: Patience for early stopping
        use_gpu: Whether to use GPU
        loss_type: Type of loss function
        output_size: Number of outputs
        test_size: Fraction of data to use for testing
        val_size: Fraction of data to use for validation
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Import modules
    from src.data.preprocessing import load_orderbook_data, preprocess_data, prepare_training_data, save_processed_data
    from src.train import train_model
    from src.predict import run_prediction
    
    # Step 1: Preprocess data
    logger.info("Step 1: Preprocessing data")
    start_time = time.time()
    
    # Load data
    df = load_orderbook_data(data_path)
    
    # Preprocess data
    features_df = preprocess_data(df, window_size)
    
    # Prepare training data
    data = prepare_training_data(
        features_df,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        test_size=test_size,
        val_size=val_size
    )
    
    # Save processed data
    processed_data_path = os.path.join(output_dir, 'processed_data.pkl')
    save_processed_data(data, processed_data_path)
    
    preprocessing_time = time.time() - start_time
    logger.info(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
    
    # Step 2: Train model
    logger.info("Step 2: Training model")
    start_time = time.time()
    
    # Create model directory
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Train model
    results = train_model(
        data_path=processed_data_path,
        model_save_dir=model_dir,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        use_gpu=use_gpu,
        loss_type=loss_type,
        output_size=output_size,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Step 3: Make predictions
    logger.info("Step 3: Making predictions")
    start_time = time.time()
    
    # Create prediction directory
    prediction_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(prediction_dir, exist_ok=True)
    
    # Run prediction
    prediction_results = run_prediction(
        model_path=results['model_path'],
        data_path=data_path,
        output_dir=prediction_dir,
        window_size=window_size,
        use_gpu=use_gpu,
        batch_size=batch_size,
        visualize=True
    )
    
    prediction_time = time.time() - start_time
    logger.info(f"Prediction completed in {prediction_time:.2f} seconds")
    
    # Log total time
    total_time = preprocessing_time + training_time + prediction_time
    logger.info(f"Total pipeline completed in {total_time:.2f} seconds")
    
    # Save pipeline info
    pipeline_info = {
        'data_path': data_path,
        'output_dir': output_dir,
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon,
        'window_size': window_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'patience': patience,
        'use_gpu': use_gpu,
        'loss_type': loss_type,
        'output_size': output_size,
        'test_size': test_size,
        'val_size': val_size,
        'preprocessing_time': preprocessing_time,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'total_time': total_time,
        'model_path': results['model_path'],
        'test_metrics': results['test_metrics'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save pipeline info
    import json
    with open(os.path.join(output_dir, 'pipeline_info.json'), 'w') as f:
        json.dump(pipeline_info, f, indent=4, default=str)
    
    logger.info(f"Pipeline info saved to {os.path.join(output_dir, 'pipeline_info.json')}")
    logger.info("Pipeline completed successfully")


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Run Bitcoin Perpetual Futures Prediction Pipeline')
    parser.add_argument('--data', type=str, required=True, help='Path to the order book data')
    parser.add_argument('--output_dir', type=str, default='pipeline_results', help='Directory to save the results')
    parser.add_argument('--sequence_length', type=int, default=100, help='Length of input sequences')
    parser.add_argument('--prediction_horizon', type=int, default=1, help='How many steps ahead to predict')
    parser.add_argument('--window_size', type=int, default=20, help='Window size for feature extraction')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and prediction')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for training')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--loss_type', type=str, default='hybrid', choices=['mse', 'hybrid', 'student_t'],
                       help='Type of loss function')
    parser.add_argument('--output_size', type=int, default=3, help='Number of outputs')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1, help='Fraction of data to use for validation')
    
    args = parser.parse_args()
    
    try:
        # Run pipeline
        run_pipeline(
            data_path=args.data,
            output_dir=args.output_dir,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
            window_size=args.window_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            use_gpu=not args.no_gpu,
            loss_type=args.loss_type,
            output_size=args.output_size,
            test_size=args.test_size,
            val_size=args.val_size
        )
    
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main() 