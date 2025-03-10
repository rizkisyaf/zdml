import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import load_processed_data
from src.models.lstm_model import OrderBookLSTM, HybridLoss, StudentTLoss, get_model_config
from src.utils.metrics import calculate_model_metrics, calculate_trading_metrics
from src.visualization.visualize import plot_training_history, plot_price_predictions, plot_direction_accuracy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Validation loss
            model: Model to save
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_model = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False
    
    def restore_model(self, model: nn.Module) -> None:
        """
        Restore the best model weights.
        
        Args:
            model: Model to restore
        """
        if self.restore_best_weights and self.best_model is not None:
            model.load_state_dict(self.best_model)


def train_model(
    data_path: str,
    model_save_dir: str,
    batch_size: int = 512,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 10,
    use_gpu: bool = True,
    loss_type: str = 'hybrid',
    output_size: int = 3,
    sequence_length: int = 100,
    prediction_horizon: int = 1
) -> Dict[str, Any]:
    """
    Train the LSTM model.
    
    Args:
        data_path: Path to the processed data
        model_save_dir: Directory to save the model
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        patience: Patience for early stopping
        use_gpu: Whether to use GPU
        loss_type: Type of loss function ('mse', 'hybrid', or 'student_t')
        output_size: Number of outputs (1 for price, 2 for price and direction, 3 for price, direction, and volatility)
        sequence_length: Length of input sequences
        prediction_horizon: How many steps ahead to predict
        
    Returns:
        Dictionary with training results
    """
    # Create model save directory
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = load_processed_data(data_path)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    feature_names = data['feature_names']
    
    # Adjust output size based on y_train shape
    if output_size != y_train.shape[1]:
        logger.warning(f"Adjusting output_size from {output_size} to {y_train.shape[1]} based on data")
        output_size = y_train.shape[1]
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Get model configuration
    input_size = X_train.shape[2]
    model_config = get_model_config(input_size)
    model_config['output_size'] = output_size
    
    # Create model
    logger.info(f"Creating model with config: {model_config}")
    model = OrderBookLSTM(**model_config).to(device)
    
    # Create loss function
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'hybrid':
        criterion = HybridLoss()
    elif loss_type == 'student_t':
        criterion = StudentTLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    # Train model
    logger.info(f"Starting training for {epochs} epochs")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for batch_X, batch_y in train_loader:
            # Move data to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * batch_X.size(0)
            train_predictions.append(outputs.cpu().detach().numpy())
            train_targets.append(batch_y.cpu().detach().numpy())
        
        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        
        # Concatenate predictions and targets
        train_predictions = np.vstack(train_predictions)
        train_targets = np.vstack(train_targets)
        
        # Calculate training metrics
        train_metrics = calculate_model_metrics(train_targets, train_predictions)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Move data to device
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Update statistics
                val_loss += loss.item() * batch_X.size(0)
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())
        
        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        
        # Concatenate predictions and targets
        val_predictions = np.vstack(val_predictions)
        val_targets = np.vstack(val_targets)
        
        # Calculate validation metrics
        val_metrics = calculate_model_metrics(val_targets, val_predictions)
        
        # Update learning rate
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        
        # Print progress
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.6f}, "
                   f"Val Loss: {val_loss:.6f}, "
                   f"Train Metrics: {train_metrics}, "
                   f"Val Metrics: {val_metrics}")
        
        # Check early stopping
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    early_stopping.restore_model(model)
    
    # Calculate training time
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            # Move data to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Update statistics
            test_loss += loss.item() * batch_X.size(0)
            test_predictions.append(outputs.cpu().numpy())
            test_targets.append(batch_y.cpu().numpy())
    
    # Calculate average test loss
    test_loss /= len(test_loader.dataset)
    
    # Concatenate predictions and targets
    test_predictions = np.vstack(test_predictions)
    test_targets = np.vstack(test_targets)
    
    # Calculate test metrics
    test_metrics = calculate_model_metrics(test_targets, test_predictions)
    
    logger.info(f"Test Loss: {test_loss:.6f}, Test Metrics: {test_metrics}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_save_dir, f"lstm_model_{timestamp}.pth")
    
    # Save model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': model_config,
        'feature_names': feature_names,
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon,
        'output_size': output_size,
        'loss_type': loss_type,
        'test_metrics': test_metrics,
        'training_time': training_time
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    # Save training history
    history_path = os.path.join(model_save_dir, f"training_history_{timestamp}.json")
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_history = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_metrics': [{k: float(v) for k, v in m.items()} for m in history['train_metrics']],
        'val_metrics': [{k: float(v) for k, v in m.items()} for m in history['val_metrics']]
    }
    
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f, indent=4)
    
    logger.info(f"Training history saved to {history_path}")
    
    # Plot training history
    plot_training_history(history, title='Training History', 
                         save_path=os.path.join(model_save_dir, f"training_history_{timestamp}.png"))
    
    # Plot price predictions
    plot_price_predictions(test_targets, test_predictions, title='Test Price Predictions',
                          save_path=os.path.join(model_save_dir, f"price_predictions_{timestamp}.png"))
    
    # Plot direction accuracy
    plot_direction_accuracy(test_targets, test_predictions, title='Test Direction Accuracy',
                           save_path=os.path.join(model_save_dir, f"direction_accuracy_{timestamp}.png"))
    
    # Return results
    return {
        'model': model,
        'model_path': model_path,
        'history': history,
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'test_predictions': test_predictions,
        'test_targets': test_targets
    }


def main():
    """
    Main function for training the model.
    """
    parser = argparse.ArgumentParser(description='Train LSTM model for order book prediction')
    parser.add_argument('--data', type=str, required=True, help='Path to processed data')
    parser.add_argument('--model_dir', type=str, default='saved_models', help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--loss_type', type=str, default='hybrid', choices=['mse', 'hybrid', 'student_t'],
                       help='Type of loss function')
    parser.add_argument('--output_size', type=int, default=3, help='Number of outputs')
    parser.add_argument('--sequence_length', type=int, default=100, help='Sequence length')
    parser.add_argument('--prediction_horizon', type=int, default=1, help='Prediction horizon')
    
    args = parser.parse_args()
    
    try:
        # Train model
        train_model(
            data_path=args.data,
            model_save_dir=args.model_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            use_gpu=not args.no_gpu,
            loss_type=args.loss_type,
            output_size=args.output_size,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon
        )
        
        logger.info("Training completed successfully")
    
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
