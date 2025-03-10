import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_price_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    title: str = 'Price Predictions',
    save_path: Optional[str] = None
) -> None:
    """
    Plot price predictions against true values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        timestamps: Optional timestamps for x-axis
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Extract price values if multi-output
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_price = y_true[:, 0]
        y_pred_price = y_pred[:, 0]
    else:
        y_true_price = y_true
        y_pred_price = y_pred
    
    # Plot data
    if timestamps is not None:
        plt.plot(timestamps, y_true_price, label='True Price', color='blue')
        plt.plot(timestamps, y_pred_price, label='Predicted Price', color='red', linestyle='--')
        plt.xlabel('Time')
    else:
        plt.plot(y_true_price, label='True Price', color='blue')
        plt.plot(y_pred_price, label='Predicted Price', color='red', linestyle='--')
        plt.xlabel('Time Step')
    
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved price prediction plot to {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_direction_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = 'Direction Prediction Accuracy',
    save_path: Optional[str] = None
) -> None:
    """
    Plot direction prediction accuracy.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Extract direction values if multi-output
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_dir = y_true[:, 1]
        y_pred_dir = y_pred[:, 1]
    else:
        # Calculate direction from consecutive values
        y_true_dir = np.zeros(len(y_true))
        y_pred_dir = np.zeros(len(y_pred))
        
        for i in range(1, len(y_true)):
            y_true_dir[i] = 1 if y_true[i] > y_true[i-1] else 0
            y_pred_dir[i] = 1 if y_pred[i] > y_pred[i-1] else 0
    
    # Convert to binary
    y_true_binary = y_true_dir > 0.5
    y_pred_binary = y_pred_dir > 0.5
    
    # Calculate confusion matrix
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    
    # Create confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.xlabel('Predicted Direction')
    plt.ylabel('True Direction')
    plt.title(title)
    
    # Calculate and display accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    plt.figtext(0.5, 0.01, f'Accuracy: {accuracy:.4f}', ha='center', fontsize=12)
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved direction accuracy plot to {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    model_weights: np.ndarray,
    feature_names: List[str],
    title: str = 'Feature Importance',
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance based on model weights.
    
    Args:
        model_weights: Model weights for the first layer
        feature_names: Names of the features
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate feature importance
    importance = np.abs(model_weights).mean(axis=1)
    
    # Sort features by importance
    sorted_idx = np.argsort(importance)
    
    # Plot feature importance
    plt.barh(range(len(sorted_idx)), importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Importance')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = 'Training History',
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot metrics if available
    if 'train_metrics' in history and 'val_metrics' in history:
        plt.subplot(1, 2, 2)
        
        # Find a common metric
        common_metrics = set(history['train_metrics'][0].keys()) & set(history['val_metrics'][0].keys())
        if common_metrics:
            metric = list(common_metrics)[0]
            train_metric = [m[metric] for m in history['train_metrics']]
            val_metric = [m[metric] for m in history['val_metrics']]
            
            plt.plot(train_metric, label=f'Training {metric}')
            plt.plot(val_metric, label=f'Validation {metric}')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.title(metric)
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_trading_performance(
    returns: np.ndarray,
    positions: np.ndarray,
    prices: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    title: str = 'Trading Performance',
    save_path: Optional[str] = None
) -> None:
    """
    Plot trading performance.
    
    Args:
        returns: Array of returns
        positions: Array of positions
        prices: Array of prices
        timestamps: Optional timestamps for x-axis
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Plot cumulative returns
    plt.subplot(3, 1, 1)
    cum_returns = np.cumprod(1 + returns) - 1
    
    if timestamps is not None:
        plt.plot(timestamps, cum_returns * 100, color='green')
        plt.xlabel('Time')
    else:
        plt.plot(cum_returns * 100, color='green')
        plt.xlabel('Time Step')
    
    plt.ylabel('Cumulative Return (%)')
    plt.title('Cumulative Returns')
    plt.grid(True, alpha=0.3)
    
    # Plot positions
    plt.subplot(3, 1, 2)
    
    if timestamps is not None:
        plt.step(timestamps, positions, color='blue')
        plt.xlabel('Time')
    else:
        plt.step(range(len(positions)), positions, color='blue')
        plt.xlabel('Time Step')
    
    plt.ylabel('Position')
    plt.title('Trading Positions')
    plt.grid(True, alpha=0.3)
    
    # Plot prices
    plt.subplot(3, 1, 3)
    
    if timestamps is not None:
        plt.plot(timestamps, prices, color='red')
        plt.xlabel('Time')
    else:
        plt.plot(prices, color='red')
        plt.xlabel('Time Step')
    
    plt.ylabel('Price')
    plt.title('Price')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved trading performance plot to {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_order_book_snapshot(
    df: pd.DataFrame,
    index: int,
    levels: int = 5,
    title: str = 'Order Book Snapshot',
    save_path: Optional[str] = None
) -> None:
    """
    Plot order book snapshot.
    
    Args:
        df: DataFrame with order book data
        index: Index of the snapshot to plot
        levels: Number of price levels to show
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Get snapshot data
    snapshot = df.iloc[index]
    
    # Extract bid and ask data
    bid_prices = [snapshot[f'bid_price{i+1}'] for i in range(levels)]
    bid_quantities = [snapshot[f'bid_qty{i+1}'] for i in range(levels)]
    ask_prices = [snapshot[f'ask_price{i+1}'] for i in range(levels)]
    ask_quantities = [snapshot[f'ask_qty{i+1}'] for i in range(levels)]
    
    # Plot bid side (negative quantities for visual separation)
    plt.barh(bid_prices, [-q for q in bid_quantities], color='green', alpha=0.7, label='Bids')
    
    # Plot ask side
    plt.barh(ask_prices, ask_quantities, color='red', alpha=0.7, label='Asks')
    
    # Add mid-price line
    mid_price = (bid_prices[0] + ask_prices[0]) / 2
    plt.axhline(y=mid_price, color='black', linestyle='--', label=f'Mid-Price: {mid_price:.2f}')
    
    # Add timestamp if available
    if 'timestamp' in snapshot:
        timestamp = pd.to_datetime(snapshot['timestamp'], unit='ms')
        plt.title(f"{title} - {timestamp}")
    else:
        plt.title(title)
    
    plt.xlabel('Quantity')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved order book snapshot plot to {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_attention_weights(
    attention_weights: np.ndarray,
    sequence_length: int,
    feature_names: List[str],
    title: str = 'Attention Weights',
    save_path: Optional[str] = None
) -> None:
    """
    Plot attention weights.
    
    Args:
        attention_weights: Attention weights matrix
        sequence_length: Length of input sequence
        feature_names: Names of the features
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Reshape attention weights if needed
    if len(attention_weights.shape) > 2:
        # Average across heads if multi-head attention
        attention_weights = attention_weights.mean(axis=0)
    
    # Plot attention weights
    sns.heatmap(attention_weights, cmap='viridis')
    
    # Set labels
    if len(feature_names) <= 10:
        plt.yticks(np.arange(0.5, len(feature_names)), feature_names)
    else:
        plt.yticks([])
    
    # Set x-axis labels (time steps)
    if sequence_length <= 20:
        plt.xticks(np.arange(0.5, sequence_length), [f't-{sequence_length-i}' for i in range(sequence_length)])
    else:
        # Show only a few time steps
        step = sequence_length // 10
        plt.xticks(
            np.arange(0.5, sequence_length, step),
            [f't-{sequence_length-i}' for i in range(0, sequence_length, step)]
        )
    
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Feature')
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention weights plot to {save_path}")
    
    plt.tight_layout()
    plt.show()
