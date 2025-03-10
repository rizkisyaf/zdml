import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Directional accuracy score
    """
    # Convert to direction (1 for up, 0 for down or no change)
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        # Multi-output case, use the direction column
        true_direction = y_true[:, 1]
        pred_direction = y_pred[:, 1]
    else:
        # Single output case, compute direction from consecutive values
        true_direction = np.zeros_like(y_true)
        pred_direction = np.zeros_like(y_pred)
        
        for i in range(1, len(y_true)):
            true_direction[i] = 1 if y_true[i] > y_true[i-1] else 0
            pred_direction[i] = 1 if y_pred[i] > y_pred[i-1] else 0
    
    # Calculate accuracy
    return accuracy_score(true_direction, pred_direction > 0.5)


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        annualization_factor: Annualization factor (252 for daily returns)
        
    Returns:
        Sharpe ratio
    """
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    # Calculate mean and standard deviation
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Avoid division by zero
    if std_return == 0:
        return 0.0
    
    # Calculate Sharpe ratio
    sharpe = (mean_return - risk_free_rate) / std_return
    
    # Annualize
    sharpe_annualized = sharpe * np.sqrt(annualization_factor)
    
    return sharpe_annualized


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Array of returns
        
    Returns:
        Maximum drawdown
    """
    # Convert returns to cumulative returns
    cum_returns = np.cumprod(1 + returns)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cum_returns)
    
    # Calculate drawdown
    drawdown = (cum_returns - running_max) / running_max
    
    # Calculate maximum drawdown
    max_drawdown = np.min(drawdown)
    
    return max_drawdown


def calculate_calmar_ratio(returns: np.ndarray, period: int = 252) -> float:
    """
    Calculate Calmar ratio (annualized return / maximum drawdown).
    
    Args:
        returns: Array of returns
        period: Annualization period
        
    Returns:
        Calmar ratio
    """
    # Calculate annualized return
    total_return = np.prod(1 + returns) - 1
    num_periods = len(returns)
    annualized_return = (1 + total_return) ** (period / num_periods) - 1
    
    # Calculate maximum drawdown
    max_drawdown = abs(calculate_max_drawdown(returns))
    
    # Avoid division by zero
    if max_drawdown == 0:
        return 0.0
    
    # Calculate Calmar ratio
    calmar_ratio = annualized_return / max_drawdown
    
    return calmar_ratio


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate Sortino ratio (similar to Sharpe but only considers downside risk).
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        annualization_factor: Annualization factor
        
    Returns:
        Sortino ratio
    """
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    # Calculate mean return
    mean_return = np.mean(returns)
    
    # Calculate downside deviation (standard deviation of negative returns)
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        downside_deviation = 1e-10  # Avoid division by zero
    else:
        downside_deviation = np.std(negative_returns)
    
    # Calculate Sortino ratio
    sortino = (mean_return - risk_free_rate) / downside_deviation
    
    # Annualize
    sortino_annualized = sortino * np.sqrt(annualization_factor)
    
    return sortino_annualized


def calculate_trading_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             prices: np.ndarray, transaction_cost: float = 0.0005) -> Dict[str, float]:
    """
    Calculate trading metrics for a simple strategy based on predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        prices: Price series
        transaction_cost: Transaction cost as a fraction of trade value
        
    Returns:
        Dictionary with trading metrics
    """
    # Convert to direction if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        # Multi-output case, use the direction column
        pred_direction = y_pred[:, 1] > 0.5
    else:
        # Single output case, compute direction from consecutive predictions
        pred_direction = np.zeros(len(y_pred), dtype=bool)
        for i in range(1, len(y_pred)):
            pred_direction[i] = y_pred[i] > y_pred[i-1]
    
    # Initialize position and returns
    position = np.zeros(len(prices))
    returns = np.zeros(len(prices))
    
    # Simulate trading
    for i in range(1, len(prices)):
        # Update position based on prediction
        if pred_direction[i-1]:
            position[i] = 1  # Long position
        else:
            position[i] = -1  # Short position
        
        # Calculate return
        price_return = (prices[i] - prices[i-1]) / prices[i-1]
        
        # Apply position to return
        returns[i] = position[i] * price_return
        
        # Apply transaction cost if position changed
        if position[i] != position[i-1]:
            returns[i] -= transaction_cost
    
    # Calculate metrics
    total_return = np.sum(returns)
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_max_drawdown(returns)
    calmar = calculate_calmar_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    
    # Calculate win rate
    wins = np.sum(returns > 0)
    losses = np.sum(returns < 0)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    
    # Calculate profit factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
        'sortino_ratio': sortino,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }


def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate model performance metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with model metrics
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Check if multi-output
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        # Price prediction metrics
        metrics['price_mae'] = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
        metrics['price_rmse'] = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
        
        # Direction prediction metrics
        metrics['direction_accuracy'] = accuracy_score(y_true[:, 1] > 0.5, y_pred[:, 1] > 0.5)
        metrics['direction_precision'] = precision_score(y_true[:, 1] > 0.5, y_pred[:, 1] > 0.5)
        metrics['direction_recall'] = recall_score(y_true[:, 1] > 0.5, y_pred[:, 1] > 0.5)
        metrics['direction_f1'] = f1_score(y_true[:, 1] > 0.5, y_pred[:, 1] > 0.5)
        
        # Volatility prediction metrics
        if y_true.shape[1] > 2:
            metrics['volatility_mae'] = mean_absolute_error(y_true[:, 2], y_pred[:, 2])
            metrics['volatility_rmse'] = np.sqrt(mean_squared_error(y_true[:, 2], y_pred[:, 2]))
    else:
        # Single output (price prediction)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['directional_accuracy'] = calculate_directional_accuracy(y_true, y_pred)
    
    return metrics
