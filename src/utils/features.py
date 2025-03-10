import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional


def calculate_mid_price(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate mid-price from order book data.
    
    Args:
        df: DataFrame with order book data
        
    Returns:
        numpy array of mid-prices
    """
    best_bid = df['bid_price1'].values
    best_ask = df['ask_price1'].values
    return (best_bid + best_ask) / 2


def calculate_weighted_mid_price(df: pd.DataFrame, levels: int = 10) -> np.ndarray:
    """
    Calculate volume-weighted mid-price from order book data.
    
    Args:
        df: DataFrame with order book data
        levels: Number of price levels to consider
        
    Returns:
        numpy array of weighted mid-prices
    """
    weighted_bid_sum = np.zeros(len(df))
    weighted_ask_sum = np.zeros(len(df))
    bid_qty_sum = np.zeros(len(df))
    ask_qty_sum = np.zeros(len(df))
    
    for i in range(1, levels + 1):
        bid_price = df[f'bid_price{i}'].values
        bid_qty = df[f'bid_qty{i}'].values
        ask_price = df[f'ask_price{i}'].values
        ask_qty = df[f'ask_qty{i}'].values
        
        weighted_bid_sum += bid_price * bid_qty
        weighted_ask_sum += ask_price * ask_qty
        bid_qty_sum += bid_qty
        ask_qty_sum += ask_qty
    
    # Avoid division by zero
    bid_qty_sum = np.where(bid_qty_sum == 0, 1e-10, bid_qty_sum)
    ask_qty_sum = np.where(ask_qty_sum == 0, 1e-10, ask_qty_sum)
    
    weighted_bid = weighted_bid_sum / bid_qty_sum
    weighted_ask = weighted_ask_sum / ask_qty_sum
    
    return (weighted_bid + weighted_ask) / 2


def calculate_order_book_imbalance(df: pd.DataFrame, levels: int = 10) -> np.ndarray:
    """
    Calculate order book imbalance.
    
    Args:
        df: DataFrame with order book data
        levels: Number of price levels to consider
        
    Returns:
        numpy array of order book imbalance values
    """
    bid_qty_sum = np.zeros(len(df))
    ask_qty_sum = np.zeros(len(df))
    
    for i in range(1, levels + 1):
        bid_qty = df[f'bid_qty{i}'].values
        ask_qty = df[f'ask_qty{i}'].values
        
        bid_qty_sum += bid_qty
        ask_qty_sum += ask_qty
    
    # Avoid division by zero
    denominator = bid_qty_sum + ask_qty_sum
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    return (bid_qty_sum - ask_qty_sum) / denominator


def calculate_spread(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate bid-ask spread.
    
    Args:
        df: DataFrame with order book data
        
    Returns:
        numpy array of spread values
    """
    return df['ask_price1'].values - df['bid_price1'].values


def calculate_microprice(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate microprice (volume-weighted mid-price at best level).
    
    Args:
        df: DataFrame with order book data
        
    Returns:
        numpy array of microprice values
    """
    bid_price = df['bid_price1'].values
    bid_qty = df['bid_qty1'].values
    ask_price = df['ask_price1'].values
    ask_qty = df['ask_qty1'].values
    
    # Avoid division by zero
    denominator = bid_qty + ask_qty
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    return (bid_price * ask_qty + ask_price * bid_qty) / denominator


def calculate_price_diff(prices: np.ndarray) -> np.ndarray:
    """
    Calculate normalized price differences.
    
    Args:
        prices: numpy array of prices
        
    Returns:
        numpy array of normalized price differences
    """
    diffs = np.diff(prices, prepend=prices[0])
    # Avoid division by zero
    denominator = np.where(prices == 0, 1e-10, prices)
    return diffs / denominator


def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calculate log returns.
    
    Args:
        prices: numpy array of prices
        
    Returns:
        numpy array of log returns
    """
    # Avoid log(0)
    prices_safe = np.where(prices <= 0, 1e-10, prices)
    log_prices = np.log(prices_safe)
    return np.diff(log_prices, prepend=log_prices[0])


def calculate_volume_acceleration(df: pd.DataFrame, window: int = 5) -> np.ndarray:
    """
    Calculate volume acceleration.
    
    Args:
        df: DataFrame with order book data
        window: Window size for acceleration calculation
        
    Returns:
        numpy array of volume acceleration values
    """
    total_qty = np.zeros(len(df))
    
    for i in range(1, 11):
        total_qty += df[f'bid_qty{i}'].values + df[f'ask_qty{i}'].values
    
    # Calculate rolling difference
    qty_diff = np.zeros_like(total_qty)
    for i in range(window, len(total_qty)):
        qty_diff[i] = (total_qty[i] - total_qty[i-window]) / window
    
    return qty_diff


def calculate_cumulative_depth(df: pd.DataFrame, side: str = 'bid', levels: int = 10) -> np.ndarray:
    """
    Calculate cumulative depth at each price level.
    
    Args:
        df: DataFrame with order book data
        side: 'bid' or 'ask'
        levels: Number of price levels to consider
        
    Returns:
        numpy array of shape (n_samples, levels) with cumulative depth
    """
    n_samples = len(df)
    cum_depth = np.zeros((n_samples, levels))
    
    for i in range(1, levels + 1):
        qty = df[f'{side}_qty{i}'].values
        cum_depth[:, i-1] = qty
        
        if i > 1:
            cum_depth[:, i-1] += cum_depth[:, i-2]
    
    return cum_depth


def calculate_price_range(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate the price range (difference between highest ask and lowest bid).
    
    Args:
        df: DataFrame with order book data
        
    Returns:
        numpy array of price range values
    """
    highest_ask = df['ask_price10'].values
    lowest_bid = df['bid_price10'].values
    return highest_ask - lowest_bid


def calculate_price_slope(df: pd.DataFrame, side: str = 'bid', levels: int = 10) -> np.ndarray:
    """
    Calculate the slope of the price curve.
    
    Args:
        df: DataFrame with order book data
        side: 'bid' or 'ask'
        levels: Number of price levels to consider
        
    Returns:
        numpy array of price slope values
    """
    n_samples = len(df)
    slopes = np.zeros(n_samples)
    
    for i in range(n_samples):
        prices = np.array([df[f'{side}_price{j}'].iloc[i] for j in range(1, levels + 1)])
        # Use linear regression to calculate slope
        x = np.arange(levels)
        # For bids, prices are decreasing, so we need to reverse the order
        if side == 'bid':
            prices = prices[::-1]
        # Calculate slope using least squares
        slope, _ = np.polyfit(x, prices, 1)
        slopes[i] = slope
    
    return slopes


def calculate_liquidity_imbalance(df: pd.DataFrame, levels: int = 10) -> np.ndarray:
    """
    Calculate liquidity imbalance at different price levels.
    
    Args:
        df: DataFrame with order book data
        levels: Number of price levels to consider
        
    Returns:
        numpy array of liquidity imbalance values
    """
    n_samples = len(df)
    imbalance = np.zeros(n_samples)
    
    for i in range(n_samples):
        bid_prices = np.array([df[f'bid_price{j}'].iloc[i] for j in range(1, levels + 1)])
        bid_qtys = np.array([df[f'bid_qty{j}'].iloc[i] for j in range(1, levels + 1)])
        ask_prices = np.array([df[f'ask_price{j}'].iloc[i] for j in range(1, levels + 1)])
        ask_qtys = np.array([df[f'ask_qty{j}'].iloc[i] for j in range(1, levels + 1)])
        
        # Calculate mid-price
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        
        # Calculate distance from mid-price
        bid_distance = mid_price - bid_prices
        ask_distance = ask_prices - mid_price
        
        # Weight quantities by distance
        weighted_bid_qty = np.sum(bid_qtys / (bid_distance + 1e-10))
        weighted_ask_qty = np.sum(ask_qtys / (ask_distance + 1e-10))
        
        # Calculate imbalance
        total = weighted_bid_qty + weighted_ask_qty
        if total > 0:
            imbalance[i] = (weighted_bid_qty - weighted_ask_qty) / total
        else:
            imbalance[i] = 0
    
    return imbalance


def calculate_rolling_volatility(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Args:
        prices: numpy array of prices
        window: Window size for volatility calculation
        
    Returns:
        numpy array of volatility values
    """
    returns = calculate_log_returns(prices)
    vol = np.zeros_like(returns)
    
    for i in range(window, len(returns)):
        vol[i] = np.std(returns[i-window:i])
    
    return vol


def extract_features(df: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
    """
    Extract all features from order book data.
    
    Args:
        df: DataFrame with order book data
        window_size: Window size for rolling calculations
        
    Returns:
        DataFrame with extracted features
    """
    # Basic price features
    mid_price = calculate_mid_price(df)
    weighted_mid_price = calculate_weighted_mid_price(df)
    spread = calculate_spread(df)
    microprice = calculate_microprice(df)
    price_range = calculate_price_range(df)
    
    # Order book imbalance
    imbalance = calculate_order_book_imbalance(df)
    liquidity_imbalance = calculate_liquidity_imbalance(df)
    
    # Price dynamics
    price_diff = calculate_price_diff(mid_price)
    log_returns = calculate_log_returns(mid_price)
    volatility = calculate_rolling_volatility(mid_price, window_size)
    
    # Volume dynamics
    volume_accel = calculate_volume_acceleration(df)
    
    # Price curve features
    bid_slope = calculate_price_slope(df, side='bid')
    ask_slope = calculate_price_slope(df, side='ask')
    
    # Depth features
    bid_depth = calculate_cumulative_depth(df, side='bid')
    ask_depth = calculate_cumulative_depth(df, side='ask')
    
    # Create feature DataFrame
    features = pd.DataFrame({
        'timestamp': df['timestamp'],
        'mid_price': mid_price,
        'weighted_mid_price': weighted_mid_price,
        'spread': spread,
        'microprice': microprice,
        'price_range': price_range,
        'imbalance': imbalance,
        'liquidity_imbalance': liquidity_imbalance,
        'price_diff': price_diff,
        'log_returns': log_returns,
        'volatility': volatility,
        'volume_accel': volume_accel,
        'bid_slope': bid_slope,
        'ask_slope': ask_slope
    })
    
    # Add cumulative depth features
    for i in range(10):
        features[f'bid_depth_{i+1}'] = bid_depth[:, i]
        features[f'ask_depth_{i+1}'] = ask_depth[:, i]
    
    return features


def create_sequences(features: pd.DataFrame, 
                     sequence_length: int = 100, 
                     target_column: str = 'mid_price',
                     prediction_horizon: int = 1,
                     step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Args:
        features: DataFrame with extracted features
        sequence_length: Length of input sequences
        target_column: Column to predict
        prediction_horizon: How many steps ahead to predict
        step: Step size for creating sequences
        
    Returns:
        Tuple of (X, y) where X is the input sequences and y is the target values
    """
    data = features.drop('timestamp', axis=1).values
    X, y = [], []
    
    for i in range(0, len(data) - sequence_length - prediction_horizon, step):
        X.append(data[i:i+sequence_length])
        # For price prediction
        if target_column == 'mid_price':
            # Predict the actual price
            y.append(data[i+sequence_length+prediction_horizon-1, 
                         features.columns.get_loc(target_column)-1])  # -1 for timestamp
        # For direction prediction
        elif target_column == 'direction':
            current_price = data[i+sequence_length-1, 
                               features.columns.get_loc('mid_price')-1]  # -1 for timestamp
            future_price = data[i+sequence_length+prediction_horizon-1, 
                              features.columns.get_loc('mid_price')-1]  # -1 for timestamp
            y.append(1.0 if future_price > current_price else 0.0)
    
    return np.array(X), np.array(y)


def normalize_features(X_train: np.ndarray, X_val: np.ndarray = None, X_test: np.ndarray = None) -> Tuple:
    """
    Normalize features using percentage changes for price data and standard scaling for others.
    """
    # Identify price-related columns (assuming they're in the first few columns)
    price_cols = [0]  # Adjust based on your feature order
    
    # Initialize normalized arrays
    X_train_norm = X_train.copy()
    X_val_norm = X_val.copy() if X_val is not None else None
    X_test_norm = X_test.copy() if X_test is not None else None
    
    # Convert prices to percentage changes
    for col in price_cols:
        # Training data
        X_train_pct = np.zeros_like(X_train[:, :, col])
        X_train_pct[:, 1:] = np.diff(X_train[:, :, col], axis=1) / X_train[:, :-1, col]
        X_train_norm[:, :, col] = X_train_pct
        
        # Validation data
        if X_val is not None:
            X_val_pct = np.zeros_like(X_val[:, :, col])
            X_val_pct[:, 1:] = np.diff(X_val[:, :, col], axis=1) / X_val[:, :-1, col]
            X_val_norm[:, :, col] = X_val_pct
        
        # Test data
        if X_test is not None:
            X_test_pct = np.zeros_like(X_test[:, :, col])
            X_test_pct[:, 1:] = np.diff(X_test[:, :, col], axis=1) / X_test[:, :-1, col]
            X_test_norm[:, :, col] = X_test_pct
    
    # Standard scaling for non-price features
    non_price_cols = [i for i in range(X_train.shape[2]) if i not in price_cols]
    if non_price_cols:
        # Reshape to 2D for scaling
        X_train_2d = X_train[:, :, non_price_cols].reshape(-1, len(non_price_cols))
        
        # Calculate mean and std
        mean = np.mean(X_train_2d, axis=0)
        std = np.std(X_train_2d, axis=0)
        std[std == 0] = 1  # Prevent division by zero
        
        # Apply scaling
        for i, col in enumerate(non_price_cols):
            X_train_norm[:, :, col] = (X_train[:, :, col] - mean[i]) / std[i]
            if X_val is not None:
                X_val_norm[:, :, col] = (X_val[:, :, col] - mean[i]) / std[i]
            if X_test is not None:
                X_test_norm[:, :, col] = (X_test[:, :, col] - mean[i]) / std[i]
    
    return X_train_norm, X_val_norm, X_test_norm
