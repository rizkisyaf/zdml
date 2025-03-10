# Bitcoin Perpetual Futures Prediction

A deep learning system for predicting Bitcoin perpetual futures price movements using LSTM networks and order book data.

## Project Overview

This project implements a sophisticated prediction system for Bitcoin perpetual futures using high-frequency order book data. The system leverages LSTM neural networks with attention mechanisms to capture temporal patterns and market microstructure effects.

## Features

- **Comprehensive Order Book Feature Engineering**: Extracts rich features from all 10 levels of L2 order book data
- **Advanced LSTM Architecture**: Multi-layer LSTM with multi-head attention mechanisms and feature projection
- **Financial-Specific Training**: Custom loss functions and evaluation metrics
- **Market Microstructure Modeling**: Accounts for order flow imbalance and liquidity dynamics
- **Deployment Optimizations**: Quantization and latency-aware design

## Project Structure

```
├── notebooks/              # Jupyter notebooks for exploration and visualization
├── src/                    # Source code
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model architecture definitions
│   ├── utils/              # Utility functions and feature engineering
│   ├── visualization/      # Visualization utilities
│   ├── train.py            # Training script
│   └── predict.py          # Prediction script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd bitcoin-futures-prediction

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

To preprocess the order book data:

```bash
python -m src.data.preprocessing --input futures_orderbook_data.csv --output processed_data.pkl --sequence_length 100 --prediction_horizon 1
```

### Model Training

To train the LSTM model:

```bash
python -m src.train --data processed_data.pkl --model_dir saved_models --epochs 100 --batch_size 512
```

### Making Predictions

To make predictions with a trained model:

```bash
python -m src.predict --model saved_models/lstm_model_YYYYMMDD_HHMMSS.pth --data futures_orderbook_data.csv --output_dir prediction_results
```

### Running the Complete Pipeline

To run the complete pipeline from data preprocessing to prediction:

```bash
./run_pipeline.py --data futures_orderbook_data.csv --output_dir pipeline_results --epochs 100
```

### Quick Example

For a quick demonstration of the system:

```bash
./example.py
```

## Key Components

### Enhanced Feature Engineering

The system extracts a comprehensive set of features from all 10 levels of order book data:

- **Basic Price Features**:
  - Mid-price: `(best_bid + best_ask)/2`
  - Weighted Mid-Price: `(Σ(bid_p*Q)/ΣQ + Σ(ask_p*Q)/ΣQ)/2` (using all 10 levels)
  - Spread: `ask_price1 - bid_price1`
  - Microprice: `(bid_price1*ask_qty1 + ask_price1*bid_qty1)/(bid_qty1 + ask_qty1)`
  - Price Range: `ask_price10 - bid_price10`

- **Order Book Imbalance Features**:
  - Standard Imbalance: `(sum(bid_qty) - sum(ask_qty))/(sum(bid_qty) + sum(ask_qty))` (using all 10 levels)
  - Liquidity Imbalance: Distance-weighted imbalance across all price levels

- **Price Dynamics**:
  - Price Differences: `(p_t - p_{t-1})/p_{t-1}`
  - Log Returns: `log(p_t) - log(p_{t-1})`
  - Rolling Volatility: Standard deviation of returns over a window

- **Volume and Depth Features**:
  - Volume Acceleration: `(qty_t - qty_{t-5})/5`
  - Cumulative Depth: Total quantity at each price level (all 10 levels)
  - Bid/Ask Slope: Linear regression slope of price curve across all levels

### Advanced LSTM Architecture

The LSTM model architecture includes:

- **Feature Projection Layer**: Reduces dimensionality of large feature sets
- **Multi-layer LSTM**: 3 layers with dropout for regularization
- **Multi-head Attention**: 8 attention heads to capture complex dependencies
- **Layer Normalization**: Stabilizes training with deep networks
- **Custom Loss Functions**: Hybrid loss for financial data with fat tails

### Evaluation Metrics

The system evaluates model performance using:

- **Price prediction accuracy**: MAE, RMSE
- **Directional prediction accuracy**: Accuracy, Precision, Recall, F1
- **Trading performance**: Sharpe Ratio, Maximum Drawdown, Calmar Ratio, Sortino Ratio
- **Profitability metrics**: Win Rate, Profit Factor

## Jupyter Notebook

For an interactive demonstration of the system, see the Jupyter notebook:

```bash
jupyter notebook notebooks/bitcoin_futures_prediction.ipynb
```

## License

MIT
