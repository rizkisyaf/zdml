import nbformat as nbf
import os

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create markdown cell with introduction
markdown_intro = nbf.v4.new_markdown_cell('''# Bitcoin Futures Prediction Model Training

This notebook trains the LSTM model for Bitcoin futures prediction using Google Colab's GPU and pushes it to Hugging Face Hub.

## Overview
1. Setup Environment
2. Load and Preprocess Data
3. Train Model
4. Push to Hugging Face Hub

Make sure you have:
- Your Hugging Face token ready
- The order book data file (`futures_orderbook_data.csv`)
- Access to your Hugging Face Space (rizkisyaf/zdml)''')

# Setup Environment
setup_code = '''# Install required packages
!pip install -q transformers huggingface_hub gradio torch pandas numpy scikit-learn matplotlib seaborn tqdm pywavelets tensorboard pytest jupyter

# Clone or update repository
import os
if os.path.exists('zdml'):
    # Update existing repository
    %cd zdml
    !git pull origin main
else:
    # Clone repository
    !git clone https://github.com/rizkisyaf/zdml.git
    %cd zdml

# Import required libraries
import os
import sys
import torch
import pandas as pd
import numpy as np
from huggingface_hub import login
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check GPU availability
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
# Add project root to Python path
if not os.path.abspath('.') in sys.path:
    sys.path.append(os.path.abspath('.'))'''

setup_cell = nbf.v4.new_code_cell(setup_code)

# Data loading markdown
data_loading_md = nbf.v4.new_markdown_cell('''## Data Loading and Preprocessing

Upload your order book data and prepare it for training.''')

# Data loading code
data_loading_code = '''# Upload data file
from google.colab import files
print("Please upload futures_orderbook_data.csv")
uploaded = files.upload()

# Save the uploaded file
!mkdir -p data
!mv futures_orderbook_data.csv data/futures_orderbook_data.csv

# Load and preprocess data
from src.data.preprocessing import load_orderbook_data, preprocess_data, prepare_training_data, save_processed_data

# Load data
df = load_orderbook_data('data/futures_orderbook_data.csv')
print(f"Loaded {len(df)} rows of order book data")

# Preprocess data
features_df = preprocess_data(df, window_size=20)
print(f"Preprocessed data shape: {features_df.shape}")

# Display first few rows of key features
key_features = ['mid_price', 'weighted_mid_price', 'spread', 'imbalance', 'liquidity_imbalance', 'price_range']
print("\nFirst few rows of key features:")
print(features_df[key_features].head())'''

data_loading_cell = nbf.v4.new_code_cell(data_loading_code)

# Data preparation markdown
data_prep_md = nbf.v4.new_markdown_cell('''## Data Preparation

Prepare sequences for LSTM training and split into train/val/test sets.''')

# Data preparation code
data_prep_code = '''# Prepare training data
data = prepare_training_data(
    features_df,
    sequence_length=100,
    prediction_horizon=1,
    test_size=0.2,
    val_size=0.1
)

# Save processed data
save_processed_data(data, 'processed_data.pkl')

print("Data shapes:")
print(f"X_train: {data['X_train'].shape}")
print(f"y_train: {data['y_train'].shape}")
print(f"X_val: {data['X_val'].shape}")
print(f"y_val: {data['y_val'].shape}")
print(f"X_test: {data['X_test'].shape}")
print(f"y_test: {data['y_test'].shape}")

# Print feature names
print("\nFeatures used for training:")
for i, feature in enumerate(data['feature_names']):
    print(f"{i+1}. {feature}")'''

data_prep_cell = nbf.v4.new_code_cell(data_prep_code)

# Model training markdown
training_md = nbf.v4.new_markdown_cell('''## Model Training

Train the LSTM model using GPU acceleration.''')

# Hugging Face login code
hf_login_code = '''# Login to Hugging Face
from huggingface_hub import login
print("Please enter your Hugging Face token when prompted")
login()'''

hf_login_cell = nbf.v4.new_code_cell(hf_login_code)

# Training code
training_code = '''# Import training modules
from src.models.lstm_model import OrderBookLSTM, OrderBookConfig, HybridLoss
from src.train import train_model
from src.utils.push_to_hub import push_model_to_hub

# Set training parameters
training_params = {
    'data_path': 'processed_data.pkl',
    'model_save_dir': 'saved_models',
    'batch_size': 512,
    'epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'patience': 10,
    'use_gpu': torch.cuda.is_available(),
    'loss_type': 'hybrid',
    'output_size': 3,  # Price, direction, volatility
    'sequence_length': 100,
    'prediction_horizon': 1
}

# Create model directory
os.makedirs(training_params['model_save_dir'], exist_ok=True)

# Train model
print("Starting model training...")
results = train_model(**training_params)

print("\nTraining completed!")
print(f"Model saved to: {results['model_path']}")
print("\nTest metrics:")
for k, v in results['test_metrics'].items():
    print(f"{k}: {v:.4f}")

# Push model to Hugging Face Hub
print("\nPushing model to Hugging Face Hub...")
model_url = push_model_to_hub(
    model_path=results['model_path'],
    repo_name="zdml",
    organization="rizkisyaf"
)

print(f"\nModel successfully pushed to {model_url}")
print("You can now use the model in your Gradio app!")'''

training_cell = nbf.v4.new_code_cell(training_code)

# Add visualization markdown
viz_md = nbf.v4.new_markdown_cell('''## Visualizations

Let's visualize the model's predictions and performance.''')

# Add visualization code
viz_code = '''from src.visualization.visualize import (
    plot_price_predictions,
    plot_direction_accuracy,
    plot_trading_performance,
    plot_order_book_snapshot
)

# Load test predictions
test_preds = results['test_predictions']
test_targets = data['y_test']

# Plot predictions
plot_price_predictions(
    test_targets,
    test_preds,
    title='Bitcoin Price Predictions (Test Set)'
)

# Plot direction accuracy
plot_direction_accuracy(
    test_targets,
    test_preds,
    title='Direction Prediction Accuracy'
)

# Plot trading performance
plot_trading_performance(
    results['test_returns'],
    results['test_positions'],
    data['test_prices'],
    title='Trading Performance (Test Set)'
)

# Plot sample order book
plot_order_book_snapshot(
    df,
    index=len(df)//2,  # Middle of the dataset
    title='Sample Order Book Snapshot'
)'''

viz_cell = nbf.v4.new_code_cell(viz_code)

# Add all cells to notebook
nb.cells.extend([
    markdown_intro,
    setup_cell,
    data_loading_md,
    data_loading_cell,
    data_prep_md,
    data_prep_cell,
    training_md,
    hf_login_cell,
    training_cell,
    viz_md,
    viz_cell
])

# Create notebooks directory if it doesn't exist
os.makedirs('notebooks', exist_ok=True)

# Write the notebook
with open('notebooks/train_bitcoin_model.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Training notebook created successfully!") 