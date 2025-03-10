import nbformat as nbf
import os

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create markdown cell with introduction
markdown_cell = nbf.v4.new_markdown_cell('''# Bitcoin Perpetual Futures Prediction

This notebook demonstrates the implementation of a Bitcoin perpetual futures prediction system using LSTM networks and order book data.

## Table of Contents
1. Setup and Data Loading
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Making Predictions
6. Evaluation and Visualization''')

# Create code cell with imports and setup
setup_code = '''# Import required libraries
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Set plotting style
plt.style.use('seaborn')
%matplotlib inline'''

setup_cell = nbf.v4.new_code_cell(setup_code)

# Create markdown cell for data loading section
data_loading_md = nbf.v4.new_markdown_cell('''## 1. Setup and Data Loading

First, we'll load the order book data and check its structure.''')

# Create code cell for data loading
data_loading_code = '''# Import our modules
from src.data.preprocessing import load_orderbook_data, preprocess_data, prepare_training_data
from src.utils.features import extract_features
from src.visualization.visualize import plot_order_book_snapshot

# Load data
data_path = '../futures_orderbook_data.csv'
if not os.path.exists(data_path):
    logger.error(f"Data file not found: {data_path}")
    logger.info("Please download the data file and place it in the correct directory.")
else:
    df = load_orderbook_data(data_path)
    logger.info(f"Loaded {len(df)} rows of order book data with {len(df.columns)} columns")

# Display first few rows (showing only first level for brevity)
columns_to_show = ['timestamp', 'bid_price1', 'bid_qty1', 'ask_price1', 'ask_qty1']
df[columns_to_show].head()'''

data_loading_cell = nbf.v4.new_code_cell(data_loading_code)

# Create markdown cell for preprocessing section
preprocessing_md = nbf.v4.new_markdown_cell('''## 2. Data Preprocessing

Now we'll preprocess the data and extract features from all 10 levels of the order book.''')

# Create code cell for preprocessing
preprocessing_code = '''# Preprocess data
window_size = 20
features_df = preprocess_data(df, window_size)
logger.info(f"Preprocessed data has {len(features_df)} rows and {features_df.shape[1]} columns")

# Display first few rows of preprocessed data (key features)
key_features = ['mid_price', 'weighted_mid_price', 'spread', 'imbalance', 'liquidity_imbalance', 'price_range']
features_df[key_features].head()'''

preprocessing_cell = nbf.v4.new_code_cell(preprocessing_code)

# Create markdown cell for feature engineering section
feature_eng_md = nbf.v4.new_markdown_cell('''## 3. Feature Engineering

Let's prepare the sequences for our LSTM model.''')

# Create code cell for feature engineering
feature_eng_code = '''# Prepare training data
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
print("Data shapes:")
print(f"X_train: {data['X_train'].shape}")
print(f"y_train: {data['y_train'].shape}")
print(f"X_val: {data['X_val'].shape}")
print(f"y_val: {data['y_val'].shape}")
print(f"X_test: {data['X_test'].shape}")
print(f"y_test: {data['y_test'].shape}")'''

feature_eng_cell = nbf.v4.new_code_cell(feature_eng_code)

# Create markdown cell for model training section
model_training_md = nbf.v4.new_markdown_cell('''## 4. Model Training

Now we'll train our LSTM model with attention mechanism.''')

# Create code cell for model training
model_training_code = '''from src.train import train_model

# Save processed data for training
processed_data_path = 'processed_data.pkl'
with open(processed_data_path, 'wb') as f:
    import pickle
    pickle.dump(data, f)

# Create model directory
model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)

# Train model
results = train_model(
    data_path=processed_data_path,
    model_save_dir=model_dir,
    batch_size=512,
    epochs=10,  # Reduced for demonstration
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
print("\\nModel Architecture:")
print(f"Input size: {data['X_train'].shape[2]} features")
print(f"Hidden size: {512 if data['X_train'].shape[2] > 100 else 256}")
print(f"Projection size: 128")
print(f"Number of LSTM layers: 3")
print(f"Number of attention heads: 8")
print(f"Output size: 3 (price, direction, volatility)")'''

model_training_cell = nbf.v4.new_code_cell(model_training_code)

# Create markdown cell for predictions section
predictions_md = nbf.v4.new_markdown_cell('''## 5. Making Predictions

Let's use our trained model to make predictions.''')

# Create code cell for predictions
predictions_code = '''from src.predict import run_prediction

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
)'''

predictions_cell = nbf.v4.new_code_cell(predictions_code)

# Create markdown cell for evaluation section
evaluation_md = nbf.v4.new_markdown_cell('''## 6. Evaluation and Visualization

Finally, let's evaluate our model's performance and visualize the results.''')

# Create code cell for evaluation
evaluation_code = '''# Print metrics
print("Model Metrics:")
for k, v in prediction_results['metrics'].items():
    print(f"{k}: {v:.4f}")

print("\\nTrading Metrics:")
for k, v in prediction_results['trading_metrics'].items():
    print(f"{k}: {v:.4f}")

# Load and display saved visualizations
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
img = plt.imread(os.path.join(prediction_dir, 'price_predictions.png'))
plt.imshow(img)
plt.axis('off')
plt.title('Price Predictions')

plt.subplot(2, 2, 2)
img = plt.imread(os.path.join(prediction_dir, 'direction_accuracy.png'))
plt.imshow(img)
plt.axis('off')
plt.title('Direction Accuracy')

plt.subplot(2, 2, 3)
img = plt.imread(os.path.join(prediction_dir, 'trading_performance.png'))
plt.imshow(img)
plt.axis('off')
plt.title('Trading Performance')

plt.subplot(2, 2, 4)
img = plt.imread(os.path.join(prediction_dir, 'order_book_first.png'))
plt.imshow(img)
plt.axis('off')
plt.title('Order Book Snapshot')

plt.tight_layout()
plt.show()'''

evaluation_cell = nbf.v4.new_code_cell(evaluation_code)

# Add all cells to notebook
nb.cells.extend([
    markdown_cell,
    setup_cell,
    data_loading_md,
    data_loading_cell,
    preprocessing_md,
    preprocessing_cell,
    feature_eng_md,
    feature_eng_cell,
    model_training_md,
    model_training_cell,
    predictions_md,
    predictions_cell,
    evaluation_md,
    evaluation_cell
])

# Create notebooks directory if it doesn't exist
os.makedirs('notebooks', exist_ok=True)

# Write the notebook to a file
with open('notebooks/bitcoin_futures_prediction.ipynb', 'w') as f:
    nbf.write(nb, f) 