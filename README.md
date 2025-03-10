# Bitcoin Futures Prediction

This Space provides a web interface for predicting Bitcoin perpetual futures price movements using order book data. The model uses a sophisticated LSTM architecture with attention mechanisms to analyze market microstructure and predict:

1. Price Movement
2. Direction (Up/Down)
3. Volatility

## Usage

1. Upload your order book data CSV file
2. The model will process the data and provide predictions
3. Results include predicted price movement, direction, and volatility

## Model Architecture

- Multi-layer LSTM with attention mechanisms
- Feature projection for dimensionality reduction
- Hybrid loss function combining price, direction, and volatility predictions
- Trained on high-frequency order book data

## Input Format

The CSV file should contain order book data with the following columns:
- timestamp
- bid_price1-10, bid_qty1-10
- ask_price1-10, ask_qty1-10

## Example

An example order book data file is provided in the interface.

## Links

- [Model Repository](https://huggingface.co/rizkisyaf/zdml)
- [GitHub Repository](https://github.com/your-username/DataResearch)
- [Documentation](https://github.com/your-username/DataResearch/blob/main/README.md) 