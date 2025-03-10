import gradio as gr
import torch
import pandas as pd
import numpy as np
from transformers import AutoConfig, AutoModel
from src.data.preprocessing import preprocess_data
from src.utils.features import extract_features

def predict(csv_file):
    """
    Make predictions using the model.
    """
    try:
        # Load and preprocess data
        df = pd.read_csv(csv_file.name)
        features_df = preprocess_data(df, window_size=20)
        
        # Load model from Hugging Face Hub
        model_name = "rizkisyaf/zdml"  # Your Hugging Face repository
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        # Prepare input
        sequence_length = 100
        X = features_df.values[-sequence_length:].reshape(1, sequence_length, -1)
        X = torch.FloatTensor(X)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(X)
        
        # Process outputs
        price_pred = outputs[0, 0].item()
        direction_pred = "Up" if outputs[0, 1].item() > 0.5 else "Down"
        volatility_pred = outputs[0, 2].item()
        
        return {
            "Predicted Price Movement": price_pred,
            "Predicted Direction": direction_pred,
            "Predicted Volatility": volatility_pred
        }
    
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.File(label="Upload Order Book CSV"),
    outputs=gr.JSON(label="Predictions"),
    title="Bitcoin Futures Prediction",
    description="Upload order book data to predict price movements, direction, and volatility.",
    examples=[["example_data/sample_orderbook.csv"]],
    cache_examples=True
)

# Launch the app
if __name__ == "__main__":
    iface.launch() 