import os
import torch
from src.train import train_model
from src.utils.push_to_hub import push_model_to_hub
from huggingface_hub import login

def train_and_push():
    """
    Train the model on Colab and push to Hugging Face Hub.
    """
    # Check if running on Colab with GPU
    if not torch.cuda.is_available():
        raise RuntimeError("This script should be run on Google Colab with GPU runtime")
    
    print("Training model on Colab GPU...")
    
    # Train model
    results = train_model(
        data_path="processed_data.pkl",
        model_save_dir="colab_models",
        batch_size=512,
        epochs=100,  # Increased epochs since we're using Colab's GPU
        learning_rate=1e-4,
        weight_decay=1e-5,
        patience=10,
        use_gpu=True,
        loss_type='hybrid',
        output_size=3,
        sequence_length=100,
        prediction_horizon=1
    )
    
    print("Training completed. Pushing to Hugging Face Hub...")
    
    # Login to Hugging Face (you'll need to provide your token)
    login()
    
    # Push to your repository
    model_url = push_model_to_hub(
        model_path=results['model_path'],
        repo_name="zdml",  # Your repository name
        organization="rizkisyaf"  # Your Hugging Face username
    )
    
    print(f"Model successfully pushed to {model_url}")
    print("\nYou can now use the model in your Gradio app!")

if __name__ == "__main__":
    train_and_push() 