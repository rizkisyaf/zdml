import os
import torch
from transformers import AutoConfig
from src.models.lstm_model import OrderBookLSTM, OrderBookConfig

def push_model_to_hub(
    model_path: str,
    repo_name: str,
    token: str = None,
    organization: str = None
) -> str:
    """
    Push a trained model to Hugging Face Hub.
    
    Args:
        model_path: Path to the saved model
        repo_name: Name for the Hugging Face repository
        token: Hugging Face API token
        organization: Optional organization name
        
    Returns:
        URL of the model on Hugging Face Hub
    """
    # Load the model state
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Create config
    config = OrderBookConfig(
        input_size=state_dict['config']['input_size'],
        hidden_size=state_dict['config']['hidden_size'],
        projection_size=state_dict['config']['projection_size'],
        num_layers=state_dict['config']['num_layers'],
        dropout=state_dict['config']['dropout'],
        num_heads=state_dict['config']['num_heads'],
        output_size=state_dict['config']['output_size'],
        bidirectional=state_dict['config']['bidirectional']
    )
    
    # Initialize model with config
    model = OrderBookLSTM(config)
    
    # Load state dict
    model.load_state_dict(state_dict['model_state_dict'])
    
    # Push to hub
    if organization:
        repo_id = f"{organization}/{repo_name}"
    else:
        repo_id = repo_name
        
    model.push_to_hub(
        repo_id,
        use_auth_token=token,
        commit_message="Upload Bitcoin futures prediction model"
    )
    
    # Push config
    config.push_to_hub(
        repo_id,
        use_auth_token=token,
        commit_message="Upload model config"
    )
    
    return f"https://huggingface.co/{repo_id}" 