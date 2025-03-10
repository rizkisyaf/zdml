import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from transformers import PreTrainedModel, PretrainedConfig


class AttentionLayer(nn.Module):
    """
    Multi-head attention layer for LSTM outputs.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention to the input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Apply attention (query, key, value are all the same for self-attention)
        attn_output, _ = self.attention(x, x, x)
        
        # Add residual connection and layer normalization
        output = self.layer_norm(x + attn_output)
        
        return output


class FeatureProjection(nn.Module):
    """
    Feature projection layer to reduce dimensionality of input features.
    """
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LayerNorm(output_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input features to lower dimension.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Tensor of shape (batch_size, seq_len, output_size)
        """
        return self.projection(x)


class OrderBookConfig(PretrainedConfig):
    """Configuration class for OrderBookLSTM model."""
    model_type = "orderbook_lstm"
    
    def __init__(
        self,
        input_size: int = 34,
        hidden_size: int = 256,
        projection_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_heads: int = 8,
        output_size: int = 3,
        bidirectional: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.output_size = output_size
        self.bidirectional = bidirectional


class OrderBookLSTM(PreTrainedModel):
    """
    Enhanced LSTM model for order book prediction with Hugging Face compatibility.
    """
    config_class = OrderBookConfig
    base_model_prefix = "orderbook_lstm"
    
    def __init__(self, config: OrderBookConfig):
        super().__init__(config)
        
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.projection_size = config.projection_size
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.output_size = config.output_size
        
        # Input normalization
        self.input_norm = nn.LayerNorm(self.input_size)
        
        # Feature projection layer
        self.feature_projection = FeatureProjection(
            input_size=self.input_size,
            output_size=self.projection_size,
            dropout=self.dropout
        ) if self.input_size > self.projection_size else None
        
        # LSTM layers with residual connections
        lstm_input_size = self.projection_size if self.feature_projection is not None else self.input_size
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=lstm_input_size if i == 0 else self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                dropout=0,
                batch_first=True,
                bidirectional=self.bidirectional
            ) for i in range(self.num_layers)
        ])
        self.lstm_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size * (2 if self.bidirectional else 1))
            for _ in range(self.num_layers)
        ])
        self.lstm_dropouts = nn.ModuleList([
            nn.Dropout(self.dropout) for _ in range(self.num_layers)
        ])
        
        # Attention layer
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.attention = AttentionLayer(
            embed_dim=lstm_output_size,
            num_heads=config.num_heads,
            dropout=self.dropout
        )
        
        # Output layers with residual connections
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.layer_norm1 = nn.LayerNorm(lstm_output_size // 2)
        self.dropout1 = nn.Dropout(self.dropout)
        
        self.fc2 = nn.Linear(lstm_output_size // 2, lstm_output_size // 4)
        self.layer_norm2 = nn.LayerNorm(lstm_output_size // 4)
        self.dropout2 = nn.Dropout(self.dropout)
        
        self.fc3 = nn.Linear(lstm_output_size // 4, self.output_size)
        self.output_norm = nn.LayerNorm(self.output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.
        """
        # Input normalization
        x = self.input_norm(x)
        
        # Project features if needed
        if self.feature_projection is not None:
            x = self.feature_projection(x)
        
        # LSTM layers with residual connections
        lstm_out = x
        for i, (lstm, norm, drop) in enumerate(zip(self.lstm_layers, self.lstm_norms, self.lstm_dropouts)):
            residual = lstm_out
            lstm_out, _ = lstm(lstm_out)
            lstm_out = norm(lstm_out)
            lstm_out = drop(lstm_out)
            if i > 0:  # Skip first layer for residual
                lstm_out = lstm_out + residual
        
        # Apply attention
        attn_out = self.attention(lstm_out)
        
        # Take the last timestep
        out = attn_out[:, -1, :]
        
        # Fully connected layers with residual connections
        residual = out
        out = self.fc1(out)
        out = self.layer_norm1(out)
        out = F.leaky_relu(out)
        out = self.dropout1(out)
        
        residual2 = out
        out = self.fc2(out)
        out = self.layer_norm2(out)
        out = F.leaky_relu(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        out = self.output_norm(out)
        
        # Apply different activations based on output type
        if self.output_size == 1:
            return out
        elif self.output_size == 2:
            price_pred = out[:, 0].unsqueeze(1)
            direction_pred = torch.sigmoid(out[:, 1]).unsqueeze(1)
            return torch.cat([price_pred, direction_pred], dim=1)
        else:
            price_pred = out[:, 0].unsqueeze(1)
            direction_pred = torch.sigmoid(out[:, 1]).unsqueeze(1)
            volatility_pred = F.softplus(out[:, 2]).unsqueeze(1)
            return torch.cat([price_pred, direction_pred, volatility_pred], dim=1)
    
    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Prediction step for inference.
        
        Args:
            batch: Input tensor
            
        Returns:
            Model predictions
        """
        self.eval()
        with torch.no_grad():
            return self(batch)


class HybridLoss(nn.Module):
    """
    Hybrid loss function combining MSE for price prediction and BCE for direction prediction.
    """
    def __init__(self, price_weight: float = 0.4, direction_weight: float = 0.5, volatility_weight: float = 0.1):
        super().__init__()
        self.price_weight = price_weight
        self.direction_weight = direction_weight
        self.volatility_weight = volatility_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the hybrid loss with emphasis on directional prediction.
        
        Args:
            pred: Predictions tensor of shape (batch_size, output_size)
            target: Target tensor of shape (batch_size, output_size)
            
        Returns:
            Scalar loss value
        """
        if pred.shape[1] == 1:
            # Single output (price prediction)
            return self.mse_loss(pred, target)
        elif pred.shape[1] == 2:
            # Two outputs (price and direction)
            price_loss = self.mse_loss(pred[:, 0].unsqueeze(1), target[:, 0].unsqueeze(1))
            direction_loss = self.bce_loss(pred[:, 1].unsqueeze(1), target[:, 1].unsqueeze(1))
            
            # Add directional accuracy penalty
            direction_penalty = torch.mean(torch.abs(pred[:, 1] - target[:, 1]))
            
            return (self.price_weight * price_loss + 
                   self.direction_weight * direction_loss + 
                   0.1 * direction_penalty)  # Additional penalty term
        else:
            # Three outputs (price, direction, volatility)
            price_loss = self.mse_loss(pred[:, 0].unsqueeze(1), target[:, 0].unsqueeze(1))
            direction_loss = self.bce_loss(pred[:, 1].unsqueeze(1), target[:, 1].unsqueeze(1))
            volatility_loss = self.mse_loss(pred[:, 2].unsqueeze(1), target[:, 2].unsqueeze(1))
            
            # Add directional accuracy penalty
            direction_penalty = torch.mean(torch.abs(pred[:, 1] - target[:, 1]))
            
            return (self.price_weight * price_loss + 
                   self.direction_weight * (direction_loss + 0.1 * direction_penalty) + 
                   self.volatility_weight * volatility_loss)


class StudentTLoss(nn.Module):
    """
    Student's t-distribution loss for handling fat tails in financial data.
    """
    def __init__(self, df: float = 5.0):
        super().__init__()
        self.df = df  # Degrees of freedom
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Student's t-distribution loss.
        
        Args:
            pred: Predictions tensor
            target: Target tensor
            
        Returns:
            Scalar loss value
        """
        # Calculate squared error
        squared_error = (pred - target) ** 2
        
        # Student's t-distribution loss
        loss = torch.log(1 + squared_error / self.df)
        
        return loss.mean()


def get_model_config(input_size: int) -> Dict[str, Any]:
    """
    Get model configuration.
    
    Args:
        input_size: Number of input features
        
    Returns:
        Dictionary with model configuration
    """
    # Adjust hidden size based on input size
    hidden_size = 512 if input_size > 100 else 256
    projection_size = 128
    
    return {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'projection_size': projection_size,
        'num_layers': 3,
        'dropout': 0.3,
        'num_heads': 8,
        'output_size': 3,  # Price, direction, volatility
        'bidirectional': False
    }
