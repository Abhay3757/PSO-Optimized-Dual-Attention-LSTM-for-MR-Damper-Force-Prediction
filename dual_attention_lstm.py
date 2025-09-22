"""
Dual-Attention LSTM Model for MR Damper Force Prediction

This module implements a sophisticated LSTM model with dual attention mechanisms:
1. Feature Attention: Learn importance weights of input variables at each timestep
2. Temporal Attention: Learn importance weights of past timesteps for current prediction

The model is designed for interpretability and high performance in time-series prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import math


class FeatureAttention(nn.Module):
    """
    Feature Attention Mechanism
    
    Learns importance weights for input features at each timestep.
    This allows the model to focus on the most relevant features for prediction.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Initialize Feature Attention
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension for attention computation
        """
        super(FeatureAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Attention network
        self.attention_linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for feature attention
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        # Compute attention weights for each feature at each timestep
        attention_weights = self.attention_linear(x)  # (batch_size, seq_len, input_dim)
        
        # Apply attention weights
        attended_features = x * attention_weights  # Element-wise multiplication
        
        return attended_features, attention_weights


class TemporalAttention(nn.Module):
    """
    Temporal Attention Mechanism
    
    Learns importance weights for different timesteps in the sequence.
    This allows the model to focus on the most relevant past timesteps.
    """
    
    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        """
        Initialize Temporal Attention
        
        Args:
            hidden_dim: Hidden dimension of LSTM outputs
            attention_dim: Dimension for attention computation
        """
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Attention components
        self.key_layer = nn.Linear(hidden_dim, attention_dim)
        self.query_layer = nn.Linear(hidden_dim, attention_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)
        self.attention_weights = nn.Linear(attention_dim, 1)
        
    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for temporal attention
        
        Args:
            lstm_outputs: LSTM outputs of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        batch_size, seq_len, hidden_dim = lstm_outputs.shape
        
        # Generate keys, queries, and values
        keys = self.key_layer(lstm_outputs)  # (batch_size, seq_len, attention_dim)
        queries = self.query_layer(lstm_outputs)  # (batch_size, seq_len, attention_dim)
        values = self.value_layer(lstm_outputs)  # (batch_size, seq_len, hidden_dim)
        
        # Compute attention scores
        # Use the last timestep's query to attend over all timesteps
        last_query = queries[:, -1:, :]  # (batch_size, 1, attention_dim)
        
        # Compute scaled dot-product attention
        scores = torch.matmul(last_query, keys.transpose(-2, -1))  # (batch_size, 1, seq_len)
        scores = scores / math.sqrt(self.attention_dim)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, 1, seq_len)
        
        # Compute context vector as weighted sum of values
        context_vector = torch.matmul(attention_weights, values)  # (batch_size, 1, hidden_dim)
        context_vector = context_vector.squeeze(1)  # (batch_size, hidden_dim)
        
        return context_vector, attention_weights.squeeze(1)  # (batch_size, seq_len)


class DualAttentionLSTM(nn.Module):
    """
    Dual-Attention LSTM Model
    
    Combines LSTM with both feature and temporal attention mechanisms
    for enhanced performance and interpretability in time-series prediction.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 feature_attention_dim: int = 64,
                 temporal_attention_dim: int = 64,
                 output_dim: int = 1):
        """
        Initialize Dual-Attention LSTM
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            feature_attention_dim: Hidden dimension for feature attention
            temporal_attention_dim: Hidden dimension for temporal attention
            output_dim: Output dimension (1 for force prediction)
        """
        super(DualAttentionLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Feature Attention
        self.feature_attention = FeatureAttention(
            input_dim=input_dim,
            hidden_dim=feature_attention_dim
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Temporal Attention
        self.temporal_attention = TemporalAttention(
            hidden_dim=hidden_dim,
            attention_dim=temporal_attention_dim
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'weight' in name and len(param.shape) >= 2:
                torch.nn.init.xavier_uniform_(param.data)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary containing:
                - output: Predicted values
                - feature_attention_weights: Feature attention weights
                - temporal_attention_weights: Temporal attention weights
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Apply feature attention
        attended_features, feature_attention_weights = self.feature_attention(x)
        
        # Pass through LSTM
        lstm_outputs, (hidden, cell) = self.lstm(attended_features)
        
        # Apply temporal attention
        context_vector, temporal_attention_weights = self.temporal_attention(lstm_outputs)
        
        # Apply dropout
        context_vector = self.dropout(context_vector)
        
        # Generate final output
        output = self.output_projection(context_vector)
        
        return {
            'output': output,
            'feature_attention_weights': feature_attention_weights,
            'temporal_attention_weights': temporal_attention_weights,
            'lstm_outputs': lstm_outputs,
            'context_vector': context_vector
        }
    
    def get_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention weights for interpretability analysis
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary with feature and temporal attention weights
        """
        with torch.no_grad():
            results = self.forward(x)
            return {
                'feature_attention': results['feature_attention_weights'],
                'temporal_attention': results['temporal_attention_weights']
            }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions without attention weights
        
        Args:
            x: Input tensor
            
        Returns:
            Predictions only
        """
        with torch.no_grad():
            results = self.forward(x)
            return results['output']


class ModelTrainer:
    """
    Training utilities for the Dual-Attention LSTM model
    """
    
    def __init__(self, model: DualAttentionLSTM, device: torch.device):
        """
        Initialize trainer
        
        Args:
            model: The model to train
            device: Device to use for training
        """
        self.model = model.to(device)
        self.device = device
        
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def estimate_inference_time(self, sample_input: torch.Tensor, num_runs: int = 100) -> float:
        """
        Estimate inference time in milliseconds
        
        Args:
            sample_input: Sample input tensor
            num_runs: Number of runs for timing
            
        Returns:
            Average inference time in milliseconds
        """
        self.model.eval()
        sample_input = sample_input.to(self.device)
        
        # Warm up
        for _ in range(10):
            _ = self.model.predict(sample_input)
        
        # Time inference
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        
        if self.device.type == 'cuda':
            start_time.record()
            for _ in range(num_runs):
                _ = self.model.predict(sample_input)
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time) / num_runs
        else:
            import time
            start = time.time()
            for _ in range(num_runs):
                _ = self.model.predict(sample_input)
            end = time.time()
            elapsed_time = (end - start) * 1000 / num_runs
        
        return elapsed_time


def create_model(config: Dict) -> DualAttentionLSTM:
    """
    Create a Dual-Attention LSTM model from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model = DualAttentionLSTM(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        feature_attention_dim=config.get('feature_attention_dim', 64),
        temporal_attention_dim=config.get('temporal_attention_dim', 64),
        output_dim=config.get('output_dim', 1)
    )
    
    return model


def main():
    """Test the Dual-Attention LSTM model"""
    # Test configuration
    config = {
        'input_dim': 4,  # A, D, Y, V
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'feature_attention_dim': 64,
        'temporal_attention_dim': 64,
        'output_dim': 1
    }
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config)
    trainer = ModelTrainer(model, device)
    
    print(f"Model created on device: {device}")
    print(f"Total parameters: {trainer.count_parameters():,}")
    print(f"Model size: {trainer.get_model_size_mb():.2f} MB")
    
    # Test with dummy data
    batch_size, seq_len, input_dim = 32, 20, 4
    dummy_input = torch.randn(batch_size, seq_len, input_dim).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        results = model(dummy_input)
        
    print(f"\nTest Forward Pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {results['output'].shape}")
    print(f"Feature attention shape: {results['feature_attention_weights'].shape}")
    print(f"Temporal attention shape: {results['temporal_attention_weights'].shape}")
    
    # Test inference time
    inference_time = trainer.estimate_inference_time(dummy_input[:1])
    print(f"Average inference time: {inference_time:.2f} ms")
    
    # Test attention weights
    attention_weights = model.get_attention_weights(dummy_input[:1])
    print(f"\nAttention Analysis:")
    print(f"Feature attention (sample): {attention_weights['feature_attention'][0, -1, :].cpu().numpy()}")
    print(f"Temporal attention (sample): {attention_weights['temporal_attention'][0, :].cpu().numpy()}")


if __name__ == "__main__":
    main()