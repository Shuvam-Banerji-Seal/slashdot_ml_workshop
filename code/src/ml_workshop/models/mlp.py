"""
Multi-Layer Perceptron (MLP) implementation.

A simple but effective feedforward neural network for classification tasks.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for classification.
    
    Architecture: Input -> [Linear -> Activation -> Dropout] x N -> Output
    
    Args:
        input_size: Size of input features (784 for flattened MNIST)
        hidden_sizes: List of hidden layer sizes, e.g., [256, 128]
        num_classes: Number of output classes (10 for MNIST)
        dropout_rate: Dropout probability (default: 0.2)
        activation: Activation function ('relu', 'leaky_relu', 'elu', 'gelu')
    
    Example:
        >>> model = MLP(784, [256, 128], 10)
        >>> x = torch.randn(32, 1, 28, 28)  # Batch of MNIST images
        >>> output = model(x)  # Shape: (32, 10)
    """
    
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: List[int] = [256, 128],
        num_classes: int = 10,
        dropout_rate: float = 0.2,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.input_size = input_size
        self.flatten = nn.Flatten()
        
        # Choose activation function
        activations = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
        }
        act_fn = activations.get(activation, nn.ReLU)
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                act_fn(),
                nn.Dropout(dropout_rate),
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width) or (batch, features)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        x = self.flatten(x)
        return self.layers(x)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        return f"MLP(input_size={self.input_size}, params={self.count_parameters():,})"


class MLPWithBatchNorm(nn.Module):
    """
    MLP with Batch Normalization for improved training stability.
    
    Architecture: Input -> [Linear -> BatchNorm -> Activation -> Dropout] x N -> Output
    """
    
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: List[int] = [256, 128],
        num_classes: int = 10,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        
        self.flatten = nn.Flatten()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.layers(x)


if __name__ == "__main__":
    # Test MLP
    model = MLP(784, [256, 128, 64], 10)
    print(model)
    print(f"Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(32, 1, 28, 28)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
