"""
Convolutional Neural Network (CNN) implementation.

A CNN architecture for image classification tasks.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class CNNClassifier(nn.Module):
    """
    Convolutional Neural Network for image classification.
    
    Architecture:
        Conv Block 1: Conv -> BatchNorm -> ReLU -> MaxPool
        Conv Block 2: Conv -> BatchNorm -> ReLU -> MaxPool
        Classifier: Flatten -> Linear -> ReLU -> Dropout -> Linear
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        
    Example:
        >>> model = CNNClassifier(num_classes=10)
        >>> x = torch.randn(32, 1, 28, 28)
        >>> output = model(x)  # Shape: (32, 10)
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        conv_channels: List[int] = [32, 64],
        fc_hidden: int = 128,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        )
        
        # Calculate flattened size (for 28x28 input)
        self._flat_size = conv_channels[1] * 7 * 7
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.classifier(x)
        return x
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_feature_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get intermediate feature maps for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of feature maps from conv1 and conv2
        """
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        return f1, f2


class LeNet5(nn.Module):
    """
    LeNet-5: The classic CNN architecture by Yann LeCun (1998).
    
    Original architecture for digit recognition.
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            # C1: 6 feature maps 28x28 (with padding)
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            # S2: 6 feature maps 14x14
            nn.AvgPool2d(2, 2),
            # C3: 16 feature maps 10x10
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            # S4: 16 feature maps 5x5
            nn.AvgPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Test CNN
    model = CNNClassifier(num_classes=10)
    print(f"Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(32, 1, 28, 28)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test feature maps
    f1, f2 = model.get_feature_maps(x)
    print(f"Feature map 1 shape: {f1.shape}")
    print(f"Feature map 2 shape: {f2.shape}")
