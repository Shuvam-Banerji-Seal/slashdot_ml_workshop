"""
ML Workshop - From Counting Numbers to Neural Networks

A comprehensive machine learning workshop covering mathematical foundations
through neural network implementation.

Author: Shuvam Banerji Seal
"""

__version__ = "0.1.0"
__author__ = "Shuvam Banerji Seal"

from .data import MNISTDataModule
from .models import MLP, CNNClassifier
from .training import Trainer

__all__ = ["MNISTDataModule", "MLP", "CNNClassifier", "Trainer"]
