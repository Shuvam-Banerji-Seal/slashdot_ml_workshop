"""
MNIST Data Module

Provides easy-to-use data loaders for MNIST dataset with proper preprocessing.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Optional, Tuple


class MNISTDataModule:
    """
    Data module for MNIST dataset with train/val/test splits.
    
    Attributes:
        data_dir: Directory to store/load data
        batch_size: Batch size for data loaders
        val_split: Fraction of training data for validation
        num_workers: Number of workers for data loading
    """
    
    # MNIST statistics (precomputed)
    MEAN = 0.1307
    STD = 0.3081
    
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        val_split: float = 0.1,
        num_workers: int = 4,
        augment: bool = False
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.augment = augment
        
        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None
        self.test_dataset: Optional[torch.utils.data.Dataset] = None
        
    def _get_transforms(self, train: bool = True) -> transforms.Compose:
        """Get data transformations."""
        transform_list = []
        
        if train and self.augment:
            transform_list.extend([
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((self.MEAN,), (self.STD,))
        ])
        
        return transforms.Compose(transform_list)
    
    def setup(self):
        """Download and setup datasets."""
        # Training data with augmentation
        full_train = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self._get_transforms(train=True)
        )
        
        # Split into train and validation
        val_size = int(len(full_train) * self.val_split)
        train_size = len(full_train) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Test data (no augmentation)
        self.test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self._get_transforms(train=False)
        )
        
    def train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample batch for visualization."""
        loader = self.train_dataloader()
        return next(iter(loader))


if __name__ == "__main__":
    # Quick test
    dm = MNISTDataModule(batch_size=32)
    dm.setup()
    
    print(f"Training samples: {len(dm.train_dataset)}")
    print(f"Validation samples: {len(dm.val_dataset)}")
    print(f"Test samples: {len(dm.test_dataset)}")
    
    images, labels = dm.get_sample_batch()
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
