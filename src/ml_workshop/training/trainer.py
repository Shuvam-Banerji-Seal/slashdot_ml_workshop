"""
Training module with complete training loop implementation.

Provides a clean, reusable trainer class for neural network training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time


@dataclass
class TrainingHistory:
    """Container for training history metrics."""
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)


class Trainer:
    """
    Neural Network Trainer with training loop, evaluation, and checkpointing.
    
    Features:
        - Automatic device selection (CPU/CUDA/MPS)
        - Learning rate scheduling
        - Early stopping
        - Model checkpointing
        - Training history tracking
    
    Args:
        model: PyTorch model to train
        criterion: Loss function
        optimizer: Optimizer (if None, Adam with lr=0.001 is used)
        scheduler: Learning rate scheduler (optional)
        device: Device to train on (auto-detected if None)
        
    Example:
        >>> model = MLP(784, [256, 128], 10)
        >>> trainer = Trainer(model, nn.CrossEntropyLoss())
        >>> history = trainer.fit(train_loader, val_loader, epochs=10)
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None
    ):
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        
        # Default optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Training state
        self.history = TrainingHistory()
        self.best_val_acc = 0.0
        self.current_epoch = 0
        
        print(f"Training on: {device}")
        print(f"Model parameters: {self._count_parameters():,}")
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)
        
        return total_loss / total, correct / total
    
    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data)
            loss = self.criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)
        
        return total_loss / total, correct / total
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        early_stopping: int = 0,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True
    ) -> TrainingHistory:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            early_stopping: Stop after N epochs without improvement (0 = disabled)
            checkpoint_dir: Directory to save checkpoints (optional)
            verbose: Print progress
            
        Returns:
            TrainingHistory with all metrics
        """
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        no_improve_count = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                val_loss, val_acc = self._evaluate(val_loader)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Step scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Record history
            epoch_time = time.time() - start_time
            self.history.train_loss.append(train_loss)
            self.history.train_acc.append(train_acc)
            self.history.val_loss.append(val_loss)
            self.history.val_acc.append(val_acc)
            self.history.learning_rates.append(current_lr)
            self.history.epoch_times.append(epoch_time)
            
            # Checkpointing
            if val_loader and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                no_improve_count = 0
                if checkpoint_dir:
                    self.save_checkpoint(f"{checkpoint_dir}/best_model.pth")
            else:
                no_improve_count += 1
            
            # Logging
            if verbose:
                msg = f"Epoch {epoch+1:3d}/{epochs}"
                msg += f" | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
                if val_loader:
                    msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                msg += f" | Time: {epoch_time:.1f}s"
                print(msg)
            
            # Early stopping
            if early_stopping > 0 and no_improve_count >= early_stopping:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with loss and accuracy
        """
        test_loss, test_acc = self._evaluate(test_loader)
        return {"loss": test_loss, "accuracy": test_acc}
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.history = checkpoint.get("history", TrainingHistory())
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class labels
        """
        self.model.eval()
        x = x.to(self.device)
        output = self.model(x)
        return output.argmax(dim=1)
    
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Softmax probabilities
        """
        self.model.eval()
        x = x.to(self.device)
        output = self.model(x)
        return torch.softmax(output, dim=1)


if __name__ == "__main__":
    # Quick test
    from ml_workshop.models import MLP
    from ml_workshop.data import MNISTDataModule
    
    # Setup
    dm = MNISTDataModule(batch_size=64)
    dm.setup()
    
    model = MLP(784, [128, 64], 10)
    trainer = Trainer(model, nn.CrossEntropyLoss())
    
    # Train for 2 epochs as a test
    history = trainer.fit(
        dm.train_dataloader(),
        dm.val_dataloader(),
        epochs=2
    )
    
    # Test
    results = trainer.evaluate(dm.test_dataloader())
    print(f"Test accuracy: {results['accuracy']:.4f}")
