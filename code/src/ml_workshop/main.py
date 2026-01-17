"""
ML Workshop - Main Entry Point

A comprehensive machine learning workshop demonstrating the complete pipeline
from mathematical foundations to neural network training.

Author: Shuvam Banerji Seal
"""

import torch
import torch.nn as nn
from pathlib import Path

from ml_workshop.data import MNISTDataModule
from ml_workshop.models import MLP, CNNClassifier
from ml_workshop.training import Trainer
from ml_workshop.visualization import plot_training_history, plot_mnist_samples


def main():
    """Main training script."""
    print("=" * 60)
    print("ML Workshop - MNIST Classification")
    print("Author: Shuvam Banerji Seal")
    print("=" * 60)
    
    # Configuration
    config = {
        "batch_size": 64,
        "epochs": 10,
        "learning_rate": 0.001,
        "model": "cnn",  # or "mlp"
        "hidden_sizes": [256, 128],  # for MLP
        "dropout": 0.2,
        "early_stopping": 5,
        "checkpoint_dir": "./checkpoints",
        "output_dir": "./outputs",
    }
    
    # Create output directories
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Setup data
    print("\n📊 Loading MNIST dataset...")
    dm = MNISTDataModule(batch_size=config["batch_size"])
    dm.setup()
    
    print(f"  Training samples: {len(dm.train_dataset):,}")
    print(f"  Validation samples: {len(dm.val_dataset):,}")
    print(f"  Test samples: {len(dm.test_dataset):,}")
    
    # Create model
    print(f"\n🧠 Creating {config['model'].upper()} model...")
    if config["model"] == "mlp":
        model = MLP(
            input_size=784,
            hidden_sizes=config["hidden_sizes"],
            num_classes=10,
            dropout_rate=config["dropout"]
        )
    else:
        model = CNNClassifier(
            num_classes=10,
            dropout_rate=config["dropout"]
        )
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    trainer = Trainer(model, criterion, optimizer)
    
    # Train
    print(f"\n🚀 Training for {config['epochs']} epochs...")
    history = trainer.fit(
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        epochs=config["epochs"],
        early_stopping=config["early_stopping"],
        checkpoint_dir=config["checkpoint_dir"]
    )
    
    # Evaluate on test set
    print("\n📈 Evaluating on test set...")
    results = trainer.evaluate(dm.test_dataloader())
    print(f"  Test Loss: {results['loss']:.4f}")
    print(f"  Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    # Plot training history
    print("\n📊 Saving training history plot...")
    history_dict = {
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
        "train_acc": history.train_acc,
        "val_acc": history.val_acc,
    }
    fig = plot_training_history(history_dict, save_path=f"{config['output_dir']}/training_history.png")
    
    # Show some predictions
    print("\n🔮 Generating sample predictions...")
    images, labels = dm.get_sample_batch()
    predictions = trainer.predict(images)
    fig = plot_mnist_samples(
        images[:16], labels[:16], predictions[:16],
        save_path=f"{config['output_dir']}/sample_predictions.png"
    )
    
    print("\n✅ Training complete!")
    print(f"   Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"   Checkpoints saved to: {config['checkpoint_dir']}")
    print(f"   Plots saved to: {config['output_dir']}")
    
    return trainer, history


if __name__ == "__main__":
    trainer, history = main()
