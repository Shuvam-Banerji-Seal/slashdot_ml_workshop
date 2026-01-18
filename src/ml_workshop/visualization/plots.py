"""
Plotting utilities for ML Workshop visualizations.

Provides beautiful, consistent visualizations for machine learning concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn


# Set style for nice plots
plt.style.use('dark_background')
COLORS = {
    'primary': '#00d9ff',      # Cyan
    'secondary': '#ff6b9d',    # Pink  
    'accent': '#c8ff00',       # Yellow-green
    'warning': '#ffcc00',      # Yellow
    'success': '#00ff88',      # Green
    'error': '#ff4444',        # Red
    'bg': '#1a1a2e',           # Dark blue
    'grid': '#333355',         # Grid color
}


def set_plot_style():
    """Apply consistent styling to plots."""
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['bg'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.3,
        'font.size': 12,
    })


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], color=COLORS['primary'], 
             label='Train Loss', linewidth=2)
    if 'val_loss' in history and any(history['val_loss']):
        ax1.plot(epochs, history['val_loss'], color=COLORS['secondary'], 
                 label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], color=COLORS['primary'], 
             label='Train Acc', linewidth=2)
    if 'val_acc' in history and any(history['val_acc']):
        ax2.plot(epochs, history['val_acc'], color=COLORS['secondary'], 
                 label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix
    
    set_plot_style()
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap='viridis')
    
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")
    
    # Add labels
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_mnist_samples(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    n_samples: int = 16,
    figsize: Tuple[int, int] = (12, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot grid of MNIST samples.
    
    Args:
        images: Batch of images (N, C, H, W)
        labels: True labels
        predictions: Predicted labels (optional)
        n_samples: Number of samples to show
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    n_samples = min(n_samples, len(images))
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_samples):
        img = images[i].squeeze().numpy()
        ax = axes[i]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        title = f"Label: {labels[i].item()}"
        color = 'white'
        if predictions is not None:
            pred = predictions[i].item()
            title += f"\nPred: {pred}"
            if pred != labels[i].item():
                color = COLORS['error']
            else:
                color = COLORS['success']
        
        ax.set_title(title, fontsize=10, color=color)
    
    # Hide remaining axes
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('MNIST Samples', fontsize=14, color='white')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_decision_boundary(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    figsize: Tuple[int, int] = (10, 8),
    resolution: int = 100,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot decision boundary for a 2D classification model.
    
    Args:
        model: PyTorch model (expects 2D input)
        X: Data points (N, 2)
        y: Labels
        figsize: Figure size
        resolution: Grid resolution
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        Z = model(grid)
        if Z.shape[1] > 1:
            Z = Z.argmax(dim=1)
        else:
            Z = (Z > 0).float().squeeze()
        Z = Z.numpy().reshape(xx.shape)
    
    # Plot decision boundary
    cmap = ListedColormap([COLORS['primary'], COLORS['secondary'], COLORS['accent']][:len(np.unique(y))])
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='white', s=50)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Boundary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_function(
    f,
    x_range: Tuple[float, float] = (-5, 5),
    n_points: int = 200,
    derivative: bool = False,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Function Plot",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a function and optionally its derivative.
    
    Args:
        f: Function to plot (callable)
        x_range: Range of x values
        n_points: Number of points
        derivative: Whether to also plot numerical derivative
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = f(x)
    
    ax.plot(x, y, color=COLORS['primary'], linewidth=2, label='f(x)')
    
    if derivative:
        dx = x[1] - x[0]
        dy = np.gradient(y, dx)
        ax.plot(x, dy, color=COLORS['secondary'], linewidth=2, 
                linestyle='--', label="f'(x)")
    
    ax.axhline(y=0, color='white', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='white', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_gradient_descent(
    f,
    start: float = 2.0,
    learning_rate: float = 0.1,
    n_steps: int = 20,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize gradient descent optimization.
    
    Args:
        f: Function to minimize (callable)
        start: Starting point
        learning_rate: Step size
        n_steps: Number of steps
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Perform gradient descent
    x = start
    history = [x]
    
    for _ in range(n_steps):
        # Numerical gradient
        h = 1e-5
        grad = (f(x + h) - f(x - h)) / (2 * h)
        x = x - learning_rate * grad
        history.append(x)
    
    history = np.array(history)
    
    # Plot function with descent path
    x_plot = np.linspace(min(history) - 1, max(start, 3) + 1, 200)
    y_plot = f(x_plot)
    
    ax1.plot(x_plot, y_plot, color=COLORS['primary'], linewidth=2)
    ax1.scatter(history, f(history), c=range(len(history)), cmap='plasma', 
                s=100, zorder=5, edgecolors='white')
    ax1.plot(history, f(history), 'w--', alpha=0.5, linewidth=1)
    ax1.scatter(history[-1], f(history[-1]), c=COLORS['success'], s=200, 
                zorder=6, marker='*')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Gradient Descent on f(x)')
    ax1.grid(True, alpha=0.3)
    
    # Plot convergence
    ax2.plot(range(len(history)), f(history), color=COLORS['secondary'], 
             linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('f(x)')
    ax2.set_title('Convergence')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_loss_landscape(
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a 3D loss landscape visualization.
    
    Args:
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create loss surface
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Example loss function: sum of two Gaussians (multiple minima)
    Z = 3 * (1-X)**2 * np.exp(-(X**2) - (Y+1)**2) \
        - 10 * (X/5 - X**3 - Y**5) * np.exp(-X**2 - Y**2) \
        - 1/3 * np.exp(-(X+1)**2 - Y**2)
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8, 
                           linewidth=0, antialiased=True)
    
    ax.set_xlabel('Weight 1')
    ax.set_ylabel('Weight 2')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Landscape')
    
    # Adjust view
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_neural_network_diagram(
    layer_sizes: List[int] = [4, 8, 6, 2],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Draw a neural network architecture diagram.
    
    Args:
        layer_sizes: Number of neurons in each layer
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)
    
    layer_colors = [COLORS['primary'], COLORS['accent'], COLORS['accent'], COLORS['secondary']]
    if len(layer_colors) < n_layers:
        layer_colors = layer_colors + [COLORS['accent']] * (n_layers - len(layer_colors))
    
    # Draw layers
    for layer_idx, n_neurons in enumerate(layer_sizes):
        x = layer_idx / (n_layers - 1) if n_layers > 1 else 0.5
        
        # Center neurons vertically
        y_start = 0.5 - (n_neurons - 1) / (2 * max_neurons)
        y_spacing = 1 / max_neurons if n_neurons > 1 else 0
        
        for neuron_idx in range(n_neurons):
            y = y_start + neuron_idx * y_spacing
            
            # Draw neuron
            circle = plt.Circle((x, y), 0.02, color=layer_colors[layer_idx], 
                               ec='white', linewidth=1.5, zorder=3)
            ax.add_patch(circle)
            
            # Draw connections to next layer
            if layer_idx < n_layers - 1:
                next_n = layer_sizes[layer_idx + 1]
                next_x = (layer_idx + 1) / (n_layers - 1)
                next_y_start = 0.5 - (next_n - 1) / (2 * max_neurons)
                next_y_spacing = 1 / max_neurons if next_n > 1 else 0
                
                for next_neuron_idx in range(next_n):
                    next_y = next_y_start + next_neuron_idx * next_y_spacing
                    ax.plot([x, next_x], [y, next_y], 'w-', alpha=0.2, 
                           linewidth=0.5, zorder=1)
    
    # Add layer labels
    labels = ['Input'] + [f'Hidden {i+1}' for i in range(n_layers - 2)] + ['Output']
    for i, label in enumerate(labels):
        x = i / (n_layers - 1) if n_layers > 1 else 0.5
        ax.text(x, -0.1, label, ha='center', va='top', fontsize=12, color='white')
        ax.text(x, 1.05, f'{layer_sizes[i]} neurons', ha='center', va='bottom', 
               fontsize=10, color='gray')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Neural Network Architecture', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Test plots
    print("Testing visualization functions...")
    
    # Test function plot
    fig1 = plot_function(lambda x: x**2 - 4*x + 3, derivative=True, 
                        title="Quadratic Function")
    plt.show()
    
    # Test gradient descent
    fig2 = plot_gradient_descent(lambda x: x**2, start=3.0, learning_rate=0.3)
    plt.show()
    
    # Test neural network diagram
    fig3 = plot_neural_network_diagram([784, 256, 128, 10])
    plt.show()
    
    print("All visualizations working!")
