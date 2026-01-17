# ML Workshop: From Numbers to Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LaTeX](https://img.shields.io/badge/LaTeX-Beamer-green.svg)](https://www.latex-project.org/)

A comprehensive machine learning workshop covering fundamental mathematics through neural network implementation. This repository contains both presentation materials (LaTeX Beamer slides) and hands-on Python code demonstrating concepts from first principles.

**Organized by:** Slashdot - The Programming Club  
**Workshop Lead:** Shuvam Banerji Seal (sbs22ms076@iiserkol.ac.in)

## 📚 Overview

This workshop takes participants on a journey from basic mathematical foundations to implementing neural networks from scratch. The material is designed to build intuition and deep understanding rather than just using libraries.

### What You'll Learn

- **Mathematical Foundations** (Sections 1-6)
  - Number systems and complex numbers
  - Functions and parameters
  - Limits and continuity
  - Differentiation and chain rule
  - Integration and fundamental theorem of calculus

- **Graph Theory & Setup** (Sections 7-8)
  - Graph theory basics for neural networks
  - Modern Python environment setup with UV

- **Neural Networks** (Sections 9-15)
  - Neural network architecture
  - Forward propagation
  - Loss functions
  - Backpropagation algorithm
  - Optimization algorithms (SGD, Momentum, Adam)
  - Regularization techniques (L1/L2, Dropout)

- **Classical ML & Theory** (Sections 16-21)
  - Decision trees and ensemble methods
  - Boosting algorithms
  - Statistical learning theory
  - Information theory and KL divergence
  - MNIST dataset and training pipeline

## 🗂️ Repository Structure

```
ml_workshop/
├── presentation/          # LaTeX Beamer presentation
│   ├── day_0/            # Main workshop slides
│   │   ├── main.tex      # Main presentation file
│   │   └── sections/     # Individual section files
│   └── global/           # Custom Beamer theme
│       └── sbs_dark_beamer.sty
│
├── code/                 # Python implementation
│   ├── notebooks/        # Jupyter notebooks with visualizations
│   ├── src/             # Source code modules
│   │   ├── activations.py
│   │   ├── layers.py
│   │   ├── network.py
│   │   ├── losses.py
│   │   ├── optimizers.py
│   │   └── utils.py
│   └── examples/        # Example scripts
│
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.11 or higher
- **UV**: Modern Python package manager
- **LaTeX**: Full TeXLive distribution with Beamer
- **Git**: For cloning the repository

### Setting Up Python Environment

Using UV (recommended):

```bash
# Clone the repository
git clone git@github.com:Shuvam-Banerji-Seal/slashdot_ml_workshop.git
cd slashdot_ml_workshop

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Building the Presentation

```bash
cd presentation/day_0
pdflatex -shell-escape -interaction=nonstopmode main.tex
pdflatex -shell-escape -interaction=nonstopmode main.tex  # Second pass for references
```

The compiled PDF will be `main.pdf` (251 pages).

### Running Code Examples

```bash
# Run a basic neural network example
python code/examples/mnist_from_scratch.py

# Launch Jupyter notebooks
jupyter lab code/notebooks/
```

## 📖 Presentation Features

The presentation uses a custom dark cyberpunk-themed Beamer style with:

- **Neon color scheme** optimized for projection
- **Animated TikZ diagrams** for visualizing concepts
- **Interactive examples** with code snippets
- **Mathematical rigor** with theorem boxes and proofs
- **Fun facts** to maintain engagement

### Key Sections

1. **Numbers & Counting** - From natural numbers to complex numbers
2. **Real & Complex Numbers** - Completeness and Euler's formula
3. **Functions & Parameters** - Domain, codomain, and function composition
4. **Limits** - ε-δ definition and limit laws
5. **Differentiation** - Derivatives and the chain rule
6. **Integration** - Fundamental theorem of calculus
7. **Graph Theory** - Graphs as neural network foundations
8. **Python Setup** - Modern development environment with UV
9. **Neural Network Intro** - Perceptrons and activation functions
10. **Neural Architectures** - Layer types and network design
11. **Forward Propagation** - How data flows through networks
12. **Loss Functions** - MSE, cross-entropy, and their derivatives
13. **Backpropagation** - Computing gradients via chain rule
14. **Optimization** - Gradient descent, momentum, and Adam
15. **Regularization** - Preventing overfitting
16. **Decision Trees** - Classical ML algorithms
17. **Boosting** - Ensemble methods and XGBoost
18. **Statistical Learning** - Bias-variance tradeoff
19. **KL Divergence** - Information theory for ML
20. **MNIST Dataset** - The "Hello World" of ML
21. **Training Pipeline** - Putting it all together

## 💻 Code Implementation

The Python code demonstrates:

- **Neural networks from scratch** (no PyTorch/TensorFlow initially)
- **Vectorized operations** with NumPy
- **Modular design** for easy experimentation
- **Type hints** for clarity
- **Comprehensive tests** for correctness
- **Jupyter notebooks** for interactive exploration

### Example: Simple Neural Network

```python
from src.network import Network
from src.layers import Dense
from src.activations import ReLU, Softmax
from src.losses import CrossEntropyLoss

# Create a simple network
model = Network([
    Dense(784, 128),
    ReLU(),
    Dense(128, 64),
    ReLU(),
    Dense(64, 10),
    Softmax()
])

# Train on MNIST
model.train(X_train, y_train, 
           loss_fn=CrossEntropyLoss(),
           epochs=10, 
           batch_size=32,
           lr=0.001)
```

## 🎯 Learning Objectives

By the end of this workshop, participants will:

1. ✅ Understand the mathematical foundations of neural networks
2. ✅ Implement backpropagation from scratch
3. ✅ Grasp why neural networks work (universal approximation)
4. ✅ Train models on real datasets (MNIST)
5. ✅ Debug and optimize neural network training
6. ✅ Understand modern optimization algorithms
7. ✅ Apply regularization techniques effectively

## 📚 Additional Resources

### Books
- *Deep Learning* by Goodfellow, Bengio, and Courville
- *Neural Networks and Deep Learning* by Michael Nielsen
- *Pattern Recognition and Machine Learning* by Christopher Bishop

### Online Courses
- [3Blue1Brown Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Stanford CS231n](http://cs231n.stanford.edu/)

### Papers
- *Attention Is All You Need* (Vaswani et al., 2017)
- *Adam: A Method for Stochastic Optimization* (Kingma & Ba, 2014)
- *Understanding Deep Learning Requires Rethinking Generalization* (Zhang et al., 2016)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contact

**Shuvam Banerji Seal**  
Email: sbs22ms076@iiserkol.ac.in  
GitHub: [@Shuvam-Banerji-Seal](https://github.com/Shuvam-Banerji-Seal)

**Slashdot - The Programming Club**  
IISER Kolkata

## 🙏 Acknowledgments

- **3Blue1Brown** for inspiring visualizations and pedagogy
- **Manim** community for animation framework examples
- **Fast.ai** for democratizing deep learning education
- **IISER Kolkata** for supporting this initiative

---

**Star ⭐ this repository if you find it helpful!**

Happy Learning! 🚀🧠
