# ML Workshop Codebase

This directory contains the Python implementation for the ML Workshop.

## Setup Instructions

### Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the code directory
cd code

# Create virtual environment and install dependencies
uv sync

# Run a Python script
uv run python src/ml_workshop/main.py

# Or activate the environment manually
source .venv/bin/activate
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -e .
```

## Project Structure

```
code/
├── pyproject.toml          # Project configuration
├── README.md               # This file
├── src/
│   └── ml_workshop/        # Main package
│       ├── __init__.py
│       ├── main.py         # Entry point
│       ├── data/           # Data loading utilities
│       ├── models/         # Neural network models
│       ├── training/       # Training utilities
│       └── visualization/  # Plotting functions
├── notebooks/              # Jupyter notebooks
│   ├── 01_numbers_and_functions.ipynb
│   ├── 02_calculus_foundations.ipynb
│   ├── 03_neural_network_basics.ipynb
│   ├── 04_training_mnist.ipynb
│   └── 05_decision_trees_boosting.ipynb
└── tests/                  # Unit tests
```

## Running Notebooks

```bash
# Start Jupyter
uv run jupyter notebook notebooks/

# Or with Jupyter Lab
uv run jupyter lab notebooks/
```

## Author

Shuvam Banerji Seal
