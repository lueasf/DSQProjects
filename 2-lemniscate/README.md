# 2. Lemniscate : 2D Shape Classification with PyTorch

This project demonstrates a neural network's ability to learn and classify complex 2D shapes (a combination of lemniscate and ellipse) using PyTorch.

## Project Description

The goal is to train a neural network to classify points in a 2D plane into two categories:
- Class 1 (Red): Points with f(x,y) <= 0 
- Class 0 (Blue): Points with f(x,y) > 0

The target function combines:
1. A lemniscate : `(x² + y²)² - 4(x² - y²)`
2. An ellipse: `(x - 0.5)² + 4(y - 1/3)² - 2`

### Requirements
- Python 3
- PyTorch
- NumPy
- Matplotlib

### Key Components

1. **Data Generation**:
   - Creates a 200×200 grid in [-2,2]×[-1,1] space
   - Labels points based on the combined shape function

2. **Neural Network Architecture**:
   - Input layer: 2 neurons (x,y coordinates)
   - 3 hidden layers with ReLU activation (20 neurons each)
   - Output layer: 1 neuron with sigmoid activation

3. **Training Configuration**:
   - Loss: Binary Cross Entropy (BCELoss)
   - Optimizer: Stochastic Gradient Descent (SGD)
   - Learning rate: 0.1
   - Batch size: 256
   - Epochs: 500
