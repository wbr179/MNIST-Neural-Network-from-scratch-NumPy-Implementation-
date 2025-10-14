MNIST Neural Network from Scratch

This project implements a simple feedforward neural network from scratch using NumPy — no deep learning frameworks (like TensorFlow or PyTorch) are used.
It demonstrates the full machine learning workflow, including data loading, forward propagation, backpropagation, loss calculation, accuracy evaluation, and even early stopping and grid search for hyperparameter tuning.

Features

Loads the original MNIST dataset (digit images 0–9)

Implements:

Sigmoid activation and its derivative

Forward and backward propagation

Mini-batch gradient descent

Mean Squared Error (MSE) as the loss function

Includes:

Early stopping to prevent overfitting

Grid search for hyperparameter optimization

Visualization of training curves and misclassified images



Requirements

Install the dependencies using:

pip install numpy matplotlib idx2numpy pillow


How to Run

Download the MNIST dataset (in .gz format) and place it in the folder mnist_data/.
The expected files are:

train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
