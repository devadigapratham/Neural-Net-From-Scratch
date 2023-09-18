import numpy as np
import matplotlib.pyplot as plt
from evaluate_model import evaluate_model
from load_data import load_data
from neural_network import Layer_Dense, Activation_ReLU, Activation_SoftMax

# Load the synthetic spiral data
X, y = load_data('spiral_data.pkl')

# Create and train the neural network model with the best hyperparameters
dense1 = Layer_Dense(2, 4)  # Use the best number of neurons found during tuning
activation1 = Activation_ReLU()
dense2 = Layer_Dense(4, 3)
activation2 = Activation_SoftMax()

for epoch in range(10001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

# Evaluate the model on test data
test_accuracy = evaluate_model((dense1, activation1, dense2, activation2), X, y)

# Visualize the data and decision boundaries
def plot_decision_boundary(X, y, model, title):
    dense1, activation1, dense2, activation2 = model
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    xx, yy = np.meshgrid(np.arange(-1.5, 1.5, 0.01), np.arange(-1.5, 1.5, 0.01))
    Z = np.argmax(activation2.output, axis=1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title(title)
    plt.show()

plot_decision_boundary(X, y, (dense1, activation1, dense2, activation2), f"Test Accuracy: {test_accuracy:.4f}")
