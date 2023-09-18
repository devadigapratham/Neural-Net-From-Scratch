import os
import numpy as np
from generate_spiral_data import create_spiral_data, save_data
from load_data import load_data
from neural_network import Layer_Dense, Activation_ReLU, Activation_SoftMax, Loss_CategoricalCrossEntropy, Optimizer_SGD
from evaluate_model import evaluate_model
import matplotlib.pyplot as plt
from visualize_data import plot_decision_boundary

if not os.path.exists('spiral_data.pkl'):
    X, y = create_spiral_data(samples=1000, classes=3)
    save_data(X, y, 'spiral_data.pkl')

X, y = load_data('spiral_data.pkl')

dense1 = Layer_Dense(2, 4)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(4, 3)
activation2 = Activation_SoftMax()
loss_function = Loss_CategoricalCrossEntropy()
optimizer = Optimizer_SGD(learning_rate=0.01)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.forward(activation2.output, y)

    loss_function.backward(activation2.output, y)
    dense2.backward(loss_function.dvalues)
    activation1.backward(dense2.dvalues)
    dense1.backward(activation1.dvalues)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

test_accuracy = evaluate_model((dense1, activation1, dense2, activation2), X, y)
print(f"Test Accuracy: {test_accuracy:.4f}")

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
