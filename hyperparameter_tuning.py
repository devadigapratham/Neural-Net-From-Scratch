import numpy as np
from itertools import product
from load_data import load_data
from neural_network import Layer_Dense, Activation_ReLU, Activation_SoftMax, Loss_CategoricalCrossEntropy, Optimizer_SGD
from evaluate_model import evaluate_model

# Load the synthetic spiral data
X, y = load_data('spiral_data.pkl')

# Define hyperparameter grid
param_grid = {
    'num_neurons': [3, 4, 5],
    'learning_rate': [0.01, 0.001, 0.0001]
}

best_accuracy = 0
best_params = None

# Perform grid search
for num_neurons, learning_rate in product(param_grid['num_neurons'], param_grid['learning_rate']):
    # Create and train the neural network model
    dense1 = Layer_Dense(2, num_neurons)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(num_neurons, 3)
    activation2 = Activation_SoftMax()
    loss_function = Loss_CategoricalCrossEntropy()
    optimizer = Optimizer_SGD(learning_rate=learning_rate)

    for epoch in range(10001):
        # Forward pass
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        # Calculate loss
        loss = loss_function.forward(activation2.output, y)

        # Backward pass
        loss_function.backward(activation2.output, y)
        dense2.backward(loss_function.dvalues)
        activation1.backward(dense2.dvalues)
        dense1.backward(activation1.dvalues)

        # Update weights and biases
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)

        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss:.4f}")

    # Evaluate the model on validation data
    accuracy = evaluate_model((dense1, activation1, dense2, activation2), X, y)
    print(f"Hyperparameters: Neurons={num_neurons}, Learning Rate={learning_rate}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print()

    # Update best hyperparameters if needed
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = (num_neurons, learning_rate)

print(f"Best Hyperparameters: Neurons={best_params[0]}, Learning Rate={best_params[1]}")
