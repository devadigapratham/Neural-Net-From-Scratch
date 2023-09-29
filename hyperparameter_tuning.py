import numpy as np
from itertools import product
from load_data import load_data
from neural_network import Layer_Dense, Activation_ReLU, Activation_SoftMax, Loss_CategoricalCrossEntropy, Optimizer_SGD
from evaluate_model import evaluate_model

# Load the synthetic spiral data
X, y = load_data('spiral_data.pkl')

# Get user input for the number of dense layers
num_dense_layers = int(input("Enter the number of dense layers: "))

# Define hyperparameter grid
param_grid = {
    'num_neurons': [3, 4, 5],
    'learning_rate': [0.01, 0.001, 0.0001]
}

best_accuracy = 0
best_params = None

# Perform grid search
for num_neurons, learning_rate in product(param_grid['num_neurons'], param_grid['learning_rate']):
    # Create dense layers based on user input
    dense_layers = [Layer_Dense(2, num_neurons) for _ in range(num_dense_layers)]
    activation_layers = [Activation_ReLU() for _ in range(num_dense_layers)]
    
    output_layer = Layer_Dense(num_neurons, 3)
    softmax_activation = Activation_SoftMax()
    loss_function = Loss_CategoricalCrossEntropy()
    optimizer = Optimizer_SGD(learning_rate=learning_rate)

    for epoch in range(10001):
        # Forward pass
        layer_output = X
        for i in range(num_dense_layers):
            dense_layer = dense_layers[i]
            activation_layer = activation_layers[i]
            dense_layer.forward(layer_output)
            activation_layer.forward(dense_layer.output)
            layer_output = activation_layer.output
        
        output_layer.forward(layer_output)
        softmax_activation.forward(output_layer.output)

        # Calculate loss
        loss = loss_function.calculate(softmax_activation.output, y)

        # Backward pass
        loss_function.backward(softmax_activation.output, y)
        output_layer.backward(loss_function.dvalues)
        
        for i in range(num_dense_layers - 1, -1, -1):
            activation_layer = activation_layers[i]
            dense_layer = dense_layers[i]
            activation_layer.backward(dense_layer.dvalues)
            dense_layer.backward(activation_layer.dvalues)

        # Update weights and biases
        for i in range(num_dense_layers):
            optimizer.update_params(dense_layers[i])

        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss:.4f}")

    # Evaluate the model on validation data
    accuracy = evaluate_model((*dense_layers, output_layer, softmax_activation), X, y)
    print(f"Hyperparameters: Neurons={num_neurons}, Learning Rate={learning_rate}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print()

    # Update best hyperparameters if needed
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = (num_neurons, learning_rate)

print(f"Best Hyperparameters: Neurons={best_params[0]}, Learning Rate={best_params[1]}")
