import numpy as np
import nnfs
from nnfs.datasets import spiral_data  

np.random.seed(0)

# Define some input data (X)
X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

# Define a class for a dense layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Define an activation function (ReLU)
class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

# Define an activation function (Softmax)
class Activation_SoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Define a base class for loss functions
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Define a categorical cross-entropy loss
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        neg_log_likelihoods = -np.log(correct_confidences)
        return neg_log_likelihoods

# Generate some synthetic spiral data
X, y = spiral_data(samples=1000, classes=3)

# Create and forward pass through the first dense layer and ReLU activation
dense1 = Layer_Dense(4, 3)  
activation1 = Activation_ReLU()

dense1.forward(X)
activation1.forward(dense1.output)

# Create and forward pass through the second dense layer and Softmax activation
dense2 = Layer_Dense(3, 3)
activation2 = Activation_SoftMax()

dense2.forward(activation1.output)  # Use the output of the previous activation
activation2.forward(dense2.output)

# Calculate the loss
loss_function = Loss_CategoricalCrossEntropy()  
loss = loss_function.calculate(activation2.output, y)

print(activation2.output[:5])
