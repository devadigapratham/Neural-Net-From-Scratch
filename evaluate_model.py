import numpy as np
from neural_network import Activation_SoftMax

def evaluate_model(model, X, y):
    dense1, activation1, dense2, activation2 = model
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    y_pred = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(y_pred == y)
    return accuracy
