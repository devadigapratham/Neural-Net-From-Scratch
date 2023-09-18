import numpy as np
import pickle

def create_spiral_data(samples, classes):
    np.random.seed(0)
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, samples) + np.random.randn(samples) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y

def save_data(X, y, filename):
    data = {'X': X, 'y': y}
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    X, y = create_spiral_data(samples=1000, classes=3)
    save_data(X, y, 'spiral_data.pkl')
