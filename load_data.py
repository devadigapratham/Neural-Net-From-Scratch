import pickle

def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data['X'], data['y']
