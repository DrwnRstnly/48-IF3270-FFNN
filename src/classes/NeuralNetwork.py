import Layer
import numpy as np

# Loss Functions & Derivatives

def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def mse_loss_deriv(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

def binary_crossentropy_loss(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def binary_crossentropy_loss_deriv(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    grad = (-y_true / y_pred + (1 - y_true) / (1 - y_pred)) / y_true.shape[0]
    return grad

def categorical_crossentropy_loss(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return loss

def categorical_crossentropy_loss_deriv(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

#  Neural Network Class

class NeuralNetwork:
    def __init__(self, input_size, layers_config, loss_function='mse', 
                 weight_init='random_uniform', weight_init_params={}):
        self.layers = []
        prev_size = input_size
        for neurons, activation in layers_config:
            layer = Layer(prev_size, neurons, activation, weight_init, weight_init_params)
            self.layers.append(layer)
            prev_size = neurons
        
        self.loss_name = loss_function.lower()
        if self.loss_name == 'mse':
            self.loss = mse_loss
            self.loss_deriv = mse_loss_deriv
        elif self.loss_name == 'binary_crossentropy':
            self.loss = binary_crossentropy_loss
            self.loss_deriv = binary_crossentropy_loss_deriv
        elif self.loss_name == 'categorical_crossentropy':
            self.loss = categorical_crossentropy_loss
            self.loss_deriv = categorical_crossentropy_loss_deriv
        else:
            raise ValueError("Unknown loss function.")

    def forward(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, X, y, learning_rate):
        output = self.forward(X)
        delta = self.loss_deriv(y, output)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)

    def train(self, X, y, batch_size=32, learning_rate=0.01, max_epoch=100, verbose=1):
        n_samples = X.shape[0]
        for epoch in range(max_epoch):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                self.backward(X_batch, y_batch, learning_rate)
            if verbose:
                y_pred = self.forward(X)
                loss_value = self.loss(y, y_pred)
                print(f"Epoch {epoch+1}/{max_epoch}, Loss: {loss_value}")

    def predict(self, X):
        return self.forward(X)