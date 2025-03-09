import numpy as np

# Activation Functions & Derivatives

def linear(z):
    return z

def d_linear(z, a):
    return np.ones_like(z)

def relu(z):
    return np.maximum(0, z)

def d_relu(z, a):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def d_sigmoid(z, a):
    return a * (1 - a)

def tanh(z):
    return np.tanh(z)

def d_tanh(z, a):
    return 1 - a**2

def softmax(z):
    exps = np.exp(z)
    return exps / np.sum(exps, axis=1, keepdims=True)

def d_softmax(softmax_output, upstream_grad):
    grad = np.zeros_like(softmax_output)
    for i in range(softmax_output.shape[0]):
        s = softmax_output[i].reshape(-1, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)
        grad[i] = np.dot(jacobian, upstream_grad[i])
    return grad

activation_functions = {
    'linear': (linear, d_linear),
    'relu': (relu, d_relu),
    'sigmoid': (sigmoid, d_sigmoid),
    'tanh': (tanh, d_tanh),
    'softmax': (softmax, None)  
}

# Weight Initialization

def initialize_weight(shape, method, params):
    if method == 'zero':
        return np.zeros(shape)
    elif method == 'random_uniform':
        low = params.get('low', -1.0)
        high = params.get('high', 1.0)
        seed = params.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(low, high, size=shape)
    elif method == 'random_normal':
        mean = params.get('mean', 0.0)
        variance = params.get('variance', 1.0)
        seed = params.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
        std = np.sqrt(variance)
        return np.random.normal(mean, std, size=shape)
    else:
        raise ValueError("Unknown weight initialization method.")

# Layer Class

class Layer:
    def __init__(self, input_size, output_size, activation, 
                 weight_init='random_uniform', weight_init_params={}):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        if self.activation_name not in activation_functions:
            raise ValueError(f"Activation '{activation}' not supported.")
        self.activation, self.activation_deriv = activation_functions[self.activation_name]
        self.W = initialize_weight((input_size, output_size), weight_init, weight_init_params)
        self.b = initialize_weight((1, output_size), weight_init, weight_init_params)
    
    def forward(self, X):
        self.X = X 
        self.z = np.dot(X, self.W) + self.b
        self.a = self.activation(self.z)
        return self.a

    def backward(self, delta, learning_rate):
        if self.activation_name == 'softmax' and self.activation_deriv is None:
            delta = d_softmax(self.a, delta)
        else:
            delta = delta * self.activation_deriv(self.z, self.a)
        
        dW = np.dot(self.X.T, delta)
        db = np.sum(delta, axis=0, keepdims=True)

        delta_prev = np.dot(delta, self.W.T)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return delta_prev
