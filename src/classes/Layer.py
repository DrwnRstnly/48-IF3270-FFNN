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

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def d_leaky_relu(z, a, alpha=0.01):
    return np.where(z > 0, 1, alpha)

def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))

def d_elu(z, a, alpha=1.0):
    return np.where(z > 0, 1, a + alpha)

def swish(z):
    s = sigmoid(z)
    return z * s

def d_swish(z, a):
    s = sigmoid(z)
    return s + z * s * (1 - s)


activation_functions = {
    'linear': (linear, d_linear),
    'relu': (relu, d_relu),
    'leaky_relu': (leaky_relu, d_leaky_relu),
    'elu': (elu, d_elu),
    'swish': (swish, d_swish),
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
    elif method == 'xavier':
        seed = params.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)
    elif method == 'he':
        seed = params.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
        fan_in = shape[0]
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, size=shape)
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
        self.dW = None
        self.db = None

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
        
        self.dW = np.dot(self.X.T, delta)
        self.db = np.sum(delta, axis=0, keepdims=True)

        delta_prev = np.dot(delta, self.W.T)

        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        return delta_prev
