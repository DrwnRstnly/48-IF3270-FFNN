import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ---------------------------
# Activation Functions & Derivatives using torch
# ---------------------------
def linear(z):
    return z

def d_linear(z, a):
    return torch.ones_like(z)

def relu(z):
    return torch.clamp(z, min=0)

def d_relu(z, a):
    return (z > 0).float()

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

def d_sigmoid(z, a):
    return a * (1 - a)

def tanh(z):
    return torch.tanh(z)

def d_tanh(z, a):
    return 1 - a**2

def softmax(z):
    exps = torch.exp(z)
    return exps / torch.sum(exps, dim=1, keepdim=True)

def d_softmax(softmax_output, upstream_grad):
    grad = torch.zeros_like(softmax_output)
    for i in range(softmax_output.shape[0]):
        s = softmax_output[i].view(-1, 1)
        jacobian = torch.diagflat(s) - s @ s.t()
        grad[i] = jacobian @ upstream_grad[i]
    return grad

activation_functions = {
    'linear': (linear, d_linear),
    'relu': (relu, d_relu),
    'sigmoid': (sigmoid, d_sigmoid),
    'tanh': (tanh, d_tanh),
    'softmax': (softmax, None)  # We'll use special handling for softmax derivative.
}

# ---------------------------
# Weight Initialization using torch
# ---------------------------
def initialize_weight(shape, method, params):
    if method == 'zero':
        return torch.zeros(shape, dtype=torch.float32)
    elif method == 'random_uniform':
        low = params.get('low', -1.0)
        high = params.get('high', 1.0)
        seed = params.get('seed', None)
        if seed is not None:
            torch.manual_seed(seed)
        return (high - low) * torch.rand(shape, dtype=torch.float32) + low
    elif method == 'random_normal':
        mean = params.get('mean', 0.0)
        variance = params.get('variance', 1.0)
        std = variance ** 0.5
        seed = params.get('seed', None)
        if seed is not None:
            torch.manual_seed(seed)
        return torch.normal(mean=mean, std=std, size=shape)
    else:
        raise ValueError("Unknown weight initialization method.")

# ---------------------------
# Layer Class using torch tensors and manual gradients
# ---------------------------
class Layer:
    def __init__(self, input_size, output_size, activation, 
                 weight_init='random_uniform', weight_init_params={}):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        if self.activation_name not in activation_functions:
            raise ValueError(f"Activation '{activation}' not supported.")
        self.activation, self.activation_deriv = activation_functions[self.activation_name]
        # Create weights and biases as torch tensors with requires_grad = False.
        self.W = initialize_weight((input_size, output_size), weight_init, weight_init_params)
        self.b = initialize_weight((1, output_size), weight_init, weight_init_params)
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X  # Cache input
        self.z = X @ self.W + self.b  # Matrix multiplication and addition
        self.a = self.activation(self.z)
        return self.a

    def backward(self, delta, learning_rate):
        # For softmax in a hidden layer, use the special derivative function.
        if self.activation_name == 'softmax' and self.activation_deriv is None:
            delta = d_softmax(self.a, delta)
        else:
            delta = delta * self.activation_deriv(self.z, self.a)
        self.dW = self.X.t() @ delta
        self.db = torch.sum(delta, dim=0, keepdim=True)
        delta_prev = delta @ self.W.t()
        # Update weights and biases manually.
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        return delta_prev

# ---------------------------
# Loss Functions & Derivatives using torch
# ---------------------------
def mse_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred)**2)

def mse_loss_deriv(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

def binary_crossentropy_loss(y_true, y_pred, epsilon=1e-12):
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    loss = -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return loss

def binary_crossentropy_loss_deriv(y_true, y_pred, epsilon=1e-12):
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    grad = (-y_true / y_pred + (1 - y_true) / (1 - y_pred)) / y_true.shape[0]
    return grad

def categorical_crossentropy_loss(y_true, y_pred, epsilon=1e-12):
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    loss = -torch.mean(torch.sum(y_true * torch.log(y_pred), dim=1))
    return loss

def categorical_crossentropy_loss_deriv(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

# ---------------------------
# Neural Network Class using torch and manual gradients (no autograd)
# ---------------------------
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

    def train(self, X_train, y_train, batch_size=32, learning_rate=0.01, max_epoch=100, verbose=1, X_val=None, y_val=None):
        n_samples = X_train.shape[0]
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(max_epoch):
            indices = torch.randperm(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                self.backward(X_batch, y_batch, learning_rate)
            y_pred_train = self.forward(X_train)
            train_loss = self.loss(y_train, y_pred_train).item()
            history["train_loss"].append(train_loss)
            if X_val is not None and y_val is not None:
                y_pred_val = self.forward(X_val)
                val_loss = self.loss(y_val, y_pred_val).item()
                history["val_loss"].append(val_loss)
                if verbose:
                    print(f"Epoch {epoch+1}/{max_epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{max_epoch}, Train Loss: {train_loss:.4f}")
        return history

    def predict(self, X):
        return self.forward(X)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}.")

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}.")
        return model

    def display_model(self):
        print("Model Structure:")
        for idx, layer in enumerate(self.layers):
            print(f"\nLayer {idx+1}:")
            print(f"  Input size: {layer.input_size}, Output size: {layer.output_size}")
            print(f"  Activation: {layer.activation_name}")
            print("  Weights:\n", layer.W)
            print("  Biases:\n", layer.b)
            if layer.dW is not None:
                print("  Weight Gradients (dW):\n", layer.dW)
                print("  Bias Gradients (db):\n", layer.db)
            else:
                print("  Gradients not computed yet.")

    def plot_weight_distribution(self, layers_to_plot):
        for idx in layers_to_plot:
            if idx < 0 or idx >= len(self.layers):
                print(f"Layer index {idx} is out of range.")
                continue
            plt.figure()
            plt.hist(self.layers[idx].W.flatten().detach().cpu().numpy(), bins=20, color="blue", label='Weights')
            plt.hist(self.layers[idx].b.flatten().detach().cpu().numpy(), bins=20, color="yellow", label='Biases')
            plt.legend()
            plt.title(f"Weight and Bias Distribution for Layer {idx+1}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.show()

    def plot_gradient_distribution(self, layers_to_plot):
        for idx in layers_to_plot:
            if idx < 0 or idx >= len(self.layers):
                print(f"Layer index {idx} is out of range.")
                continue
            if self.layers[idx].dW is None:
                print(f"No gradient available for Layer {idx+1}. Run a backward pass first.")
                continue
            plt.figure()
            plt.hist(self.layers[idx].dW.flatten().detach().cpu().numpy(), bins=20, color="blue", label='dW')
            plt.hist(self.layers[idx].db.flatten().detach().cpu().numpy(), bins=20, color="yellow", label='db')
            plt.legend()
            plt.title(f"Gradient Distribution for Layer {idx+1}")
            plt.xlabel("Gradient Value")
            plt.ylabel("Frequency")
            plt.show()

# ---------------------------
# Example: Iris Dataset with PyTorch (manual gradients, no autograd)
# ---------------------------
if __name__ == "__main__":
    # Load Iris dataset.
    iris = load_iris()
    X = iris.data  # shape: (150, 4)
    y = iris.target.reshape(-1, 1)  # shape: (150, 1)
    # One-hot encode labels.
    encoder = OneHotEncoder(sparse_output=False)
    y_one_hot = encoder.fit_transform(y)
    # Split into training and validation sets.
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
    # Convert to torch tensors.
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)

    # Define a multi-class neural network.
    layers_config = [
        (8, 'relu'),      # Hidden layer with 8 neurons, ReLU activation.
        (3, 'softmax')    # Output layer with 3 neurons (for 3 classes), Softmax activation.
    ]
    nn_model = NeuralNetwork(input_size=4, layers_config=layers_config, loss_function='categorical_crossentropy',
                             weight_init='random_uniform', weight_init_params={'low': -1.0, 'high': 1.0, 'seed': 42})
    # Train the model.
    history = nn_model.train(X_train, y_train, batch_size=20, learning_rate=0.01, max_epoch=500, verbose=1,
                             X_val=X_val, y_val=y_val)
    # Predictions on validation set.
    y_pred = nn_model.predict(X_val)
    y_pred_labels = torch.argmax(y_pred, dim=1).detach().cpu().numpy()
    y_true_labels = torch.argmax(y_val, dim=1).detach().cpu().numpy()
    # Display model structure.
    nn_model.display_model()
    # Plot weight & gradient distributions.
    nn_model.plot_weight_distribution([0, 1])
    nn_model.plot_gradient_distribution([0, 1])
    # Print final accuracy.
    accuracy = (y_pred_labels == y_true_labels).mean()
    print(f"Validation Accuracy: {accuracy:.2f}")
