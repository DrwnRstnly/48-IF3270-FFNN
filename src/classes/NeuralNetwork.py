from classes.Layer import Layer
import numpy as np
import matplotlib.pyplot as plt
import pickle 

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
                 weight_init='random_uniform', weight_init_params={}, regularization=None, reg_lambda=0.0):
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
        
        self.regularization = regularization # 'L1' atau 'L2' atau None
        self.reg_lambda = reg_lambda

    def forward(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, X, y, learning_rate):
        output = self.forward(X)
        delta = self.loss_deriv(y, output)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate, reg_type=self.regularization, reg_lambda=self.reg_lambda)

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, learning_rate=0.01, max_epoch=100, verbose=1):
        n_samples = X_train.shape[0]
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(max_epoch):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                self.backward(X_batch, y_batch, learning_rate)

            y_pred = self.forward(X_train)
            y_pred_val = self.forward(X_val)
            train_loss = self.loss(y_train, y_pred)
            val_loss = self.loss(y_val, y_pred_val)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{max_epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")
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
            plt.hist(self.layers[idx].W.flatten(), bins=20, color="blue", label='Weights')
            plt.hist(self.layers[idx].b.flatten(), bins=20, color="yellow", label='Biases')
            plt.legend()
            plt.title(f"Weight and Bias Distribution for Layer {idx+1}")
            plt.xlabel("Weight value")
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
            plt.hist(self.layers[idx].dW.flatten(), bins=20,color="blue", label='dW')
            plt.hist(self.layers[idx].db.flatten(), bins=20,color="yellow", label='db')
            plt.legend()
            plt.title(f"Gradient Distribution (dW) for Layer {idx+1}")
            plt.xlabel("Gradient value")
            plt.ylabel("Frequency")
            plt.show()

    def plot_training_loss(self, history):
        epochs = range(1, len(history["train_loss"]) + 1)

        plt.figure(figsize=(10, 6))

        plt.plot(epochs, history["train_loss"], label="Training Loss", marker="o")
        
        plt.plot(epochs, history["val_loss"], label="Validation Loss", marker="x")
        
        plt.title("Grafik Training dan Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        plt.legend()
        plt.grid(True)

        plt.show()
