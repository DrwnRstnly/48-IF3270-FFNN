from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from classes.NeuralNetwork import NeuralNetwork

import numpy as np

if __name__ == "__main__":
    # Load the dataset
    iris = load_iris()
    X = iris.data  # Features: Sepal length, Sepal width, Petal length, Petal width
    y = iris.target.reshape(-1, 1)  # Labels: 0 (Setosa), 1 (Versicolor), 2 (Virginica)

    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y_one_hot = encoder.fit_transform(y)

    # Split into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    # Define a multi-class neural network
    layers_config = [
        (8, 'relu'),  # Hidden layer with 8 neurons, ReLU activation
        (3, 'softmax')  # Output layer with 3 neurons (for 3 classes), Softmax activation
    ]

    # Initialize the neural network
    nn = NeuralNetwork(input_size=4, layers_config=layers_config, loss_function='categorical_crossentropy',
                    weight_init='random_uniform', weight_init_params={'low': -1.0, 'high': 1.0, 'seed': 42})

    # Train the model with validation data
    history = nn.train(X_train, y_train, batch_size=20, learning_rate=0.01, max_epoch=500, verbose=1,
                    X_val=X_val, y_val=y_val)

    # Evaluate predictions on validation set
    y_pred = nn.predict(X_val)
    y_pred_labels = np.argmax(y_pred, axis=1)  # Convert softmax probabilities to class labels
    y_true_labels = np.argmax(y_val, axis=1)

    # Display model structure
    nn.display_model()

    # Plot weight & gradient distributions for each layer
    nn.plot_weight_distribution([0, 1])  # Hidden layer (0) and Output layer (1)
    nn.plot_gradient_distribution([0, 1])  # Gradients of weights

    # Print final accuracy
    accuracy = np.mean(y_pred_labels == y_true_labels)
    print(f"Validation Accuracy: {accuracy:.2f}")
