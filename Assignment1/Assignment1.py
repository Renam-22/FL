"""
Federated Learning Simulation
==============================
A basic implementation of Federated Learning where multiple clients train local models
on their own data without sharing raw datasets. A central server aggregates the locally
trained model parameters using averaging to form a global model.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import copy


class SimpleNeuralNetwork:
    """
    A simple neural network for classification using one hidden layer.
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize the neural network with random weights.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output classes
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        return x * (1 - x)
    
    def softmax(self, x):
        """Softmax activation function for output layer."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation.
        
        Args:
            X: Input data
            
        Returns:
            Output predictions
        """
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        """
        Backward propagation to compute gradients.
        
        Args:
            X: Input data
            y: True labels (one-hot encoded)
            output: Predicted output from forward pass
        """
        m = X.shape[0]
        
        # Compute gradients
        dz2 = output - y
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = np.dot(dz2, self.weights2.T) * self.sigmoid_derivative(self.a1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.weights1 -= self.learning_rate * dw1
        self.bias1 -= self.learning_rate * db1
        self.weights2 -= self.learning_rate * dw2
        self.bias2 -= self.learning_rate * db2
    
    def train(self, X, y, epochs=1):
        """
        Train the neural network.
        
        Args:
            X: Training data
            y: Training labels (one-hot encoded)
            epochs: Number of training epochs
        """
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            Predicted class labels
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def get_parameters(self):
        """
        Get model parameters.
        
        Returns:
            Dictionary containing all model parameters
        """
        return {
            'weights1': copy.deepcopy(self.weights1),
            'bias1': copy.deepcopy(self.bias1),
            'weights2': copy.deepcopy(self.weights2),
            'bias2': copy.deepcopy(self.bias2)
        }
    
    def set_parameters(self, parameters):
        """
        Set model parameters.
        
        Args:
            parameters: Dictionary containing model parameters
        """
        self.weights1 = copy.deepcopy(parameters['weights1'])
        self.bias1 = copy.deepcopy(parameters['bias1'])
        self.weights2 = copy.deepcopy(parameters['weights2'])
        self.bias2 = copy.deepcopy(parameters['bias2'])


class FederatedClient:
    """
    Represents a client in the federated learning system.
    Each client has its own local dataset and trains a local model.
    """
    def __init__(self, client_id, X_train, y_train, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize a federated learning client.
        
        Args:
            client_id: Unique identifier for the client
            X_train: Local training data
            y_train: Local training labels
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output classes
            learning_rate: Learning rate for training
        """
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.model = SimpleNeuralNetwork(input_size, hidden_size, output_size, learning_rate)
        print(f"Client {client_id} initialized with {len(X_train)} training samples")
    
    def train_local_model(self, epochs=5):
        """
        Train the local model on client's data.
        
        Args:
            epochs: Number of local training epochs
            
        Returns:
            Trained model parameters
        """
        print(f"Client {self.client_id} training locally for {epochs} epochs...")
        self.model.train(self.X_train, self.y_train, epochs)
        return self.model.get_parameters()
    
    def update_model(self, global_parameters):
        """
        Update local model with global parameters from server.
        
        Args:
            global_parameters: Parameters from the global model
        """
        self.model.set_parameters(global_parameters)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the local model.
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Accuracy score
        """
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(np.argmax(y_test, axis=1), predictions)
        return accuracy


class FederatedServer:
    """
    Central server that aggregates client models in federated learning.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the federated learning server.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output classes
        """
        self.global_model = SimpleNeuralNetwork(input_size, hidden_size, output_size)
        print("Federated Server initialized")
    
    def aggregate_parameters(self, client_parameters_list):
        """
        Aggregate parameters from multiple clients using Federated Averaging (FedAvg).
        
        Args:
            client_parameters_list: List of parameter dictionaries from clients
            
        Returns:
            Aggregated global parameters
        """
        print(f"Aggregating parameters from {len(client_parameters_list)} clients...")
        
        # Initialize aggregated parameters
        aggregated_params = {}
        num_clients = len(client_parameters_list)
        
        # Average all parameters
        for key in client_parameters_list[0].keys():
            aggregated_params[key] = sum(params[key] for params in client_parameters_list) / num_clients
        
        return aggregated_params
    
    def update_global_model(self, aggregated_parameters):
        """
        Update the global model with aggregated parameters.
        
        Args:
            aggregated_parameters: Aggregated parameters from clients
        """
        self.global_model.set_parameters(aggregated_parameters)
        print("Global model updated")
    
    def get_global_parameters(self):
        """
        Get current global model parameters.
        
        Returns:
            Global model parameters
        """
        return self.global_model.get_parameters()
    
    def evaluate_global_model(self, X_test, y_test):
        """
        Evaluate the global model.
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Accuracy score
        """
        predictions = self.global_model.predict(X_test)
        accuracy = accuracy_score(np.argmax(y_test, axis=1), predictions)
        return accuracy


def prepare_data(num_clients=5, test_size=0.2):
    """
    Load and prepare data for federated learning simulation.
    Splits data among multiple clients to simulate distributed data.
    
    Args:
        num_clients: Number of clients to split data among
        test_size: Proportion of data to use for testing
        
    Returns:
        Tuple of (client_data_list, X_test, y_test, input_size, output_size)
    """
    print("\n=== Preparing Data ===")
    
    # Load digits dataset (0-9 digit classification)
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert labels to one-hot encoding
    num_classes = len(np.unique(y))
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]
    
    # Distribute training data among clients (simulating non-IID distribution)
    samples_per_client = len(X_train) // num_clients
    client_data = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(X_train)
        
        client_X = X_train[start_idx:end_idx]
        client_y = y_train_onehot[start_idx:end_idx]
        
        client_data.append((client_X, client_y))
    
    print(f"Data distributed among {num_clients} clients")
    print(f"Test set: {X_test.shape[0]} samples\n")
    
    return client_data, X_test, y_test_onehot, X.shape[1], num_classes


def run_federated_learning(num_clients=5, num_rounds=10, local_epochs=5, hidden_size=32, learning_rate=0.1):
    """
    Run the complete federated learning simulation.
    
    Args:
        num_clients: Number of clients in the federation
        num_rounds: Number of federated learning rounds
        local_epochs: Number of epochs each client trains locally
        hidden_size: Size of hidden layer in neural network
        learning_rate: Learning rate for training
        
    Returns:
        Dictionary containing training history
    """
    print("\n" + "="*60)
    print("FEDERATED LEARNING SIMULATION")
    print("="*60)
    
    # Prepare data
    client_data, X_test, y_test, input_size, output_size = prepare_data(num_clients=num_clients)
    
    # Initialize server
    print("\n=== Initializing Federated Learning System ===")
    server = FederatedServer(input_size, hidden_size, output_size)
    
    # Initialize clients
    clients = []
    for i, (X_client, y_client) in enumerate(client_data):
        client = FederatedClient(
            client_id=i+1,
            X_train=X_client,
            y_train=y_client,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            learning_rate=learning_rate
        )
        clients.append(client)
    
    # Track accuracy over rounds
    global_accuracies = []
    
    print("\n" + "="*60)
    print("STARTING FEDERATED LEARNING ROUNDS")
    print("="*60)
    
    # Federated learning rounds
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num}/{num_rounds} ---")
        
        # Step 1: Distribute global model to clients
        global_params = server.get_global_parameters()
        for client in clients:
            client.update_model(global_params)
        
        # Step 2: Clients train locally
        client_params_list = []
        for client in clients:
            local_params = client.train_local_model(epochs=local_epochs)
            client_params_list.append(local_params)
        
        # Step 3: Server aggregates client models
        aggregated_params = server.aggregate_parameters(client_params_list)
        server.update_global_model(aggregated_params)
        
        # Step 4: Evaluate global model
        global_accuracy = server.evaluate_global_model(X_test, y_test)
        global_accuracies.append(global_accuracy)
        
        print(f"Global Model Accuracy: {global_accuracy:.4f} ({global_accuracy*100:.2f}%)")
    
    print("\n" + "="*60)
    print("FEDERATED LEARNING COMPLETED")
    print("="*60)
    print(f"\nFinal Global Model Accuracy: {global_accuracies[-1]:.4f} ({global_accuracies[-1]*100:.2f}%)")
    
    return {
        'global_accuracies': global_accuracies,
        'num_rounds': num_rounds,
        'num_clients': num_clients,
        'local_epochs': local_epochs
    }


def plot_results(history):
    """
    Plot the federated learning training results.
    
    Args:
        history: Dictionary containing training history
    """
    plt.figure(figsize=(10, 6))
    
    rounds = range(1, history['num_rounds'] + 1)
    accuracies = history['global_accuracies']
    
    plt.plot(rounds, accuracies, 'b-o', linewidth=2, markersize=8, label='Global Model Accuracy')
    plt.xlabel('Federated Learning Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Federated Learning Performance\n({history["num_clients"]} clients, {history["local_epochs"]} local epochs)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim([0, 1])
    
    # Add accuracy values on points
    for i, acc in enumerate(accuracies):
        if i % 2 == 0:  # Show every other value to avoid crowding
            plt.annotate(f'{acc:.3f}', 
                        (rounds[i], accuracies[i]), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8)
    
    plt.tight_layout()
    plt.savefig('federated_learning_results.png', dpi=300, bbox_inches='tight')
    print(f"\nResults plot saved as 'federated_learning_results.png'")
    plt.show()


def main():
    """
    Main function to run the federated learning simulation.
    """
    # Simulation parameters
    NUM_CLIENTS = 5          # Number of clients in the federation
    NUM_ROUNDS = 15          # Number of federated learning rounds
    LOCAL_EPOCHS = 5         # Number of epochs each client trains locally
    HIDDEN_SIZE = 32         # Hidden layer size
    LEARNING_RATE = 0.1      # Learning rate
    
    # Run federated learning
    history = run_federated_learning(
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # Plot results
    plot_results(history)
    
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Number of Clients: {NUM_CLIENTS}")
    print(f"Federated Rounds: {NUM_ROUNDS}")
    print(f"Local Epochs per Round: {LOCAL_EPOCHS}")
    print(f"Hidden Layer Size: {HIDDEN_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Final Accuracy: {history['global_accuracies'][-1]*100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
