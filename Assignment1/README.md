# Federated Learning Simulation

## Overview

This project implements a basic simulation of a **Federated Learning (FL)** system in Python. In this system, multiple clients train local models on their own data without sharing raw datasets. A central server aggregates the locally trained model parameters using **Federated Averaging (FedAvg)** to form a global model, demonstrating the core federated learning workflow.

## What is Federated Learning?

Federated Learning is a machine learning approach where:
- **Privacy is preserved**: Raw data never leaves client devices
- **Distributed training**: Multiple clients train models on their local data
- **Centralized aggregation**: A server combines model updates (not data) to create a global model
- **Iterative process**: The global model is redistributed to clients for further training

## Project Structure

```
Assignment1/
├── federated_learning.py      # Main implementation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── federated_learning_results.png  # Generated results plot
```

## Core Components

### 1. **SimpleNeuralNetwork**
- A basic neural network with one hidden layer
- Uses sigmoid activation for hidden layer and softmax for output
- Implements forward and backward propagation
- Supports parameter getting/setting for federated aggregation

### 2. **FederatedClient**
- Represents a client in the federated learning system
- Has its own local dataset
- Trains a local model without sharing data
- Can update its model with global parameters from the server

### 3. **FederatedServer**
- Central coordinator of the federated learning process
- Aggregates client model parameters using averaging (FedAvg algorithm)
- Maintains and updates the global model
- Evaluates global model performance

## Federated Learning Workflow

1. **Initialization**
   - Server initializes a global model
   - Data is distributed among clients (simulating real-world distributed data)

2. **For each federated learning round:**
   
   a. **Distribution Phase**
      - Server sends current global model parameters to all clients
   
   b. **Local Training Phase**
      - Each client updates their local model with global parameters
      - Clients train on their local data for several epochs
      - No raw data is shared with the server
   
   c. **Aggregation Phase**
      - Clients send trained model parameters back to server
      - Server aggregates parameters using Federated Averaging:
        ```
        θ_global = (1/N) * Σ(θ_client_i)
        ```
      - Server updates the global model
   
   d. **Evaluation Phase**
      - Global model is evaluated on test data
      - Accuracy is recorded for this round

3. **Completion**
   - After all rounds, final global model is evaluated
   - Results are visualized

## Features

- ✅ **Privacy-Preserving**: Clients never share raw data
- ✅ **Distributed Training**: Multiple clients train in parallel
- ✅ **Federated Averaging**: Standard FedAvg algorithm for aggregation
- ✅ **Performance Tracking**: Monitors accuracy across rounds
- ✅ **Visualization**: Generates plots showing learning progress
- ✅ **Configurable**: Easy to adjust number of clients, rounds, epochs, etc.

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Simply run the main script:

```bash
python federated_learning.py
```

### Configuration

You can modify the simulation parameters in the `main()` function:

```python
NUM_CLIENTS = 5          # Number of clients in the federation
NUM_ROUNDS = 15          # Number of federated learning rounds
LOCAL_EPOCHS = 5         # Number of epochs each client trains locally
HIDDEN_SIZE = 32         # Hidden layer size in neural network
LEARNING_RATE = 0.1      # Learning rate for training
```

### Advanced Usage

You can also import and use the components programmatically:

```python
from federated_learning import run_federated_learning, plot_results

# Run custom simulation
history = run_federated_learning(
    num_clients=10,
    num_rounds=20,
    local_epochs=3,
    hidden_size=64,
    learning_rate=0.05
)

# Plot results
plot_results(history)
```

## Dataset

The simulation uses the **Digits dataset** from scikit-learn:
- 1797 samples of 8x8 pixel handwritten digits (0-9)
- 64 features per sample
- 10 classes (digits 0-9)
- Split: 80% training (distributed among clients), 20% testing

## Output

When you run the simulation, you'll see:

1. **Console Output:**
   - Data preparation details
   - Client initialization
   - Training progress for each round
   - Global model accuracy after each round
   - Final summary statistics

2. **Visualization:**
   - A plot showing global model accuracy over federated learning rounds
   - Saved as `federated_learning_results.png`

### Example Output:
```
============================================================
FEDERATED LEARNING SIMULATION
============================================================

=== Preparing Data ===
Dataset: 1797 samples, 64 features, 10 classes
Data distributed among 5 clients
Test set: 360 samples

=== Initializing Federated Learning System ===
Federated Server initialized
Client 1 initialized with 287 training samples
Client 2 initialized with 287 training samples
...

============================================================
STARTING FEDERATED LEARNING ROUNDS
============================================================

--- Round 1/15 ---
Client 1 training locally for 5 epochs...
Client 2 training locally for 5 epochs...
...
Aggregating parameters from 5 clients...
Global model updated
Global Model Accuracy: 0.8528 (85.28%)

--- Round 2/15 ---
...

============================================================
FEDERATED LEARNING COMPLETED
============================================================

Final Global Model Accuracy: 0.9556 (95.56%)
```

## Key Concepts Demonstrated

1. **Data Privacy**: Raw data never leaves client devices
2. **Federated Averaging**: Simple yet effective aggregation method
3. **Distributed Learning**: Multiple clients learning in parallel
4. **Model Aggregation**: Combining knowledge without sharing data
5. **Iterative Improvement**: Global model improves over rounds

## Algorithm: Federated Averaging (FedAvg)

The core aggregation algorithm:

```
For each round t = 1, 2, ..., T:
    1. Server broadcasts global model θ_t to all clients
    2. For each client k = 1, ..., K (in parallel):
        - Download θ_t
        - Train on local data: θ_k^(t+1) = LocalTrain(θ_t, D_k)
        - Send θ_k^(t+1) to server
    3. Server aggregates:
        θ_(t+1) = (1/K) * Σ(θ_k^(t+1))
    4. Evaluate θ_(t+1)
```

## Advantages of Federated Learning

- **Privacy Preservation**: Data remains on client devices
- **Reduced Communication Costs**: Only model parameters are shared
- **Leverages Distributed Data**: Can learn from data across many sources
- **Regulatory Compliance**: Helps meet data privacy regulations (GDPR, HIPAA)

## Limitations & Future Enhancements

Current limitations:
- Assumes all clients are available in each round
- Uses simple averaging (doesn't weight by data size)
- No handling of stragglers or failures
- Assumes IID-like data distribution

Possible enhancements:
- Weighted averaging based on client data sizes
- Client sampling (random subset per round)
- Non-IID data handling
- Differential privacy mechanisms
- Secure aggregation protocols
- Support for more complex models (CNNs, etc.)

## References

- **FedAvg Paper**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
- **Federated Learning Survey**: Kairouz et al., "Advances and Open Problems in Federated Learning" (2019)

## License

This project is for educational purposes.

## Author

Created for Federated Learning Assignment - VIIT, 6th Semester

## Contact

For questions or suggestions, please open an issue or contact the course instructor.
