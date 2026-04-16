import numpy as np

# -----------------------------
# Simple Linear Model
# y = wx + b
# -----------------------------
class LinearModel:
    def __init__(self):
        self.w = np.random.randn()
        self.b = np.random.randn()

    def predict(self, x):
        return self.w * x + self.b

    def get_params(self):
        return np.array([self.w, self.b])

    def set_params(self, params):
        self.w, self.b = params


# -----------------------------
# Client
# -----------------------------
class Client:
    def __init__(self, client_id, x_data, y_data):
        self.client_id = client_id
        self.x = x_data
        self.y = y_data
        self.model = LinearModel()

    def local_train(self, epochs=5, lr=0.01):
        """Train model locally"""
        for _ in range(epochs):
            y_pred = self.model.predict(self.x)
            error = y_pred - self.y

            # gradients
            dw = np.mean(error * self.x)
            db = np.mean(error)

            # update
            self.model.w -= lr * dw
            self.model.b -= lr * db

        return self.model.get_params()


# -----------------------------
# Server
# -----------------------------
class Server:
    def __init__(self):
        self.global_model = LinearModel()

    def aggregate(self, client_params):
        """FedAvg aggregation"""
        avg_params = np.mean(client_params, axis=0)
        self.global_model.set_params(avg_params)

    def broadcast(self, clients):
        """Send global model to all clients"""
        global_params = self.global_model.get_params()
        for client in clients:
            client.model.set_params(global_params)


# -----------------------------
# Data Generator
# -----------------------------
def generate_client_data(n_samples=100, noise=0.5):
    x = np.random.rand(n_samples) * 10
    y = 2.5 * x + 5 + np.random.randn(n_samples) * noise
    return x, y


# -----------------------------
# Federated Learning Simulation
# -----------------------------
NUM_CLIENTS = 5
ROUNDS = 10

# Create clients
clients = []
for i in range(NUM_CLIENTS):
    x, y = generate_client_data()
    clients.append(Client(i, x, y))

# Create server
server = Server()

# Initial broadcast
server.broadcast(clients)

# Federated rounds
for round_num in range(ROUNDS):
    print(f"\n🌍 Federated Round {round_num+1}")

    client_params = []

    # Local training
    for client in clients:
        params = client.local_train(epochs=5, lr=0.01)
        client_params.append(params)
        print(f"Client {client.client_id} local params: {params}")

    # Aggregation
    server.aggregate(client_params)
    global_params = server.global_model.get_params()
    print(f"\n🧠 Global model params after aggregation: {global_params}")

    # Broadcast updated model
    server.broadcast(clients)

print("\n✅ Federated Learning Simulation Complete")
