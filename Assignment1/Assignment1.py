import numpy as np

client_data = {
    "Client_1": {
        "x": np.array([1, 2, 3]),
        "y": np.array([2, 4, 6])
    },
    
    "Client_2": {
        "x": np.array([4, 5, 6]),
        "y": np.array([8, 10, 12])
    },
    
    "Client_3": {
        "x": np.array([7, 8, 9]),
        "y": np.array([14, 16, 18])
    }
}


global_weight = 0.0

learning_rate = 0.01
epochs = 5

print("Initial Global Weight:", global_weight)


def train_local_model(x, y, weight):

    local_weight = weight

    for epoch in range(epochs):

        predictions = local_weight * x

        error = predictions - y

        gradient = (2 / len(x)) * np.sum(error * x)

        local_weight = local_weight - learning_rate * gradient

    return local_weight



local_weights = []

for client, data in client_data.items():

    print(f"\nTraining on {client}")

    updated_weight = train_local_model(
        data["x"],
        data["y"],
        global_weight
    )

    print("Updated Local Weight:", updated_weight)

    local_weights.append(updated_weight)


global_weight = np.mean(local_weights)

print("\nAggregated Global Weight:", global_weight)


test_x = 10

prediction = global_weight * test_x

print("\nTesting Global Model")
print("Input:", test_x)
print("Predicted Output:", prediction)