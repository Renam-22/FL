import numpy as np





client_data = {

    "Client_1": {
        "x": np.array([1000, 1200, 1500]),
        "y": np.array([200, 240, 300])
    },

    "Client_2": {
        "x": np.array([1800, 2000, 2200]),
        "y": np.array([360, 400, 440])
    },

    "Client_3": {
        "x": np.array([2500, 2700, 3000]),
        "y": np.array([500, 540, 600])
    }
}






class LinearRegressionModel:

    def __init__(self):

        self.weight = 0.0
        self.bias = 0.0


    def predict(self, x):

        return self.weight * x + self.bias


    def train(self, x, y, learning_rate, epochs):

        n = len(x)

        for epoch in range(epochs):

            predictions = self.predict(x)

            error = predictions - y

            
            dw = (2/n) * np.sum(error * x)
            db = (2/n) * np.sum(error)

            
            self.weight = self.weight - learning_rate * dw
            self.bias = self.bias - learning_rate * db






global_model = LinearRegressionModel()

learning_rate = 0.0000001
epochs = 10

print("Initial Global Weight:", global_model.weight)
print("Initial Global Bias:", global_model.bias)







local_weights = []
local_biases = []

for client, data in client_data.items():

    print(f"\nTraining on {client}")
    local_model = LinearRegressionModel()

    local_model.weight = global_model.weight
    local_model.bias = global_model.bias


    local_model.train(
        data["x"],
        data["y"],
        learning_rate,
        epochs
    )

    print("Local Weight:", local_model.weight)
    print("Local Bias:", local_model.bias)


    local_weights.append(local_model.weight)
    local_biases.append(local_model.bias)





global_model.weight = np.mean(local_weights)
global_model.bias = np.mean(local_biases)

print("\nUpdated Global Model")
print("Global Weight:", global_model.weight)
print("Global Bias:", global_model.bias)






test_house_size = 2800

predicted_price = global_model.predict(test_house_size)

print("\nTesting Global Model")
print("House Size:", test_house_size)
print("Predicted House Price:", predicted_price)