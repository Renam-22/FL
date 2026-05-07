


import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




data = load_diabetes()

X = data.data
y = data.target




y = (y > y.mean()).astype(int)



scaler = StandardScaler()

X = scaler.fit_transform(X)





X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)




X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1,1)





hospital1_X = X_train[:100]
hospital1_y = y_train[:100]

hospital2_X = X_train[100:200]
hospital2_y = y_train[100:200]

hospital3_X = X_train[200:]
hospital3_y = y_train[200:]

hospital_datasets = [

    TensorDataset(hospital1_X, hospital1_y),

    TensorDataset(hospital2_X, hospital2_y),

    TensorDataset(hospital3_X, hospital3_y)
]

hospital_sizes = [100, 100, len(hospital3_X)]





class DiabetesModel(nn.Module):

    def __init__(self):

        super(DiabetesModel, self).__init__()

        self.fc1 = nn.Linear(10, 16)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.relu(self.fc1(x))

        x = self.sigmoid(self.fc2(x))

        return x





def local_training(model, dataset):

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(3):

        for X, y in loader:

            optimizer.zero_grad()

            output = model(X)

            loss = criterion(output, y)

            loss.backward()

            optimizer.step()

    return model.state_dict()





def federated_averaging(global_model, local_models, hospital_sizes):

    total_data = sum(hospital_sizes)

    global_state = copy.deepcopy(global_model.state_dict())

    for key in global_state.keys():

        global_state[key] = torch.zeros_like(global_state[key])

        for local_model, size in zip(local_models, hospital_sizes):

            weight = size / total_data

            global_state[key] += local_model[key] * weight

    global_model.load_state_dict(global_state)

    return global_model





global_model = DiabetesModel()

rounds = 3

for round_num in range(rounds):

    print(f"\nFederated Round {round_num+1}")

    local_models = []


    for i in range(3):

        print(f"Hospital {i+1} Training...")

        local_model = copy.deepcopy(global_model)

        updated_weights = local_training(
            local_model,
            hospital_datasets[i]
        )

        local_models.append(updated_weights)



    global_model = federated_averaging(
        global_model,
        local_models,
        hospital_sizes
    )

    print("Global Model Updated")






global_model.eval()

with torch.no_grad():

    predictions = global_model(X_test)

    predicted = (predictions > 0.5).float()

    accuracy = (predicted == y_test).float().mean()

print("\nTest Accuracy:", accuracy.item())