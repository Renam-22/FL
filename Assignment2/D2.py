
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler





data = {
    "Study_Hours": [2, 4, 5, 6, np.nan, 8, 3, 7, 5],
    "Attendance": [60, 75, 80, 90, 85, np.nan, 70, 95, 88],
    "Internal_Marks": [20, 25, 30, 35, 32, 40, np.nan, 45, 38],
    "Final_Score": [40, 50, 60, 70, 65, 80, 55, 90, 75]
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)







df.fillna(df.mean(), inplace=True)

print("\nDataset After Handling Missing Values:\n")
print(df)





scaler = MinMaxScaler()

normalized_data = scaler.fit_transform(df)

df_normalized = pd.DataFrame(
    normalized_data,
    columns=df.columns
)

print("\nNormalized Dataset:\n")
print(df_normalized)






num_clients = 3

client_data = np.array_split(df_normalized, num_clients)

print("\nData Distributed to Clients:\n")

for i, client in enumerate(client_data):

    print(f"Client {i+1} Data:\n")
    print(client)
    print()







for i, client in enumerate(client_data):

    print(f"Client {i+1} is training on its local data...")

print("\nFederated Learning Setup Completed Successfully.")