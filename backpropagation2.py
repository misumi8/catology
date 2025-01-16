import torch
import torch.nn as nn
import torch.optim as optim
import time
from math import exp
from pprint import pprint
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def get_k_folds(k):
    df = pd.read_excel('xlsx/new_main.xlsx', engine='openpyxl')
    k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    X = df.drop(columns=["Breed"])
    y = df["Breed"]
    folds = []
    for train_index, test_index in k_fold.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        folds.append((X_train, X_test, y_train, y_test))
    return folds

def normalize_data(data):
    columns = data.columns.drop("Breed")
    breeds = data["Breed"].unique()
    for column in columns:
        content = data[column].unique()
        mapping = {k: (v / (len(content) - 1)) for v, k in enumerate(sorted(content))}
        data[column] = data[column].map(mapping)
    mapping = {breed: i + 1 for i, breed in enumerate(breeds)}
    data["Breed"] = data["Breed"].map(mapping)
    return data.drop(columns = ["Breed"]), data["Breed"]

def get_input_output_size():
    df = pd.read_excel('xlsx/new_main.xlsx', engine='openpyxl')
    return len(df.columns) - 1, len(df["Breed"].unique())

from sklearn.preprocessing import OneHotEncoder

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        hidden_out = torch.relu(self.hidden(x))
        output = self.softmax(self.output(hidden_out))
        return output

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Предположим, что X — ваши данные без целевой переменной (признаки)
df = pd.read_excel('xlsx/new_main.xlsx', engine='openpyxl')

scaler = StandardScaler()
X_scaled, y = normalize_data(df)

print(X_scaled) # if worse use normalization function

# Применяем t-SNE для уменьшения размерности до 2
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Визуализация
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=15)  # y — целевая переменная (классы пород)
plt.colorbar(label='Breed')
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

# k = 5
# folds = get_k_folds(k)
# input_size, output_size = get_input_output_size()
# hidden_size = round((input_size + output_size) / 2)
#
# model = NeuralNetwork(input_size, hidden_size, output_size)
# criterion = nn.CrossEntropyLoss()  # For multiclass classification
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# for (X_train, X_test, y_train, y_test) in folds:
#     X_train, y_train = normalize_data(pd.concat([X_train, y_train], axis=1))
#     X_test, y_test = normalize_data(pd.concat([X_test, y_test], axis=1))
#
#     for epoch in range(500):
#         # Convert data to torch tensors
#         inputs = torch.tensor(X_train.values, dtype=torch.float32)
#         targets = torch.tensor(y_train.values, dtype=torch.long)
#
#         # Forward pass
#         outputs = model(inputs)
#         print(outputs)
#         loss = criterion(outputs, targets)
#
#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if epoch % 20 == 0:
#             print(f'Epoch {epoch}, Loss: {loss.item()}')
#
#     # Prediction
#     with torch.no_grad():
#         outputs = model(torch.tensor(X_test.values, dtype=torch.float32))
#         _, predicted = torch.max(outputs, 1)
#         accuracy = (predicted == torch.tensor(y_test.values)).sum().item() / len(y_test)
#         print(f'Accuracy: {accuracy}')


# for (X_train, X_test, y_train, y_test) in folds:
#     X_train, y_train = normalize_data(pd.concat([X_train, y_train], axis=1))
#
#     nn = NeuralNetwork(input_size, hidden_size, output_size)
#     nn.train(X_train, y_train, epochs=500, learning_rate=0.01)
#
#     X_test, y_test = normalize_data(pd.concat([X_test, y_test], axis=1))
#     output = nn.feedforward(X_test)
#
#     correct_preditions = []
#     for i in range(len(output)):
#         if output.iloc[i] == y_test.iloc[i]:
#             correct_preditions[i] = 1
#         else:
#             correct_preditions[i] = 0
#
#     print(f"Score: {sum(correct_preditions) / len(correct_preditions)}")
#
