import time
from idlelib.iomenu import errors
from math import exp

import numpy as np
import pandas as pd
import random

from fontTools.varLib.avar import mappings_from_avar
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def get_k_folds(k):
    df = pd.read_excel('xlsx/main.xlsx', engine='openpyxl')
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    X = df.drop(columns=["Breed"])
    y = df["Breed"]
    folds = []
    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        folds.append((X_train, X_test, y_train, y_test))
    return folds

# layer = строка из нашего датасета
def init_network(input_size, hidden_size, output_size):
    network = []
    hidden_layer = [{"weights": [random.uniform(-1,1) for i in range(input_size + 1)]} for i in range(hidden_size)]
    output_layer = [{"weights": [random.uniform(-1,1) for i in range(hidden_size + 1)]} for i in range(output_size)]
    network.append(hidden_layer)
    network.append(output_layer)
    return network

def activate_neuron(weights, inputs):
    return np.dot(weights[:-1], inputs) + weights[-1]
    # activation = weights[-1] # We assume that bias is the last weight
    # for i in range(len(weights) - 1):
    #     activation += weights[i] * inputs[i]
    # return activation

# Non-linear activation function
def sigmoid(z): # ReLu try
    return 1 / (1 + exp(-z))

def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            z = activate_neuron(neuron["weights"], inputs)
            neuron["output"] = sigmoid(z)
            new_inputs.append(neuron["output"])
        inputs = new_inputs
    return inputs

def sigmoid_derivative(s):
    return s * (1 - s)

def backward_propagate(network, expected):
    # print(f"Backward_propagate: {expected}")
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0
                for neuron in network[i + 1]:
                    error += (neuron["weights"][j] * neuron["delta"])
                errors.append(error)
        else:  # output layer in network
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron["output"] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron["delta"] = errors[j] * sigmoid_derivative(neuron["output"])

def update_weights(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1] # all but the last (target) attribute
        if i != 0:
            inputs = [neuron["output"] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron["weights"][j] -= learning_rate * neuron["delta"] * inputs[j]
            neuron["weights"][-1] -= learning_rate * neuron["delta"]

def train_network(network, train_data, learning_rate, epoch_count, outputs_count):
    # best_epoch_error = max_float
    # best_epoch_network = network
    for i in range(epoch_count):
        sum_error = 0
        # print(train_data)
        for row in train_data:
            row = row.tolist()
            outputs = forward_propagate(network, row)
            expected = [0 for k in range(outputs_count)]
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[j] - outputs[j]) ** 2 for j in range(len(expected))])
            backward_propagate(network, expected)
            update_weights(network, row, learning_rate)
        print(f"Epoch: {i}; Learning rate: {learning_rate}; MSE: {sum_error / len(train_data)}") # Mean Squared Error (MSE)
    #     if sum_error < best_epoch_error:
    #         best_epoch_error = sum_error
    #         best_epoch_network = network
    # return best_epoch_network
def normalize_data(data):
    data = data.drop(columns=["ID"])
    columns = data.columns
    for column in columns:
        content = data[column].unique()
        mapping = {k: (v / (len(content) - 1)) for v, k in enumerate(sorted(content))}  # (i / (len(breeds) - 1))
        data[column] = data[column].map(mapping)
        # if(column == "PredBird"):
        #     print(sorted(content), "\n", mapping)
    breeds = data["Breed"].unique()
    mapping = {breed: i for i, breed in enumerate(breeds)} # (i / (len(breeds) - 1))
    data["Breed"] = data["Breed"].map(mapping)
    print(data)
    return data

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# for (a,b,c,d) in folds:
#     print(len(a), len(b), len(c), len(d))

random.seed(30)
k = 5
learning_rate = 0.01
epoch_count = 500

start = time.time()
folds = get_k_folds(k)
scores = []
x = k - 1

for (X_train, X_test, y_train, y_test) in folds:
    with open("results/" + str(epoch_count) + "_" + str(learning_rate) + "_fold" + str(k - x) + ".txt", "w") as f:
        n_inputs = len(X_train.values[0])
        n_outputs = len(y_train.unique())
        network = init_network(n_inputs, (n_inputs + n_outputs) // 2, n_outputs)
        training_data = normalize_data(pd.concat([X_train, y_train], axis=1))
        train_network(network, training_data.values, learning_rate, epoch_count, n_outputs)

        testing_data = normalize_data(pd.concat([X_test, y_test], axis=1))
        predictions = []
        breeds_count = len(testing_data["Breed"].unique()) - 1
        for row in testing_data.values:
            row = row.tolist()
            prediction = predict(network, row)
            # print(prediction)
            predictions.append(prediction == int(row[-1]))
            f.write("\nExpected: " + str(row[-1]) + " Got: " + str(prediction))
        scores.append(sum(predictions) / len(predictions))
        f.write("\nAccuracy: " + str(sum(predictions) / len(predictions)))
        print(sum(predictions) / len(predictions))
        if(x == 1):
            f.write("\nFinal score: " + str(sum(scores) / len(scores)))
        x -= 1

print(sum(scores) / len(scores))

print("Done")
end = time.time()
print(f"Execution time: {end -  start}")

# folds = get_k_folds(3)
# learning_rate = 0
# epochs = 0

