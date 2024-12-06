import time
from idlelib.iomenu import errors
from math import exp
import numpy as np
import pandas as pd
import random
from fontTools.varLib.avar import mappings_from_avar
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt

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
    # print(1 / (1 + exp(-z)), end = "\n------\n")
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

def train_network(network, train_data, test_data, learning_rate, epoch_count, outputs_count, fold_ord):
    # best_epoch_error = max_float
    # best_epoch_network = network
    train_errors = []
    test_errors = []
    mean_train_errors = []
    mean_test_errors = []
    for i in range(epoch_count):
        sum_error_train = 0
        sum_error_test = 0
        wrong_predictions_train = 0
        wrong_predictions_test = 0
        # print(train_data)
        for row in train_data:
            row = row.tolist()
            outputs = forward_propagate(network, row)
            expected = [0 for k in range(outputs_count)]
            # print(int(row[-1]))
            expected[int(row[-1]) - 1] = 1
            sum_error_train += sum([(expected[j] - outputs[j]) ** 2 for j in range(len(expected))])
            backward_propagate(network, expected)
            update_weights(network, row, learning_rate)
            if (int(row[-1]) - 1) != outputs.index(max(outputs)):
                wrong_predictions_train += 1
        mse_train = sum_error_train / len(train_data)
        train_errors.append(mse_train)
        mean_train_errors.append(wrong_predictions_train / len(train_data))

        # testing network on test_data
        for row in test_data:
            row = row.tolist()
            outputs = forward_propagate(network, row)
            expected = [0 for l in range(outputs_count)]
            expected[int(row[-1]) - 1] = 1
            sum_error_test += sum([(expected[j] - outputs[j]) ** 2 for j in range(len(expected))])
            if (int(row[-1]) - 1) != outputs.index(max(outputs)):
                wrong_predictions_test += 1
        mse_test = sum_error_test / len(test_data)
        test_errors.append(mse_test)
        mean_test_errors.append(wrong_predictions_test / len(test_data))
        print(f"Epoch: {i}; Learning rate: {learning_rate}; MSE_train: {mse_train}; MSE_test: {mse_test}") # Mean Squared Error (MSE)
    #     if sum_error < best_epoch_error:
    #         best_epoch_error = sum_error
    #         best_epoch_network = network
    # return best_epoch_network

    # Plotting the training error convergence
    plt.plot(range(epoch_count), train_errors, label="Training Error", color="limegreen")
    plt.plot(range(epoch_count), test_errors, label="Testing Error", color="cornflowerblue")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    # plt.title("Training Error Convergence")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("plots/mse/" + str(epoch_count) + "_" + str(learning_rate) + "_fold" + str(fold_ord) + ".png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.plot(range(epoch_count), mean_train_errors, label="Training Error", color="limegreen")
    plt.plot(range(epoch_count), mean_test_errors, label="Testing Error", color="cornflowerblue")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Error")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/mean_err/" + str(epoch_count) + "_" + str(learning_rate) + "_fold" + str(fold_ord) + ".png", dpi=300, bbox_inches='tight')
    plt.close()

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
    mapping = {breed: i + 1 for i, breed in enumerate(breeds)} # (i / (len(breeds) - 1))
    data["Breed"] = data["Breed"].map(mapping)
    print(data)
    return data

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs)) + 1

# def visualize_misclassified_points(test_data, predictions):
#     misclassified = []
#     correctly_classified = []
#
#     for i, row in enumerate(test_data.values):
#         row_features = row[:-1]
#         true_label = row[-1]
#         predicted_label = predictions[i]
#
#         if true_label != predicted_label:
#             misclassified.append(row_features)
#         else:
#             correctly_classified.append(row_features)
#
#     misclassified = np.array(misclassified)
#     correctly_classified = np.array(correctly_classified)
#
#     # Plotting misclassified and correctly classified points
#     if misclassified.shape[0] > 0:
#         plt.scatter(misclassified[:, 0], misclassified[:, 1], color='red', label='Misclassified')
#     if correctly_classified.shape[0] > 0:
#         plt.scatter(correctly_classified[:, 0], correctly_classified[:, 1], color='green', label='Correctly Classified')
#     print("Misclassified 0:\n", misclassified[:, 0])
#     print("Misclassified 1:\n", misclassified[:, 1])
#     print("Correctly classified 0:\n", correctly_classified[:, 0])
#     print("Correctly classified 1:\n", correctly_classified[:, 1])
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.title('Misclassified vs Correctly Classified Points')
#     plt.legend()
#     plt.show()

random.seed(60)
k = 5
learning_rate = 0.01
epoch_count = 500

start = time.time()
folds = get_k_folds(k)
scores = []
x = k - 1

df = pd.read_excel('xlsx/main.xlsx', engine='openpyxl')
unique_breeds = df["Breed"].unique()
breed_ids = {breed: i for i, breed in enumerate(unique_breeds)}

for (X_train, X_test, y_train, y_test) in folds:
    output_str = ""
    n_inputs = len(X_train.values[0])
    n_outputs = len(y_train.unique())
    network = init_network(n_inputs, (n_inputs + n_outputs) // 2, n_outputs)
    training_data = normalize_data(pd.concat([X_train, y_train], axis=1))
    testing_data = normalize_data(pd.concat([X_test, y_test], axis=1))
    train_network(network, training_data.values, testing_data.values, learning_rate, epoch_count, n_outputs, k - x)

    predictions = []
    breeds = testing_data["Breed"].unique()
    # breeds_count = len(testing_data["Breed"].unique())
    each_breed_misc_count = [0 for i in range(len(breeds))]
    for row in testing_data.values:
        row = row.tolist()
        prediction = predict(network, row)
        # print(prediction)
        predictions.append(prediction == (int(row[-1])))
        if prediction != (int(row[-1])):
            each_breed_misc_count[int(row[-1]) - 1] += 1
        output_str += "Expected: " + str(row[-1]) + " Got: " + str(prediction) + "\n"
    scores.append(sum(predictions) / len(predictions))
    output_str += "\nAccuracy: " + str(sum(predictions) / len(predictions))
    print(sum(predictions) / len(predictions))

    breed_counts = testing_data["Breed"].value_counts(sort=False)
    sorted_counts = breed_counts.sort_index()
    # print("each_breed_misc_count:", each_breed_misc_count)
    # print("breed_counts:", breed_counts)
    # print("sorted_counts:", sorted_counts)
    for br in range(len(each_breed_misc_count)):
        each_breed_misc_count[br] /= breed_counts.iloc[br]
    breed_ids_sorted = sorted(breed_ids.items(), key=lambda w: w[1])

    plt.figure(figsize=(12, 6))
    plt.bar([key for key, value in breed_ids_sorted], each_breed_misc_count, color='skyblue', edgecolor='black', alpha=0.8)
    plt.xlabel('Breeds')
    plt.ylabel('Error rate')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/" + str(epoch_count) + "_" + str(learning_rate) + "_fold" + str(k - x) + ".png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save results
    with open("results/" + str(epoch_count) + "_" + str(learning_rate) + "_fold" + str(k - x) + ".txt", "w") as f:
        f.write(output_str)
        if(x == 0):
            f.write("\nFinal score: " + str(sum(scores) / len(scores)))
        x -= 1

print(sum(scores) / len(scores))

print("Done")
end = time.time()
print(f"Execution time: {end -  start}")


