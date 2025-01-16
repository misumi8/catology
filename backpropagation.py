import json
import time
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def get_k_folds(k):
    df = pd.read_excel('xlsx/new_main.xlsx', engine='openpyxl')
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    X = df.drop(columns=["Breed"])
    y = df["Breed"]
    folds = []
    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        folds.append((X_train, X_test, y_train, y_test))
    return folds

def init_layer(n_neurons, n_inputs):
    return [{"weights": np.random.uniform(-1, 1, n_inputs + 1).tolist()} for n in range(n_neurons)]

def init_network(input_size, hidden_size, output_size):
    return [init_layer(hidden_size, input_size), init_layer(output_size, hidden_size)]

def activate_neuron(weights, inputs):
    # print(weights[:-1], end="\n____________________\n")
    # print(inputs)
    return np.dot(weights[:-1], inputs) + weights[-1]
    # activation = weights[-1] # We assume that bias is the last weight
    # for i in range(len(weights) - 1):
    #     activation += weights[i] * inputs[i]
    # return activation

# Non-linear activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagate(network, row):
    inputs = row
    # print([i for i in row if i > 1])
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
    best_network = None
    best_test_error = float("inf")
    train_errors, test_errors = [], []

    for epoch in range(epoch_count):
        sum_error_train, sum_error_test = 0, 0
        wrong_predictions_train, wrong_predictions_test = 0, 0

        for row in train_data:
            row = np.around(row, decimals=4).tolist()
            outputs = forward_propagate(network, row[:-1])
            expected = [0] * outputs_count
            expected[int(row[-1]) - 1] = 1

            sum_error_train += sum((expected[j] - outputs[j]) ** 2 for j in range(outputs_count))
            backward_propagate(network, expected)
            update_weights(network, row, learning_rate)

            if (int(row[-1]) - 1) != outputs.index(max(outputs)):
                wrong_predictions_train += 1

        for row in test_data:
            row = np.around(row, decimals=4).tolist()
            outputs = forward_propagate(network, row[:-1])
            expected = [0] * outputs_count
            expected[int(row[-1]) - 1] = 1

            sum_error_test += sum((expected[j] - outputs[j]) ** 2 for j in range(outputs_count))
            if (int(row[-1]) - 1) != outputs.index(max(outputs)):
                wrong_predictions_test += 1

        mse_train = sum_error_train / len(train_data)
        mse_test = sum_error_test / len(test_data)
        train_errors.append(mse_train)
        test_errors.append(mse_test)

        # Early stopping
        if mse_test < best_test_error:
            best_test_error = mse_test
            best_network = [layer.copy() for layer in network]

        print(f"Epoch {epoch + 1}/{epoch_count}, Train Error: {(wrong_predictions_train/len(train_data)):.4f}, Test Error: {(wrong_predictions_test/len(test_data)):.4f}")

    # MSE training and testing plot
    plt.plot(range(epoch_count), train_errors, label="Train Error", color="limegreen")
    plt.plot(range(epoch_count), test_errors, label="Test Error", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/MSE_{str(epoch_count)}_{str(learning_rate)}_fold{str(k - x)}.png", dpi=300)
    plt.close()

    return best_network

def normalize_data(data):
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    if "Breed" in data.columns:
        label_encoder = LabelEncoder()
        data["Breed"] = label_encoder.fit_transform(data["Breed"]) + 1

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

if __name__ == "__main__":
    random.seed(60) #60
    k = 15
    learning_rate = 0.01
    epoch_count = 1000

    start = time.time()
    folds = get_k_folds(k)
    scores = []
    x = k - 1

    df = pd.read_excel('xlsx/new_main.xlsx', engine='openpyxl')
    unique_breeds = df["Breed"].unique()
    breed_ids = {breed: i for i, breed in enumerate(unique_breeds)}

    for (X_train, X_test, y_train, y_test) in folds:
        output_str = ""
        n_inputs = len(X_train.values[0])
        n_outputs = len(y_train.unique())
        network = init_network(n_inputs, int((n_inputs + n_outputs) / 2) * 4 + 1 , n_outputs)
        training_data = normalize_data(pd.concat([X_train, y_train], axis=1))
        testing_data = normalize_data(pd.concat([X_test, y_test], axis=1))
        best_network = train_network(network, training_data.values, testing_data.values, learning_rate, epoch_count, n_outputs, k - x)

        predictions = []
        breeds = testing_data["Breed"].unique()
        # breeds_count = len(testing_data["Breed"].unique())
        each_breed_misc_count = [0 for i in range(len(breeds))]
        for row in testing_data.values:
            row = np.around(row, decimals=4).tolist()
            prediction = predict(network, row[:-1])
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

        data = {
            "score": str(sum(scores) / len(scores)),
            "input_layer_size": n_inputs,
            "hidden_layer_size": int((n_inputs + n_outputs) / 2) * 4 + 1 ,
            "output_layer_size": n_outputs,
            "weights": {
                "input_to_hidden": best_network[0],
                "hidden_to_output": best_network[1]
            }
        }

        with open("results/networks/" + str(epoch_count) + "_" + str(learning_rate) + ".json", "w") as jf:
            json.dump(data, jf, indent=4)

    print(sum(scores) / len(scores))

    end = time.time()
    print(f"Execution time: {end -  start}")


