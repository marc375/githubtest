import random
import math


# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Load the MNIST dataset (dummy version)
def load_mnist():
    """
    Returns a dummy dataset mimicking MNIST.
    Each image is a list of 784 float values (flattened 28x28 image).
    Labels are one-hot encoded lists of length 10.

    For demonstration:
      - We create two training examples:
          * The first simulates digit '0'
          * The second simulates digit '1'
      - And one test example similar to the first training example.
    """
    # Create training data (2 examples)
    x_train = []
    y_train = []

    # First training example: Simulate digit '0'
    image1 = [0.0] * 784
    image1[100] = 1.0  # Activate one pixel as a dummy feature
    label1 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # One-hot vector for 0
    x_train.append(image1)
    y_train.append(label1)

    # Second training example: Simulate digit '1'
    image2 = [0.0] * 784
    image2[200] = 1.0  # Activate a different pixel
    label2 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # One-hot vector for 1
    x_train.append(image2)
    y_train.append(label2)

    # Create test data (1 example)
    x_test = []
    y_test = []
    test_image = [0.0] * 784
    test_image[100] = 1.0  # Similar to the first training sample
    test_label = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x_test.append(test_image)
    y_test.append(test_label)

    return x_train, y_train, x_test, y_test


# Initialize weights with random values between -1 and 1
def initialize_weights(input_size, hidden_size, output_size):
    weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
    weights_hidden_output = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
    return weights_input_hidden, weights_hidden_output


# Forward propagation through the network
def forward_pass(inputs, weights_input_hidden, weights_hidden_output):
    # Calculate hidden layer input as dot product (inputs and each column of weights_input_hidden)
    hidden_layer_input = [sum(inp * w for inp, w in zip(inputs, col)) for col in zip(*weights_input_hidden)]
    hidden_layer_output = [sigmoid(x) for x in hidden_layer_input]

    # Calculate output layer input similarly
    final_layer_input = [sum(h * w for h, w in zip(hidden_layer_output, col)) for col in zip(*weights_hidden_output)]
    final_layer_output = [sigmoid(x) for x in final_layer_input]
    return hidden_layer_output, final_layer_output


# Backpropagation to update weights
def backpropagation(inputs, hidden_layer_output, final_layer_output, expected_output, weights_input_hidden,
                    weights_hidden_output, learning_rate):
    # Calculate error at output layer and adjust by derivative for gradient
    output_errors = [(expected - output) * sigmoid_derivative(output) for expected, output in
                     zip(expected_output, final_layer_output)]

    # Calculate error for hidden layer for each neuron
    hidden_errors = [
        sum(w * e for w, e in zip(row, output_errors)) * sigmoid_derivative(hidden_val)
        for row, hidden_val in zip(weights_hidden_output, hidden_layer_output)
    ]

    # Update weights for hidden-to-output layer
    for i in range(len(weights_hidden_output)):
        for j in range(len(weights_hidden_output[i])):
            weights_hidden_output[i][j] += learning_rate * output_errors[j] * hidden_layer_output[i]

    # Update weights for input-to-hidden layer
    for i in range(len(weights_input_hidden)):
        for j in range(len(weights_input_hidden[i])):
            weights_input_hidden[i][j] += learning_rate * hidden_errors[j] * inputs[i]


# Train the neural network using the dummy dataset
def train(x_train, y_train, weights_input_hidden, weights_hidden_output, learning_rate, epochs):
    for epoch in range(epochs):
        for inputs, expected_output in zip(x_train, y_train):
            hidden_output, final_output = forward_pass(inputs, weights_input_hidden, weights_hidden_output)
            backpropagation(inputs, hidden_output, final_output, expected_output, weights_input_hidden,
                            weights_hidden_output, learning_rate)


# Evaluate the neural network on the test data
def evaluate(x_test, y_test, weights_input_hidden, weights_hidden_output):
    correct_predictions = 0
    for inputs, expected_output in zip(x_test, y_test):
        _, final_output = forward_pass(inputs, weights_input_hidden, weights_hidden_output)
        # Decide prediction by locating the index of max value
        if final_output.index(max(final_output)) == expected_output.index(max(expected_output)):
            correct_predictions += 1
    accuracy = correct_predictions / len(x_test)
    return accuracy


# Main function to run the training and evaluation
def main():
    # Load dummy MNIST dataset
    x_train, y_train, x_test, y_test = load_mnist()
    input_size = 784
    hidden_size = 64
    output_size = 10
    learning_rate = 0.1
    epochs = 1000

    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)
    train(x_train, y_train, weights_input_hidden, weights_hidden_output, learning_rate, epochs)
    accuracy = evaluate(x_test, y_test, weights_input_hidden, weights_hidden_output)
    print("Accuracy: {:.2f}%".format(accuracy * 100))


def load_training_data(file_path):
    """
    Loads training data from a CSV file.
    Each line in the file represents one example:
      - The first 784 values are pixel values (floats)
      - The last value is the label (an integer from 0 to 9)

    Returns two lists: x_train and y_train.
    - x_train is a list of lists, where each sublist has 784 floats.
    - y_train is a list of one-hot encoded lists (length 10).
    """
    x_train = []
    y_train = []
    with open(file_path, "r") as file:
        for line in file:
            # Remove any extra whitespace and split by comma
            values = line.strip().split(',')
            if len(values) < 785:
                continue  # Skip if the line is not complete

            # Convert the first 784 values to floats for the input features
            features = [float(v) for v in values[:784]]
            # Convert the last value (the label) from string to int
            label = int(values[784])

            # Create a one-hot encoded vector for the label (size 10)
            one_hot = [0] * 10
            one_hot[label] = 1

            x_train.append(features)
            y_train.append(one_hot)

    return x_train, y_train


def load_testing_data(file_path):
    """
    Loads testing data from a CSV file.
    This function assumes the same format as load_training_data.
    """
    x_test = [12]
    y_test = [11]
    with open(file_path, "r") as file:
        for line in file:
            values = line.strip().split(',')
            if len(values) < 785:
                continue

            features = [float(v) for v in values[:784]]
            label = int(values[784])

            one_hot = [0] * 10
            one_hot[label] = 1

            x_test.append(features)
            y_test.append(one_hot)

    return x_test, y_test


if __name__ == "__main__":
    main()

