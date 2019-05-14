"""
Name: Ines Ayara
Project 4: Classification with Neural Networks.
How to use this program: To collect accuracy data from running the NN on a data
set, follow these instructions:
    1. In main, set input and output sizes where specified.
    2. Run with: python3 project4.py <my_data_set>.csv
It should produce the accuracy for a range of network strcutures, along with the
number of layers and nodes per layer that were used.
"""
import csv, sys, random, math
from sklearn import preprocessing
import numpy as np
import pandas as pd

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom

def logistic_vect(v1):
    """Logistic / sigmoid function"""
    v2 = []
    for val in v1:
        v2.append(logistic(val))
    return v2

def accuracy(nn, pairs):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.

    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.

    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""
    #print("Testing Accuracy...")
    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        nn.forward_propagate(x)
        class_prediction = nn.predict_class()
        # For bit-incrementer:
        if class_prediction != y:
        # If dealing with a non-classification problem, uncomment this:
        #if class_prediction != y[0]:
        # If dealing with a classification problem, uncomment this:
        #if class_prediction != y.index(max(y)):
            true_positives += 1

        outputs = nn.get_outputs()
        #print("y =", y, ",class_pred =", class_prediction, "outputs =", outputs)

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here
class NeuralNetwork:
    def __init__(self, nn_dims):

        self.layers = nn_dims
        self.total_layers = len(nn_dims)
        self.input_size = nn_dims[0]
        self.output_size = nn_dims[-1]
        self.num_hidden_layers = len(nn_dims[1:-1])
        self.hidden_layers = nn_dims[1:-1]

        self.link_weights = []
        self.inputs = []
        self.output = []
        self.outputs = []
        self.deltas = []

    def get_matrix_of_weights(self, rows, cols):
        """ Given a number of rows and columns, returns a mtrix of random
            weights uniformly distributed between 0 and 1. """

        weights = []
        for row in range(rows):
            weights.append([])
            for col in range(cols):
                weights[row].append(random.uniform(0, 1))

        return weights

    def set_network_random_weights(self):
        """ Sets the weights for all node/layer links based on number of hidden
        layers provided and their respective number of nodes."""

        layer_num = 0
        current_layer = self.hidden_layers[layer_num]
        w_matrix = self.get_matrix_of_weights(current_layer,
                                              self.input_size + 1)
        # Set weights for the links between the input layer and the first
        # hidden layer.
        self.link_weights.append(w_matrix)

        layer_num += 1
        # Set weights of the links between consecutive hidden layers.
        while layer_num < self.num_hidden_layers:
            prev_layer = current_layer
            current_layer = self.hidden_layers[layer_num]

            self.link_weights.append(self.get_matrix_of_weights(current_layer,
                                                                prev_layer + 1))
            layer_num += 1

        # Set weights for the links between the last hidden layer and the
        # output layer.
        self.link_weights.append(self.get_matrix_of_weights(self.output_size,
                                                           current_layer + 1))

    def forward_propagate(self, x):
        """ Propagate the inputs forward to compute the outputs."""

        self.outputs = [x]
        self.inputs = [x]
        self.output = x
        for l in range(self.total_layers - 1):
            z = []
            for row in self.link_weights[l]:
                sum = 0
                for node in range(len(self.output)):
                    sum += self.output[node] * row[node]
                z.append(sum)
            self.inputs.append(z)
            self.output = logistic_vect(z) + [1] # Add dummy node
            self.outputs.append(self.output)

        # We don't want to have a dummy value (1) with our final output.
        self.output = self.output[:-1]

    def back_propagation_learning(self, training):
        """ Propagate deltas backward from the output layer to input layer."""

        alpha = 1
        t = 0
        self.set_network_random_weights()
        while t < 1000:
            for (x, y) in training:
                self.forward_propagate(x)
                self.deltas = []
                # Calculate the error vect delta from the output layer.
                delta = [0 for _ in range(len(self.output))]
                for j in range(len(self.output)):
                    deriv_sigmoid = logistic(self.inputs[-1][j]) * (1 - logistic(self.inputs[-1][j]))
                    delta[j] = deriv_sigmoid * (y[j] - self.output[j])

                # Propagate deltas from last hidden layer to input layer.
                self.deltas.append(delta)
                for l in range(self.total_layers - 2, -1, -1):
                    delta = [0 for _ in range(len(self.outputs[l]))]
                    for i in range(len(self.inputs[l])):
                        deriv_sigmoid = logistic(self.inputs[l][i]) * (1 - logistic(self.inputs[l][i]))
                        s = 0
                        for j in range(len(self.inputs[l + 1])):
                            s += float(self.deltas[-1][j]) * self.link_weights[l][j][i]
                        delta[i] = deriv_sigmoid * s
                    self.deltas.append(delta)

                self.deltas = self.deltas[::-1]

                # Update every weight in network using deltas
                for l in range(self.total_layers - 1):
                    for j in range(len(self.inputs[l + 1])):
                        for i in range(len(self.outputs[l])):
                            self.link_weights[l][j][i] += alpha * self.outputs[l][i] * self.deltas[l + 1][j]

            alpha = 1000 / (1000 + t)
            t += 1

    def predict_class(self):
        """ Returns the predicted value, whether it's a class or a rounded
            output value. """
        # If dealing with a non-classification problem, uncomment this:
        return float(round(self.output[0]))
        # If dealing with a classification problem, uncomment this:
        #return self.output.index(max(self.output))
        # If testing the 3-bit-increment:
        #return [float(round(x)) for x in self.output]

    def get_outputs(self):
        return self.output

def normalize_wine_dataset():
    # Get dataset
    df = pd.read_csv("wine.csv", sep=",")
    cols = []
    rows = [[]]

    # Normalize input columns
    for num in range(1, 14):
        rows[0].append('x' + str(num))
        x_array = np.array(df['x' + str(num)])
        normalized_X = preprocessing.normalize([x_array])
        df['x' + str(num)] = pd.Series(normalized_X[0])
        cols.append(df['x' + str(num)].tolist())

    # Get output columns
    cols.append(np.array(df['target1']).tolist())
    cols.append(np.array(df['target2']).tolist())
    cols.append(np.array(df['target3']).tolist())

    # Write normalized data to new .csv
    rows[0].append('target1')
    rows[0].append('target2')
    rows[0].append('target3')
    for i in range(len(cols[0])):
        row = []
        for col in cols:
            row.append(col[i])
        rows.append(row)

    with open('normalized_wine_data.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(rows)
    writeFile.close()

def main():
    header, data = read_data(sys.argv[1], ",")
    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]

    # Uncomment only if running the wine data (normalized_wine_data.csv).
    #random.shuffle(training)

    # K-fold Cross Validation: (not for 3-bit-increment)
    k = 5
    offset = len(training) // k
    data = [training[x:x+offset] for x in range(0, len(training), offset)]

    training_sets = data[:-1]
    validation_set = data[-1]

    # Data collection Script:
    # in = 2, out = 1 should work for generated and banana.
    input = [2]  # Set length of input.
    output = [1] # Set length of output.
    for node in range(1, 11):
        for layer in range(1, 4):
            dims = input + [node * layer] + output
            nn = NeuralNetwork(dims)
            avg_accuracy = 0
            for set in training_sets:
                nn.back_propagation_learning(set)
                avg_accuracy += accuracy(nn, validation_set)
            avg_accuracy = avg_accuracy / len(training_sets)

            print("Layers:", layer, "Nodes:", node)
            print("Accuracy:", avg_accuracy)
            print("**********")

if __name__ == "__main__":
    main()
