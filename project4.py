"""
Name: Ines Ayara
Project 4: Classification with Neural Networks.
How to use this program:
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

def print_matrix(m):
    for r in m:
        row = ''
        for c in r:
            row += str(c) + ' '
        print(row)

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
        #if class_prediction != y:
        if class_prediction != y[0]:
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
        self.link_weights.append(w_matrix)

        layer_num += 1
        while layer_num < self.num_hidden_layers:
            prev_layer = current_layer
            current_layer = self.hidden_layers[layer_num]

            self.link_weights.append(self.get_matrix_of_weights(current_layer,
                                                                prev_layer + 1))
            layer_num += 1

        self.link_weights.append(self.get_matrix_of_weights(self.output_size,
                                                           current_layer + 1))

    def forward_propagate(self, x):
        """ Given an input vector and the matrix of all weights, updates:
            - self.output = output value
            - self.inputs = outputs before activation
            - self.outputs = outputs after activation
        """

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
            #self.outputs.append(logistic_vect(z))
            # Add dummy node
            self.output = logistic_vect(z) + [1]
            self.outputs.append(self.output)

        self.output = self.output[:-1]

    def back_propagation_learning(self, training):
        alpha = 1
        t = 0
        self.set_network_random_weights()
        while t < 1000:
            for (x, y) in training:
                self.forward_propagate(x)
                self.deltas = []

                # Propagate deltas backward from output layer to input layer
                delta = [0 for _ in range(len(self.output))]
                for j in range(len(self.output)):
                    deriv_sigmoid = logistic(self.inputs[-1][j]) * (1 - logistic(self.inputs[-1][j]))
                    delta[j] = deriv_sigmoid * (y[j] - self.output[j])

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
        #return [float(round(out)) for out in self.output]
        return float(round(self.output[0]))
        #return self.output.index(max(self.output))

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
    #random.shuffle(training)


    # K-fold Cross Validation:
    k = 5
    offset = len(training) // k
    data = [training[x:x+offset] for x in range(0, len(training), offset)]

    training_sets = data[:-1]
    validation_set = data[-1]

    # Data collection Script
    input = [2]
    output = [1]
    file = "Banana:"
    for node in range(1, 11):
        for layer in range(1, 4):
            dims = input + [node * layer] + output
            nn = NeuralNetwork(dims)
            avg_accuracy = 0
            for set in training_sets:
                nn.back_propagation_learning(set)
                avg_accuracy += accuracy(nn, validation_set)
            avg_accuracy = avg_accuracy / len(training_sets)

            print(file)
            print("Layers:", layer, "Nodes:", node)
            print("Accuracy:", avg_accuracy)
            print("**********")

    # Generated: classification
    #nn = NeuralNetwork([2, 3, 1])
    # Wine: non-classification
    #random.shuffle(training)
    #nn = NeuralNetwork([13, 4, 3])
    # banana: classification
    #nn = NeuralNetwork([2, 2, 1])
    # bits: non-classification
    #nn = NeuralNetwork([3, 6, 3])
    # breast cancer: classification
    #nn = NeuralNetwork([30, 15, 1]) #weird

    #nn.back_propagation_learning(training, test_pairs)

    #print(accuracy(nn, pairs))
    #print(accuracy(nn, test_pairs))

if __name__ == "__main__":
    main()
