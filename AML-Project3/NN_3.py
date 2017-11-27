import pickle
import numpy as np
import dill

class layer:
    def __init__(self, num_nodes_in_layer, num_nodes_in_next_layer, activation_function):
        self.num_nodes_in_layer = num_nodes_in_layer
        self.activation_function = activation_function
        self.activations = np.zeros([num_nodes_in_layer, 1])
        if num_nodes_in_next_layer != 0:
            self.weights_for_layer = np.random.normal(0, 0.001, size=(num_nodes_in_layer, num_nodes_in_next_layer))
        else:
            self.weights_for_layer = None

class neural_network:
    def __init__(self, num_layers, num_nodes, activation_function, cost_function):
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.layers = []
        self.cost_function = cost_function

        for i in range(num_layers):
            if i != num_layers - 1:
                layer_i = layer(num_nodes[i], num_nodes[i + 1], activation_function[i])
            else:
                layer_i = layer(num_nodes[i], 0, activation_function[i])
            self.layers.append(layer_i)

    def train(self, batch_size, inputs, labels, num_epochs, learning_rate, filename):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        for j in range(num_epochs):
            i = 0
            print("== EPOCH: ", j, " ==")
            while i + batch_size <= len(inputs):
                # print i
                self.error = 0
                self.forward_pass(inputs[i:i + batch_size])
                self.calculate_error(labels[i:i + batch_size])
                self.back_pass(labels[i:i + batch_size])
                i += batch_size
        dill.dump_session(filename)

    def forward_pass(self, inputs):
        self.layers[0].activations = inputs
        for i in range(self.num_layers - 1):
            temp = np.matmul(self.layers[i].activations, self.layers[i].weights_for_layer)
            if self.layers[i + 1].activation_function == "softmax":
                self.layers[i + 1].activations = self.sigmoid(temp)
            else:
                self.layers[i + 1].activations = temp

    def softmax(self, layer):
        exp = np.exp(layer)
        if isinstance(layer[0], np.ndarray):
            return exp / np.sum(exp, axis=1, keepdims=True)
        else:
            return exp / np.sum(exp, keepdims=True)


    def calculate_error(self, labels):
        if len(labels[0]) != self.layers[self.num_layers - 1].num_nodes_in_layer:
            print("Label: ", len(labels), " : ", len(labels[0]))
            print("Out: ", len(self.layers[self.num_layers - 1].activations), " : ",
                  len(self.layers[self.num_layers - 1].activations[0]))
            return

        if self.cost_function == "mean_squared":
            self.error = np.mean(
                np.divide(np.square(np.subtract(labels, self.layers[self.num_layers - 1].activations)), 2))
        elif self.cost_function == "cross_entropy":
            self.error = np.negative(np.sum(np.multiply(labels, np.log(self.layers[self.num_layers - 1].activations))))

    def back_pass(self, labels):
        targets = labels
        i = self.num_layers - 1
        y = self.layers[i].activations
        deltaw = np.matmul(np.asarray(self.layers[i - 1].activations).T,
                           np.multiply(y, np.multiply(1 - y, targets - y)))
        new_weights = self.layers[i - 1].weights_for_layer - self.learning_rate * deltaw
        for i in range(i - 1, 0, -1):
            y = self.layers[i].activations
            deltaw = np.matmul(np.asarray(self.layers[i - 1].activations).T, np.multiply(y, np.multiply(1 - y, np.sum(
                np.multiply(new_weights, self.layers[i].weights_for_layer), axis=1).T)))
            self.layers[i].weights_for_layer = new_weights
            new_weights = self.layers[i - 1].weights_for_layer - self.learning_rate * deltaw
        self.layers[0].weights_for_layer = new_weights

    def predict(self, filename, input):
        dill.load_session(filename)
        self.batch_size = 1
        self.forward_pass(input)
        a = self.layers[self.num_layers - 1].activations
        a[np.where(a == np.max(a))] = 1
        a[np.where(a != np.max(a))] = 0
        return a


with open("pickle_dump_x.txt", "r")as fp:
    train_x = pickle.load(fp)
with open("pickle_dump_y.txt", "r")as fp:
    train_y = pickle.load(fp)
with open("pickle_test_x.txt", "r")as fp:
    test_x = pickle.load(fp)

class_list =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20,
               21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

train_y = np.array(train_y, dtype=int)
tmp_y = np.empty(50000,)
for i in range(50000):
    tmp_y[i] = class_list.index(train_y[i])

train_y = np.array(tmp_y, dtype=int)

b = np.zeros((train_y.size, train_y.max()+1))
b[np.arange(train_y.size), train_y] = 1
train_y = b

NN = neural_network(4, [4096, 30, 40, 40], 'softmax', 'mean_squared')
NN.train(200, train_x, train_y, 100, 0.01, 'training.pkl')
print 'output'

output = []
for i in range(len(test_x)):
    output.append(np.argmax(NN.predict("training.pkl", test_x[i])))

with open("result.csv", "w") as fp:
    count = 1
    for i in output:
        fp.write(str(count) + ',' + str(i))
        count += 1
        fp.write('\n')