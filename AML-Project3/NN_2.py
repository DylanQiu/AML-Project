
from package import *

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

# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([[1., 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]])
# print np.shape(X)
# print np.shape(y)
X = train_x
y = train_y

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

inputlayer_neurons = X.shape[1]
hiddenlayer_neurons = 20
output_neurons = y.shape[1]

# wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
# bh=np.random.uniform(size=(1,hiddenlayer_neurons))
# wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
# bout=np.random.uniform(size=(1,output_neurons))

def train(X, y, epoch=5000, lr=0.1):
    wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
    bh = np.random.uniform(size=(1, hiddenlayer_neurons))
    wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
    bout = np.random.uniform(size=(1, output_neurons))

    for i in range(epoch):
        print "epoch: ", str(i)
        # Feed Forward
        hidden_layer_input1 = np.dot(X, wh)
        hidden_layer_input = hidden_layer_input1 + bh
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1 = np.dot(hiddenlayer_activations, wout)
        output_layer_input = output_layer_input1 + bout
        output = sigmoid(output_layer_input)

        # Back Propagation
        E = y - output
        slope_output_layer = derivatives_sigmoid(output)
        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
        d_output = E * slope_output_layer
        Error_at_hidden_layer = d_output.dot(wout.T)
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        wout += hiddenlayer_activations.T.dot(d_output) * lr
        bout += np.sum(d_output, axis=0, keepdims=True) * lr
        wh += X.T.dot(d_hiddenlayer) * lr
        bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr
    return wh, bh, wout, bout

def predict(x, wh, bh, wout, bout):
    hidden_layer_input1 = np.dot(x, wh)
    hidden_layer_input = hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1 = np.dot(hiddenlayer_activations, wout)
    output_layer_input = output_layer_input1 + bout
    output_matrix = sigmoid(output_layer_input)
    output = [np.argmax(output_matrix[i]) for i in range(len(output_matrix))]
    return output

wh, bh, wout, bout = train(X, y)
predicted = predict(test_x, wh, bh, wout, bout)

with open("result.csv", "w") as fp:
    count = 1
    for i in predicted:
        fp.write(str(count) + ',' + str(class_list[i]))
        count += 1
        fp.write('\n')