from package import *

# with open("pickle_dump_x.txt", "r")as fp:
#     x = pickle.load(fp)
# with open("pickle_dump_y.txt", "r")as fp:
#     y = pickle.load(fp)

x = np.loadtxt("train_x.csv", delimiter=",")
y = np.loadtxt("train_y.csv", delimiter=",")
test_x = np.loadtxt("test_x.csv", delimiter=",")
# print y
# test_x = np.loadtxt("test_x.csv", delimiter=",")
# test_x = test_x.reshape(-1, 64, 64)
#
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf.fit(x, y)
predicted = clf.predict(test_x)
with open("result.csv", "w") as fp:
    for i in predicted:
        fp.write(str(i))
        fp.write('\n')