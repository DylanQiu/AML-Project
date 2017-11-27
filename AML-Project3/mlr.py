import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

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

# change to one hot format
b = np.zeros((train_y.size, train_y.max()+1))
b[np.arange(train_y.size), train_y] = 1
train_y = b


x = tf.placeholder(tf.float32, [None, 4096])
W = tf.Variable(tf.zeros([4096, 40]))
b = tf.Variable(tf.zeros([40]))

y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 40])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(250):
    idx = np.random.randint(45000, size=100)
    batch_xs = train_x[idx,:]
    batch_ys = train_y[idx,:]
    print 'processing %s loop' %i
    # batch_xs, batch_ys = train_x
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # sess.run(train_step, feed_dict={x: train_x, y_: train_y})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: train_x[45000:], y_: train_y[45000:]}))

predicted = sess.run(tf.argmax(y, 1), feed_dict={x: test_x})

with open("result.csv", "w") as fp:
    count = 1
    for i in predicted:
        fp.write(str(count) + ',' + str(class_list[i]))
        count += 1
        fp.write('\n')

