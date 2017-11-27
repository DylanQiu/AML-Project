from __future__ import print_function
import keras
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

###############################################################################
# Load data
###############################################################################

print("Load")

num_classes = 82
input_shape = (64, 64, 1)

this = {}
this['train_data']   = np.genfromtxt('train_x.csv', delimiter=',')
this['train_data']   = this['train_data'].reshape((50000, 64, 64, 1)).astype('float32') / 255
this['train_label']  = np.genfromtxt('train_y.csv', delimiter=',').astype('int')
this['train_label2'] = keras.utils.to_categorical(this['train_label'], num_classes)
this['test_data']    = np.genfromtxt('test_x.csv', delimiter=',')
this['test_data']    = this['test_data'].reshape((10000, 64, 64, 1)).astype('float32') / 255

x_train  = this['train_data']
x_train[x_train < 220. / 255.] = 0
y_train  = this['train_label']
y_train2 = this['train_label2']
x_test   = this['test_data']
x_test[x_test < 220. / 255.] = 0

###############################################################################
# Define helper functions
###############################################################################

# classes = y labels in integer format
# train_ratio = % of obs allocated to training partition of the split
# x = either x_train or y_train
def psplit(classes, train_ratio, x = None):
    # Partition classes indices per unique values
    indices_by_class = [list(np.where(classes == i))[0] for i in np.unique(classes)]
    # Pick out indices of train split (ratio)
    train_split_indices = [i[np.arange(0, int(train_ratio * len(i)))] for i in indices_by_class]
    # Flatten list of lists into lists
    train_split_indices = [item for sublist in train_split_indices for item in sublist]
    # Pick out indices of test split (1 - ratio)
    test_split_indices = [i[np.arange(int(train_ratio * len(i)), len(i))] for i in indices_by_class]
    # Flatten list of lists into lists
    test_split_indices = [item for sublist in test_split_indices for item in sublist]
    # Return indices
    if x == None:
      return (train_split_indices, test_split_indices)
    else:
      return (x[[train_split_indices]], x[[test_split_indices]])

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


###############################################################################
# Model construction
###############################################################################

batch_size = 100
epochs = 21

model = Sequential()
model.add(Conv2D(32, (5, 5), strides=(1, 1), input_shape=input_shape))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(32, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

BatchNormalization(axis=-1)
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

BatchNormalization(axis=-1)
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

BatchNormalization()
model.add(Dense(512))
model.add(Activation('relu'))
BatchNormalization()
model.add(Dense(512))
model.add(Activation('relu'))
BatchNormalization()
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()

history = AccuracyHistory()

#my_train_idx, my_test_idx = psplit(y_train, 0.8)
my_train_idx = np.arange(0, len(y_train))
my_test_idx  = np.arange(0, len(y_train))
my_x_train = x_train[my_train_idx]
my_y_train = y_train2[my_train_idx]
my_x_test  = x_train[my_test_idx]
my_y_test  = y_train2[my_test_idx]

model.fit(my_x_train, my_y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(my_x_test, my_y_test),
          callbacks=[history])

module.summary()

preds = list(map(np.argmax, model.predict(x_test)))
idx   = np.arange(1, len(preds)+1)
pd.DataFrame({ "Id": idx, "Label": preds }).to_csv("predictions_ndejay_171111_test12.csv", index=False)
