""" 
   Author: Nicholas Heim
   email: nwh8
"""

"""
   import pandas as pd
   import numpy as np
   from keras.utils import to_categorical
   from keras.datasets import mnist
   from keras import models
   from keras import layers

   (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

   train_images = train_images.reshape((60000, 28 * 28))
   train_images = train_images.astype('float32') / 255

   test_images = test_images.reshape((10000, 28 * 28))
   test_images = test_images.astype('float32') / 255

   train_labels = to_categorical(train_labels)
   test_labels = to_categorical(test_labels)

   network = models.Sequential
   network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
   network.add(layers.Dense(10, activation='softmax'))

   network.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
                   metrics=['accuracy'])
   network.fit(train_images, train_labels, epochs=5, batch_size=128)

   test_loss, test_acc = network.evaluate(test_images, test_labels) 
"""

from keras.utils import to_categorical
from keras.models import Sequential
from keras.datasets import cifar10
from keras.optimizers import sgd
from keras import optimizers
from keras import layers
import pandas as pd
import numpy as np
import os


def genBaseline():
   # Model creation:
   # 3-block VGG method as a starting baseline
   model = Sequential()
   model.add(layers.Conv2D(32, (3, 3), 
                           activation='relu',
                           kernel_initializer='he_uniform',
                           padding='same',
                           input_shape=(32, 32, 3)))
   model.add(layers.Conv2D(32, (3, 3),
                           activation='relu',
                           kernel_initializer='he_uniform',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))

   model.add(layers.Conv2D(64, (3, 3),
                           activation='relu',
                           kernel_initializer='he_uniform',
                           padding='same'))
   model.add(layers.Conv2D(64, (3, 3),
                           activation='relu',
                           kernel_initializer='he_uniform',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))

   model.add(layers.Conv2D(128, (3, 3),
                           activation='relu',
                           kernel_initializer='he_uniform',
                           padding='same'))
   model.add(layers.Conv2D(128, (3, 3),
                           activation='relu',
                           kernel_initializer='he_uniform',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))

   # Finalizing model
   model.add(layers.Flatten())
   model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
   model.add(layers.Dense(10, activation='softmax'))

   opti = optimizers.SGD(lr=0.001, momentum=0.9)
   model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])

   return model


def orgData():
   (x_train, y_train), (x_test, y_test) = cifar10.load_data()

   x_train = x_train.astype('float32')
   x_test = x_test.astype('float32')
   x_train /= 255.0
   x_test /= 255.0

   y_train = to_categorical(y_train, 10)
   y_test = to_categorical(y_test, 10)

   return (x_train, y_train), (x_test, y_test)

def baseline():
   (x_train, y_train), (x_test, y_test) = orgData()
   model = genBaseline()
   model.fit(x_train, y_train, epochs=10, batch_size=32,
             validation_data=(x_test, y_test))
   model.evaluate((x_test, y_test))
