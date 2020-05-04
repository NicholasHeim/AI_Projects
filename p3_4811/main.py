""" 
   Author: Nicholas Heim
   email: nwh8
"""

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers
from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import os

def orgData():
   (x_train, y_train), (x_test, y_test) = cifar10.load_data()

   x_train = x_train.astype('float32')
   x_test = x_test.astype('float32')
   x_train /= 255.0
   x_test /= 255.0

   y_train = to_categorical(y_train, 10)
   y_test = to_categorical(y_test, 10)

   return (x_train, y_train), (x_test, y_test)


def loadModel(folder):
   model = load_model("logs/" + folder + "/cifar-10.h5")
   (x_train, y_train), (x_test, y_test) = orgData()
   loss, accuracy = model.evaluate(x_test, y_test)
   return model, loss, accuracy


def genBaseline():
   # Model creation:
   # 3-block VGG method as a starting baseline
   model = Sequential()
   model.add(layers.Conv2D(32, (3, 3), 
                           activation='relu',
                           padding='same',
                           input_shape=(32, 32, 3)))
   model.add(layers.Conv2D(32, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))

   model.add(layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))

   model.add(layers.Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))

   # Finalizing model
   model.add(layers.Flatten())
   model.add(layers.Dense(128, activation='relu'))
   model.add(layers.Dense(10, activation='softmax'))

   optim = optimizers.SGD(lr=0.001, momentum=0.9)
   model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

   return model

def computeBaseline():
   (x_train, y_train), (x_test, y_test) = orgData()

   #create a tensorboard log directory
   log_dir="logs/baseline/"
   tensorboard_callback = TensorBoard(log_dir=log_dir, 
                                      histogram_freq=1)
   
   model = genBaseline()
   model.summary()
   model.fit(x=x_train,
             y=y_train, 
             epochs=100,
             batch_size=64,
             validation_data=(x_test, y_test)),
             callbacks=[tensorboard_callback])
   model.reset_metrics()
   model.save(log_dir + '/cifar-10.h5')


def gen2():
   # Model creation:
   # Added dropout layer at 0.1
   model = Sequential()
   model.add(layers.Conv2D(32, (3, 3), 
                           activation='relu',
                           padding='same',
                           input_shape=(32, 32, 3)))
   model.add(layers.Conv2D(32, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.1))

   model.add(layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.1))

   model.add(layers.Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.1))

   # Finalizing model
   model.add(layers.Flatten())
   model.add(layers.Dense(128, activation='relu'))
   model.add(layers.Dense(10, activation='softmax'))

   optim = optimizers.SGD(lr=0.001, momentum=0.9)
   model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

   return model

def compute2():
   (x_train, y_train), (x_test, y_test) = orgData()

   # setup a tensorboard log directory
   log_dir="logs\\model2\\"
   tensorboard_callback = TensorBoard(log_dir=log_dir, 
                                      histogram_freq=1)
   
   model = gen2()
   model.summary()
   model.fit(x=x_train,
             y=y_train, 
             epochs=100,
             batch_size=64,
             validation_data=(x_test, y_test)),
             callbacks=[tensorboard_callback])
   model.reset_metrics()
   model.save(log_dir + '\\cifar-10.h5')



def gen3():
   # Model creation:
   # Added dropout layer at 0.3
   model = Sequential()
   model.add(layers.Conv2D(32, (3, 3), 
                           activation='relu',
                           padding='same',
                           input_shape=(32, 32, 3)))
   model.add(layers.Conv2D(32, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.3))

   model.add(layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.3))

   model.add(layers.Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.3))

   # Finalizing model
   model.add(layers.Flatten())
   model.add(layers.Dense(128, activation='relu'))
   model.add(layers.Dense(10, activation='softmax'))

   optim = optimizers.SGD(lr=0.001, momentum=0.9)
   model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

   return model

def compute3():
   (x_train, y_train), (x_test, y_test) = orgData()

   # setup a tensorboard log directory
   log_dir="logs\\model3\\"
   tensorboard_callback = TensorBoard(log_dir=log_dir, 
                                      histogram_freq=1)
   
   model = gen3()
   model.summary()
   model.fit(x=x_train,
             y=y_train, 
             epochs=100,
             batch_size=64,
             validation_data=(x_test, y_test)),
             callbacks=[tensorboard_callback])
   model.reset_metrics()
   model.save(log_dir + '\\cifar-10.h5')



def gen4():
   # Model creation:
   # Added dropout layer at 0.5
   model = Sequential()
   model.add(layers.Conv2D(32, (3, 3), 
                           activation='relu',
                           padding='same',
                           input_shape=(32, 32, 3)))
   model.add(layers.Conv2D(32, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.5))

   model.add(layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.5))

   model.add(layers.Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.5))

   # Finalizing model
   model.add(layers.Flatten())
   model.add(layers.Dense(128, activation='relu'))
   model.add(layers.Dense(10, activation='softmax'))

   optim = optimizers.SGD(lr=0.001, momentum=0.9)
   model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

   return model

def compute4():
   (x_train, y_train), (x_test, y_test) = orgData()

   # setup a tensorboard log directory
   log_dir="logs\\model4\\"
   tensorboard_callback = TensorBoard(log_dir=log_dir, 
                                      histogram_freq=1)
   
   model = gen4()
   model.summary()
   model.fit(x=x_train,
             y=y_train, 
             epochs=100,
             batch_size=64,
             validation_data=(x_test, y_test)),
             callbacks=[tensorboard_callback])
   model.reset_metrics()
   model.save(log_dir + '\\cifar-10.h5')



def gen5():
   # Model creation:
   # Added increasing values to dropout layers
   model = Sequential()
   model.add(layers.Conv2D(32, (3, 3), 
                           activation='relu',
                           padding='same',
                           input_shape=(32, 32, 3)))
   model.add(layers.Conv2D(32, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.2))

   model.add(layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.35))

   model.add(layers.Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.5))

   # Finalizing model
   model.add(layers.Flatten())
   model.add(layers.Dense(128, activation='relu'))
   model.add(layers.Dense(10, activation='softmax'))

   optim = optimizers.SGD(lr=0.001, momentum=0.9)
   model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

   return model

def compute5():
   (x_train, y_train), (x_test, y_test) = orgData()

   # setup a tensorboard log directory
   log_dir="logs\\model5\\"
   tensorboard_callback = TensorBoard(log_dir=log_dir, 
                                      histogram_freq=1)
   
   model = gen5()
   model.summary()
   model.fit(x=x_train,
             y=y_train, 
             epochs=100,
             batch_size=64,
             validation_data=(x_test, y_test)),
             callbacks=[tensorboard_callback])
   model.reset_metrics()
   model.save(log_dir + '\\cifar-10.h5')



def gen6():
   # Model creation:
   # Added dropout layer after first dense layer
   model = Sequential()
   model.add(layers.Conv2D(32, (3, 3), 
                           activation='relu',
                           padding='same',
                           input_shape=(32, 32, 3)))
   model.add(layers.Conv2D(32, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.2))

   model.add(layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.35))

   model.add(layers.Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.5))

   # Finalizing model
   model.add(layers.Flatten())
   model.add(layers.Dense(128, activation='relu'))
   model.add(layers.Dropout(0.3))
   model.add(layers.Dense(10, activation='softmax'))

   optim = optimizers.SGD(lr=0.001, momentum=0.9)
   model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

   return model

def compute6():
   (x_train, y_train), (x_test, y_test) = orgData()

   # setup a tensorboard log directory
   log_dir="logs\\model6\\"
   tensorboard_callback = TensorBoard(log_dir=log_dir, 
                                      histogram_freq=1)
   
   model = gen6()
   model.summary()
   model.fit(x=x_train,
             y=y_train, 
             epochs=100,
             batch_size=64,
             validation_data=(x_test, y_test)),
             callbacks=[tensorboard_callback])
   model.reset_metrics()
   model.save(log_dir + '\\cifar-10.h5')



def gen7():
   # Model creation:
   # Added dropout layer after first dense layer, dropout is 0.5
   model = Sequential()
   model.add(layers.Conv2D(32, (3, 3), 
                           activation='relu',
                           padding='same',
                           input_shape=(32, 32, 3)))
   model.add(layers.Conv2D(32, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.2))

   model.add(layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.35))

   model.add(layers.Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same'))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Dropout(0.5))

   # Finalizing model
   model.add(layers.Flatten())
   model.add(layers.Dense(128, activation='relu'))
   model.add(layers.Dropout(0.5))
   model.add(layers.Dense(10, activation='softmax'))

   optim = optimizers.SGD(lr=0.001, momentum=0.9)
   model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

   return model

def compute7():
   (x_train, y_train), (x_test, y_test) = orgData()

   # setup a tensorboard log directory
   log_dir="logs\\model7\\"
   tensorboard_callback = TensorBoard(log_dir=log_dir, 
                                      histogram_freq=1)
   
   model = gen7()
   model.summary()
   model.fit(x=x_train,
             y=y_train, 
             epochs=100,
             batch_size=64,
             validation_data=(x_test, y_test)),
             callbacks=[tensorboard_callback])
   model.reset_metrics()
   model.save(log_dir + '\\cifar-10.h5')
