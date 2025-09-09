import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid') # Correct
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

# Terminal: cd .\lesson_2\

concrete_file_path = '../concrete.csv'
concrete = pd.read_csv(concrete_file_path)
print("\n", concrete.head())

# The target is 'CompressiveStranght', the remaining columns will be used as inputs
input_shape = [8]

import tensorflow as tf
from tensorflow import keras
from keras import layers

# Model with three hidden layers,
# each layer having 512 units and the ReLU activation
"""
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=input_shape), # input_shape as argument to the first layer
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1) # output layer
])
"""
# Model with two hidden layers,
# each layer having 32 units and the ReLU activation
model = keras.Sequential([
    layers.Dense(32, input_shape=[8]),
    layers.Activation('relu'),
    layers.Dense(32),
    layers.Activation('relu'),
    layers.Dense(1)
])
print("\nModel succesfully created!")

# --- GRAPHIC ---

# Change 'relu' to 'elu', 'selu', 'swish'... or something else
activation_layer = layers.Activation('relu')

x = tf.linspace(-3.0, 3.0, 100)
y = activation_layer(x) # once created, a layer is callable just like a function

plt.figure(dpi=100)
plt.plot(x, y)
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")

# --- Save graphic as png file ---
print("\nSaving gaphic as 'activation_graphic.png'...")
plt.savefig('activation_graphic.png')
print("Graphic succesfully saved!\n")

# plt.show()