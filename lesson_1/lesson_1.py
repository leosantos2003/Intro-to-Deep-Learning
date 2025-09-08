import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Terminal: cd .\lesson_1\

red_wine_file_path = '../red-wine.csv'
red_wine = pd.read_csv(red_wine_file_path)

print('\n', red_wine.head())

# Number of rows and columns of the dataframe
print('\nRows and columns of the dataframe:', red_wine.shape)

# The target is 'quality' (the twelfth), so let's take all the others as input features
input_shape = [11]

from tensorflow import keras
from keras import layers

# LINEAR MODEL:
# the individual neuron of a neural network.

model = keras.Sequential([
    layers.Dense(units=1, input_shape=[11])
])

# The weights of the neural network
w, b = model.weights
print("\nWeights\n{}\n\nBias\n{}\n".format(w, b))

# --- GRAPHIC ---

# Simple Model with 1 entry for graphic
model_simple = keras.Sequential([
    layers.Dense(1, input_shape=[1]),
])

x = tf.linspace(-1.0, 1.0, 100)
y = model_simple.predict(x)

plt.figure(dpi=100)
plt.plot(x, y, 'k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Input: x")
plt.ylabel("Target y")

w_simple, b_simple = model_simple.weights
plt.title("Weight: {:0.2f}\nBias: {:0.2f}".format(w_simple.numpy()[0][0], b_simple.numpy()[0]))

# --- Save graphic as png file ---
print("\nSaving gaphic as 'grafico_reta_neuronio.png'...")
plt.savefig('linear_neuron_graphic.png')
print("Graphic succesfully saved!\n")

#plt.show()