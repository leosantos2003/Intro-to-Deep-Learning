import pandas as pd

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
print("\nWeights\n{}\n\nBias\n{}".format(w, b))