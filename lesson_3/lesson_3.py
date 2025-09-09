# Setup plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid') # Correct
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

# Terminal: cd .\lesson_3\

fuel_file_path = '../fuel.csv'
fuel = pd.read_csv(fuel_file_path)

X = fuel.copy()
# Remove target
y = X.pop('FE')

preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse_output=False), # sparse_output
     make_column_selector(dtype_include=object)),
)

X = preprocessor.fit_transform(X)
y = np.log(y) # log transform target instead of standardizing

input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))

# print("\n", fuel.head())
# print("\n", pd.DataFrame(X[:10,:]).head())

from tensorflow import keras
from keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),    
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])

# Choosing the Adam Optimizer and the MAE loss
model.compile(
    optimizer="adam",
    loss="mae"
)

# Training the network for 200 epochs with a batch size of 128.
# The input data is X with target y.
history = model.fit(
    X, y,
    batch_size=128,
    epochs=200
)

history_df = pd.DataFrame(history.history)

# --- GRAPHIC ---

plt.figure(figsize=(10, 6))

history_df['loss'].plot(label='Training Loss')
plt.title('Training Loss per Epoch')
plt.ylabel('Loss (MAE)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('training_loss_graphic.png')
print("\nGraphic succesfully saved as 'training_loss_graphic.png'")
# plt.show()