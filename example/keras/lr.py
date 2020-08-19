from tensorflow import keras

from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential();
model.add(layers.Dense(32, activation = "relu", input_shape = (784,)));
model.add(layers.Dense(10, activation = "softmax"));
input_tensor = layers.Input(shape=(784,));

x = layers.Dense(32, activation = "relu")
