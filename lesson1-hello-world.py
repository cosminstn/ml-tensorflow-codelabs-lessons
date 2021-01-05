import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# We can see that the loss is really small even after 50 epochs, we don't need 500
model.fit(xs, ys, epochs=500)

# Neural networks deal with probabilities, so it calculated that there is a very high probability that the relationship between X and Y is Y=3X+1, but it can't know for sure with only six data points. As a result, 10 is very close to 31, but not necessarily 31.
print(model.predict([10.0]))