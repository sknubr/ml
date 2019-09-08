import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

## Provide the input and output data respectively 
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype= float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

## fit the data and run 500 times
model.fit(xs,ys,epochs=500)

## make prediction 
prediction = model.predict([10])

## print
print (prediction)
