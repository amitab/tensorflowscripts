import keras
import numpy as np
import tensorflow as tf

def make_model(features):
    model = tf.keras.Sequential([
        # tf.keras.Input(shape=(features,)),
        tf.keras.layers.Dense(1000,
                              activation='relu',
                              input_shape=(features, )),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2000, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model


features = [200000, 400000, 600000, 800000, 1000000]

for f in features:
    model = make_model(f)
    w1 = np.random.rand(f, 1000)
    b1 = np.random.rand(1000)
    w2 = np.random.rand(1000, 2000)
    b2 = np.random.rand(2000)
    wo = np.random.rand(2000, 2)
    bo = np.random.rand(2)

    model.layers[0].set_weights([w1, b1])
    model.layers[2].set_weights([w2, b2])
    model.layers[4].set_weights([wo, bo])

    model.save('./simple_ff_{}.h5'.format(f))
    del model
    model = None