import tensorflow as tf
import numpy as np

model1_dir = 'models/clip_1_5_NO_0_50_2.h5'
model2_dir = 'models/clip_1_5_NO_1_50_2.h5'

model1 = tf.keras.models.load_model(model1_dir)
model2 = tf.keras.models.load_model(model2_dir)

def save_weights(model, prefix):
    m = model.layers[0].get_weights()
    m[0] = m[0].reshape(64, 16)
    np.savetxt('weights/{}-w{}.out'.format(prefix, 0), m[0], header="{},{}".format(*m[0].shape), delimiter=",")
    np.savetxt('weights/{}-b{}.out'.format(prefix, 0), m[1], header="{},{}".format(m[1].shape, 1), delimiter=",")
    m = model.layers[2].get_weights()
    m[0] = m[0].reshape(16, 512)
    np.savetxt('weights/{}-w{}.out'.format(prefix, 2), m[0], header="{},{}".format(*m[0].shape), delimiter=",")
    np.savetxt('weights/{}-b{}.out'.format(prefix, 2), m[1], header="{},{}".format(m[1].shape, 1), delimiter=",")
    m = model.layers[5].get_weights()
    np.savetxt('weights/{}-w{}.out'.format(prefix, 5), m[0], header="{},{}".format(*m[0].shape), delimiter=",")
    np.savetxt('weights/{}-b{}.out'.format(prefix, 5), m[1], header="{},{}".format(m[1].shape, 1), delimiter=",")
    m = model.layers[6].get_weights()
    np.savetxt('weights/{}-w{}.out'.format(prefix, 6), m[0], header="{},{}".format(*m[0].shape), delimiter=",")
    np.savetxt('weights/{}-b{}.out'.format(prefix, 6), m[1], header="{},{}".format(m[1].shape, 1), delimiter=",")


save_weights(model1, "clip_1_5_NO_0_50_2.h5")
save_weights(model2, "clip_1_5_NO_1_50_2.h5")