import keras
import tensorflow as tf
import numpy as np
import sys
import time


def load_model():
    model = keras.models.load_model('simple_ff.h5')
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                label_smoothing=0,
                                                reduction="auto",
                                                name="binary_crossentropy"),
        metrics=['accuracy'],
    )
    return model

def print_stats(data):
    print("File open time: {}".format(data['file_open_time']))
    print("feature data load time: {}".format(data['feat_open_time']))
    print("model load time: {}".format(data['modl_open_time']))
    print("prediction time for {} points: {}".format(data['points'], data['inference_time']))

def avg_data(data):
    data['file_open_time'] = np.mean(data['file_open_time'])
    data['feat_open_time'] = np.mean(data['feat_open_time'])
    data['modl_open_time'] = np.mean(data['modl_open_time'])
    data['inference_time'] = np.mean(data['inference_time'])

    print_stats(data)

def main(stats):
    if stats == None:
        stats = {
            'file_open_time': [],
            'feat_open_time': [],
            'modl_open_time': [],
            'inference_time': [],
            'points': 0,
        }

    start = time.process_time()
    f = open(sys.argv[1])
    end = time.process_time()
    stats['file_open_time'].append(end - start)

    start = time.process_time()
    data = np.loadtxt(f, delimiter=",")
    end = time.process_time()
    stats['feat_open_time'].append(end - start)
    start = time.process_time()
    model = load_model()
    end = time.process_time()
    stats['modl_open_time'].append(end - start)
    model.summary()
    start = time.process_time()
    op = model.predict(data)
    end = time.process_time()
    stats['points'] = data.shape[0]
    stats['inference_time'].append(end - start)

    del data
    del model
    model = None
    data = None

    return stats


# Usage example: python ff_classifier.py 2. 0. -0.32348594 1. 1. -0.04435613 0. 0. 1.
if __name__ == "__main__":
    data = None
    for _ in range(1):
        data = main(data)

    avg_data(data)