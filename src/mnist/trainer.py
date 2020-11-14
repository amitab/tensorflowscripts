# https://www.kaggle.com/ngbolin/mnist-dataset-digit-recognizer/data?select=sample_submission.csv
import pandas as pd
import numpy as np

np.random.seed(1212)

import keras
from keras.models import Model
from keras.layers import *
from keras import optimizers

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

df_features = df_train.iloc[:, 1:785]
df_label = df_train.iloc[:, 0]

X_test = df_test.iloc[:, 0:784]

from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(df_features, df_label, 
                                                test_size = 0.2,
                                                random_state = 1212)

X_train = X_train.to_numpy().reshape(33600, 784) #(33600, 784)
X_cv = X_cv.to_numpy().reshape(8400, 784) #(8400, 784)

X_test = X_test.to_numpy().reshape(28000, 784)


# Feature Normalization 
X_train = X_train.astype('float32'); X_cv= X_cv.astype('float32'); X_test = X_test.astype('float32')
X_train /= 255; X_cv /= 255; X_test /= 255

# Convert labels to One Hot Encoded
num_digits = 10
y_train = keras.utils.to_categorical(y_train, num_digits)
y_cv = keras.utils.to_categorical(y_cv, num_digits)


shared_percent = 75 / 100
shared_train_data_size = int(X_train.shape[0] * shared_percent)
num_models = 2

shared_train_data = X_train[:shared_train_data_size]
shared_train_labels = y_train[:shared_train_data_size]

private_train_data = np.split(X_train[shared_train_data_size:], num_models)
private_train_labels = np.split(
    y_train[shared_train_data_size:], num_models)

# Input Parameters
n_input = 784 # number of features
n_hidden_1 = 300
n_hidden_2 = 100
n_hidden_3 = 100
n_hidden_4 = 200
num_digits = 10

learning_rate = 0.1
training_epochs = 20
batch_size = 100

def save_weights(model, prefix):
    for i, layer in enumerate(model.layers):
        if len(layer.get_weights()) == 0:
            continue
        m = layer.get_weights()
        np.savetxt('saved_weights/{}-w{}.out'.format(prefix, i), m[0], header="{},{}".format(*m[0].shape), delimiter=",")
        np.savetxt('saved_weights/{}-b{}.out'.format(prefix, i), m[1], header="{},{}".format(m[1].shape, 1), delimiter=",")


main_model = keras.Sequential([
    Input(shape=(784,)),
    Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1"),
    Dropout(0.3),
    Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2"),
    Dropout(0.3),
    Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3"),
    Dropout(0.3),
    Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4"),
    Dense(num_digits, activation='softmax', name = "Output_Layer")
])
main_model.summary()
og_file_name = 'saved_model/main_model_{}_{}.h5'.format(int(shared_percent * 100), num_models)
main_model.save(og_file_name)

for i in range(num_models):
    print("Training model - {}".format(i))
    
    model_train_data = np.vstack((shared_train_data, private_train_data[i]))
    model_train_labels = np.vstack(
        (shared_train_labels, private_train_labels[i]))

    model = keras.models.load_model(og_file_name)
    # Will compile mess up optimizer state?
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.fit(model_train_data, model_train_labels, batch_size = batch_size, epochs = training_epochs, validation_data=(X_cv, y_cv))

    test_pred = pd.DataFrame(model.predict(X_test, batch_size=200))
    test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))
    test_pred.index.name = 'ImageId'
    test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()
    test_pred['ImageId'] = test_pred['ImageId'] + 1

    test_pred.head()
    
    model_name = "saved_model/model_{}_{}_{}.h5".format(i, int(shared_percent * 100), num_models)
    model.save(model_name)

    save_weights(model, 'model_{}_{}_{}'.format(i, int(shared_percent * 100), num_models))