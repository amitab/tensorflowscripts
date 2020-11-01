import numpy as np
import pandas as pd        # For loading and processing the dataset
import tensorflow as tf    # Of course, we need TensorFlow.
from sklearn.model_selection import train_test_split

# Read the CSV input file and show first 5 rows
df_train = pd.read_csv('input/train.csv')
df_train.head(5)

# To make 'Sex' numeric, we replace 'female' by 0 and 'male' by 1
df_train['Sex'] = df_train['Sex'].map({'female':0, 'male':1}).astype(int) 

# We replace 'Embarked' by three dummy variables 'Embarked_S', 'Embarked_C', and 'Embarked Q',
# which are 1 if the person embarked there, and 0 otherwise.
df_train = pd.concat([df_train, pd.get_dummies(df_train['Embarked'], prefix='Embarked')], axis=1)
df_train = df_train.drop('Embarked', axis=1)

# We normalize the age and the fare by subtracting their mean and dividing by the standard deviation
age_mean = df_train['Age'].mean()
age_std = df_train['Age'].std()
df_train['Age'] = (df_train['Age'] - age_mean) / age_std

fare_mean = df_train['Fare'].mean()
fare_std = df_train['Fare'].std()
df_train['Fare'] = (df_train['Fare'] - fare_mean) / fare_std

# In many cases, the 'Age' is missing - which can cause problems. Let's look how bad it is:
print("Number of missing 'Age' values: {:d}".format(df_train['Age'].isnull().sum()))

# A simple method to handle these missing values is to replace them by the mean age.
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

# With that, we're almost ready for training
df_train.head()

# Finally, we convert the Pandas dataframe to a NumPy array, and split it into a training and test set
X_train = df_train.drop('Survived', axis=1).to_numpy()
y_train = df_train['Survived'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# We'll build a classifier with two classes: "survived" and "didn't survive",
# so we create the according labels
# This is taken from https://www.kaggle.com/klepacz/titanic/tensor-flow
labels_train = (np.arange(2) == y_train[:,None]).astype(np.float32)
labels_test = (np.arange(2) == y_test[:,None]).astype(np.float32)

inputs = tf.compat.v1.placeholder(tf.float32, shape=(None, X_train.shape[1]), name='inputs')
label = tf.compat.v1.placeholder(tf.float32, shape=(None, 2), name='labels')

# First layer
hid1_size = 128
w1 = tf.Variable(tf.random.normal([hid1_size, X_train.shape[1]], stddev=0.01), name='w1')
b1 = tf.Variable(tf.constant(0.1, shape=(hid1_size, 1)), name='b1')
y1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w1, tf.transpose(a=inputs)), b1)), rate=1 - (0.5))

# Second layer
hid2_size = 256
w2 = tf.Variable(tf.random.normal([hid2_size, hid1_size], stddev=0.01), name='w2')
b2 = tf.Variable(tf.constant(0.1, shape=(hid2_size, 1)), name='b2')
y2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w2, y1), b2)), rate=1 - (0.5))

# Output layer
wo = tf.Variable(tf.random.normal([2, hid2_size], stddev=0.01), name='wo')
bo = tf.Variable(tf.random.normal([2, 1]), name='bo')
yo = tf.transpose(a=tf.add(tf.matmul(wo, y2), bo))

# Loss function and optimizer
lr = tf.compat.v1.placeholder(tf.float32, shape=(), name='learning_rate')
loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=yo, labels=tf.stop_gradient(label)))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr).minimize(loss)

# Prediction
pred = tf.nn.softmax(yo)
pred_label = tf.argmax(input=pred, axis=1)
correct_prediction = tf.equal(tf.argmax(input=pred, axis=1), tf.argmax(input=label, axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, "float"))

# Create operation which will initialize all variables
init = tf.compat.v1.global_variables_initializer()

# Configure GPU not to use all memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# Start a new tensorflow session and initialize variables
sess = tf.compat.v1.InteractiveSession(config=config)
sess.run(init)

# This is the main training loop: we train for 50 epochs with a learning rate of 0.05 and another 
# 50 epochs with a smaller learning rate of 0.01
for learning_rate in [0.05, 0.01]:
    for epoch in range(50):
        avg_cost = 0.0

        # For each epoch, we go through all the samples we have.
        for i in range(X_train.shape[0]):
            # Finally, this is where the magic happens: run our optimizer, feed the current example into X and the current target into Y
            _, c = sess.run([optimizer, loss], feed_dict={lr:learning_rate, 
                                                          inputs: X_train[i, None],
                                                          label: labels_train[i, None]})
            avg_cost += c
        avg_cost /= X_train.shape[0]    

        # Print the cost in this epcho to the console.
        if epoch % 10 == 0:
            print("Epoch: {:3d}    Train Cost: {:.4f}".format(epoch, avg_cost))

acc_train = accuracy.eval(feed_dict={inputs: X_train, label: labels_train})
print("Train accuracy: {:3.2f}%".format(acc_train*100.0))

acc_test = accuracy.eval(feed_dict={inputs: X_test, label: labels_test})
print("Test accuracy:  {:3.2f}%".format(acc_test*100.0))

df_test = pd.read_csv('input/test.csv')
df_test.head()

# Do all pre-processing steps as above
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_test['Sex'] = df_test['Sex'].map({'female':0, 'male':1}).astype(int)
df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')], axis=1)
df_test = df_test.drop('Embarked', axis=1)
df_test['Age'] = (df_test['Age'] - age_mean) / age_std
df_test['Fare'] = (df_test['Fare'] - fare_mean) / fare_std
df_test.head()
X_test = df_test.drop('PassengerId', axis=1).to_numpy()

# Predict
for i in range(X_test.shape[0]):
    df_test.loc[i, 'Survived'] = sess.run(pred_label, feed_dict={inputs: X_test[i, None]}).squeeze()

# Important: close the TensorFlow session, now that we're finished.
sess.close()

output = pd.DataFrame()
output['PassengerId'] = df_test['PassengerId']
output['Survived'] = df_test['Survived'].astype(int)
output.to_csv('output/prediction.csv', index=False)
output.head()
