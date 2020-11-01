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

df_train = df_train.drop(['Name', 'Ticket', 'Cabin'], axis=1)

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

model = tf.keras.Sequential([
    # tf.keras.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
  optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
  loss=tf.keras.losses.BinaryCrossentropy(
    from_logits=True, label_smoothing=0, reduction="auto", name="binary_crossentropy"),
  metrics=['accuracy'],
)

model.fit(
  X_train,
  labels_train, # training targets
  epochs=50,
)