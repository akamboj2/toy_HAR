import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import scipy.io
import os
from dataset import IMUDataset

# Load the training data
data_dir = "/home/abhi/data/utd-mhad/Inertial/"

# data = []
# for f in os.listdir(data_dir):
#     d = scipy.io.loadmat(os.path.join(data_dir,f))
#     data.append(d['d_iner'])
# data = np.array(data)
# print(data.shape)

# # data = pd.read_csv('/home/abhi/research/action_recognition/toy_HAR/IMU/train.csv')

# # Split the data into features and labels
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]

train_dir = "/home/abhi/data/utd-mhad/Inertial_splits/train.txt"
val_dir = "/home/abhi/data/utd-mhad/Inertial_splits/val.txt"
train_dataset = IMUDataset(train_dir)
val_dataset = IMUDataset(val_dir)

# Read the pytorch datasets into np arrays
X_train = np.array([elt[0].numpy() for elt in train_dataset])
y_train = np.array([elt[1] for elt in train_dataset])
X_val = np.array([elt[0].numpy() for elt in val_dataset])
y_val = np.array([elt[1] for elt in val_dataset])

# Flatten the data across time
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)

# Create a RandomForestClassifier object with desired hyperparameters
rfc = RandomForestClassifier(n_estimators=1000, max_depth=50, random_state=76)

# Fit the classifier to the training data
rfc.fit(X_train, y_train)

# Predict the labels for the testing data
y_pred = rfc.predict(X_val)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_val, y_pred)
print('Accuracy:', accuracy)
