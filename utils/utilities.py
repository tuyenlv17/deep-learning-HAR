import pandas as pd 
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

WINDOWN_SIZE = 128
CHANNEL_LIST = ["x_sensor_acc","y_sensor_acc","z_sensor_acc","x_watch_acc","y_watch_gyr","z_watch_acc","x_watch_gyr","y_watch_acc","z_watch_gyr"]
NUM_CHANNEL = len(CHANNEL_LIST)

class_label = ["wrist","ankle","shoulder","haunch","knee","walking","running","cycling","brushing","washing_hand","slice","mixing","upstair","downstair"]
NUM_CLASS = len(class_label)
class_label_int = {val:str(idx) for idx, val in enumerate(class_label)}

def read_data(data_path, split = "train"):
    """ Read data """

    # Fixed params
    n_class = NUM_CLASS
    n_steps = WINDOWN_SIZE

    # Paths
    path_ = os.path.join(data_path, split)
    path_signals = os.path.join(path_, "sensor")

    # Read labels and one-hot encode
    label_path = os.path.join(path_, "class.txt")
    labels = pd.read_csv(label_path, header = None)
    
    # Read time-series data
    channel_files = os.listdir(path_signals)
    channel_files.sort()
    n_channels = len(channel_files)
    posix = len(split) + 5

    # Initiate array
    list_of_channels = []
    X = np.zeros((len(labels), n_steps, n_channels))
    i_ch = 0
    for fil_ch in channel_files:
        channel_name = fil_ch[:-4]
        print channel_name
        dat_ = pd.read_csv(os.path.join(path_signals,fil_ch), delim_whitespace = True, header = None)
        X[:,:,i_ch] = dat_.as_matrix()

        # Record names
        list_of_channels.append(channel_name)

        # iterate
        i_ch += 1

    # Return 
    return X, labels[0].values, list_of_channels

def standardize(train, test):
    """ Standardize data """

    # Standardize train and test
    X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
    X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]

    return X_train, X_test

def one_hot(labels, n_class = NUM_CLASS):
    """ One-hot encoding """
    expansion = np.eye(n_class)
    y = expansion[:, labels-1].T
    assert y.shape[1] == n_class, "Wrong number of labels!"

    return y

def get_batches(X, y, batch_size = 100):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]
    

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

