
# coding: utf-8

# # HAR CNN training 

# In[292]:
# HAR classification 
# Author: Burak Himmetoglu
# 8/15/2017

import pandas as pd 
import numpy as np
import os

NUM_CLASS = 18
WINDOWN_SIZE = 128
CHANNEL_LIST = ["x_sensor_acc","y_sensor_acc","z_sensor_acc","x_watch_acc","y_watch_gyr","z_watch_acc","x_watch_gyr","y_watch_acc","z_watch_gyr"]
NUM_CHANNEL = len(CHANNEL_LIST)

class_label = ["wrist","ankle","shoulder","haunch","knee","walking","running","kicking","cycling","brushing","washing_hand","wipe","sweep_floor","peel","slice","mixing","upstair","downstair"]
class_label_int = {val:str(idx) for idx, val in enumerate(class_label)}

def read_data(data_path, split = "train"):
    """ Read data """

    # Fixed params
    n_class = 18
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

def one_hot(labels, n_class = 18):
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
    






# Imports
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score,     recall_score, confusion_matrix, classification_report,     accuracy_score, f1_score
# from utils.utilities import *


# In[293]:



# In[294]:


class_label_vn = [u"Cổ tay",u"Cổ chân",u"Bả vai",u"Xoay người",u"Xoay đầu gối",u"Đi bộ",u"Chạy",u"Đá bóng",u"Đạp xe",u"Đánh răng",u"Rửa tay",u"Lau bàn",u"Quét nhà",u"Nạo",u"Thái",u"Trộn",u"Lên cầu thang",u"Xuống cầu thang"]


# ## Prepare data

# In[295]:


rootDatasetDir = "./datasets/PTIT/normalized"
X_train, labels_train, list_ch_train = read_data(data_path="./datasets/PTIT/normalized", split="train") # train
X_test, labels_test, list_ch_test = read_data(data_path=rootDatasetDir, split="test") # test
assert list_ch_train == list_ch_test, "Mistmatch in channels!"


# In[296]:


# Normalize?
X_train, X_test = standardize(X_train, X_test)


# Train/Validation Split

# In[297]:


X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train, 
                                                stratify = labels_train, random_state = 123)


# One-hot encoding:

# In[298]:


y_tr = one_hot(lab_tr)
y_vld = one_hot(lab_vld)
y_test = one_hot(labels_test)


# In[299]:


# Imports
import tensorflow as tf


# ### Hyperparameters

# In[300]:


batch_size = 600       # Batch size
seq_len = WINDOWN_SIZE          # Number of steps or window size
learning_rate = 0.0001
epochs = 1000

n_classes = NUM_CLASS
n_channels = NUM_CHANNEL
print "n_channels %d" % n_channels


# ### Construct the graph
# Placeholders

# In[301]:


graph = tf.Graph()

# Construct placeholders
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
    labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')


# Build Convolutional Layers

# In[302]:


with graph.as_default():
    # (batch, 128, 9) --> (batch, 64, 18)
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=18, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')
    
    # (batch, 64, 18) --> (batch, 32, 36)
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
    
    # (batch, 32, 36) --> (batch, 16, 72)
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')
    
    # (batch, 16, 72) --> (batch, 8, 144)
    conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=144, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
    max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')


# Now, flatten and pass to the classifier

# In[303]:


with graph.as_default():
    # Flatten and add dropout
    flat = tf.reshape(max_pool_4, (-1, 8*144))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    
    # Predictions
    logits = tf.layers.dense(flat, n_classes)
    
    # Cost function and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
    
    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


# ### Train the network

# In[ ]:


if (os.path.exists('checkpoints-cnn') == False):
    get_ipython().system(u'mkdir checkpoints-cnn')


# In[ ]:


validation_acc = []
validation_loss = []

train_acc = []
train_loss = []

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
   
    # Loop over epochs
    for e in range(epochs):
        
        # Loop over batches
        for x,y in get_batches(X_tr, y_tr, batch_size):
            
            # Feed dictionary
            feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5, learning_rate_ : learning_rate}
            
            # Loss
            loss, _ , acc = sess.run([cost, optimizer, accuracy], feed_dict = feed)
            train_acc.append(acc)
            train_loss.append(loss)
            
            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))
            
            # Compute validation loss at every 10 iterations
            if (iteration%10 == 0):                
                val_acc_ = []
                val_loss_ = []
                
                for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                    # Feed
                    feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0}  
                    
                    # Loss
                    loss_v, acc_v = sess.run([cost, accuracy], feed_dict = feed)                    
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)
                
                # Print info
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))
                
                # Store
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))
            
            # Iterate 
            iteration += 1
    
    saver.save(sess,"checkpoints-cnn/har.ckpt")


# In[ ]:


# Plot training and test loss
t = np.arange(iteration-1)


# ## Evaluate on test set

# In[ ]:


test_acc = []
with graph.as_default():
    prediction=tf.argmax(logits,1)
with tf.Session(graph=graph) as sess:
    # Restore
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints-cnn'))
    y_pred = []
    y_true = []
    for x_t, y_t in get_batches(X_test, y_test, batch_size):
        feed = {inputs_: x_t,
                labels_: y_t,
                keep_prob_: 1}
        batch_acc, batch_y_pred = sess.run([accuracy, prediction], feed_dict=feed)
        y_pred.extend(batch_y_pred)
        y_true.extend(np.where(r==1)[0][0] for r in y_t )
        
        test_acc.append(batch_acc)
    print("Test accuracy: {:.6f}".format(np.mean(test_acc)))
#     print y_true
#     print y_pred
    sk_class_labels = [i for i in range(NUM_CLASS)]
    print precision_recall_fscore_support(y_true, y_pred, average=None, labels=sk_class_labels)
    print 'Accuracy:', accuracy_score(y_true, y_pred)
    print 'F1 score:', f1_score(y_true, y_pred, average='micro')
    print 'Recall:', recall_score(y_true, y_pred, average='micro')
    print 'Precision:', precision_score(y_true, y_pred, average='micro')
    print '\n clasification report:\n', classification_report(y_true,y_pred)
    print '\n confussion matrix:\n',confusion_matrix(y_true, y_pred)

