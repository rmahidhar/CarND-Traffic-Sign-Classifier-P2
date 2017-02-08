import tensorflow as tf
from tensorflow.contrib.layers import flatten
import tensorflowvisu
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import DataSet
import cv2
import numpy as np
import numpy
import pickle
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def vis_conv(v,ix,iy,ch,cy,cx, p = 0) :
    v = np.reshape(v,(iy,ix,ch))
    ix += 2
    iy += 2
    npad = ((1,1), (1,1), (0,0))
    v = np.pad(v, pad_width=npad, mode='constant', constant_values=p)
    v = np.reshape(v,(iy,ix,cy,cx))
    v = np.transpose(v,(2,0,3,1)) #cy,iy,cx,ix
    v = np.reshape(v,(cy*iy,cx*ix))
    return v

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

with open('train.p', mode='rb') as f:
    train = pickle.load(f)
with open('test.p', mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
# convert to B/W
X_train_bw = numpy.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in X_train])
X_test_bw = numpy.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in X_test])

# apply histogram equalization
X_train_hst_eq = numpy.array([cv2.equalizeHist(image) for image in X_train_bw])
X_test_hst_eq = numpy.array([cv2.equalizeHist(image) for image in X_test_bw])

# reshape for conv layer
X_train = X_train_hst_eq[..., numpy.newaxis]
X_test = X_test_hst_eq[..., numpy.newaxis]
print('Before shaping:', X_train_hst_eq.shape)
print('After shaping:', X_train.shape)

X = tf.placeholder(tf.float32, (None, 32, 32, 1))
Y_ = tf.placeholder(tf.int32, (None))

# SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = 0, stddev = 0.1))
conv1_b = tf.Variable(tf.zeros(6))
conv1   = tf.nn.conv2d(X, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
# SOLUTION: Activation.
conv1 = tf.nn.relu(conv1)

# SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = 0, stddev = 0.1))
conv2_b = tf.Variable(tf.zeros(16))
conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
# SOLUTION: Activation.
conv2 = tf.nn.relu(conv2)

# SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# SOLUTION: Flatten. Input = 5x5x16. Output = 400.
fc0   = flatten(conv2)

# SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = 0, stddev = 0.1))
fc1_b = tf.Variable(tf.zeros(120))
fc1   = tf.matmul(fc0, fc1_W) + fc1_b
# SOLUTION: Activation.
fc1    = tf.nn.relu(fc1)

# SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = 0, stddev = 0.1))
fc2_b  = tf.Variable(tf.zeros(84))
fc2    = tf.matmul(fc1, fc2_W) + fc2_b
# SOLUTION: Activation.
fc2    = tf.nn.relu(fc2)
#fc2d   = tf.nn.dropout(fc2, pkeep)

# SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = 0, stddev = 0.1))
fc3_b  = tf.Variable(tf.zeros(43))
logits = tf.matmul(fc2, fc3_W) + fc3_b

lr = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(Y_, 43)
rate = 0.001

#X_train -= np.mean(X_train, axis = 0) # zero-center
#X_train /= np.std(X_train, axis = 0) # normalize

#X_test -= np.mean(X_test, axis = 0) # zero-center
#X_test /= np.std(X_test, axis = 0) # normalize

#X_train_normalized = (X_train - np.mean(X_train)) / 128.0
#X_test_normalized = (X_test - np.mean(X_test)) / 128.0
#print('Mean before normalizing:', np.mean(X_train), np.mean(X_test))
#print('Mean after normalizing:', np.mean(X_train_normalized), np.mean(X_test_normalized))

# init
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

train = X_train[0]
train = train[numpy.newaxis,...]

conv1 = sess.run([conv1], {X : train})

#ch = 6
#cy = 4   # grid from channels:  32 = 4x8
#cx = 8


# W_conv1 - weights
#ix = 5  # data size
#iy = 5
#v  = vis_conv(vv1,ix,iy,ch,cy,cx)
#plt.figure(figsize = (8,8))
#plt.imshow(v,cmap="Greys_r",interpolation='nearest')

#  h_conv1 - processed image
#ix = 28  # data size
#iy = 28
#v  = vis_conv(conv1,ix,iy,ch,cy,cx)
#plt.figure(figsize = (8,8))
#plt.imshow(v,cmap="Greys_r",interpolation='nearest')

h_conv1_features = tf.unpack(conv1, axis=3)
h_conv1_max = tf.reduce_max(conv1)
h_conv1_features_padded = map(lambda t: tf.pad(t-h_conv1_max, [[0,0],[0,1],[0,0]])+h_conv1_max, h_conv1_features)
h_conv1_imgs = tf.expand_dims(tf.concat(1, h_conv1_features_padded), -1)
print(h_conv1_imgs.shape)
#datavis.animate(training_step, 20001, train_data_update_freq=20, test_data_update_freq=200)
