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
import random

def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]


def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)

def read_dataset(train, test):
    with open(train, mode='rb') as f:
        train = pickle.load(f)
    with open(test, mode='rb') as f:
        test = pickle.load(f)

    return train['features'], train['labels'], test['features'], test['labels']

def grayscale(images):
    gray_images = numpy.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images])
    return gray_images

def equalize_hist(images):
    hist_images = numpy.array([cv2.equalizeHist(image) for image in images])
    hist_images = hist_images[..., numpy.newaxis]
    return hist_images

def normalize(images):
    norm_images = (images - np.mean(images))
    norm_images = norm_images / np.std(norm_images)
    return norm_images

def image_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_hist(image):
    return cv2.equalizeHist(image)

def image_normalize(image):
    norm_image = (image - np.mean(image))
    norm_image = norm_image / np.std(norm_image)
    return norm_image

# Translation is the shifting of object's location.
# tx, ty is the point where the point (0,0) will be
# shifted.
def image_random_translate(img):
    rows, cols, _ = img.shape
    # allow translation up to px pixels in x and y directions
    px = 2
    tx, ty = np.random.randint(-px, px, 2)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    dst = dst[:, :, np.newaxis]
    return dst

# For perspective transformation, 3x3 transformation matrix is required.
# Straight lines will remain straight even after the transformation. For finding
# the transformation matrix, we need 4 points on the input image and corresponding
# points on the output image. Among these 4 points, 3 of them should not be
# collinear. The transformation matrix can be found by the function
# cv2.getPerspectiveTransform. Then apply cv2.warpPerspective with the 3x3
# transformation matrix.
def image_random_scaling(img):
    rows, cols, _ = img.shape
    # transform limits
    px = np.random.randint(-2, 2)
    # ending locations
    pts1 = np.float32([[px, px], [rows - px, px], [px, cols - px], [rows - px, cols - px]])
    # starting locations (4 corners)
    pts2 = np.float32([[0, 0], [rows, 0], [0, cols], [rows, cols]])
    # pts1 are scale to pts2.
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (rows, cols))
    dst = dst[:, :, np.newaxis]
    return dst

# In affine transformation, all parallel lines in the original image will still be parallel
# in the output image. we need three points from the input image and their corresponding
# locations in output image.
def image_random_warp(img):
    rows, cols, _ = img.shape
    # random scaling coefficients
    rndx = np.random.rand(3) - 0.5
    rndx *= cols * 0.06  # this coefficient determines the degree of warping
    rndy = np.random.rand(3) - 0.5
    rndy *= rows * 0.06
    # 3 starting points for transform, 1/4 way from edges
    x1 = cols / 4
    x2 = 3 * cols / 4
    y1 = rows / 4
    y2 = 3 * rows / 4
    pts1 = np.float32([[y1, x1],
                       [y2, x1],
                       [y1, x2]])
    pts2 = np.float32([[y1 + rndy[0], x1 + rndx[0]],
                       [y2 + rndy[1], x1 + rndx[1]],
                       [y1 + rndy[2], x2 + rndx[2]]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))
    dst = dst[:, :, np.newaxis]
    return dst

def image_random_brightness(img):
    shifted = img + 1.0   # shift to (0,2) range
    img_max_value = max(shifted.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = shifted * coef - 1.0
    return dst

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def create_variant(image):
    if (random.choice([True, False])):
        image = scipy.ndimage.interpolation.shift(image, [random.randrange(-2, 2), random.randrange(-2, 2), 0])
    else:
        image = scipy.ndimage.interpolation.rotate(image, random.randrange(-10, 10), reshape=False)
    return image

def transform_image(img,ang_range=20,shear_range=10,trans_range=5):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.
    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    #rows,cols,ch = img.shape
    rows = img.shape[0]
    cols = img.shape[1]
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    # Brightness
    if random.choice([True, False]):
        img = image_random_brightness(img)

    return img

def subplot_images_transform(images,
                             num_images,
                             transform):
    #subplots_adjust(left=None, bottom=None, right=None,
    # top=None, wspace=None, hspace=None)
    #left = 0.125  # the left side of the subplots of the figure
    #right = 0.9  # the right side of the subplots of the figure
    #bottom = 0.1  # the bottom of the subplots of the figure
    #top = 0.9  # the top of the subplots of the figure
    #wspace = 0.2  # the amount of width reserved for blank space between subplots
    #hspace = 0.2  # the amount of height reserved for white space between subplots
    #fig.tight_layout()

    offset = random.randint(1, images.shape[0] - num_images)
    rows, cols = get_grid_dim(num_images * 2)
    fig, axes = plt.subplots(rows, cols, figsize=(32, 32))
    axes = axes.ravel()
    fig.tight_layout()
    for image_idx in range(offset, offset + num_images):
        idx = image_idx - offset
        image = images[image_idx]
        axes[idx * 2 + 0].axis('off')
        axes[idx * 2 + 0].imshow(image.squeeze())
        axes[idx * 2 + 0].set_title('original')
        axes[idx * 2 + 1].axis('off')
        transform_image = transform(image)
        axes[idx * 2 + 1].imshow(transform_image.squeeze(), cmap='gray')
        axes[idx * 2 + 1].set_title('transformed')

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_everages

def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()

def FlatNet(x):
    W = tf.Variable(tf.zeros([32*32, 43]))
    # biases b[10]
    b = tf.Variable(tf.zeros([43]))

    # flatten the images into a single line of pixels
    # -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
    XX = tf.reshape(X, [-1, 32 * 32])

    # The model
    logits = tf.matmul(XX, W) + b;
    Y = tf.nn.softmax(logits)
    return logits, None, None, None, None, None

def generate_augmented_samples(x_train, y_train):
    labels = y_train.tolist()
    signs = [labels.count(y) for y in range(43)]
    required_samples = [int(math.ceil(1000 / labels.count(y))) for y in range(43)]

    augmented_samples = []
    augmented_labels = []
    for idx, label in enumerate(labels, start=0):
        if required_samples[label] > 1:
            for i in range(required_samples[label]):
                augmented_samples.append(transform_image(x_train[idx]))
                augmented_labels.append(label)
                print(i, idx)
    x_train_augmented = np.append(np.array(x_train), np.array(augmented_samples), axis=0)
    y_train_augmented = np.append(np.array(y_train), np.array(augmented_labels), axis=0)
    print("Generated augmented samples", len(augmented_samples))
    print("new data set", x_train_augmented.shape)
    return x_train_augmented, y_train_augmented

x_train, y_train, x_test, y_test = read_dataset('train.p', 'test.p')
generate_augmented_samples(x_train, y_train)

def MyNet(x):
    # three convolutional layers with their channel counts, and a
    # fully connected layer (tha last layer has 10 softmax neurons)
    K = 6  # first convolutional layer output depth
    L = 12  # second convolutional layer output depth
    M = 24  # third convolutional layer
    N = 200  # fully connected layer

    W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
    B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))

    W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
    B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))

    W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
    B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

    W4 = tf.Variable(tf.truncated_normal([8 * 8 * M, N], stddev=0.1))
    B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))

    W5 = tf.Variable(tf.truncated_normal([N, 43], stddev=0.1))
    B5 = tf.Variable(tf.constant(0.1, tf.float32, [43]))

    # The model
    stride = 1  # output is 32x32
    Y1l = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1
    Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1, convolutional=True)
    Y1 = tf.nn.relu(Y1bn)
    stride = 2  # output is 16x16
    Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2
    Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2, convolutional=True)
    Y2 = tf.nn.relu(Y2bn)
    stride = 2  # output is 8x8
    Y3l = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3
    Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3, convolutional=True)
    Y3 = tf.nn.relu(Y3bn)

    # reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 8 * 8 * M])

    Y4l = tf.matmul(YY, W4) + B4
    Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)
    Y4 = tf.nn.relu(Y4bn)
    Y4d = tf.nn.dropout(Y4, pkeep)
    logits = tf.matmul(Y4d, W5) + B5
    #Y = tf.nn.softmax(logits)

    allweights = tf.concat(0, [tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])])
    allbiases  = tf.concat(0, [tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])])
    conv_activations = tf.concat(0, [tf.reshape(tf.reduce_max(Y1, [0]), [-1]), tf.reshape(tf.reduce_max(Y2, [0]), [-1]), tf.reshape(tf.reduce_max(Y3, [0]), [-1])])
    dense_activations = tf.reduce_max(Y4, [0])
    update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)
    #update_ema = None

    return logits, allweights, allbiases, conv_activations, dense_activations, update_ema


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1bn, update_ema1 = batchnorm(conv1, tst, iter, conv1_b, convolutional=True)

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1bn)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2bn, update_ema2 = batchnorm(conv2, tst, iter, conv2_b, convolutional=True)


    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2bn)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1bn, update_ema3 = batchnorm(fc1, tst, iter, fc1_b)


    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1bn)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2bn, update_ema4 = batchnorm(fc2, tst, iter, fc2_b)

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2bn)
    fc2d   = tf.nn.dropout(fc2, pkeep)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2d, fc3_W) + fc3_b

    allweights = tf.concat(0, [tf.reshape(conv1_W, [-1]), tf.reshape(conv2_W, [-1]), tf.reshape(fc1_W, [-1]), tf.reshape(fc2_W, [-1]), tf.reshape(fc3_W, [-1])])
    allbiases  = tf.concat(0, [tf.reshape(conv1_b, [-1]), tf.reshape(conv2_b, [-1]), tf.reshape(fc1_b, [-1]), tf.reshape(fc2_b, [-1]), tf.reshape(fc3_b, [-1])])
    conv_activations = tf.concat(0, [tf.reshape(tf.reduce_max(conv1, [0]), [-1]), tf.reshape(tf.reduce_max(conv2, [0]), [-1]), tf.reshape(tf.reduce_max(fc1, [0]), [-1])])
    dense_activations = tf.concat(0, [tf.reduce_max(fc1, [0]), tf.reduce_max(fc2, [0])])
    update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)
    #update_ema = None

    return logits, allweights, allbiases, conv_activations, dense_activations, update_ema

"""
# test flag for batch norm
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
# dropout probability
pkeep = tf.placeholder(tf.float32)
pkeep_conv = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, (None, 32, 32, 1))
Y_ = tf.placeholder(tf.int32, (None))
lr = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(Y_, 43)
rate = 0.001

logits, allweights, allbiases, conv_activations, dense_activations, update_ema = FlatNet(X)
Y = tf.nn.softmax(logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
cross_entropy = tf.reduce_mean(cross_entropy) * 100
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(one_hot_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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

#X_train = tf.cast(X_train, tf.float32)
#X_train -= np.mean(X_train) # zero-center
#X_train /= np.std(X_train) # normalize

#X_test = tf.cast(X_test, tf.float32)
#X_test -= np.mean(X_test) # zero-center
#X_test /= np.std(X_test) # normalize

X_train_normalized = (X_train - np.mean(X_train))
X_train_normalized = X_train_normalized/np.std(X_train_normalized)

X_test_normalized = (X_test - np.mean(X_test))
X_test_normalized = X_test_normalized/np.std(X_test_normalized)

print('Mean before normalizing:', np.mean(X_train), np.mean(X_test))
print('Mean after normalizing:', np.mean(X_train_normalized), np.mean(X_test_normalized))

X_train, X_validation, y_train, y_validation = train_test_split(X_train_normalized, y_train, test_size=0.20, random_state=7)

dataset = DataSet.DataSet(X_train, y_train, reshape=False)

datavis = tensorflowvisu.MnistDataVis()

# init
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = dataset.next_batch(200)

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000
    #max_learning_rate = 0.02
    #min_learning_rate = 0.00015
    #decay_speed = 1000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    # compute training values for visualisation
    if update_train_data:
        a, c, = sess.run([accuracy, cross_entropy], {X: batch_X, Y_: batch_Y, tst: False, pkeep:1.0})
        #a, c, ca, da = sess.run([accuracy, cross_entropy, conv_activations, dense_activations], {X: batch_X, Y_: batch_Y, tst: False, pkeep:1.0})
        #print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        datavis.append_training_curves_data(i, a, c)
        #datavis.update_image1(im)
        #datavis.append_data_histograms(i, ca, da)

    # compute test values for visualisation
    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], {X: X_test , Y_: y_test, tst: True, pkeep: 1.0})
        print(str(i) + ": ********* epoch " + str(i*200//X_train.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        datavis.append_test_curves_data(i, a, c)
        #datavis.update_image2(im)

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False, pkeep: 0.75})
    #sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 1.0})

#datavis.animate(training_step, 20001, train_data_update_freq=20, test_data_update_freq=200)
"""