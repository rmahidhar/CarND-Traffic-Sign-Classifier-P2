from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from sklearn.utils import shuffle
import tensorflowvisu
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import os
import numpy
import matplotlib.gridspec as gridspec
import csv
import cv2
import matplotlib.image as mpimg

class Model(object):
    def __init__(self, input_shape, num_classes):
        tf.set_random_seed(0.0)
        self._x = tf.placeholder(tf.float32, (None,) + input_shape)
        print('xshape {}'.format(self._x.get_shape()))
        print()
        self._y = tf.placeholder(tf.int32, (None))
        self._one_hot_y = tf.one_hot(self._y, num_classes)
        self._keep_prob = tf.placeholder(tf.float32)
        self._keep_prob_conv = tf.placeholder(tf.float32)
        self._learning_rate = tf.placeholder(tf.float32)
        self._batch_norm_test = tf.placeholder(tf.bool)
        self._batch_norm_iter = tf.placeholder(tf.int32)
        self._activation = None
        self._output = None
        self._cost = None
        self._optimizer = None
        self._correct_predictions = None
        self._test_accuracy = None
        self._test_loss = None
        self._sess = None
        self._saver = None
        self._x_train = self._y_train = self._x_validation = self._y_validation = None
        self._x_test = self._y_test = None
        self._conv_activations = []
        self._dense_activations = []
        self._update_ema = []
        self._dataset = None

        # training accuracy and loss
        self._training_iterations = []
        self._training_accuracy = []
        self._validation_accuracy = []
        self._validation_iterations = []
        self._training_loss = []
        self._validation_loss = []

    def x(self):
        return self._x

    def y(self):
        return self._y

    def one_hot_y(self):
        return self._one_hot_y

    def activation(self):
        return self._activation

    def keep_prob(self):
        return self._keep_prob

    def compatible_convolutional_noise_shape(self, y):
        noiseshape = tf.shape(y)
        noiseshape = noiseshape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])
        return noiseshape

    def conv2d(self, ksize, nfeatures, stride=1, batch_norm=False, dropout=False):

        if self._activation is not None:
            x = self._activation
        else:
            x = self._x

        channels = x.get_shape()[3].value
        print('convd input shape {}'.format(x.get_shape()))
        weights = tf.Variable(tf.truncated_normal(shape=[ksize, ksize, channels, nfeatures],
                                                  mean=0,
                                                  stddev=0.1))
        bias = tf.Variable(tf.zeros(nfeatures))

        conv = tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding='SAME')

        if batch_norm:
            conv, update_ema = self.batchnorm(conv, self._batch_norm_test,
                                              self._batch_norm_iter,
                                              bias, convolutional=True)
            self._update_ema.append(update_ema)
        else:
            conv = tf.nn.bias_add(conv, bias)

        self._activation = tf.nn.relu(conv)

        self._conv_activations.append(tf.reshape(tf.reduce_max(self._activation, [0]), [-1]))

        if dropout:
            self._activation = tf.nn.dropout(self._activation,
                                             self._keep_prob_conv,
                                             self.compatible_convolutional_noise_shape(self._activation))

        print('convd output shape {}'.format(self._activation.get_shape()))
        print()

    def maxpool(self, ksize, stride):
        print('maxpool input shape {}'.format(self._activation.get_shape()))
        self._activation = tf.nn.max_pool(self._activation,
                                          ksize=[1, ksize, ksize, 1],
                                          strides=[1, stride, stride, 1],
                                          padding='VALID')
        print('maxpool output shape {}'.format(self._activation.get_shape()))
        print()

    def fc(self, nodes, batch_norm=False, dropout=False, act=tf.nn.relu):
        if self._activation is not None:
            x = self._activation
        else:
            x = self._x

        shape = x.get_shape()
        print('fc input shape {}'.format(shape))
        # if rank of input tensor is greater than 2
        # we need to reshape it
        if shape.ndims > 2:
            n = 1
            for s in shape[1:]:
                n *= s.value
            x = tf.reshape(x, tf.pack([-1, n]))
            x.set_shape([None, n])

        # get number of column in input tensor
        n = x.get_shape()[1].value

        weights = tf.Variable(tf.truncated_normal(shape=[n, nodes],
                                                  mean=0,
                                                  stddev=0.1))

        bias = tf.Variable(tf.zeros(nodes))

        if batch_norm:
            fc = tf.matmul(x, weights)
            fc, update_ema = self.batchnorm(fc, self._batch_norm_test,
                                            self._batch_norm_iter,
                                            bias)
            self._update_ema.append(update_ema)
        else:
            fc = tf.add(tf.matmul(x, weights), bias)

        if act is not None:
            self._activation = act(fc)
            self._dense_activations.append(tf.reshape(tf.reduce_max(self._activation, [0]), [-1]))
        else:
            self._activation = fc

        if dropout:
            self._activation = tf.nn.dropout(self._activation, self._keep_prob)


        print('fc output shape {}'.format(self._activation.get_shape()))
        print()

    def batch_trainer_1(self, x_train, y_train, x_validation, y_validation, epochs=30, batch_size=256 ,dropout=.75):
        self._saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(x_train)
            for i in range(epochs):
                x_train, y_train = shuffle(x_train, y_train)
                for offset in range(0, num_examples, batch_size):
                    end = offset + batch_size
                    batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                    sess.run(self._optimizer,
                             feed_dict={self._x: batch_x,
                                        self._y: batch_y,
                                        self._batch_norm_test: False,
                                        self._keep_prob_conv: 1.0,
                                        self._keep_prob: dropout})

                validation_accuracy = self._evaluate(x_validation, y_validation)
                print("EPOCH {} ...".format(i + 1) +
                      "Validation Accuracy = {:.3f}".format(validation_accuracy))
                print()
            self._saver.save(sess, './lenet')

    def append_training_data(self, i, a, c):
        self._training_iterations.append(i)
        self._training_accuracy.append(a)
        self._training_loss.append(c)

    def append_validation_data(self, i, a, c):
        self._validation_iterations.append(i)
        self._validation_accuracy.append(a)
        self._validation_loss.append(c)

    def batch_trainer_2(self, x_train, y_train, x_validation, y_validation,
                        epochs=10, batch_size=128, dropout=0.75):
        self._saver = tf.train.Saver()

        with tf.Session() as self._sess:
            self._sess.run(tf.global_variables_initializer())
            current_epoch = 0
            dataset = DataSet(x_train, y_train, reshape=False)
            i = 0
            while epochs >= dataset.epochs_completed:
                batch_x, batch_y = dataset.next_batch(batch_size)
                # Run optimization op (backprop)
                self._sess.run(self._optimizer,
                         feed_dict={self._x: batch_x,
                                    self._y: batch_y,
                                    self._batch_norm_test: False,
                                    self._keep_prob_conv: 1.0,
                                    self._keep_prob: dropout})
                i += 1
                # compute training values for visualisation
                if i % 20 == 0:
                    a, c, ca, da = self._sess.run(
                        [self._accuracy, self._cost, self._conv_activations, self._dense_activations],
                        {self._x: batch_x,
                         self._y: batch_y,
                         self._batch_norm_test: False,
                         self._keep_prob_conv: 1.0,
                         self._keep_prob: 1.0})
                    print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))
                    self.append_training_data(i, a, c)

                # compute test values for visualisation
                if i % 100 == 0:
                    a, c = self._evaluate(x_validation, y_validation, batch_size)
                    #a, c = self._sess.run([self._accuracy, self._cost],
                    #                      {self._x: x_validation,
                    #                       self._y: y_validation,
                    #                       self._batch_norm_test: True,
                    #                       self._keep_prob_conv: 1.0,
                    #                       self._keep_prob: 1.0})
                    print(str(i) + ": ********* epoch " + str(dataset.epochs_completed)
                          + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
                    self.append_validation_data(i, a, c)

            print("\rOptimization Finished!")
            self._saver.save(self._sess, './lenet')

    def train(self,
              x_train,
              y_train,
              x_validation,
              y_validation,
              epochs=10,
              batch_size=128,
              dropout=0.75):

        learning_rate = 0.001

        # Define loss function
        self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self._activation, self._one_hot_y))

        # Define optimizer
        self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._cost)

        # Define accuracy operation
        self._pred_labels_sm = tf.nn.softmax(self._activation)
        self._pred_labels = tf.argmax(self._pred_labels_sm, 1)
        self._correct_prediction = tf.equal(self._pred_labels, tf.argmax(self._one_hot_y, 1))
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))

        self.batch_trainer_2(x_train, y_train, x_validation, y_validation,
                             epochs, batch_size, dropout)

    def _evaluate(self, x_data, y_data, batch_size = 128):
        num_examples = len(x_data)
        total_accuracy = 0
        total_cost = 0.
        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = x_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
            accuracy, cost = sess.run([self._accuracy, self._cost],
                                feed_dict={self._x: batch_x,
                                           self._y: batch_y,
                                           self._batch_norm_test: False,
                                           self._keep_prob_conv : 1.0,
                                           self._keep_prob: 1.})
            total_accuracy += (accuracy * len(batch_x))
            total_cost += (cost * len(batch_x))
        return total_accuracy / num_examples, total_cost / num_examples

    def evaluate(self, x_test, y_test):
        with tf.Session() as sess:
            self._saver.restore(sess, tf.train.latest_checkpoint('.'))
            self._test_accuracy, self._test_loss = self._evaluate(x_test, y_test)
            print("Test Accuracy = {:.3f} Test Loss = {:.3f}".format(self._test_accuracy, self._test_loss))

    def print_confusion_matrix(self, x_test, y_test):
        with tf.Session() as sess:
            self._saver.restore(sess, tf.train.latest_checkpoint('.'))
            pred_labels = sess.run(self._pred_labels,
                                   feed_dict={self._x: x_test[:1024],
                                              self._y: y_test[:1024],
                                              self._batch_norm_test: False,
                                              self._keep_prob_conv: 1.,
                                              self._keep_prob: 1.})

            cm = confusion_matrix(y_true=y_test[:1024], y_pred=pred_labels)
            print(cm)
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

            # Make various adjustments to the plot.
            plt.tight_layout()
            plt.colorbar()
            tick_marks = numpy.arange(self._nclasses)
            plt.xticks(tick_marks, range(self._nclasses))
            plt.yticks(tick_marks, range(self._nclasses))
            plt.xlabel('Predicted')
            plt.ylabel('True')


    def pre_process_image(self, image_name):
        image = cv2.imread(image_name)
        image = cv2.resize(image,(32,32))
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img = cv2.equalizeHist(img)
        img = img[..., numpy.newaxis]
        return img

    def evaluate_extra_images(self, dir_name, labels):
        images = [self.pre_process_image(dir_name + "/" + name) for name in os.listdir(dir_name)]
        color_images = [mpimg.imread(dir_name + "/" + name) for name in os.listdir(dir_name)]
        image_names = [name for name in os.listdir(dir_name)]
        images = numpy.array(images, dtype=numpy.float32)
        
        with tf.Session() as sess:
            self._saver.restore(sess, tf.train.latest_checkpoint('.'))

            top5 = tf.nn.top_k(self._pred_labels_sm, 5)

            pred_labels = sess.run(self._pred_labels_sm,
                                   feed_dict={self._x: images,
                                                 self._batch_norm_test: False,
                                                 self._keep_prob_conv: 1.,
                                                 self._keep_prob: 1.})

            top5_pred = sess.run([self._pred_labels_sm, top5],
                                  feed_dict={self._x: images,
                                               self._batch_norm_test: False,
                                               self._keep_prob_conv: 1.,
                                               self._keep_prob: 1.})

            f = open(labels)
            names = csv.reader(f)
            names = list(names)
            names = names[1:]
            sign_names = {}
            for sign_label, sign_name in names:
                sign_names[int(sign_label)] = sign_name

            for i in range(len(image_names)):
                plt.figure(figsize=(5, 1.5))
                gs = gridspec.GridSpec(1, 2, width_ratios=[2, 3])
                plt.subplot(gs[0])
                plt.imshow(color_images[i])
                plt.axis('off')
                plt.subplot(gs[1])
                plt.barh(6 - numpy.arange(5), top5_pred[1][0][i], align='center')
                for i_label in range(5):
                    plt.text(top5_pred[1][0][i][i_label] + .02, 6 - i_label - .25,
                         sign_names[top5_pred[1][1][i][i_label]])
                plt.axis('off')
                plt.text(0, 6.95, image_names[i].split('.')[0])
                plt.show()

    def train_and_plot(self,
                       x_train,
                       y_train,
                       x_validation,
                       y_validation,
                       epochs=30,
                       batch_size=256,
                       dropout=0.75):

        # Define loss function
        self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self._activation, self._one_hot_y))

        # Define optimizer
        self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._cost)

        # Define accuracy operation
        Y = tf.nn.softmax(self._activation)
        self._correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(self._one_hot_y, 1))
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))

        if len(self._update_ema) != 0:
            self._update_ema = tf.group(*self._update_ema)
        else:
            self._update_ema = None
        self.batch_train_and_plot(x_train, y_train, x_validation, y_validation,
                                  epochs, batch_size, dropout)

    def batch_train_and_plot(self, x_train, y_train, x_validation, y_validation,
                             epochs, batch_size, dropout):
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self._datavis = tensorflowvisu.MnistDataVis()
        self._x_train = x_train
        self._y_train = y_train
        self._x_validation = x_validation
        self._y_validation = y_validation
        if len(self._conv_activations) != 0:
            self._conv_activations = tf.concat(0, self._conv_activations)
        if len(self._dense_activations) != 0:
            self._dense_activations = tf.concat(0, self._dense_activations)
        self._dataset = DataSet(x_train, y_train, reshape=False)

        self._datavis.animate(self.training_step, 10000, train_data_update_freq=20, test_data_update_freq=100)

    def training_step(self, i, update_test_data, update_train_data):
        # training on batches of 100 images with 100 labels

        batch_X, batch_Y = self._dataset.next_batch(100)

        # learning rate decay
        max_learning_rate = 0.003
        min_learning_rate = 0.0001
        decay_speed = 2000
        # max_learning_rate = 0.02
        # min_learning_rate = 0.00015
        # decay_speed = 1000.0
        lr = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)

        # compute training values for visualisation
        if update_train_data:
            a, c, ca, da = self._sess.run([self._accuracy, self._cost, self._conv_activations, self._dense_activations],
                                          {self._x: batch_X,
                                           self._y: batch_Y,
                                           self._batch_norm_test: False,
                                           self._keep_prob_conv: 1.0,
                                           self._keep_prob:1.0})
            print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(lr) + ")")
            self._datavis.append_training_curves_data(i, a, c)
            self._datavis.append_data_histograms(i, ca, da)

        # compute test values for visualisation
        if update_test_data:
            a, c = self._sess.run([self._accuracy, self._cost],
                                  {self._x: self._x_validation,
                                   self._y: self._y_validation,
                                   self._batch_norm_test: True,
                                   self._keep_prob_conv: 1.0,
                                   self._keep_prob: 1.0})
            print(str(i) + ": ********* epoch " + str(i * 100 // self._x_train.shape[0] + 1) + " ********* test accuracy:" + str(
                a) + " test loss: " + str(c))
            self._datavis.append_test_curves_data(i, a, c)

        # the backpropagation training step
        self._sess.run(self._optimizer,
                      {self._x: batch_X,
                       self._y: batch_Y,
                       self._learning_rate: lr,
                       self._batch_norm_test: False,
                       self._keep_prob_conv: 1.0,
                       self._keep_prob: 0.75})
        if self._update_ema:
            #print("computing ema")
            self._sess.run(self._update_ema,
                           {self._x: batch_X,
                            self._y: batch_Y,
                            self._batch_norm_test: False,
                            self._batch_norm_iter: i,
                            self._keep_prob_conv: 1.0,
                            self._keep_prob: 1.0})

    def batchnorm(self, logits, is_test, iteration, offset, convolutional=False):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(logits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(logits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(logits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages

    def plot_model(self):
        fig = plt.figure(figsize=(19.20,10.80), dpi=70)
        plt.gcf().canvas.set_window_title("Traffic Classifer")
        fig.set_facecolor('#FFFFFF')

        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        #ax3 = fig.add_subplot(234)
        #ax4 = fig.add_subplot(235)

        ax1.set_title("Accuracy", y=1.02)
        ax2.set_title("Cross entropy loss", y=1.02)
        #ax3.set_title("Weights", y=1.02)
        #ax4.set_title("Biases", y=1.02)

        ax1.set_ylim(0, 1)  # important: not autoscaled
        ax2.autoscale(axis='y')
        #ax2.set_ylim(0, 5)  # important: not autoscaled

        line1, = ax1.plot(self._training_iterations, self._training_accuracy, label="training accuracy")
        line2, = ax1.plot(self._validation_iterations, self._validation_accuracy, label="test accuracy")
        legend = ax1.legend(loc='lower right') # fancybox : slightly rounded corners
        legend.draggable(True)

        line3, = ax2.plot(self._training_iterations, self._training_loss, label="training loss")
        line4, = ax2.plot(self._validation_iterations, self._validation_loss, label="test loss")
        legend = ax2.legend(loc='upper right') # fancybox : slightly rounded corners
        legend.draggable(True)

        line1.set_data(self._training_iterations, self._training_accuracy)
        line2.set_data(self._validation_iterations, self._validation_accuracy)
        line3.set_data(self._training_iterations, self._training_loss)
        line4.set_data(self._validation_iterations, self._validation_loss)

        #plt.show()
        plt.savefig('myfilename.png')

    """"
    def plot_error_examples():
        correct, labels_cls_pred = session.run([correct_prediction, labels_pred_cls],
                                               feed_dict=feed_dict_test)
        incorrect = (correct == False)
        X_incorrect = X_test[incorrect]
        y_incorrect = y_test[incorrect]
        y_pred = labels_cls_pred[incorrect]

        plot_random_3C(3, 3, X_incorrect, y_incorrect)

    def plot_random_3C(n_row,n_col,X,y):

        plt.figure(figsize = (11,8))
        gs1 = gridspec.GridSpec(n_row,n_row)
        gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes.

        for i in range(n_row*n_col):
            # i = i + 1 # grid spec indexes from 0
            ax1 = plt.subplot(gs1[i])
            plt.axis('on')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            #plt.subplot(4,11,i+1)
            ind_plot = np.random.randint(1,len(y))
            plt.imshow(X[ind_plot])
            plt.text(2,4,str(y[ind_plot]),
                 color='k',backgroundcolor='c')
            plt.axis('off')
        plt.show()

    def plot_random_1C(n_row,n_col,X,y):

        plt.figure(figsize = (11,8))
        gs1 = gridspec.GridSpec(n_row,n_row)
        gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes.

        for i in range(n_row*n_col):
            # i = i + 1 # grid spec indexes from 0
            ax1 = plt.subplot(gs1[i])
            plt.axis('on')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            #plt.subplot(4,11,i+1)
            ind_plot = np.random.randint(1,len(y))
            plt.imshow(X[ind_plot],cmap='gray')
            plt.text(2,4,str(y[ind_plot]),
                 color='k',backgroundcolor='c')
            plt.axis('off')
        plt.show()

    def plot_random_preprocess(n_row,n_col,X,y):

        plt.figure(figsize = (11,8))
        gs1 = gridspec.GridSpec(n_row,n_row)
        gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes.

        for i in range(n_row*n_col):
            # i = i + 1 # grid spec indexes from 0
            ax1 = plt.subplot(gs1[i])
            plt.axis('on')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            #plt.subplot(4,11,i+1)
            ind_plot = np.random.randint(1,len(y))
            plt.imshow(pre_process_image(X[ind_plot]),cmap='gray')
            plt.text(2,4,str(y[ind_plot]),
                 color='k',backgroundcolor='c')
            plt.axis('off')
        plt.show()
    """
