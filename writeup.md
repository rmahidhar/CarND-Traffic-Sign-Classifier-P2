#**Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

[//]: # (Image References)

[image1]: ./images/train_images.png "Visualization"
[image2]: ./images/training_examples.png "Training Samples Histogram"
[image3]: ./images/test_examples.png "Test Samples Histogram"
[image4]: ./images/pre_processed_images.png "Pre Processed Images"
[image5]: ./images/generated_images.png "Generated Images"
[image6]: ./images/generated_examples.png "Generated Samples Histogram"
[image7]: ./images/augmented_training_examples.png "Total Images"
[image8]: ./images/model_plot.png "Model Plot"
[image9]: ./images/web_images.png "Web Images"
[image10]: ./images/web_image_1_sm_prob.png "Web Image 1"
[image11]: ./images/web_image_2_sm_prob.png "Web Image 2"
[image12]: ./images/web_image_3_sm_prob.png "Web Image 3"
[image13]: ./images/web_image_4_sm_prob.png "Web Image 4"
[image14]: ./images/web_image_5_sm_prob.png "Web Image 5"
[image15]: ./images/web_image_6_sm_prob.png "Web Image 6"
[image16]: ./images/web_image_7_sm_prob.png "Web Image 7"
[image17]: ./images/web_image_8_sm_prob.png "Web Image 8"

##### A python class is modeled for traffic images dataset. It provides various functions for accessing original dataset, pre processed dataset, generated dataset, and also plotting the images, histograms, e.t.c.
 
```python
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
import cv2
import math
from scipy import ndimage
import random
import matplotlib.pyplot as plt
import csv
import os

%matplotlib inline

random.seed()

class TrafficImageDataSet(object):
    def __init__(self, train_p, test_p, sign_names_csv):
        with open(train_p, mode='rb') as f:
            self._train = pickle.load(f)
        with open(test_p, mode='rb') as f:
            self._test = pickle.load(f)
	
        self._read_sign_names(sign_names_csv)
	
        self._train_f = self._train['features']
        self._train_l = self._train['labels']
        self._test_f = self._test['features']
        self._test_l = self._test['labels']
        self._train_c  = self._train['coords'] 
        self._test_c = self._test['coords']
        self._nclasses = len(np.unique(self._train_l))
        self._generate_augmented_train_dataset()
	
    def image_shape(self):
        return self._train_f[0].shape
    
    def nclasses(self):
        return self._nclasses
    
    def sign_names(self):
        return self._sign_names
    
    def plot_histogram(self, labels, title):
        labels = labels.tolist()
        samples_count = [labels.count(y) for y in range(self._nclasses)]
        plt.figure(figsize=(12,8))
        plt.bar(range(self._nclasses),height=samples_count)
        plt.ylabel('Total Samples')
        plt.xlabel('Traffic sign class')
        plt.title(title)
	                   
    def grayscale(self, images):
        gray_images = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images])
        return gray_images
    
    def normalize(self, images):
        norm_images = (images - np.mean(images))
        norm_images = norm_images / np.std(norm_images)
        return norm_images
	
    def image_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	
    def image_normalize(self, image):
        norm_image = (image - 128) /128
        #norm_image = (image - np.mean(image))
        #norm_image = norm_image / np.std(norm_image)
        return norm_image
	
    def get_train_coords(self):
        return self._train_c
    
    def get_train_dataset(self):
        return self._train_f, self._train_l
    
    def plot_pre_processed_images(self, num_images):
        images = self._train_f
        labels = self._train_l
        fig, axes = plt.subplots(num_images, 2, figsize=(32, 32))
        axes = axes.ravel()
        #fig.tight_layout()
        for row in range(num_images):
            idx = random.randint(1, images.shape[0])
            image = images[idx]
            axes[row * 2 + 0].axis('off')
            axes[row * 2 + 0].imshow(image)
            axes[row * 2 + 0].set_title(self._sign_names[labels[idx]])
            axes[row * 2 + 1].axis('off')
            image = self._pre_process_image(image)
            image = image.squeeze()
            axes[row * 2 + 1].imshow(image, cmap='gray')
            axes[row * 2 + 1].set_title('gray/histequilized')
                
    def get_test_dataset(self):
        return self._test_f, self._test_l
  
    def get_generated_dataset(self):
        return self._augmented_f, self._augmented_l
	
    def get_train_validation_dataset(self):
        train_f, train_l = shuffle(self._train_f, self._train_l)
        train_f, validation_f, train_l, validation_l = \
            train_test_split(train_f, train_l, test_size=0.20, random_state=7)
        return train_f, train_l, validation_f, validation_l
	
    def get_augmented_train_validation_dataset(self):
        augmented_f, augmented_l = self.get_augmented_train_dataset()
        train_f, train_l = shuffle(self._augmented_f, self._augmented_l)
        train_f, validation_f, train_l, validation_l = \
            train_test_split(train_f, train_l, test_size=0.20, random_state=7)
        return train_f, train_l, validation_f, validation_l
	
    def get_pre_processed_train_dataset(self):
        train_f = self._pre_process_images(self._train_f)
        return train_f, self._train_l
    
    def get_pre_processed_test_dataset(self):
        test_f = self._pre_process_images(self._test_f)
        return test_f, self._test_l
    
    def get_pre_processed_train_validation_dataset(self):
        train_f = self._pre_process_images(self._train_f)
        train_f, train_l = shuffle(train_f, self._train_l)
        train_f, validation_f, train_l, validation_l = \
            train_test_split(train_f, train_l, test_size=0.20, random_state=7)
        return train_f, train_l, validation_f, validation_l
    
    def get_pre_processed_augmented_train_dataset(self):
        augmented_f = self._pre_process_images(self._augmented_f)
        train_f = self._pre_process_images(train_f)
        augmented_f = np.append(np.array(train_f), np.array(augmented_f), axis=0)
        augmented_l = np.append(np.array(self._train_l), np.array(augmented_l), axis=0)
        return augmented_f, augmented_l
	
    def get_pre_processed_augmented_train_validation_dataset(self):
        augmented_f, augmented_l = self.get_augmented_train_dataset()
        augmented_f = self._pre_process_images(augmented_f)
        train_f, train_l = shuffle(augmented_f, augmented_l)
        train_f, validation_f, train_l, validation_l = \
            train_test_split(train_f, train_l, test_size=0.20, random_state=7)
        return train_f, train_l, validation_f, validation_l
   
    def plot_train_images(self, num_images):        
        self._plot_images(self._train_f, self._train_l, num_images, 'train_images')
	
    def plot_test_images(self, num_images):        
        self._plot_images(self._test_f, self._test_l, num_images, 'test_images')
        
    def plot_generated_images(self, num_images):        
        #subplots_adjust(left=None, bottom=None, right=None,
        # top=None, wspace=None, hspace=None)
        #left = 0.125  # the left side of the subplots of the figure
        #right = 0.9  # the right side of the subplots of the figure
        #bottom = 0.1  # the bottom of the subplots of the figure
        #top = 0.9  # the top of the subplots of the figure
        #wspace = 0.2  # the amount of width reserved for blank space between subplots
        #hspace = 0.2  # the amount of height reserved for white space between subplots
        #fig.tight_layout()
        images = self._train_f
        labels = self._train_l
        coords = self._train_c
        fig, axes = plt.subplots(num_images, 6, figsize=(32, 32))
        axes = axes.ravel()
        fig.tight_layout()
        for row in range(num_images):
            idx = random.randint(1, images.shape[0])
            image = images[idx]
            axes[row * 6 + 0].axis('off')
            axes[row * 6 + 0].imshow(image)
            axes[row * 6 + 0].set_title(self._sign_names[labels[idx]])
	
            #scaled_image = self._scale_image(image, coords[idx])
            #axes[row * 6 + 1].axis('off')
            #axes[row * 6 + 1].imshow(scaled_image)
            #axes[row * 6 + 1].set_title('scaled')                
            for col in range(1,6):
                axes[row * 6 + col].axis('off')
                transform_image = self._transform_image(image, coords[idx])
                axes[row * 6 + col].imshow(transform_image)
                axes[row * 6 + col].set_title('shifted/rotated')
        plt.savefig('images/generated_images.png')
        
    def get_additional_images(self, dir_name):
        images = [cv2.imread(dir_name + "/" + name) for name in os.listdir(dir_name)]
        return np.asarray(images)
    
    def get_pre_processed_additional_images(self, dir_name):
        images = self.get_additional_images(dir_name)
        images = self._pre_process_images(images)
        return images
                
    def plot_additional_images(self, dir_name):
        images = self.get_additional_images(dir_name)
        rows, cols = TrafficImageDataSet.get_grid_dim(len(images))
        fig, axs = plt.subplots(rows, cols, figsize=(5, 5))
        #fig.subplots_adjust(hspace = .2, wspace=.001)
        fig.tight_layout()
        axs = axs.ravel()
        for i, image in enumerate(images):
            axs[i].axis('off')
            axs[i].imshow(image)
        plt.savefig('images/web_images.png')
                
    # For perspective transformation, 3x3 transformation matrix is required.
    # Straight lines will remain straight even after the transformation. For finding
    # the transformation matrix, we need 4 points on the input image and corresponding
    # points on the output image. Among these 4 points, 3 of them should not be
    # collinear. The transformation matrix can be found by the function
    # cv2.getPerspectiveTransform. Then apply cv2.warpPerspective with the 3x3
    # transformation matrix.
    def _scale_image(self, img, coords):
        rows, cols, _ = img.shape
        # transform limits
        px = np.random.randint(-2, 2)
        # ending locations
        #(coords[0],coords[1]),coords[2]-coords[0],coords[3]-coords[1]
        # x1,y1,x2,y2
        # [x1,y1], [y2 - y1, x1], [y1, x2-x1], [x2,y2]
        x1,y1,x2,y2 = coords[0], coords[1], coords[2], coords[3] 
        pts1 = np.float32([[x1, y1], [y2-y1, x1], [y1, x2-x1], [x2,y2]])
        #pts1 = np.float32([[px, px], [rows - px, px], [px, cols - px], [rows - px, cols - px]])
        # starting locations (4 corners)
        pts2 = np.float32([[0, 0], [rows, 0], [0, cols], [rows, cols]])
        # pts1 are scale to pts2.
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (rows, cols))
        #dst = dst[:, :, np.newaxis]
        return dst
	
    def _generate_augmented_train_dataset(self):
        labels = self._train_l.tolist()
        signs = [labels.count(y) for y in range(self._nclasses)]
        #print(signs)
        required_samples = [int(math.ceil(1000 / labels.count(y))) for y in range(self._nclasses)]
        #print(required_samples)
        augmented_f = []
        augmented_l = []
        for idx, label in enumerate(labels, start=0):
            if required_samples[label] > 1:
                #scaled_image = self._scale_image(self._train_f[idx], self._train_c[idx])
                #augmented_f.append(scaled_image)
                #augmented_l.append(label)
                for i in range(required_samples[label]):
                    augmented_f.append(self._transform_image(self._train_f[idx], self._train_c[idx]))
                    augmented_l.append(label)
        self._augmented_f = np.array(augmented_f)
        self._augmented_l = np.array(augmented_l)
	
    def get_augmented_train_dataset(self):
        augmented_f = np.append(np.array(self._train_f), np.array(self._augmented_f), axis=0)
        augmented_l = np.append(np.array(self._train_l), np.array(self._augmented_l), axis=0)
        return augmented_f, augmented_l
	
    def _pre_process_images(self, images):
        return np.array([self._pre_process_image(image) for image in images])
	
    def _pre_process_image(self, image):
        img = self.image_grayscale(image)
        img = cv2.equalizeHist(img)
        #img = self.image_normalize(img)
        img = img[..., np.newaxis]
        return img
	
    def _read_sign_names(self, signames_file):
        with open(signames_file) as f:
            reader = csv.DictReader(f)
            self._sign_names = [line['SignName'] for line in reader]
            
    def _transform_image(self, img, coords):
        #image = ndimage.interpolation.shift(img, 
        #                                    [random.randrange(-2, 2), 
        #                                     random.randrange(-2, 2), 0])
        #image = ndimage.interpolation.rotate(image, 
        #                                     random.randrange(-10, 10), 
        #                                     reshape=False)
        if random.choice([False, True]):
            image = ndimage.interpolation.shift(img, 
                                                [random.randrange(-2, 2), 
                                                 random.randrange(-2, 2), 0])
        else:
            image = ndimage.interpolation.rotate(img, 
                                                 random.randrange(-10, 10), 
                                                 reshape=False)
        return image
       
    def _plot_images(self, images, labels, num_images, save_file):
        offset = random.randint(1, images.shape[0] - num_images)
        rows, cols = TrafficImageDataSet.get_grid_dim(num_images)
        fig, axes = plt.subplots(rows, cols, figsize=(32, 32))
        axes = axes.ravel()
        fig.tight_layout()
        for i in range(num_images):
            idx = random.randint(1, images.shape[0])
            image = images[idx]
            axes[i].axis('off')
            axes[i].imshow(image)
            axes[i].set_title(self._sign_names[labels[idx]])            
        plt.savefig('images/' + save_file + '.png')
	
    def get_grid_dim(x):
        """
        Transforms x into product of two integers
        """
        factors = TrafficImageDataSet.prime_powers(x)
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
```

###1. Load the Data	

	dataset = TrafficImageDataSet('train.p', 'test.p', 'signnames.csv') 

Create TrafficImageDataSet object. The constructor loads the training, test pickle files, and also the signnames.csv file. Now the dataset object member functions is used to access the dataset, plot images, e.t.c

The pickled data is a dictionary with 4 key/value pairs:

* 'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
* 'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
* 'sizes' is a list containing tuples, (width, height) representing the the original width and height the image.
* 'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image.

The traffic sign classification mainly use only features and labels. I attempted using 'coords' values for generation scaled images, but scaled images didn't help in improving the model accuracies. 

###2. Summarize, Explore and Visualize the dataset

The German Traffic Sign Dataset consists of 39,209 32×32 px color images for training, and 12,630 images for testing. Each image is a photo of a traffic sign belonging to one of 43 classes, e.g. traffic sign types.

```python
# print the train and test dataset count
train_f, train_l = dataset.get_train_dataset()
print("Training examples {}".format(len(train_f)))
test_f, test_l = dataset.get_test_dataset()
print("Test examples {}".format(len(test_f)))
print("Image shape {}".format(dataset.image_shape()))
print("Number of traffic sign classes {}".format(dataset.nclasses()))
```
		
	Training examples 39209
	Test examples 12630
	Image shape (32, 32, 3)
	Number of traffic sign classes 43

Each image is a 32×32×3 array of pixel intensities, represented as [0, 255] integer values in RGB color space.  


```python
# plot random 20 images from the training data set
dataset.plot_train_images(20)
```
![alt text][image1]

Class of each image is encoded as an integer in a 0 to 42 range. The training dataset is not balanced across the classes.

```python
# plot training examples histogram 
dataset.plot_histogram(train_l, "Training examples")
```
![alt text][image2]

```python
# plot test examples histogram 
dataset.plot_histogram(test_l, 'Test Examples')
```
![alt text][image3]

###3. Design and Test a Model Architecture

#### Preprocess the data

I used grayscale images instead of color ones based on the recommendation in the Pierre Sermanet and Yann LeCun paper.

Looking at the sample images plot above, histogram equilization will be helpful, as it adjusts the image intensistes enhancing the image contrast that can improve feature extraction. 

I didn't used normalization as it worsened the model validation and test accuracy for some reason.

```python
def _pre_process_image(self, image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img = cv2.equalizeHist(img)
    img = img[..., np.newaxis]
    return img
    
### Preprocess the data here.
dataset.plot_pre_processed_images(10)    
```       
![alt text][image4]

```python
# print original and pre processed image shapes
train_f, _ = dataset.get_train_dataset()
pre_processed_train_f, _ = dataset.get_pre_processed_train_dataset()
print("orignal image shape {}".format(train_f[0].shape))
print("pre processed image shape {}".format(pre_processed_train_f[0].shape))
```

	orignal image shape (32, 32, 3)
	pre processed image shape (32, 32, 1)
	
#### Data Augmentation

The dataset we have is not sufficient for generalizing the model well. It is also fairly unbalanced from the training histogram above. Some traffic sign classes are having significantly lower samples to others. We can fix this with data augmentation.

The additional data is generated using random image rotation and shifts. Image scaling using the cooridnates in the data set wasn't of much help.

The data generating algorithm ensure every traffic sign class has atleast 1000 samples.

```python
def _generate_augmented_train_dataset(self):
    labels = self._train_l.tolist()
    signs = [labels.count(y) for y in range(self._nclasses)]
    required_samples = [int(math.ceil(1000 / labels.count(y))) for y in range(self._nclasses)]
    augmented_f = []
    augmented_l = []
    for idx, label in enumerate(labels, start=0):
        if required_samples[label] > 1:
            #scaled_image = self._scale_image(self._train_f[idx], self._train_c[idx])
            #augmented_f.append(scaled_image)
            #augmented_l.append(label)
            for i in range(required_samples[label]):
                augmented_f.append(self._transform_image(self._train_f[idx], self._train_c[idx]))
                augmented_l.append(label)
    self._augmented_f = np.array(augmented_f)
    self._augmented_l = np.array(augmented_l)

# plot generated images        
dataset.plot_generated_images(10)
```
![alt text][image5]

```python
# get generated training examples
generated_f, generated_l = dataset.get_generated_dataset()
print("Generated Training examples {}".format(len(generated_f)))
```

	Generated Training examples 31318
	
```python
# plot generated examples histogram
dataset.plot_histogram(generated_l, 'Generated Examples')
```
![alt text][image6]

```python
# get total (train + generated) training examples
train_f, train_l = dataset.get_augmented_train_dataset()
print("Total Training examples {}".format(len(train_f)))
```
	Total Training examples 70527
```
# plot generated examples histogram
dataset.plot_histogram(train_l, "Augmented training examples")
```
![alt text][image7]

```python
# print the dataset smaples count used for training, validation and test
x_train, _, x_validation, _ = dataset.get_pre_processed_augmented_train_validation_dataset()
x_test, _ = dataset.get_pre_processed_test_dataset()
print("training data {}".format(x_train.shape))
print("validation data {}".format(x_validation.shape))
print("test data {}".format(x_test.shape))
```
	training data (56421, 32, 32, 1)
	validation data (14106, 32, 32, 1)
	test data (12630, 32, 32, 1)

####Model Design


##### A neural network model class is designed for training the convolutional neural netowrk. The class supports the following functionalites.

1. functions for adding convolutional neural network layers
2. Training the convolutional neural network layers.
3. Evaluating the convolutional neural network layers.
4. plotting the training and validation accuracy.
5. plotting the confusion matrix for test data.
6. evlauating additional images.

```python
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
from sklearn.utils import shuffle
#from pandas_ml import ConfusionMatrix
from matplotlib.ticker import MultipleLocator

class Model(object):
    def __init__(self, input_shape, num_classes):
        tf.set_random_seed(0.0)
        self._x = tf.placeholder(tf.float32, (None,) + input_shape)
        self._nclasses = num_classes
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

    def conv2d(self, ksize, nfeatures, stride=1, dropout=False):

        if self._activation is not None:
            x = self._activation
        else:
            x = self._x

        channels = x.get_shape()[3].value
        print('convd layer input shape {}'.format(x.get_shape()))
        weights = tf.Variable(tf.truncated_normal(shape=[ksize, ksize, channels, nfeatures],
                                                  mean=0,
                                                  stddev=0.1))
        bias = tf.Variable(tf.zeros(nfeatures))

        conv = tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, bias)

        self._activation = tf.nn.relu(conv)

        self._conv_activations.append(tf.reshape(tf.reduce_max(self._activation, [0]), [-1]))

        if dropout:
            self._activation = tf.nn.dropout(self._activation,
                                             self._keep_prob_conv)

        print('convd layer output shape {}'.format(self._activation.get_shape()))
        print()

    def maxpool(self, ksize, stride):
        print('maxpool layer input shape {}'.format(self._activation.get_shape()))
        self._activation = tf.nn.max_pool(self._activation,
                                          ksize=[1, ksize, ksize, 1],
                                          strides=[1, stride, stride, 1],
                                          padding='VALID')
        print('maxpool layer output shape {}'.format(self._activation.get_shape()))
        print()

    def fc(self, nodes, batch_norm=False, dropout=False, act=tf.nn.relu):
        if self._activation is not None:
            x = self._activation
        else:
            x = self._x

        shape = x.get_shape()
        print('fully connected layer input shape {}'.format(shape))
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

        fc = tf.add(tf.matmul(x, weights), bias)

        if act is not None:
            self._activation = act(fc)
            self._dense_activations.append(tf.reshape(tf.reduce_max(self._activation, [0]), [-1]))
        else:
            self._activation = fc

        if dropout:
            self._activation = tf.nn.dropout(self._activation, self._keep_prob)


        print('fully connected layer output shape {}'.format(self._activation.get_shape()))
        print()

    def append_training_data(self, i, a, c):
        self._training_iterations.append(i)
        self._training_accuracy.append(a)
        self._training_loss.append(c)

    def append_validation_data(self, i, a, c):
        self._validation_iterations.append(i)
        self._validation_accuracy.append(a)
        self._validation_loss.append(c)

    def _batch_trainer(self, x_train, y_train, x_validation, y_validation,
                       epochs, batch_size, dropout, conv_dropout):
        self._saver = tf.train.Saver()

        with tf.Session() as self._sess:
            self._sess.run(tf.global_variables_initializer())
            current_epoch = 0
            dataset = DataSet(x_train, y_train, reshape=False)
            i = 0
            while epochs > dataset.epochs_completed:
                batch_x, batch_y = dataset.next_batch(batch_size)
                # Run optimization op (backprop)
                self._sess.run(self._optimizer,
                         feed_dict={self._x: batch_x,
                                    self._y: batch_y,
                                    self._batch_norm_test: False,
                                    self._keep_prob_conv: conv_dropout,
                                    self._keep_prob: dropout})
                i += 1
                
                # compute training accuracy every 20 iterations for visualisation
                if i % 20 == 0:
                    a, c, ca, da = self._sess.run(
                        [self._accuracy, self._cost, self._conv_activations, self._dense_activations],
                        {self._x: batch_x,
                         self._y: batch_y,
                         self._batch_norm_test: False,
                         self._keep_prob_conv: 1.0,
                         self._keep_prob: 1.0})
                    #print("{} : training accuracy {}, loss {}".format(i, a, c))      
                    self.append_training_data(i, a, c)

                # compute validation accuracy every 100 iterations for visualisation
                if i % 100 == 0:
                    a, c = self._evaluate(x_validation, y_validation, batch_size)
                    #print("{} : epoch {} validation accuracy {}, loss {}".format(i, dataset.epochs_completed, a, c))
                    self.append_validation_data(i, a, c)
                    
                if current_epoch != dataset.epochs_completed:
                    current_epoch = dataset.epochs_completed
                    a, c = self._evaluate(x_validation, y_validation, batch_size)
                    print("epoch {} validation accuracy {}, loss {}".format(dataset.epochs_completed, a, c))
                    
            print("\rOptimization Finished!")
            self._saver.save(self._sess, './lenet')

    def train(self,
              x_train,
              y_train,
              x_validation,
              y_validation,
              epochs=10,
              batch_size=128,
              dropout=0.75,
              conv_dropout=1.0):

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

        self._x_train = x_train
        self._y_train = y_train
        self._x_validation = x_validation
        self._y_validation = y_validation
        
        self._batch_trainer(x_train, y_train, x_validation, y_validation,
                            epochs, batch_size, dropout, conv_dropout)

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
            x_test, y_test = shuffle(x_test, y_test)
            pred_labels = sess.run(self._pred_labels,
                                   feed_dict={self._x: x_test[:4096],
                                              self._y: y_test[:4096],
                                              self._keep_prob_conv: 1.,
                                              self._keep_prob: 1.})

            cm = confusion_matrix(y_true=y_test[:4096], y_pred=pred_labels)
            print(cm)
            fig = plt.figure(fig_size=())
            ax = fig.add_subplot(111)
            cax = ax.matshow(cm)
            plt.title('Confusion matrix of Traffic Sign classifier')
            fig.colorbar(cax)
            ax.set_xticklabels([i for i in range(self._nclasses)])
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.set_yticklabels([i for i in range(self._nclasses)])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()
                
    def evaluate_additional_images(self, orig_images, images, labels):       
        with tf.Session() as sess:
            self._saver.restore(sess, tf.train.latest_checkpoint('.'))

            topk = tf.nn.top_k(self._pred_labels_sm, 5)

            pred_labels = sess.run(self._pred_labels_sm,
                                   feed_dict={self._x: images,
                                              self._keep_prob_conv: 1.,
                                              self._keep_prob: 1.})

            topk_pred = sess.run([self._pred_labels_sm, topk],
                                  feed_dict={self._x: images,
                                             self._keep_prob_conv: 1.,
                                             self._keep_prob: 1.})

            self.plot_top_k_predictions(topk_pred[1], orig_images, labels)
            return topk_pred[1]
    
    def plot_top_k_predictions(self, topk_pred, images, labels):

        for i, image in enumerate(images):
            # Prepare the grid
            plt.figure(figsize = (6, 2))
            gridspec.GridSpec(1, 2)

            # Plot original image
            plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
            plt.imshow(image.squeeze())
            plt.axis('off')

            # Plot predictions
            plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1)
            plt.barh(np.arange(5)+.5, 
                     topk_pred[0][i], 
                     align='center')
            plt.yticks(np.arange(5)+.5, 
                       labels[topk_pred[1][i].astype(int)])
            plt.tick_params(axis='both', 
                            which='both', 
                            labelleft='off', 
                            labelright='on', 
                            labeltop='off', 
                            labelbottom='off')

            plt.show()

            
    def plot_model(self):
        fig = plt.figure(figsize=(19.20,10.80), dpi=70)
        plt.gcf().canvas.set_window_title("Traffic Sign Classifer")
        fig.set_facecolor('#FFFFFF')

        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)

        ax1.set_title("Accuracy", y=1.02)
        ax2.set_title("Cross entropy loss", y=1.02)

        ax1.set_ylim(0, 1)  # important: not autoscaled
        ax2.autoscale(axis='y')
        #ax2.set_ylim(0, 5)  # important: not autoscaled

        line1, = ax1.plot(self._training_iterations, self._training_accuracy, label="training accuracy")
        line2, = ax1.plot(self._validation_iterations, self._validation_accuracy, label="validation accuracy")
        legend = ax1.legend(loc='lower right') # fancybox : slightly rounded corners
        legend.draggable(True)

        line3, = ax2.plot(self._training_iterations, self._training_loss, label="training loss")
        line4, = ax2.plot(self._validation_iterations, self._validation_loss, label="validation loss")
        legend = ax2.legend(loc='upper right') # fancybox : slightly rounded corners
        legend.draggable(True)

        line1.set_data(self._training_iterations, self._training_accuracy)
        line2.set_data(self._validation_iterations, self._validation_accuracy)
        line3.set_data(self._training_iterations, self._training_loss)
        line4.set_data(self._validation_iterations, self._validation_loss)

        plt.show()
```

#### Network Model

I started with the basic LeNet model and then tweaked the layers and features based on the following observations.

#####Observations:
* Basic LeNet model without data augmentation

	* Color images gave validaton accuracy of 97% and test accuracy of 92%.
	* Gray scale images gave validaton accuracy of 97% and test accuracy of 92%.
	* Gray scale with histogram equilization gave validation accuracy of 97% and test accuracy of 92%.
	* Gray scale with histogram equilization and normalization worsened the validation and test accuracy for some reason.
	
* Basic LeNet model with data augmentation
	* I haven't observed any improvement in validation and test accuracy with various pre processing method, so tried with data augmentation ensuring every traffic sign class has atleast 1000 samples. Out of my suprise I didn't observed any improvement. Even with data augmentation in 25 epochs the validation accuracy was 98% and test accuracy was 92%.
	
* Modified LeNet model with data augmentation
	* I tried multiple iterations by tweaking the conv layer feature maps and layers, fully connected layers and pooling layers. Finally, the following modified LeNet model resulted in validation accuracy of 99% and test accuracy of 94-95%.

                layer | neurons | weights | biases | parameters
		------| --------| --------| -------| ----------
		5x5 convolution (32x32x1 in, 32x32x32 out) | 32768 | 800 | 32 | 1600
		ReLU | | | |
		4x4 convolution (32x32x32 in, 16x16x64 out) | 16384 | 1024 | 64 | 1088
		ReLU | | | |
		3x3 convolution (16x16x64 in, 8x8x128 out) | 8192 | 1152 | 128 | 1280
		ReLU | | | |
		Flatten (8x8x128 -> 8192) | | | |
		Fully connected (8192 in, 512 out) | 512 | 4194304 | 512 | 4194816
		ReLU | | | |
		Dropout (0.75) | | | |
		Fully connected (512 in, 256 out) |256 | 131072 | 256 | 131328
		ReLU | | | |
		Dropout (0.75) | | | |
		Fully connected (256 in, 43 out) |43 | 11008 | 43 | 11051

	* The model requires total of 58155 neurons and 4341163 parameters.

#### Training Model

I used AdamOptimizer and experimented with various hyperparameters, including batch size, epochs and regularisation techniques to prevent overfitting. I  ended up using 15 epoches with batche size of 128 examples, and applying dropout to the fully connected layers. The learning rate used was 0.001.

```python
def ModifiedLeNet(dataset):
    # VA: 98.8 TA:94.5
    n_classes = dataset.nclasses()
    model = Model(input_shape=(32,32,1), num_classes=n_classes)
    model.conv2d(ksize=5, nfeatures=42)
    model.conv2d(ksize=4, nfeatures=64, stride=2)
    model.conv2d(ksize=3, nfeatures=128, stride=2)
    model.fc(nodes=512, dropout=True)
    model.fc(nodes=256, dropout=True)
    model.fc(nodes=n_classes, act=None)
    return model

model = ModifiedLeNet(dataset)
x_train, y_train, x_validation, y_validation = dataset.get_pre_processed_augmented_train_validation_dataset()
model.train(x_train, y_train, x_validation, y_validation, epochs=15)
model.plot_model()
```        

	xshape (?, 32, 32, 1)

	convd layer input shape (?, 32, 32, 1)
	convd layer output shape (?, 32, 32, 32)
	
	convd layer input shape (?, 32, 32, 32)
	convd layer output shape (?, 16, 16, 64)
	
	convd layer input shape (?, 16, 16, 64)
	convd layer output shape (?, 8, 8, 128)
	
	fully connected layer input shape (?, 8, 8, 128)
	fully connected layer output shape (?, 512)
	
	fully connected layer input shape (?, 512)
	fully connected layer output shape (?, 256)
	
	fully connected layer input shape (?, 256)
	fully connected layer output shape (?, 43)
	
	epoch 1 validation accuracy 0.9414433575041595, loss 18.58771672080497
	epoch 2 validation accuracy 0.9693747341556784, loss 12.179819780114553
	epoch 3 validation accuracy 0.9828441797816532, loss 8.47104036573077
	epoch 4 validation accuracy 0.9849709343541755, loss 6.74870127001739
	epoch 5 validation accuracy 0.9868141216503615, loss 7.4124967901821295
	epoch 6 validation accuracy 0.9862469870976889, loss 8.06208544917091
	epoch 7 validation accuracy 0.9834113143343258, loss 10.572774684667486
	epoch 8 validation accuracy 0.989933361690061, loss 6.342123677892082
	epoch 9 validation accuracy 0.9904296044236495, loss 6.7263423507135505
	epoch 10 validation accuracy 0.9870267971076138, loss 7.703916981636179
	epoch 11 validation accuracy 0.9894371189564725, loss 7.282206263171062
	epoch 12 validation accuracy 0.9908549553381539, loss 6.441632891692592
	epoch 13 validation accuracy 0.9901460371473132, loss 6.385471919944495
	epoch 14 validation accuracy 0.990004253509145, loss 7.28184727731367
	epoch 15 validation accuracy 0.9929108180915922, loss 5.938216350508906
	Optimization Finished!

#### Model Plot

![alt text][image8]

#### Evaluate the model
```python	
# run the trained model on the test dataset
x_test, y_test = dataset.get_pre_processed_test_dataset()
model.evaluate(x_test, y_test)
```
	Test Accuracy = 0.952 Test Loss = 52.042
	
#### Evalute the model with new images

Collected images from the internet that are easily distinguishable from the original dataset. The images are bit brighter and have different colors that the model was not trained on.	

```python
### Load the images and plot them here.
dataset.plot_additional_images('./web_images')
```
![alt text][image9]

```python
# Evaluate the model on the collected new images
x_web_images = dataset.get_pre_processed_additional_images('./web_images')
y_web_images = [12,3,25,34,1,18,11,30]
model.evaluate(x_web_images, y_web_images)
```
	Test Accuracy = 0.625 Test Loss = 2204.618

The model predicted 6 out of 8 new images correctly with 100% softmax probability. The softmax probability visualization is captured below.

```
### Visualize the softmax probabilities here.
pred = model.evaluate_additional_images(dataset.get_additional_images('./web_images'), 
                                        dataset.get_pre_processed_additional_images('./web_images'), 
                                        dataset.sign_names())
```                                        

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]
