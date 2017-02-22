import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
import cv2
import math
from scipy import ndimage
import random

class DataSet(object):
    def __init__(self, train_p, test_p):
        with open(train_p, mode='rb') as f:
            self._train = pickle.load(f)
        with open(test_p, mode='rb') as f:
            self._test = pickle.load(f)
        self._train_f = self._train['features']
        self._train_l = self._train['labels']
        self._test_f = self._test['features']
        self._test_l = self._test['labels']
        self._nclasses = len(np.unique(self._train_l))

    def nclasses(self):
        return self._nclasses

    def grayscale(self, images):
        gray_images = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images])
        return gray_images

    def pre_process_images(self, images):
        return np.array([self.pre_process_image(image) for image in images])

    def normalize(self, images):
        norm_images = (images - np.mean(images))
        norm_images = norm_images / np.std(norm_images)
        return norm_images

    def image_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def pre_process_image(self, image):
        img = self.image_grayscale(image)
        img = cv2.equalizeHist(img)
        img = img[..., np.newaxis]
        return img

    def image_normalize(self, image):
        norm_image = (image - np.mean(image))
        norm_image = norm_image / np.std(norm_image)
        return norm_image

    def augment_brightness_camera_images(image):
        image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        random_bright = .25 + np.random.uniform()
        # print(random_bright)
        image1[:, :, 2] = image1[:, :, 2] * random_bright
        image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
        return image1

    def transform_image(self, img):
        if (random.choice([True, False])):
            image = ndimage.interpolation.shift(img, [random.randrange(-2, 2), random.randrange(-2, 2), 0])
        else:
            image = ndimage.interpolation.rotate(img, random.randrange(-10, 10), reshape=False)
        image = self.pre_process_image(image)    
        return image

    def get_train_validation_dataset(self):
        train_f, train_l = shuffle(self._train_f, self._train_l)
        train_f, validation_f, train_l, validation_l = \
            train_test_split(train_f, train_l, test_size=0.20, random_state=7)
        return train_f, train_l, validation_f, validation_l

    def get_pre_processed_train_validation_dataset(self):
        train_f = self.pre_process_images(self._train_f)
        train_f, train_l = shuffle(train_f, self._train_l)
        train_f, validation_f, train_l, validation_l = \
            train_test_split(train_f, train_l, test_size=0.20, random_state=7)
        return train_f, train_l, validation_f, validation_l

    def get_train_dataset(self):
        return self._train_f, self._train_l

    def get_pre_processed_test_dataset(self):
        test_f = self.pre_process_images(self._test_f)
        return test_f, self._test_l
    
    def get_test_dataset(self):
        return self._test_f, self._test_l

    def get_aug_train_dataset(self):
        labels = self._train_l.tolist()
        signs = [labels.count(y) for y in range(self._nclasses)]
        required_samples = [int(math.ceil(1000 / labels.count(y))) for y in range(self._nclasses)]
        augmented_f = []
        augmented_l = []
        for idx, label in enumerate(labels, start=0):
            if required_samples[label] > 1:
                for i in range(required_samples[label]):
                    augmented_f.append(self.transform_image(self._train_f[idx]))
                    augmented_l.append(label)
        train_f = self.pre_process_images(self._train_f)
        print("train shape {}".format(train_f.shape))
        augmented_f = np.append(np.array(train_f), np.array(augmented_f), axis=0)
        augmented_l = np.append(np.array(self._train_l), np.array(augmented_l), axis=0)
        # print("Generated augmented samples", len(augmented_samples))
        # print("new data set", x_train_augmented.shape)
        return augmented_f, augmented_l

    def get_aug_train_validation_dataset(self):
        augmented_f, augmented_l = self.get_aug_train_dataset()
        print("aug_f shape {}".format(augmented_f.shape))
        train_f, train_l = shuffle(augmented_f, augmented_l)
        train_f, validation_f, train_l, validation_l = \
            train_test_split(train_f, train_l, test_size=0.20, random_state=7)
        return train_f, train_l, validation_f, validation_l
