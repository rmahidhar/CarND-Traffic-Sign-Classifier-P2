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
        #image[:,:,0] = cv2.equalizeHist(image[:,:,0])
        #image[:,:,1] = cv2.equalizeHist(image[:,:,1])
        #image[:,:,2] = cv2.equalizeHist(image[:,:,2])
        #image = image/255. - .5
        img = self.image_grayscale(image)
        img = cv2.equalizeHist(img)
        img = img[..., np.newaxis]
        #img = self.image_normalize(img)
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

    def generate_variant_image(self, img):
        if (random.choice([True, False])):
            image = ndimage.interpolation.shift(img, [random.randrange(-2, 2), random.randrange(-2, 2), 0])
        else:
            image = ndimage.interpolation.rotate(img, random.randrange(-10, 10), reshape=False)
        image = self.pre_process_image(image)    
        return image

    def transform_image(self, img, ang_range=20, shear_range=10, trans_range=5):
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
        ang_rot = np.random.uniform(ang_range) - ang_range / 2
        rows, cols, ch = img.shape
        rows = img.shape[0]
        cols = img.shape[1]
        Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

        # Translation
        tr_x = trans_range * np.random.uniform() - trans_range / 2
        tr_y = trans_range * np.random.uniform() - trans_range / 2
        Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

        # Shear
        pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
        pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
        pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

        pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

        shear_M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, Rot_M, (cols, rows))
        img = cv2.warpAffine(img, Trans_M, (cols, rows))
        img = cv2.warpAffine(img, shear_M, (cols, rows))

        img = self.pre_process_image(img)

        # Brightness
        #if random.choice([True, False]):
        #    img = augment_brightness_camera_images(img)

        # print(img.shape)
        # img.reshape([-1, rows, cols, 1])
        return img

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
                    #augmented_f.append(self.transform_image(self._train_f[idx]))
                    augmented_f.append(self.generate_variant_image(self._train_f[idx]))
                    augmented_l.append(label)
                    # print(i, idx)
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
