import pickle
from Model import Model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import ImageDataSet

def LeNet():
    dataset = ImageDataSet.DataSet('train.p', 'test.p')
    n_classes = dataset.nclasses()
    x_train, y_train, x_validation, y_validation = dataset.get_pre_processed_train_validation_dataset()
    x_test, y_test = dataset.get_pre_processed_test_dataset()

    print('nclasses {}'.format(n_classes))
    lenet = Model(input_shape=(32,32,1), num_classes=n_classes)
    lenet.conv2d(ksize=5, nfeatures=6)
    lenet.maxpool(ksize=2, stride=2)
    lenet.conv2d(ksize=5, nfeatures=16)
    lenet.maxpool(ksize=2, stride=2)
    lenet.fc(nodes=120, dropout=True)
    lenet.fc(nodes=84, dropout=True)
    lenet.fc(nodes=n_classes, act=None)
    lenet.train_and_plot(x_train, y_train, x_validation, y_validation)
    #lenet.train(x_train, y_train, x_validation, y_validation)
    #lenet.evaluate(x_test, y_test)

def LeNetBn():
    # this is not improving the accuracy. the validation accuracy stuck at 90.
    dataset = ImageDataSet.DataSet('train.p', 'test.p')
    n_classes = dataset.nclasses()
    x_train, y_train, x_validation, y_validation = dataset.get_pre_processed_train_validation_dataset()
    x_test, y_test = dataset.get_pre_processed_test_dataset()

    print('nclasses {}'.format(n_classes))
    lenet = Model(input_shape=(32,32,1), num_classes=n_classes)
    lenet.conv2d(ksize=5, nfeatures=6, batch_norm=False, dropout=False)
    lenet.maxpool(ksize=2, stride=2)
    lenet.conv2d(ksize=5, nfeatures=16, batch_norm=False, dropout=False)
    lenet.maxpool(ksize=2, stride=2)
    lenet.fc(nodes=120, batch_norm=True, dropout=True)
    lenet.fc(nodes=84, batch_norm=True, dropout=True)
    lenet.fc(nodes=n_classes, batch_norm=False, act=None, dropout=False)
    lenet.train_and_plot(x_train, y_train, x_validation, y_validation)
    #lenet.train(x_train, y_train, x_validation, y_validation)
    #lenet.evaluate(x_test, y_test)

def LeNetWithAugData_1():
    # VA: 98.8 TA:93.3
    dataset = ImageDataSet.DataSet('train.p', 'test.p')
    n_classes = dataset.nclasses()
    x_train, y_train, x_validation, y_validation = dataset.get_aug_train_validation_dataset()
    x_test, y_test = dataset.get_pre_processed_test_dataset()

    print('nclasses {}'.format(n_classes))
    lenet = Model(input_shape=(32,32,1), num_classes=n_classes)
    #lenet.conv2d(ksize=1, nfeatures=3)
    lenet.conv2d(ksize=5, nfeatures=24)
    #lenet.maxpool(ksize=2, stride=2)
    lenet.conv2d(ksize=4, nfeatures=48, stride=2)
    #lenet.maxpool(ksize=2, stride=2)
    lenet.conv2d(ksize=3, nfeatures=96, stride=2)
    lenet.fc(nodes=512, batch_norm=True, dropout=True)
    lenet.fc(nodes=256, batch_norm=True, dropout=True)
    lenet.fc(nodes=n_classes, batch_norm=False, act=None, dropout=False)
    #lenet.train_and_plot(x_train, y_train, x_validation, y_validation)
    lenet.train(x_train, y_train, x_validation, y_validation, epochs=15)
    lenet.plot_model()
    lenet.evaluate(x_test, y_test)

def LeNetWithAugData_2():
    # VA: 98.8 TA:93.3
    dataset = ImageDataSet.DataSet('train.p', 'test.p')
    n_classes = dataset.nclasses()
    x_train, y_train, x_validation, y_validation = dataset.get_aug_train_validation_dataset()
    x_test, y_test = dataset.get_pre_processed_test_dataset()

    print('nclasses {}'.format(n_classes))
    lenet = Model(input_shape=(32,32,1), num_classes=n_classes)
    lenet.conv2d(ksize=1, nfeatures=3)
    lenet.conv2d(ksize=5, nfeatures=24, batch_norm=False, dropout=False)
    #lenet.maxpool(ksize=2, stride=2)
    lenet.conv2d(ksize=4, nfeatures=48, batch_norm=False, dropout=False, stride=2)
    #lenet.maxpool(ksize=2, stride=2)
    #lenet.conv2d(ksize=3, nfeatures=64, batch_norm=False, dropout=False, stride=2)
    #lenet.fc(nodes=512, batch_norm=False, dropout=True)
    lenet.fc(nodes=256, batch_norm=False, dropout=True)
    lenet.fc(nodes=n_classes, batch_norm=False, act=None, dropout=False)
    lenet.train_and_plot(x_train, y_train, x_validation, y_validation)
    #lenet.train(x_train, y_train, x_validation, y_validation)
    #lenet.evaluate(x_test, y_test)

def FlatNet():
    dataset = ImageDataSet.DataSet('train.p', 'test.p')
    n_classes = dataset.nclasses()
    x_train, y_train, x_validation, y_validation = dataset.get_aug_train_validation_dataset()
    x_test, y_test = dataset.get_pre_processed_test_dataset()

    lenet = Model(input_shape=(32,32,1), num_classes=n_classes)
    #lenet.fc(nodes=1024)
    lenet.fc(nodes=512)
    lenet.fc(nodes=n_classes, act=None)
    lenet.train_and_plot(x_train, y_train, x_validation, y_validation)
    #lenet.evaluate(x_test, y_test)


def main():
    #LeNet()
    LeNetWithAugData_1()
    #FlatNet()
    
if __name__ == "__main__":
    main()
