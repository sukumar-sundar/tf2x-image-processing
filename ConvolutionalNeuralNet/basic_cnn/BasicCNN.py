import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras import models, layers, datasets
import matplotlib.pyplot as plt
# Download the data and explore the dataset
import pathlib
import os, ssl

class BasicCNN():

    def __init__(self):
        pass

    def collectCIFAR10Data(self):
        try:
            # The code snippet to avoid the SSL  error
            if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
                    getattr(ssl, '_create_unverified_context', None)):
                ssl._create_default_https_context = ssl._create_unverified_context

            (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar10.load_data()
            #normalize the pixel values to be 0 and 1.
            self.train_images, self.test_images = self.train_images/255.0 , self.test_images/255.0

            print(len(self.train_images))

        except Exception as exc:
            print("Caught in the unexpected exception at collectCIFAR10Data {0} ".format(exc))

    def verifyData(self):
        try:
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']

            plt.figure(figsize=(10,10))
            for i in range(25):
                plt.subplot(5,5,i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(self.train_images[i], cmap=plt.cm.binary)
                plt.xlabel(class_names[self.train_labels[i][0]])
            plt.show()


        except Exception as exc:
            print("Caught in the unexpected exception at verifyData {0} ".format(exc))

    def create_convolutional_model(self):
        try:
            self.crnt_model = models.Sequential()
            self.crnt_model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3)))
            self.crnt_model.add(layers.MaxPooling2D(2,2))
            self.crnt_model.add(layers.Conv2D(64,(3,3), activation='relu'))
            self.crnt_model.add(layers.MaxPooling2D(2,2))
            self.crnt_model.add(layers.Conv2D(64,(3,3), activation='relu'))
            self.crnt_model.summary()
        except Exception as exc:
            print("Caught in the exception at create_convolutional_model {0}".format(exc))

