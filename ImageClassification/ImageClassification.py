import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
#For returning the history
from tensorflow.keras import callbacks

# Download the data and explore the dataset
import pathlib
import os, ssl

#To the Save the training History as aphysical file.
import pandas as pd

class ImageClassification:

    #The basic Model.
    crnt_model = None
    #Improved model.
    improved_model = None


    def __init__(self, class_names, image_height, image_weight):
        #Code will be added later.
        self.class_name = class_names
        self.img_height = image_height
        self.img_weight = image_weight


    def get_files_fromstorage(self):
        """
        This function collects the files from the google storage api.
        :return: The downloaded file path.
        """
        path_downloaded_file = None
        try:
            #The code snippet to avoid the SSL  error
            if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
                getattr(ssl, '_create_unverified_context', None)):
                ssl._create_default_https_context = ssl._create_unverified_context


            dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
            data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
            path_downloaded_file = pathlib.Path(data_dir)
            print("The file downloaded : {0}".format(path_downloaded_file))
            image_count = len(list(path_downloaded_file.glob('*/*.jpg')))
            print("The image count {0}".format(image_count))

        except Exception as exc:
            print("Caught in the exception at get_files_fromstorage {0}".format(exc))
        return path_downloaded_file

    def display_images_to_validate(self):
        try:
            pass
        except Exception as exc:
            print("Caught in the exception at display_images_to_validate {0}".format(exc))

    def create_dataset(self, data_dir):
        batch_size = 32
        img_height = 180
        img_width = 180
        class_names = None
        try:
            training_ds = tf.keras.preprocessing.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset='training',
                seed=123,
                image_size= (img_height, img_width),
                batch_size=batch_size)


            validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset='validation',
                seed=123,
                image_size=(img_height, img_width),
                batch_size=batch_size)

            self.class_names = training_ds.class_names

            print("Displaying the class names {0}:".format(self.class_names))
            print("No of class names {0}".format(len(self.class_names)))

            #Confiure the dataset for performance.
            AUTO_TUNE = tf.data.experimental.AUTOTUNE

            training_ds = training_ds.cache().shuffle(1000).prefetch(buffer_size=AUTO_TUNE)
            validation_ds = validation_ds.cache().prefetch(buffer_size=AUTO_TUNE)

        except Exception as exc:
            print("Caught in the exception at create_dataset {0}".format(exc))

        return training_ds, validation_ds, len(self.class_names)


    def standardize_data(self):
        try:
            print("Standardize data")
            pass
        except Exception as exc:
            print("Caught in the exception at display_images_to_validate {0}".format(exc))


    def ceate_model_and_run_fit(self, num_of_classes, train_ds, validate_ds):
        img_height = 180
        img_width = 180
        no_of_epochs = 5
        history = callbacks.History()

        try:
            ImageClassification.crnt_model = tf.keras.models.Sequential([
                layers.experimental.preprocessing.Rescaling(1/.255, input_shape=(img_height,img_width, 3)),
                layers.Conv2D(16,3,padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(num_of_classes)
                ])
            # Compile the model
            ImageClassification.crnt_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
            ImageClassification.crnt_model.summary()

            print(ImageClassification.crnt_model)

            print("Starting the training ....")
            history =ImageClassification.crnt_model.fit(
                train_ds,
                validation_data=validate_ds,
                epochs=no_of_epochs
            )

            # Save the trained model
            ImageClassification.crnt_model.save("image_classification")

            # Save the model training to physical file.
            hist_df = pd.DataFrame(history.history)
            self.save_history_as_file(hist_df)

            #Plot the history value
            #self.plot_training_results(history, no_of_epochs)

        except Exception as exc:
            print("Caught in the exception at display_images_to_validate {0}".format(exc))



    def create_improved_model(self, num_of_classes, train_ds, validate_ds, no_of_epochs):
        """
        This model will have the ImageAugmentation.
        :return:
        """
        try:
            data_augmentation = keras.Sequential([
                layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(self.img_height, self.img_weight,3)),
                layers.experimental.preprocessing.RandomRotation(0.1),
                layers.experimental.preprocessing.RandomZoom(0.1)
            ])

            ImageClassification.improved_model = tf.keras.models.Sequential([
                data_augmentation,
                layers.experimental.preprocessing.Rescaling(1/.255),
                layers.Conv2D(16, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Dropout(0.2),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(len(self.class_names))
            ])

            #Compile and train the model.
            ImageClassification.improved_model.compile(optimizer='adam',
                                                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                                       metrics=['accuracy']
                                                       )

            ImageClassification.improved_model.summary()


        except Exception as exc:
            print("Caught in the unexpected exception at create_improved_model {0} ".format(exc))



    def plot_training_results(self, crnt_run_hist, no_of_epochs):

        try:
            acc = crnt_run_hist.history['accuracy']
            val_acc = crnt_run_hist.history['val_accuracy']

            loss = crnt_run_hist.history['loss']
            val_loss = crnt_run_hist.history['val_loss']

            epochs_range = range(no_of_epochs)

            plt.figure(figsize=(8,8))
            plt.subplot(1,2,1)

            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')

            plt.subplot(1, 2, 2)

            plt.plot(epochs_range, acc, label='Training Loss')
            plt.plot(epochs_range, val_acc, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')

            plt.show()

        except Exception as exc:
            print("Caught in the exception at plot_training_results {0} ".format(exc))


    def save_history_as_file(self, df_hist):

        try:
            print("The histry has been saved as csv file")
            #Save as a csv file
            hist_csv_file = '/Users/vvsukumarsundararajan/PyCharmTF2xCoding/TF2XImages/ImageClassification/MiscFiles/image_clsfcn.csv'
            with open(hist_csv_file, mode='w') as f:
                df_hist.to_csv(hist_csv_file)

        except Exception as exc:
            print("Caught in the exception at save_history_as_file {0} ".format(exc))

    def predict_on_new_data(self):
        #convert it as class variables.
        img_height = 180
        img_width = 180

        try:
            # The code snippet to avoid the SSL  error
            if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
                    getattr(ssl, '_create_unverified_context', None)):
                ssl._create_default_https_context = ssl._create_unverified_context

            sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
            sunflower_path = tf.keras.utils.get_file("Red_sunflower", origin=sunflower_url)

            img = keras.preprocessing.image.load_img(
                sunflower_path, target_size=(img_height, img_width)
            )

            img_array =keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            #Load the saved model.
            restored_model = tf.keras.models.load_model("image_classification")
            predictions = restored_model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence."
                    .format(self.class_names[np.argmax(score)], 100 * np.max(score))
            )

        except Exception as exc:
            print("Caught in the unexpected exception at predict_on_new_data {0}".format(exc))





