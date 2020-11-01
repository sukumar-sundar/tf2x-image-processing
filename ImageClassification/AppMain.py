
from TF2XImages.ImageClassification.ImageClassification import ImageClassification


def image_classification_caller(name):
    try:
        print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

        class_names = []
        #required image sizes
        imc = ImageClassification(class_names, image_height=180, image_weight=180)
        #Step 1: Download the images from Google Storage.
        # path_downloaded_file = ImageClassification.get_files_fromstorage()
        path_downloaded_file = "/Users/vvsukumarsundararajan/.keras/datasets/flower_photos"

        # Create the training dataset and validation dataset.
        traing_ds, validation_ds, num_of_classes = imc.create_dataset(path_downloaded_file)

        # Create the model
        #imc.ceate_model_and_run_fit(num_of_classes, traing_ds, validation_ds)

        #Improve the model.
        imc.create_improved_model(num_of_classes, traing_ds, validation_ds, no_of_epochs=10)


        #Validate the model
        #imc.predict_on_new_data()

    except Exception as exc:
        print("Caught in the unexpected exception at image_classification_caller {0}".format(exc))




# Start from here.
if __name__ == '__main__':
    image_classification_caller('Calling the image classification function...')